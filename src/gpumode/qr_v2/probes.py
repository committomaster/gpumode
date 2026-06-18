"""Semantic structure and work-reduction probes for qr_v2 inputs."""

import torch

from gpumode.qr_v2.check import factor_rtol, matrix_one_norm
from gpumode.qr_v2.inputs import generate_from_spec
from gpumode.qr_v2.specs import QrV2Spec


EARLY_STOP_FRACTIONS = (0.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 3.0 / 4.0, 1.0)
EARLY_STOP_PANELS = (16, 32, 64, 128, 256)


def _relative(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(numerator, dtype=torch.float64)
    infs = torch.full_like(numerator, float("inf"), dtype=torch.float64)
    return torch.where(denominator > 0, numerator / denominator, torch.where(numerator == 0, zeros, infs))


def _per_matrix_float_list(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values.detach().cpu().reshape(-1)]


def _per_matrix_int_list(values: torch.Tensor) -> list[int]:
    return [int(value) for value in values.detach().cpu().reshape(-1)]


def _per_matrix_bool_list(values: torch.Tensor) -> list[bool]:
    return [bool(value) for value in values.detach().cpu().reshape(-1)]


def _mean(values: list[float] | list[int] | list[bool]) -> float | None:
    if not values:
        return None
    return float(sum(values)) / len(values)


def _quantile(values: list[float] | list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sum_squares(n: int) -> float:
    return float(n * (n + 1) * (2 * n + 1) // 6)


def _early_stop_work_saved(n: int, stop_k: int) -> float:
    total = _sum_squares(n)
    if total == 0.0:
        return 0.0
    remaining = _sum_squares(max(n - stop_k, 0))
    return remaining / total


def _stop_grid(n: int) -> list[int]:
    stops = {0, n}
    stops.update(max(0, min(n, round(n * fraction))) for fraction in EARLY_STOP_FRACTIONS)
    for panel in EARLY_STOP_PANELS:
        stops.update(range(panel, n, panel))
    return sorted(stops)


def _route_classes(
    approx_upper: list[bool],
    early_stop_k: list[int],
    zero_tau_suffix_count: list[int],
    n: int,
) -> list[str]:
    routes: list[str] = []
    for index, is_approx_upper in enumerate(approx_upper):
        if is_approx_upper:
            routes.append("upper")
        elif early_stop_k and early_stop_k[index] <= max(1, n // 4):
            routes.append("early_stop")
        elif zero_tau_suffix_count and zero_tau_suffix_count[index] >= max(1, n // 8):
            routes.append("zero_tail")
        else:
            routes.append("dense")
    return routes


def _class_counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _suffix_true_counts(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.int64).flip(-1).cumprod(dim=-1).sum(dim=-1)


def _effective_rank_probe(a: torch.Tensor) -> dict[str, object]:
    singular_values = torch.linalg.svdvals(a)
    max_singular = singular_values.amax(dim=-1)
    tolerance = torch.finfo(torch.float32).eps * max(a.shape[-1], 1) * max_singular
    rank = (singular_values > tolerance[:, None]).sum(dim=-1)
    frobenius_sq = a.to(torch.float64).square().sum(dim=(-2, -1))
    stable_rank = torch.where(
        max_singular > 0,
        frobenius_sq / max_singular.to(torch.float64).square(),
        torch.zeros_like(frobenius_sq),
    )
    rank_list = _per_matrix_int_list(rank)
    rank_fraction = [float(value) / max(a.shape[-1], 1) for value in rank_list]
    stable_rank_list = _per_matrix_float_list(stable_rank)
    return {
        "rank_probe_evaluated": True,
        "effective_rank_proxy": rank_list,
        "effective_rank_fraction": rank_fraction,
        "effective_rank_median": _quantile(rank_list, 0.5),
        "effective_rank_fraction_median": _quantile(rank_fraction, 0.5),
        "stable_rank_proxy": stable_rank_list,
        "stable_rank_median": _quantile(stable_rank_list, 0.5),
    }


def _skipped_rank_probe() -> dict[str, object]:
    return {
        "rank_probe_evaluated": False,
        "effective_rank_proxy": [],
        "effective_rank_fraction": [],
        "effective_rank_median": None,
        "effective_rank_fraction_median": None,
        "stable_rank_proxy": [],
        "stable_rank_median": None,
    }


def _early_stop_probe(a: torch.Tensor, norm1: torch.Tensor) -> dict[str, object]:
    n = a.shape[-1]
    h, tau = torch.geqrf(a)
    zero_tau = tau == 0
    zero_tau_count = _per_matrix_int_list(zero_tau.sum(dim=-1))
    zero_tau_suffix_count = _per_matrix_int_list(_suffix_true_counts(zero_tau))
    zero_tau_rate = [count / max(n, 1) for count in zero_tau_count]
    zero_tau_suffix_rate = [count / max(n, 1) for count in zero_tau_suffix_count]

    grid = _stop_grid(n)
    a64 = a.to(torch.float64)
    threshold = factor_rtol(n) * norm1
    stop_k = torch.full((a.shape[0],), n, device=a.device, dtype=torch.int64)
    found = torch.zeros((a.shape[0],), device=a.device, dtype=torch.bool)
    pass_rate_by_k: dict[str, float] = {}

    for k in grid:
        masked_tau = torch.zeros_like(tau)
        if k > 0:
            masked_tau[:, :k] = tau[:, :k]
        q = torch.linalg.householder_product(h, masked_tau)
        qt_a = q.to(torch.float64).transpose(-1, -2) @ a64
        lower_leakage = matrix_one_norm(torch.tril(qt_a, diagonal=-1))
        passed = lower_leakage <= threshold
        pass_rate_by_k[str(k)] = float(passed.to(torch.float32).mean().item())
        newly_found = passed & ~found
        stop_k = torch.where(newly_found, torch.full_like(stop_k, k), stop_k)
        found |= passed

    stop_k_list = _per_matrix_int_list(stop_k)
    work_saved = [_early_stop_work_saved(n, value) for value in stop_k_list]
    remaining_work = [1.0 - value for value in work_saved]
    mean_remaining = _mean(remaining_work)
    speedup_ceiling = None if mean_remaining is None else 1.0 / max(mean_remaining, 1.0e-9)
    n4_rate = sum(1 for value in stop_k_list if value <= max(1, n // 4)) / len(stop_k_list)
    n2_rate = sum(1 for value in stop_k_list if value <= max(1, n // 2)) / len(stop_k_list)

    return {
        "early_stop_evaluated": True,
        "early_stop_grid": grid,
        "early_stop_pass_rate_by_k": pass_rate_by_k,
        "early_stop_k": stop_k_list,
        "early_stop_k_fraction": [value / max(n, 1) for value in stop_k_list],
        "early_stop_median_k": _quantile(stop_k_list, 0.5),
        "early_stop_p90_k": _quantile(stop_k_list, 0.9),
        "early_stop_max_k": max(stop_k_list) if stop_k_list else None,
        "early_stop_n4_rate": n4_rate,
        "early_stop_n2_rate": n2_rate,
        "early_stop_work_saved": work_saved,
        "early_stop_work_saved_mean": _mean(work_saved),
        "estimated_speedup_ceiling": speedup_ceiling,
        "zero_tau_reflector_count": zero_tau_count,
        "zero_tau_reflector_rate": zero_tau_rate,
        "zero_tau_reflector_rate_mean": _mean(zero_tau_rate),
        "zero_tau_suffix_count": zero_tau_suffix_count,
        "zero_tau_suffix_rate": zero_tau_suffix_rate,
        "zero_tau_suffix_median": _quantile(zero_tau_suffix_count, 0.5),
    }


def _skipped_early_stop_probe() -> dict[str, object]:
    return {
        "early_stop_evaluated": False,
        "early_stop_grid": [],
        "early_stop_pass_rate_by_k": {},
        "early_stop_k": [],
        "early_stop_k_fraction": [],
        "early_stop_median_k": None,
        "early_stop_p90_k": None,
        "early_stop_max_k": None,
        "early_stop_n4_rate": None,
        "early_stop_n2_rate": None,
        "early_stop_work_saved": [],
        "early_stop_work_saved_mean": None,
        "estimated_speedup_ceiling": None,
        "zero_tau_reflector_count": [],
        "zero_tau_reflector_rate": [],
        "zero_tau_reflector_rate_mean": None,
        "zero_tau_suffix_count": [],
        "zero_tau_suffix_rate": [],
        "zero_tau_suffix_median": None,
    }


def probe_tensor(
    a: torch.Tensor,
    *,
    early_stop_max_n: int = 512,
    rank_probe_max_n: int = 512,
) -> dict[str, object]:
    """Compute per-matrix structure and semantic work-reduction features for a batch."""

    if a.ndim != 3 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"expected a batch of square matrices, got {a.shape}")
    if a.dtype != torch.float32:
        raise ValueError(f"expected torch.float32 input, got {a.dtype}")

    n = a.shape[-1]
    a64 = a.to(torch.float64)
    abs_a = a64.abs()
    norm1 = matrix_one_norm(a64)
    strictlower_norm1 = matrix_one_norm(torch.tril(a64, diagonal=-1))
    upper_ratio = _relative(strictlower_norm1, norm1)
    factor_threshold = factor_rtol(n) * norm1
    exact_upper = strictlower_norm1 == 0
    approx_upper = strictlower_norm1 <= factor_threshold

    column_l1 = abs_a.sum(dim=-2)
    zero_columns = (column_l1 == 0).sum(dim=-1)
    positive_column_l1 = column_l1.masked_fill(column_l1 == 0, float("inf"))
    min_positive_column_l1 = positive_column_l1.amin(dim=-1)
    max_column_l1 = column_l1.amax(dim=-1)
    column_l1_span = _relative(max_column_l1, min_positive_column_l1)
    column_l1_span = torch.where(min_positive_column_l1.isinf(), torch.zeros_like(column_l1_span), column_l1_span)

    row_l1 = abs_a.sum(dim=-1)
    positive_row_l1 = row_l1.masked_fill(row_l1 == 0, float("inf"))
    min_positive_row_l1 = positive_row_l1.amin(dim=-1)
    max_row_l1 = row_l1.amax(dim=-1)
    row_l1_span = _relative(max_row_l1, min_positive_row_l1)
    row_l1_span = torch.where(min_positive_row_l1.isinf(), torch.zeros_like(row_l1_span), row_l1_span)

    tiny_column_threshold = torch.finfo(torch.float32).eps * max(n, 1) * max_column_l1.clamp_min(1e-30)
    tiny_columns = (column_l1 <= tiny_column_threshold[:, None]).sum(dim=-1)
    column_span_list = _per_matrix_float_list(column_l1_span)
    row_span_list = _per_matrix_float_list(row_l1_span)

    result: dict[str, object] = {
        "matrix_l1": _per_matrix_float_list(norm1),
        "strictlower_l1": _per_matrix_float_list(strictlower_norm1),
        "upper_triangular_ratio": _per_matrix_float_list(upper_ratio),
        "upper_triangular_exact": _per_matrix_bool_list(exact_upper),
        "upper_triangular_rate": float(exact_upper.to(torch.float32).mean().item()),
        "approx_upper_triangular": _per_matrix_bool_list(approx_upper),
        "approx_upper_triangular_rate": float(approx_upper.to(torch.float32).mean().item()),
        "zero_column_count": _per_matrix_int_list(zero_columns),
        "tiny_column_count": _per_matrix_int_list(tiny_columns),
        "column_l1_span": column_span_list,
        "column_l1_span_median": _quantile(column_span_list, 0.5),
        "row_l1_span": row_span_list,
        "row_l1_span_median": _quantile(row_span_list, 0.5),
    }

    if n <= early_stop_max_n:
        result.update(_early_stop_probe(a, norm1))
    else:
        result.update(_skipped_early_stop_probe())

    if n <= rank_probe_max_n:
        result.update(_effective_rank_probe(a))
    else:
        result.update(_skipped_rank_probe())

    approx_upper_list = _per_matrix_bool_list(approx_upper)
    early_stop_values_obj = result.get("early_stop_k")
    early_stop_values: list[int] = []
    if isinstance(early_stop_values_obj, list):
        early_stop_values = [value for value in early_stop_values_obj if isinstance(value, int)]
    zero_suffix_obj = result.get("zero_tau_suffix_count")
    zero_suffix_values: list[int] = []
    if isinstance(zero_suffix_obj, list):
        zero_suffix_values = [value for value in zero_suffix_obj if isinstance(value, int)]
    routes = _route_classes(approx_upper_list, early_stop_values, zero_suffix_values, n)
    result.update(
        {
            "routing_class": routes,
            "routing_class_counts": _class_counts(routes),
            "compaction_survivor_fraction_n4": (
                None
                if not early_stop_values
                else sum(1 for value in early_stop_values if value > max(1, n // 4)) / len(routes)
            ),
            "compaction_survivor_fraction_n2": (
                None
                if not early_stop_values
                else sum(1 for value in early_stop_values if value > max(1, n // 2)) / len(routes)
            ),
        }
    )
    return result


def run_structure_probe(
    spec: QrV2Spec,
    *,
    device: torch.device | str | None = None,
    early_stop_max_n: int = 512,
    rank_probe_max_n: int = 512,
) -> dict[str, object]:
    a = generate_from_spec(spec, device=device)
    return {
        "event": "probe_result",
        "probe_id": "semantic_v2",
        "batch": spec.batch,
        "n": spec.n,
        "cond": spec.cond,
        "seed": spec.seed,
        "case": spec.case,
        "device": str(a.device),
        "dtype": str(a.dtype).removeprefix("torch."),
        **probe_tensor(a, early_stop_max_n=early_stop_max_n, rank_probe_max_n=rank_probe_max_n),
    }
