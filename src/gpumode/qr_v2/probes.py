"""Semantic structure probes for qr_v2 inputs."""

import torch

from gpumode.qr_v2.check import matrix_one_norm
from gpumode.qr_v2.inputs import generate_from_spec
from gpumode.qr_v2.specs import QrV2Spec


def _relative(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(numerator, dtype=torch.float64)
    infs = torch.full_like(numerator, float("inf"), dtype=torch.float64)
    return torch.where(denominator > 0, numerator / denominator, torch.where(numerator == 0, zeros, infs))


def _per_matrix_float_list(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values.detach().cpu().reshape(-1)]


def _per_matrix_int_list(values: torch.Tensor) -> list[int]:
    return [int(value) for value in values.detach().cpu().reshape(-1)]


def probe_tensor(a: torch.Tensor) -> dict[str, object]:
    """Compute cheap per-matrix structure features for a batch."""

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

    return {
        "matrix_l1": _per_matrix_float_list(norm1),
        "strictlower_l1": _per_matrix_float_list(strictlower_norm1),
        "upper_triangular_ratio": _per_matrix_float_list(upper_ratio),
        "zero_column_count": _per_matrix_int_list(zero_columns),
        "tiny_column_count": _per_matrix_int_list(tiny_columns),
        "column_l1_span": _per_matrix_float_list(column_l1_span),
        "row_l1_span": _per_matrix_float_list(row_l1_span),
    }


def run_structure_probe(
    spec: QrV2Spec,
    *,
    device: torch.device | str | None = None,
) -> dict[str, object]:
    a = generate_from_spec(spec, device=device)
    return {
        "event": "probe_result",
        "probe_id": "structure_v1",
        "batch": spec.batch,
        "n": spec.n,
        "cond": spec.cond,
        "seed": spec.seed,
        "case": spec.case,
        "device": str(a.device),
        "dtype": str(a.dtype).removeprefix("torch."),
        **probe_tensor(a),
    }
