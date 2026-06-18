"""Official-style qr_v2 input generation."""

import torch

from gpumode.qr_v2.specs import QrV2Spec


MIXED_PROFILES: tuple[str, ...] = (
    "dense",
    "rankdef",
    "nearrank",
    "clustered",
    "band",
    "rowscale",
    "nearcollinear",
)
MIXED_WEIGHTS: tuple[float, ...] = (6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def apply_column_scaling(a: torch.Tensor, cond: int) -> torch.Tensor:
    """Apply the task's deterministic column dynamic-range scaling."""

    if cond:
        n = a.shape[-1]
        scales = torch.logspace(0.0, -float(cond), n, device=a.device, dtype=torch.float32)
        return a * scales
    return a.contiguous()


def band_mask(n: int, bandwidth: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(n, device=device)
    return (idx[:, None] - idx[None, :]).abs() <= bandwidth


def apply_case(a: torch.Tensor, case: str, cond: int, generator: torch.Generator) -> torch.Tensor:
    """Apply one official conditioning profile to an existing base batch."""

    m, n = a.shape[0], a.shape[-1]
    device = a.device
    case = case.lower()

    if case == "dense":
        a = apply_column_scaling(a, cond)
    elif case == "upper":
        diag_boost = torch.linspace(1.0, 0.25, n, device=device, dtype=torch.float32)
        a = torch.triu(a)
        a.diagonal(dim1=-2, dim2=-1).add_(diag_boost)
        a = apply_column_scaling(a, cond)
    elif case == "diagonal":
        diag = torch.randn((m, n), device=device, dtype=torch.float32, generator=generator)
        diag = diag.sign().clamp(min=0.0).mul(2.0).sub(1.0) * torch.logspace(
            0.0,
            -float(max(cond, 2)),
            n,
            device=device,
            dtype=torch.float32,
        )
        a = torch.diag_embed(diag)
    elif case == "rankdef":
        rank = max(1, (3 * n) // 4)
        a[:, :, rank:] = 0.0
        a = apply_column_scaling(a, cond)
    elif case == "nearrank":
        rank = max(1, (3 * n) // 4)
        tail = n - rank
        if tail > 0:
            noise = torch.randn((m, n, tail), device=device, dtype=torch.float32, generator=generator)
            a[:, :, rank:] = a[:, :, :tail] + 1.0e-5 * noise
        a = apply_column_scaling(a, cond)
    elif case == "clustered":
        scales = torch.ones((n,), device=device, dtype=torch.float32)
        scales[n // 2 :] = 4.0 * torch.finfo(torch.float32).eps
        if n >= 8:
            lo = max(0, n // 2 - 2)
            hi = min(n, n // 2 + 2)
            scales[lo:hi] = torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps, device=device))
        a = a * scales
    elif case == "band":
        bandwidth = max(2, min(32, n // 32))
        a = a * band_mask(n, bandwidth, device)
        diag_boost = torch.linspace(1.0, 0.5, n, device=device, dtype=torch.float32)
        a.diagonal(dim1=-2, dim2=-1).add_(diag_boost)
        a = apply_column_scaling(a, cond)
    elif case == "nearcollinear":
        base = torch.randn((m, n, 1), device=device, dtype=torch.float32, generator=generator)
        noise = torch.randn((m, n, n), device=device, dtype=torch.float32, generator=generator)
        a = base.expand(m, n, n) + 1.0e-4 * noise
        a = apply_column_scaling(a, cond)
    elif case == "rowscale":
        row_cond = max(cond, 4)
        scales = torch.logspace(0.0, -float(row_cond), n, device=device, dtype=torch.float32)
        a = scales.reshape(1, n, 1) * a
    else:
        raise ValueError(f"unknown QR test case: {case}")

    return a


def generate_mixed(a: torch.Tensor, cond: int, generator: torch.Generator) -> torch.Tensor:
    """Generate the official heterogeneous mixed-batch profile."""

    m = a.shape[0]
    device = a.device
    weights = torch.tensor(MIXED_WEIGHTS, dtype=torch.float32, device=device)
    labels = torch.multinomial(weights, m, replacement=True, generator=generator)

    if m >= 2:
        is_dense = labels == 0
        if not bool(is_dense.any()):
            labels[int(torch.randint(0, m, (1,), device=device, generator=generator))] = 0
        elif bool(is_dense.all()):
            pos = int(torch.randint(0, m, (1,), device=device, generator=generator))
            labels[pos] = int(torch.randint(1, len(MIXED_PROFILES), (1,), device=device, generator=generator))

    for k, profile in enumerate(MIXED_PROFILES):
        mask = labels == k
        if bool(mask.any()):
            a[mask] = apply_case(a[mask], profile, cond, generator)

    return a


def generate_input(
    batch: int,
    n: int,
    cond: int,
    seed: int,
    case: str = "dense",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate one official qr_v2 input tensor."""

    if batch <= 0:
        raise ValueError("batch must be positive")
    if n <= 0:
        raise ValueError("n must be positive")
    if cond < 0:
        raise ValueError("cond must be non-negative")

    resolved_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(seed)

    case = case.lower()
    a = torch.randn((batch, n, n), device=resolved_device, dtype=torch.float32, generator=generator)

    if case == "mixed":
        a = generate_mixed(a, cond, generator)
    else:
        a = apply_case(a, case, cond, generator)

    return a.contiguous()


def generate_from_spec(spec: QrV2Spec, device: torch.device | str | None = None) -> torch.Tensor:
    return generate_input(
        batch=spec.batch,
        n=spec.n,
        cond=spec.cond,
        seed=spec.seed,
        case=spec.case,
        device=device,
    )
