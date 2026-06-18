"""Official-style checks for compact-Householder QR witnesses."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QrCheckResult:
    passed: bool
    factor_abs: float
    factor_rel: float
    factor_scaled: float
    factor_threshold: float
    orth_abs: float
    orth_rel: float
    orth_scaled: float
    orth_threshold: float
    lower_abs: float
    lower_rel: float
    lower_scaled: float
    lower_threshold: float
    reconstruction_abs: float
    reconstruction_scaled: float
    worst_batch: int


def factor_rtol(n: int) -> float:
    return 20.0 * n * torch.finfo(torch.float32).eps


def orth_rtol(n: int) -> float:
    return 100.0 * n * torch.finfo(torch.float32).eps


def scaled_residual(residual: torch.Tensor, scale: torch.Tensor, n: int) -> torch.Tensor:
    eps = torch.finfo(torch.float32).eps
    return residual / (eps * max(n, 1) * scale.clamp_min(1e-30))


def matrix_one_norm(x: torch.Tensor) -> torch.Tensor:
    """Return the matrix 1-norm over the last two dimensions."""

    return x.abs().sum(dim=-2).amax(dim=-1)


def _relative(abs_value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(abs_value, dtype=torch.float64)
    infs = torch.full_like(abs_value, float("inf"), dtype=torch.float64)
    return torch.where(scale > 0, abs_value / scale, torch.where(abs_value == 0, zeros, infs))


def check_compact_qr(
    a_original: torch.Tensor,
    h: torch.Tensor,
    tau: torch.Tensor,
) -> QrCheckResult:
    """Validate a compact-Householder QR witness against the challenge contract."""

    if a_original.shape != h.shape:
        raise ValueError(f"a_original and h must have the same shape, got {a_original.shape} and {h.shape}")
    if a_original.ndim < 2 or a_original.shape[-1] != a_original.shape[-2]:
        raise ValueError(f"a_original must be a batch of square matrices, got {a_original.shape}")
    expected_tau_shape = (*a_original.shape[:-1],)
    if tau.shape != expected_tau_shape:
        raise ValueError(f"tau shape must be {expected_tau_shape}, got {tau.shape}")
    if a_original.dtype != torch.float32 or h.dtype != torch.float32 or tau.dtype != torch.float32:
        raise ValueError("a_original, h, and tau must be torch.float32 tensors")
    if h.device != a_original.device or tau.device != a_original.device:
        raise ValueError(f"h and tau must be on {a_original.device}")
    if not torch.isfinite(h).all().item():
        raise ValueError("h contains NaN or Inf")
    if not torch.isfinite(tau).all().item():
        raise ValueError("tau contains NaN or Inf")

    n = a_original.shape[-1]
    a64 = a_original.to(torch.float64)
    q = torch.linalg.householder_product(h, tau)
    r = torch.triu(h)
    if not torch.isfinite(q).all().item():
        raise ValueError("Q materialized from (h, tau) contains NaN or Inf")
    if not torch.isfinite(r).all().item():
        raise ValueError("R extracted from triu(h) contains NaN or Inf")
    q64 = q.to(torch.float64)
    r64 = r.to(torch.float64)

    qt_a = q64.transpose(-1, -2) @ a64
    if not torch.isfinite(qt_a).all().item():
        raise ValueError("Q.T @ A contains NaN or Inf")
    eye = torch.eye(n, dtype=torch.float64, device=a_original.device)
    eye = eye.expand(a_original.shape[:-2] + (n, n))
    qtq = q64.transpose(-1, -2) @ q64
    if not torch.isfinite(qtq).all().item():
        raise ValueError("Q.T @ Q contains NaN or Inf")

    factor_abs = matrix_one_norm(r64 - qt_a)
    orth_abs = matrix_one_norm(qtq - eye)
    lower_abs = matrix_one_norm(torch.tril(qt_a, diagonal=-1))
    reconstruction_abs = matrix_one_norm(q64 @ r64 - a64)

    a_norm = matrix_one_norm(a64)
    eye_norm = matrix_one_norm(eye)
    factor_threshold = factor_rtol(n) * a_norm
    orth_threshold = orth_rtol(n) * eye_norm
    lower_threshold = factor_threshold

    factor_rel = _relative(factor_abs, a_norm)
    orth_rel = _relative(orth_abs, eye_norm)
    lower_rel = _relative(lower_abs, a_norm)
    factor_scaled = scaled_residual(factor_abs, a_norm, n)
    orth_scaled = scaled_residual(orth_abs, eye_norm, n)
    lower_scaled = scaled_residual(lower_abs, a_norm, n)
    reconstruction_scaled = scaled_residual(reconstruction_abs, a_norm, n)

    if not torch.isfinite(factor_scaled).all().item():
        raise ValueError("R - Q.T @ A residual produced NaN or Inf")
    if not torch.isfinite(orth_scaled).all().item():
        raise ValueError("Q.T @ Q residual produced NaN or Inf")
    if not torch.isfinite(lower_scaled).all().item():
        raise ValueError("lower-triangular residual produced NaN or Inf")
    if not torch.isfinite(reconstruction_scaled).all().item():
        raise ValueError("Q @ R reconstruction residual produced NaN or Inf")

    passed_by_matrix = (factor_abs <= factor_threshold) & (orth_abs <= orth_threshold)
    worst_score = torch.maximum(
        _relative(factor_abs, factor_threshold),
        _relative(orth_abs, orth_threshold),
    )
    worst_batch = int(worst_score.reshape(-1).argmax().item())

    return QrCheckResult(
        passed=bool(passed_by_matrix.all().item()),
        factor_abs=float(factor_abs.amax().item()),
        factor_rel=float(factor_rel.amax().item()),
        factor_scaled=float(factor_scaled.amax().item()),
        factor_threshold=float(factor_threshold.amax().item()),
        orth_abs=float(orth_abs.amax().item()),
        orth_rel=float(orth_rel.amax().item()),
        orth_scaled=float(orth_scaled.amax().item()),
        orth_threshold=float(orth_threshold.amax().item()),
        lower_abs=float(lower_abs.amax().item()),
        lower_rel=float(lower_rel.amax().item()),
        lower_scaled=float(lower_scaled.amax().item()),
        lower_threshold=float(lower_threshold.amax().item()),
        reconstruction_abs=float(reconstruction_abs.amax().item()),
        reconstruction_scaled=float(reconstruction_scaled.amax().item()),
        worst_batch=worst_batch,
    )


def _smoke() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    a = torch.randn((3, 8, 8), dtype=torch.float32, generator=generator)
    h, tau = torch.geqrf(a)
    result = check_compact_qr(a, h, tau)
    print(result)
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    _smoke()
