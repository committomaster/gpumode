"""Local baseline runners for qr_v2."""

import time
from dataclasses import asdict

import torch

from gpumode.qr_v2.check import check_compact_qr
from gpumode.qr_v2.inputs import generate_from_spec
from gpumode.qr_v2.specs import BENCHMARK_SPECS, TEST_SPECS, QrV2Spec


SMOKE_SPECS: tuple[QrV2Spec, ...] = (
    QrV2Spec(batch=3, n=8, cond=1, seed=101, case="dense"),
    QrV2Spec(batch=3, n=8, cond=1, seed=102, case="upper"),
    QrV2Spec(batch=3, n=8, cond=2, seed=103, case="diagonal"),
    QrV2Spec(batch=3, n=8, cond=0, seed=104, case="rankdef"),
    QrV2Spec(batch=3, n=8, cond=0, seed=105, case="nearrank"),
    QrV2Spec(batch=3, n=8, cond=0, seed=106, case="clustered"),
    QrV2Spec(batch=3, n=8, cond=0, seed=107, case="band"),
    QrV2Spec(batch=3, n=8, cond=0, seed=108, case="rowscale"),
    QrV2Spec(batch=3, n=8, cond=0, seed=109, case="nearcollinear"),
    QrV2Spec(batch=3, n=8, cond=1, seed=110, case="mixed"),
)


def get_spec_suite(name: str) -> tuple[QrV2Spec, ...]:
    if name == "smoke":
        return SMOKE_SPECS
    if name == "tests":
        return TEST_SPECS
    if name == "benchmarks":
        return BENCHMARK_SPECS
    raise ValueError(f"unknown spec suite: {name}")


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_torch_geqrf_baseline(
    spec: QrV2Spec,
    *,
    device: torch.device | str | None = None,
) -> dict[str, object]:
    a = generate_from_spec(spec, device=device)

    _synchronize_if_cuda(a.device)
    start_ns = time.perf_counter_ns()
    h, tau = torch.geqrf(a)
    _synchronize_if_cuda(a.device)
    elapsed_ns = time.perf_counter_ns() - start_ns

    check = check_compact_qr(a, h, tau)
    return {
        "event": "baseline_result",
        "candidate_id": "torch.geqrf",
        "batch": spec.batch,
        "n": spec.n,
        "cond": spec.cond,
        "seed": spec.seed,
        "case": spec.case,
        "device": str(a.device),
        "dtype": str(a.dtype).removeprefix("torch."),
        "elapsed_ns": elapsed_ns,
        "elapsed_us": elapsed_ns / 1_000.0,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        **asdict(check),
    }
