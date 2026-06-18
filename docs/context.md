# qr_v2 Handoff

Generated: 2026-06-18

This repository is a local harness and CUDA-candidate workspace for the GPU Mode `qr_v2` problem:

- Source: https://github.com/gpu-mode/reference-kernels/tree/main/problems/linalg/qr_v2
- Task metadata: https://raw.githubusercontent.com/gpu-mode/reference-kernels/main/problems/linalg/qr_v2/task.yml

## Objective

Implement batched square compact-Householder QR for CUDA tensors.

Input is `A`, shaped `batch x n x n`, `torch.float32`, on CUDA. Output is `(H, tau)` in the same compact convention as `torch.geqrf(A)`: `triu(H)` is `R`, the lower triangle of `H` stores Householder vectors, and `tau` stores reflector coefficients. Correctness is checked against the original FP32 input using `torch.linalg.householder_product(H, tau)`, the factor residual `triu(H) - Q.T @ A`, and orthogonality of `Q`. Returned factors must be FP32 even if a future implementation uses lower precision internally.

Important shapes include `n=512` with large batch, plus `1024`, `2048`, and `4096`. The first three official `tests` cases are dense `n=32`, `n=176`, and `n=352`; those are the current correctness gate before broadening.

## Project Shape

This is a `uv` Python project inside a self-contained Flox environment.

- Package: `gpumode`
- CLI: `qr-v2 = "gpumode.qr_v2.cli:main"`
- Source: `src/gpumode/qr_v2/`
- Generated data: `data/qr_v2/...`
- Python: `>=3.13`, supplied by Flox
- uv is configured in `pyproject.toml` with `python-downloads = "never"` and `python-preference = "only-system"`
- Use `flox activate -- ...` from the repo root; do not depend on absolute `.flox/run/...` paths

Pinned runtime dependencies are `torch==2.12.1`, `numpy==2.4.6`, `polars==1.41.2`, `jinja2==3.1.6`, and `rich==15.0.0`. Dev dependencies are `ruff==0.15.17`, `ty==0.0.50`, and `lightning-sdk==2026.6.8.post0`.

The Flox manifest installs Python 3.13.13, uv 0.11.19, yq, gh, and CUDA 13.0 packages on `x86_64-linux`: `nvcc`, `cudart`, `cccl`, `cuda-gdb`, `cupti`, `cuda_sanitizer_api`, `cuobjdump`, `nvdisasm`, `nvprune`, Nsight Compute, and Nsight Systems. The NVIDIA driver is supplied by the host machine.

The activation hook sets:

```sh
export PYTHONPATH="$FLOX_ENV_PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="$FLOX_ENV_CACHE/python"
export UV_NO_MANAGED_PYTHON=1
export UV_PYTHON="$FLOX_ENV/bin/python"
```

## Implemented Harness

Core modules:

- `specs.py`: smoke/test/benchmark spec definitions.
- `inputs.py`: official-style input generation.
- `check.py`: compact Householder QR checker.
- `baseline.py`: Torch `geqrf` baseline runner.
- `probes.py`: input structure probes.
- `summary.py`: Polars/Rich summaries.
- `jsonl.py`: JSONL helpers.

Candidate pipeline:

- `render.py`: renders CUDA artifacts and a render manifest.
- `compile_plan.py`: converts render rows into `nvcc` jobs. The CLI exposes `--gpu-arch`, defaulting to `sm_89` for L4.
- `compile.py`: executes compile jobs, writes `.so` files and compile-result manifests.
- `run_plan.py`: converts compile rows into run jobs.
- `run.py`: loads compiled shared objects with `ctypes`, launches kernels, checks `(H, tau)`, and writes run-result manifests.
- `cli.py`: exposes `baseline`, `probe`, `summary`, `render`, `verify-renders`, `plan-compile`, `verify-compile-plan`, `compile-one`, `compile`, `verify-compile-results`, `plan-run`, `verify-run-plan`, `run-one`, `run`, and `verify-run-results`.

Current render candidates:

- `serial` (default): `qr_v2_geqr2_serial_tpb128_tile32`, template `cuda_geqr2_serial_v1`. Correct-first GEQR2 baseline, one block per matrix, one active thread.
- `parallel`: `qr_v2_geqr2_parallel_tpb256_tile32`, template `cuda_geqr2_parallel_v1`. One block per matrix, cooperative reductions for column norms and reflector dot products, parallel trailing-column updates.

Rendering a suite writes `data/qr_v2/renders/<suite>/manifest.jsonl`. That manifest points at the most recently rendered candidate for that suite, while artifacts live under candidate-specific subdirectories.

## Standard Pipeline

Use this from the repo root:

```sh
flox activate -- uv run qr-v2 render --suite tests --limit 3 --candidate parallel
flox activate -- uv run qr-v2 verify-renders --suite tests
flox activate -- uv run qr-v2 plan-compile --suite tests
flox activate -- uv run qr-v2 verify-compile-plan --suite tests
flox activate -- uv run qr-v2 compile --suite tests --timeout-seconds 60
flox activate -- uv run qr-v2 verify-compile-results --suite tests
flox activate -- uv run qr-v2 plan-run --suite tests
flox activate -- uv run qr-v2 verify-run-plan --suite tests
flox activate -- uv run qr-v2 run --suite tests --limit 3
flox activate -- uv run qr-v2 verify-run-results --suite tests
```

For result summaries, `yq` is available in Flox. `jq` is not installed.

## Latest L4 Results

Host used for setup: NVIDIA L4, driver 580.159.03, CUDA 13.0, target arch `sm_89`.

Serial baseline passed the first three official `tests` cases:

```text
n=32,  batch=20: passed, ~158 ms
n=176, batch=40: passed, ~300 ms
n=352, batch=40: passed, ~2839 ms
```

Parallel candidate passed all 10 smoke cases and the first three official `tests` cases:

```text
n=32,  batch=20: passed, 137.436 ms
n=176, batch=40: passed, 48.382 ms
n=352, batch=40: passed, 240.441 ms
```

The parallel candidate is already the better platform for larger early cases. L4 has served its purpose for environment setup and initial correctness gating.

Final checks passed after the parallel candidate work:

```sh
flox activate -- uv run ruff check .
flox activate -- uv run ty check
flox activate -- uv sync
flox activate -- git diff --check
```

## Alignment With Proposal And Plan

`docs/proposal.md` and `docs/plan.md` describe the larger strategy: build a semantic work-reduction tournament, not just a faster fixed QR kernel. The current repo is aligned with the early infrastructure part of that plan: it has deterministic input generation, a local checker, candidate rendering, compile/run manifests, correctness result records, and a first native CUDA candidate. It is not yet aligned with the later tournament architecture: there is no PyTorch extension API, no cuSOLVER/cuSolverDx fallback, no candidate registry, no semantic feasibility table, and no frozen dispatch system.

Treat the current parallel GEQR2 kernel as a correctness and profiling foothold. Do not let it become the whole strategy unless the semantic probes show no useful slack.

## Next Steps

Move to the H100 next. Save the B200 until after the H100 pass answers two questions: whether the current CUDA path is portable/timing-sane on a stronger GPU, and which semantic work-reduction probes deserve implementation before a real B200 tournament.

On H100, use `qr-v2 plan-compile --gpu-arch sm_90`. For B200 later, confirm the supported Blackwell target with `nvcc --list-gpu-arch` on that machine before planning compiles.

Then rerun the first-three test gate on H100 with the parallel candidate:

```sh
flox activate -- uv run qr-v2 render --suite tests --limit 3 --candidate parallel
flox activate -- uv run qr-v2 plan-compile --suite tests --gpu-arch sm_90
flox activate -- uv run qr-v2 compile --suite tests --timeout-seconds 60
flox activate -- uv run qr-v2 plan-run --suite tests
flox activate -- uv run qr-v2 run --suite tests --limit 3
flox activate -- uv run qr-v2 verify-run-results --suite tests
```

If that passes, profile `n=176` and `n=352` with Nsight Compute. Look first at time spent in reductions, reflector dot products, and trailing updates.

After the H100 gate, return to the proposal/plan priorities before spending B200 time:

- Add semantic feasibility probes and summary output for early-stop rate, approximate-upper shortcut rate, column-scale span, zero-tail reflector frequency, and routing/compaction overhead.
- Build a small baseline portfolio: current serial/parallel kernels plus a vendor fallback path if cuSOLVER/cuSolverDx is available in the environment.
- Start recording result fields needed by the tournament plan, especially `candidate_id`, runtime, residual diagnostics, fallback count, and stop panel.
- Use those data to decide whether to implement early stop, upper-triangular shortcut, power-of-two gauge, speculative fast path, or a bigger tiled/multi-block QR core first.

Likely implementation paths:

- If semantic probes show slack, implement the highest-yield semantic operator first; proposal priority is early stop, then structural shortcut, gauge/speculative fast path, and per-matrix routing.
- If semantic probes show little slack on the relevant cases, focus on backend work: tiled or multi-block QR for `n>=512`, vendor baselines, and eventually cuBLASLt/CUTLASS-style updates.
- Always keep `passed` as the gate; compare residual diagnostics against the serial candidate when changing math order.

## Gotchas

- The worktree is intentionally dirty during this session; do not reset unrelated changes.
- `plan-compile` defaults to `sm_89`; pass `--gpu-arch sm_90` on H100.
- No `pytest` dependency has been added yet; verification is CLI/manifests/JSONL-first.
- Keep generated artifacts under `data/qr_v2/`.
- Keep dependencies local to Flox/uv. Do not globally install Python packages or CUDA tools.
