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
- `run.py`: loads compiled shared objects with `ctypes`, launches kernels, checks `(H, tau)`, writes run-result manifests, and can write checker-free launcher timing records via `time-one`.
- `cli.py`: exposes `baseline`, `probe`, `summary`, `render`, `verify-renders`, `plan-compile`, `verify-compile-plan`, `compile-one`, `compile`, `verify-compile-results`, `plan-run`, `verify-run-plan`, `run-one`, `run`, `time-one`, and `verify-run-results`.

Current render candidates:

- `serial` (default): `qr_v2_geqr2_serial_tpb128_tile32`, template `cuda_geqr2_serial_v1`. Correct-first GEQR2 baseline, one block per matrix, one active thread.
- `parallel`: `qr_v2_geqr2_parallel_tpb256_tile32`, template `cuda_geqr2_parallel_v1`. One block per matrix, cooperative reductions for column norms and reflector dot products, parallel trailing-column updates.
- `parallel_profile`: `qr_v2_geqr2_parallel_profile_tpb256_tile32`, template `cuda_geqr2_parallel_profile_v1`. Same math as `parallel`, with device-side `clock64()` phase counters for non-admin profiling.
- `semantic_upper`: `qr_v2_semantic_upper_fallback_tpb256_tile32`, template `cuda_semantic_upper_fallback_v1`. Routes approximate-upper matrices to `tau=0`, `H=triu(A)`, otherwise falls back to cooperative GEQR2.
- `semantic_early_stop`: `qr_v2_semantic_early_stop_fallback_tpb256_tile32`, template `cuda_semantic_early_stop_fallback_v1`. Routes a narrow `n=512` near-collinear certificate to a 16-reflector early stop, otherwise falls back to cooperative GEQR2.
- `semantic_combo`: `qr_v2_semantic_combo_fallback_tpb256_tile32`, template `cuda_semantic_combo_fallback_v1`. Combined upper shortcut, near-collinear early stop, clustered early stop, exact zero-tail pruning, route counters, and cooperative GEQR2 fallback. This is the current best semantic prototype.
- `cusolver`: `qr_v2_cusolver_geqrf_tpb256`, template `cuda_cusolver_geqrf_v1`. Host cuSOLVER dense GEQRF baseline/fallback with row-major/column-major packing kernels. It links against Flox-provided CUDA 13.0 `libcusolver` and `libcublas` packages when available, with a `.venv` CUDA-wheel fallback for older local environments.

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

## L4 Lightning Studio Results

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

## H100 Lightning Studio Results

Host used for the current pass: Lightning.AI Studio with an NVIDIA H100 80GB HBM3, CUDA 13.0 user-space packages from Flox, and host driver 580.142-series. The Studio environment does not provide admin access. Nsight Compute performance counters are locked by the host setting `RmProfilingAdminOnly: 1`, so normal-user `ncu` reports `ERR_NVGPUCTRPERM`. Cannot run project Python as root to collect counters.

Because NCU counters were unavailable, `parallel_profile` was added with device-side `clock64()` counters. It passed the first-three official `tests` gate on `sm_90`:

```text
n=32,  batch=20: passed, 209.234 ms
n=176, batch=40: passed, 25.371 ms
n=352, batch=40: passed, 134.360 ms
```

The useful signal was the phase split, not the instrumented wall time:

```text
n=176: dot 70.56%, update 28.06%, norm 1.06%
n=352: dot 59.50%, update 39.84%, norm 0.37%
```

So norm/scale tuning is not the next lever. The hot path is reflector dot products plus trailing updates.

Semantic feasibility probes were expanded to `semantic_v2`. They now record exact/approx-upper rate, early-stop stop-k distribution, zero-`tau` reflector rate, rank/stable-rank proxies, row/column spans, route classes, compaction survivor fractions, and an estimated speedup ceiling. The important H100 probe findings on `tests` were:

```text
dense n=176/352: no semantic slack, stop_k=n
nearcollinear n=512: stop50=16, saved ~90.9%, ceiling ~11.0x
clustered n=512: stop50=256, saved ~12.5%, ceiling ~1.14x
rankdef n=512: zero_tau_rate=25%, early stop only at 384
upper n=4096: upper_rate=1.0, approx_upper=1.0
mixed n=512: mostly dense, one zero-tail route in the test seed
```

Three semantic CUDA prototypes were added and validated on H100:

```text
semantic_upper:
  dense n=32/176/352 guard: passed
  upper n=4096 official test row: passed
  pure launcher timing on pre-generated upper n=4096:
    semantic_upper 20.83 ms median vs parallel 34.93 ms

semantic_early_stop:
  dense n=32/176/352 guard: passed
  nearcollinear n=512 official test row: passed
  pure launcher timing on pre-generated nearcollinear n=512:
    semantic_early_stop 19.29 ms median vs parallel 226.10 ms

semantic_combo:
  dense n=32/176/352 guard: passed
  nearcollinear n=512 official test row: passed
  upper n=4096 official test row: passed
  pure launcher timing on pre-generated inputs:
    nearcollinear n=512: semantic_combo 19.26 ms vs parallel 226.12 ms
    upper n=4096:       semantic_combo 20.62 ms vs parallel 34.96 ms
```

These early semantic wins mostly did not hit the official benchmark distribution: the benchmark list in `docs/task.md` does not include standalone `upper n=4096` or `nearcollinear n=512`. Later H100 passes added an exact zero-tail route for the official rank-deficient structure, a conservative clustered `k=n/2` early-stop route, and device-side route counters (`upper_count`, `early_stop_count`, `zero_tail_count`, `fallback_count`) to run records via a `_launch_routes` entry point.

The zero-tail route detects an exact trailing quarter of zero columns for `n=512` and `n=1024`, caps reflector generation at `3*n/4`, and skips trailing updates into the zero tail. Validation on H100 `sm_90`:

```text
tests dense n=32/176/352: passed, fallback_count=batch
tests rankdef n=512, batch=16: passed, zero_tail_count=16
benchmarks rankdef n=512, batch=640: passed, zero_tail_count=640
benchmarks clustered n=512, batch=640: passed, early_stop_count=640
benchmarks mixed n=512, batch=640: passed, early_stop_count=57, zero_tail_count=51, fallback_count=532
benchmarks mixed n=1024, batch=60: passed, zero_tail_count=4, fallback_count=56
```

`time-one` was added to record checker-free launcher timing on pre-generated inputs. It loads the `.so` once, warms up, repeats the launcher, and can collect route counts with a separate route-counter launch. H100 timing snapshots for `semantic_combo`:

```text
benchmarks rankdef n=512, batch=640:   launcher p50 2,561,842 us, zero_tail_count=640
benchmarks clustered n=512, batch=640: launcher p50 3,671,123 us, early_stop_count=640
benchmarks mixed n=512, batch=640:     launcher p50 3,744,751 us, early_stop_count=57, zero_tail_count=51
benchmarks mixed n=1024, batch=60:     launcher p50 1,739,834 us, zero_tail_count=4
```

The older `run.py` `elapsed_ms` records include input generation, dynamic loading, kernel launch, and the checker, so they are useful for local regression tracking but not pure launcher timing. The current semantic candidate is now benchmark-relevant evidence, but it is still not plausibly in the public B200 leaderboard range by itself. Dense benchmark cases and most mixed matrices still fall back to the one-block cooperative GEQR2 kernel, which is far too slow for a ~2 ms geometric mean.

A dense cuSOLVER candidate was added next as a correctness fallback and benchmark-relevant baseline. FloxHub packages `flox-cuda/cudaPackages_13_0.libcusolver@12.0.4.66` and `flox-cuda/cudaPackages_13_0.libcublas@13.1.1.3` are now installed in the Flox manifest; `compile_plan.py` prefers `.flox/run/x86_64-linux.gpumode.dev/include` and `.flox/run/x86_64-linux.gpumode.dev/lib` for cuSOLVER artifacts. The first-three official `tests` rows and first-four dense `benchmarks` rows passed on H100 `sm_90` with the Flox-linked build. Checker-free timing for the first four dense benchmark rows:

```text
benchmarks dense n=32,  batch=20:  launcher p50 1,605.945 us
benchmarks dense n=176, batch=40:  launcher p50 17,980.749 us
benchmarks dense n=352, batch=40:  launcher p50 42,420.372 us
benchmarks dense n=512, batch=640: launcher p50 1,003,052.783 us
first-four dense geomean: 33,293.479 us
```

This is useful as a dense fallback and a reference implementation, but it is not a leaderboard path by itself. The current launcher loops over matrices and calls host cuSOLVER once per matrix, so batch-heavy n=512 is dominated by library/launch overhead. The next dense path needs cuSolverDx, a real batched interface, or a custom tiled/multi-block QR path.

Final checks passed after the H100 semantic and cuSOLVER work:

```sh
flox activate -- uv run ruff check .
flox activate -- uv run ty check
flox activate -- git diff --check
```

## Alignment With Proposal And Plan

`docs/proposal.md` and `docs/plan.md` describe the larger strategy: build a semantic work-reduction tournament, not just a faster fixed QR kernel. The repo is now aligned with the early infrastructure and semantic-feasibility parts of that plan: deterministic input generation, checker, candidate rendering, compile/run manifests, result records, native CUDA candidates, device-side profile summaries, a semantic probe table, and a local cuSOLVER dense fallback. It is still missing the later tournament architecture: no PyTorch extension API, no candidate registry, no frozen dispatch system, and no competitive dense backend.

Treat the current `semantic_combo` and `cusolver` candidates as benchmark-relevant proof points, not as leaderboard-ready implementations.

## Earlier H100 Plan

Before the move to H200/Verda, the H100 development plan targeted the actual benchmark set from `docs/task.md`:

1. Extend benchmark-relevant semantic routes:
   - Keep the conservative clustered `k=256` route enabled for `clustered n=512, batch=640`; it now passes, but the timing gain is modest because one-block GEQR2 remains the core cost.
   - `mixed n=512/1024`: per-matrix routing and compaction; whole-batch routing is invalid because the batch is heterogeneous. Current route counters show 57/640 clustered early-stop matrices plus 51/640 zero-tail matrices for n=512, and 4/60 zero-tail matrices for n=1024.
   - `nearrank n=1024, batch=60`: add a lighter feasibility probe and route if stop/slack is real.
   - Keep exact zero-tail pruning enabled for `rankdef n=512, batch=640`; it now passes the benchmark row and reports `zero_tail_count=640`.

2. Turn the dense fallback into a competitive dense path:
   - `cusolver` is now available and validated as a dense fallback, but the host loop over one cuSOLVER call per matrix is too slow for batch-heavy rows.
   - Next evaluate cuSolverDx or another device-side/batched path so n=512 batch=640 does not pay hundreds of host-library launches.
   - If a custom dense path is needed, the profiler already showed dot/update dominance, so it must address trailing updates and reflector dots, not just norm reductions.

3. Improve benchmark estimation before B200:
   - Run lightweight benchmark probes by default; the full `benchmarks` probe with SVD/early-stop enabled is too heavy because `batch=640, n=512` SVD dominates.
   - Use `qr-v2 time-one` for selected benchmark rows; current `run.py` elapsed time includes checker and setup overhead.
   - Once more routes hit benchmark cases, run a H100 benchmark slice and compute a local geometric mean.

The old B200 gate was to switch only when one of these became true:

- H100 benchmark-suite geomean is plausibly within a few times the 2-4 ms leaderboard range after architectural scaling, or
- There is a combined candidate with benchmark-relevant routes for rankdef/clustered/mixed/nearrank plus a better dense fallback, and the goal is final timing/tuning.

On B200, first confirm the supported Blackwell target with `nvcc --list-gpu-arch`; do not assume the right `sm_` value. Then run a tight validation/timing slice before the full benchmark suite. As of the current note below, the active host is H200/Verda, not H100.

### Last status note:

Implemented the next dense-backend step.

Added:
- `cublas_batched`: uses `cublasSgeqrfBatched`.
- `dense_linalg_best`: dispatch proxy. It originally used cuBLAS batched for `n=32/176/512` and cuSOLVER for `n=352` and `n>=1024`; after the H200 MathDx and blocked-QR work, it uses cuSolverDx for `n=32/176`, a blocked Householder panel path for dense `n=512`, cuBLAS batched for other `n=512` cases, and cuSOLVER for the other larger rows.

Key finding: NVIDIA’s cuBLAS docs say `Aarray`/`TauArray` are device pointer arrays, but `info` is a host scalar. Passing device `info` caused a segfault inside `cublasSgeqrfBatched`; fixed now. Source: https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-geqrfbatched

Validation completed:
- `cublas_batched` passed first-three `tests`.
- `dense_linalg_best` passed all 12 official benchmark rows via checked runs, either full slice or individual rows.
- Full 12-row H100 checker-free timing geomean for `dense_linalg_best`: `100,524.402 us`.

Best full benchmark timings:
```text
n=32:        611.090 us
n=176:    15,498.664 us
n=352:    42,476.032 us
n=512:   501,143.892 us
n=1024:  231,468.449 us
n=2048:   74,897.064 us
n=4096:   48,025.212 us
mixed512: 473,392.086 us
mixed1024:231,734.992 us
rankdef:  498,899.915 us
clustered:500,882.730 us
nearrank: 231,400.433 us
```

Follow-up verification after this note:

```sh
flox activate -- uv run ruff check .
flox activate -- uv run ty check
flox activate -- git diff --check
```

All three passed. The normal sandbox hit Flox's read-only metrics initialization issue, so the commands were rerun with escalation as the Studio user.

### Current H200/Verda status note:

The active host is now an NVIDIA H200 instance on Verda.com. Use `sm_90` for this host. Treat older H100 timings as historical; do not blend them with new H200 timing records without labeling the host.

MathDx/cuSolverDx integration is now on the Flox package-build path:

- `.flox/pkgs/mathdx.nix`: packages NVIDIA MathDx `26.03.0-cuda13` with pinned `fetchurl` hash `sha256-lcJ0QFEC9qJj956wREe9dTn2bOZJd5vY4iX8uMxxWU0=`.
- `src/gpumode/qr_v2/mathdx.py`: discovers MathDx only from package-style outputs: `$FLOX_ENV` after a published/installed package, or `result-mathdx` after `flox build mathdx`.
- `qr-v2 mathdx-info --stdout`: reports whether MathDx is visible and which compile flags will be used.
- `compile_plan.py`: adds NVIDIA's required `-dlto`, MathDx include path, and the packaged cuSolverDx library only for rendered artifacts that include `cusolverdx.hpp`.
- `docs/mathdx.md`: documents the Flox/Nix package-build route with `.flox/pkgs/mathdx.nix`, not a loose local unpack.
- `render.py`: adds candidate `cusolverdx`, using cuSolverDx GEQRF for `n=32` and `n=176` rows, with existing host cuSOLVER fallback for larger rows.

Flox catalog searches for `mathdx`, `cusolverdx`, `commonDx`, and `cublasDx` returned no package on this environment. `flox-cuda/cudaPackages_13_0.cutlass@3.9.2` exists, but NVIDIA's cuSolverDx docs say cuSolverDx is distributed as part of the MathDx package, which also ships commonDx and CUTLASS headers required by `cusolverdx.hpp`.

The Flox package has been built locally:

```sh
flox build mathdx
```

That creates `result-mathdx`, where `qr-v2 mathdx-info --stdout` detects:

```json
{"compile_args":["-dlto","-Iresult-mathdx/include","-Iresult-mathdx/external/cutlass/include","-Lresult-mathdx/lib","-lcusolverdx"],"event":"mathdx_info","found":true,"include_dir":"result-mathdx/include","lib_dir":"result-mathdx/lib","library_kind":"static","library_path":"result-mathdx/lib/libcusolverdx.a","root":"result-mathdx"}
```

Current cuSolverDx slice verification on H200:

```sh
flox activate -- uv run qr-v2 render --suite tests --limit 3 --candidate cusolverdx
flox activate -- uv run qr-v2 verify-renders --suite tests
flox activate -- uv run qr-v2 mathdx-info --stdout
flox activate -- uv run qr-v2 plan-compile --suite tests --gpu-arch sm_90
flox activate -- uv run qr-v2 verify-compile-plan --suite tests
flox activate -- uv run qr-v2 compile --suite tests --timeout-seconds 180
flox activate -- uv run qr-v2 verify-compile-results --suite tests
flox activate -- uv run qr-v2 plan-run --suite tests
flox activate -- uv run qr-v2 verify-run-plan --suite tests
flox activate -- uv run qr-v2 run --suite tests --limit 3
flox activate -- uv run qr-v2 verify-run-results --suite tests
```

All passed. The generated `n=32` and `n=176` rows compile against cuSolverDx and pass checked execution; the `n=352` row uses the existing cuSOLVER fallback and also passes.

Checker-free `time-one` snapshots on H200 with 3 warmups and 10 repeats:

```text
tests dense n=32,  batch=20: p50 20.525 us, cuSolverDx
tests dense n=176, batch=40: p50 1,402.185 us, cuSolverDx
tests dense n=352, batch=40: p50 41,071.303 us, cuSOLVER fallback
```

Fresh H200 comparison against the previous `dense_linalg_best` routing showed cuSolverDx should be folded into the dispatch candidate:

```text
previous dense_linalg_best:
  n=32:  p50 2,658.579 us, cuBLAS batched
  n=176: p50 34,395.735 us, cuBLAS batched
  n=352: p50 41,098.792 us, cuSOLVER fallback

updated dense_linalg_best:
  n=32:  p50 20.525 us, cuSolverDx
  n=176: p50 1,695.954 us, cuSolverDx
  n=352: p50 41,092.702 us, cuSOLVER fallback
```

The next cuSolverDx size probe, `n=352`, was not viable on H200/SM90. Rendering the standalone `cusolverdx` candidate through `n=352` failed at compile time with NVIDIA MathDx's static assertion that the data type and size combination does not fit the shared memory available for block execution. Keep the cuSolverDx threshold at `n <= 176` unless a different cuSolverDx execution mode or tiling strategy is introduced.

Updated `dense_linalg_best` passed the first four dense benchmark rows on H200. Checker-free `time-one` snapshots with 3 warmups and 10 repeats:

```text
benchmarks dense n=32,  batch=20:  p50 20.525 us, cuSolverDx
benchmarks dense n=176, batch=40:  p50 3,800.783 us, cuSolverDx
benchmarks dense n=352, batch=40:  p50 89,680.683 us, cuSOLVER fallback
benchmarks dense n=512, batch=640: p50 500,076.771 us, cuBLAS batched
first-four dense geomean: 7,690.825 us
```

Lane 2 dense work then added a proper blocked Householder QR prototype for dense `n=512` and `n=1024`: factor width-16 panels with cuBLAS batched panel QR, build the triangular block factor, and apply each block reflector with batched GEMM updates. It passes the official dense `n=512, batch=640` and `n=1024, batch=60` benchmark rows and is folded into `dense_linalg_best` for those dense rows.

Panel-width bracket on H200 for benchmark dense `n=512, batch=640`:

```text
panel 8:  p50 34,666.291 us
panel 16: p50 28,460.063 us
panel 32: p50 36,879.027 us
```

The dense `n=1024, batch=60` row improved from the cuSOLVER fallback p50 of `228,329.322 us` to the blocked QR p50 of `40,053.163 us`.

Integrated `dense_linalg_best` first-five dense benchmark timings on H200 with 5 warmups and 20 repeats, measured sequentially:

```text
benchmarks dense n=32,   batch=20:  p50 20.060 us, cuSolverDx
benchmarks dense n=176,  batch=40:  p50 1,403.769 us, cuSolverDx
benchmarks dense n=352,  batch=40:  p50 41,095.626 us, cuSOLVER fallback
benchmarks dense n=512,  batch=640: p50 28,468.848 us, blocked panel-16 QR
benchmarks dense n=1024, batch=60:  p50 40,053.163 us, blocked panel-16 QR
first-five dense geomean: 4,208.097 us
```

The integrated `dense_linalg_best` first-five official `tests` manifest was refreshed on H200 after the `n=1024` promotion:

```sh
flox activate -- uv run qr-v2 render --suite tests --limit 5 --candidate dense_linalg_best
flox activate -- uv run qr-v2 verify-renders --suite tests
flox activate -- uv run qr-v2 plan-compile --suite tests --gpu-arch sm_90
flox activate -- uv run qr-v2 verify-compile-plan --suite tests
flox activate -- uv run qr-v2 compile --suite tests --timeout-seconds 300
flox activate -- uv run qr-v2 verify-compile-results --suite tests
flox activate -- uv run qr-v2 plan-run --suite tests
flox activate -- uv run qr-v2 verify-run-plan --suite tests
flox activate -- uv run qr-v2 run --suite tests --limit 5
flox activate -- uv run qr-v2 verify-run-results --suite tests
```

All passed. Final handoff checks also passed:

```sh
flox activate -- uv run ruff check .
flox activate -- uv run ty check
git diff --check
```

The repository wording audit for the disallowed kernel-launch hint family returned no matches.

Dense `n=2048` blocked-QR probe on H200:

- Test row `n=2048, batch=2, cond=2` compiled and passed checked execution on the blocked panel-16 path.
- Official benchmark row `n=2048, batch=8, cond=1` also compiled and passed checked execution on the blocked panel-16 path.
- Checker-free timing with 5 warmups and 20 repeats was slower than cuSOLVER on the same H200 row:

```text
blocked panel-16 QR: p50 108,327.829 us
cuSOLVER fallback:   p50  73,311.086 us
```

Conclusion: do not promote blocked QR for dense `n=2048`; keep the cuSOLVER fallback for that row unless a different blocked/update strategy is added.

Pre-promotion structured-row H200 comparison, 2 warmups and 5 repeats:

```text
row  case       n     batch  dense_linalg_best p50  semantic_combo p50  semantic routes
7    mixed      512   640       499,876.635 us    2,670,586.769 us  early=57, zero_tail=51, fallback=532
8    mixed      1024  60        229,590.422 us    1,752,004.138 us  zero_tail=4, fallback=56
9    rankdef    512   640       499,358.842 us    1,826,577.010 us  zero_tail=640
10   clustered  512   640       499,926.017 us    2,584,820.789 us  early=640
11   nearrank   1024  60        228,589.741 us    1,752,568.907 us  fallback=60
```

Conclusion: do not combine in the old `semantic_combo` implementation directly. Its route decisions are useful evidence, but the fallback/core QR is much slower than the library-backed dense dispatch. The next semantic attempt should put the zero-tail and early-stop reductions on top of a library-backed panel/block path, or use `dense_linalg_best` for all official structured rows until that hybrid exists.

Structured blocked-WY probe then validated the simpler next step: use the blocked panel-16 QR core for all official `n=512` and `n=1024` rows, not only dense rows. The probe candidate passed checked execution for benchmark rows 7-11:

```text
row  case       n     batch  blocked panel-16 QR p50
7    mixed      512   640       28,531.823 us
8    mixed      1024  60        40,240.168 us
9    rankdef    512   640       28,320.749 us
10   clustered  512   640       28,503.128 us
11   nearrank   1024  60        40,154.099 us
```

The blocked QR launcher now reuses its cuBLAS handle and device work buffers inside each generated artifact. That trims the repeated-launch p50s for `n=512/1024` rows. A later atomic probe promoted dense `n=352` to cached blocked QR with panel width 8; panel bracketing on the official benchmark row showed:

```text
n=352 panel 8:  p50  4,473.768 us
n=352 panel 16: p50  6,399.728 us
n=352 panel 32: p50 11,546.938 us
```

The promoted `dense_linalg_best` row for benchmark `n=352, batch=40` passed checked execution and timed at `4,486.118 us` p50. This drops the current 12-row H200 geomean estimate from `41,985.158 us` to `12,173.971 us` using the measured row p50s:

```text
0 dense n=32:        20.060 us
1 dense n=176:    1,403.769 us
2 dense n=352:    4,486.118 us
3 dense n=512:   25,503.413 us
4 dense n=1024:  38,619.475 us
5 dense n=2048:  73,311.086 us
6 dense n=4096:  47,354.265 us
7 mixed n=512:   25,491.320 us
8 mixed n=1024:  38,602.679 us
9 rankdef n=512: 25,385.264 us
10 cluster n=512:25,457.409 us
11 nearrank1024: 38,578.981 us
```

`dense_linalg_best` has been promoted accordingly: it now uses cuSolverDx for `n <= 176`, cached-workspace blocked QR with panel 8 for `n=352`, cached-workspace blocked panel-16 QR for every `n=512` and `n=1024` row, and cuSOLVER fallback for the remaining larger rows. After the `n=352` promotion, the full 22-row official `tests` suite rendered, compiled, ran, and verified successfully on H200 with `sm_90`. The benchmark `n=352` row also rendered, compiled, passed checked execution, and timed successfully. Full 12-row benchmark revalidation after the `n=352` promotion is the first resume item; the run was interrupted by the command-execution usage limit immediately after benchmark render verification.

Next concrete steps:

1. Resume by running full 12-row benchmark compile/run/verify for the promoted `dense_linalg_best` manifest.
2. Profile/tune the blocked QR path: replace the custom T builder if it becomes visible in profiling, and inspect whether the repeated `cublasSgeqrfBatched` panel calls or the three GEMM updates dominate.
3. Build semantic reductions only on top of the promoted blocked/library-backed path; the old `semantic_combo` core is not competitive on H200.
4. Attack the remaining slow rows: `n=2048` and `n=4096`. The current panel-16 blocked path is not the right `n=2048` replacement, so this needs a different large-row strategy.
5. If the package should be shared across machines, publish `mathdx` to a private Flox catalog and install it into the environment so `$FLOX_ENV` contains the SDK instead of relying on the local `result-mathdx` build output.
6. Rerun `ruff`, `ty`, `git diff --check`, and the repository wording audit before handing off or committing.

## Gotchas

- The worktree is intentionally dirty during this session; do not reset unrelated changes.
- Run commands through Flox from the repo root: `flox activate -- ...`. Do not fall back to system-wide Python or packages unless Flox truly lacks the tool.
- In this Lightning Studio, the normal command sandbox may fail with `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`; escalated command execution has been needed, but commands should still run as the Studio user and inside Flox.
- `plan-compile` defaults to `sm_89`; pass `--gpu-arch sm_90` on H100/H200. On B200, confirm the target with `nvcc --list-gpu-arch`.
- Nsight Compute performance counters are unavailable without admin/host changes in this Studio; prefer device-side instrumentation or ordinary wall-clock timing.
- No `pytest` dependency has been added yet; verification is CLI/manifests/JSONL-first.
- Keep generated artifacts under `data/qr_v2/`.
- Keep dependencies local to Flox/uv. Do not globally install Python packages or CUDA tools.
