You can begin immediately without B200. The right split is:

```text
Before B200:
  build the correctness harness, candidate grammar, semantic probes,
  PyTorch extension plumbing, vendor wrappers, and most CUDA kernels.

On B200:
  validate target-specific performance, profile, tune, search, and freeze dispatch.
```

I would **not** wait for B200 access to start. For this challenge, use the search/admission philosophy: external search should explore aggressively, while any formal layer should only admit/certify a narrow set of transformations after they prove useful.

## B200 access estimate

Given the current top-three cutoff is roughly `2.57 ms`, my estimate is:

| Goal                             | B200 access needed if everything else is ready | Notes                                                                 |
| -------------------------------- | ---------------------------------------------: | --------------------------------------------------------------------- |
| Passing submission               |                        `5–15` single-GPU hours | Mostly environment, correctness, and baseline timing.                 |
| Serious top-10                   |                       `25–60` single-GPU hours | Enough for shape dispatch and a few custom paths.                     |
| Plausible top-3                  |                      `80–180` single-GPU hours | Needs candidate tournament, profiling, and multiple kernel revisions. |
| Beat current #1 / sub-ms attempt |                    `200–500+` single-GPU hours | Requires semantic work reduction plus low-level kernel wins.          |

If you rent an 8×B200 node and parallelize candidates across GPUs, the top-three range becomes roughly:

```text
10–25 wall-clock node-hours
```

but only if your harness is fully automated and you are not debugging basic build/correctness issues on the rented node. If you arrive unprepared, double those numbers.

## What you can do without B200

You can build almost everything except the final performance answer.

### 1. Build the official-style correctness harness

Implement the exact checker locally:

```python
Q = torch.linalg.householder_product(H, tau)
R = torch.triu(H)

factor_residual = norm1(R - Q.transpose(-1, -2) @ A_original)
orth_residual   = norm1(Q.transpose(-1, -2) @ Q - I)
lower_leakage   = norm1(torch.tril(Q.transpose(-1, -2) @ A_original, -1))
```

Run it in FP64 for diagnostics, but require FP32-output factors. This harness should be deterministic, log every failure, and report per-matrix diagnostics, not just pass/fail.

### 2. Implement semantic probes in pure PyTorch

Before writing kernels, test whether the proposal’s work-reduction ideas are real:

```text
early-stop feasibility
upper-triangular shortcut frequency
zero-tail reflector frequency
rank bucket distribution
column-scale span
pow2 diagonal gauge pass/fail
fast speculative path pass rate
```

The most important probe is:

```text
after k Householder panels, would tau[k:n] = 0 still pass?
```

If many rank-deficient, near-rank, clustered, or mixed matrices pass after (k \ll n), that is your massive-gain route.

### 3. Build the candidate tournament framework

Do this before kernels are fast:

```text
candidate registry
build cache
correctness gate
short benchmark gate
full benchmark gate
failure taxonomy
lineage tracking
per-shape/per-case dispatch table
```

The final object should be a frozen dispatch table, not one monolithic implementation.

### 4. Write vendor wrappers

Implement these wrappers early:

```text
cuSOLVER geqrf fallback
cuSolverDx geqrf small-matrix path
cuBLASLt update helper
CUB/CCCL reductions and prefix-sum compaction
```

cuSolverDx is especially relevant because its `geqrf` uses Householder transformations, overwrites the upper triangle with `R`, stores Householder vectors below the diagonal with `tau`, and supports row-major and column-major layouts. ([NVIDIA Docs][2])

### 5. Write portable CUDA kernels first

You can write and test many kernels on H100/H200 or even non-datacenter GPUs:

```text
scale_columns_pow2
unscale_upper_triangle
detect_upper_triangular
compute_column_scales
classify_matrix
compact_active_batch_indices
scatter_outputs
small GEQR2 reference kernel
panel-norm/reduction kernels
```

These kernels do not need SM100-specific Tensor Core code at first. They need to be correct, deterministic, and composable.

## What you cannot really do without B200

You cannot reliably answer:

```text
which nb wins on B200
whether cuBLASLt grouped GEMM beats custom CUTLASS for this shape
whether implicit-V update is worth it
whether CUDA Graph capture removes enough launch overhead
whether a kernel is register/SMEM/TMEM limited
what ptxas actually emits for sm_100
whether Nsight metrics show tensor-core, memory, or barrier stalls
```

B200 is compute capability 10.0, while RTX/Pro Blackwell cards in the public table are compute capability 12.0, so “Blackwell” testing on a workstation GPU is useful but not equivalent. ([NVIDIA Developer][3]) B200/GB200 has different SM100-targeted behavior, and the Blackwell tuning guide lists compute-capability-10.0-specific limits such as 64 resident warps/SM, 64K 32-bit registers/SM, 228 KB shared memory/SM, and 227 KB max shared memory per block. ([NVIDIA Docs][4])

## Recommended stack as of June 2026

### Primary languages

Use:

```text
Python:
  harness, candidate search, benchmark orchestration, result database

C++20/C++23:
  PyTorch extension glue, dispatch, workspace management

CUDA C++:
  kernels, cuSOLVER/cuBLASLt/cuSolverDx integration, graph capture

CUTLASS/CuTe:
  custom GEMM-like WY update kernels

Optional Lean:
  small law/admission layer only after the transformations prove useful
```

Do **not** put Lean in the hot implementation loop. Use it later for a small certificate layer if `Pow2Gauge`, `EarlyStop`, or `SpeculateThenFallback` becomes a winning family.

### CUDA and NVIDIA stack

Use **CUDA Toolkit 13.3** if the challenge environment permits it. CUDA 13.3 release notes list the 13.3 toolkit, driver requirements, C++23 support, and cuBLAS 13.3 updates; notably, cuBLAS 13.3 reports improved TF32 matrix multiplication performance on Blackwell/Blackwell Ultra by a geometric mean of 27%, with much larger gains on some small problems. ([NVIDIA Docs][5]) ([NVIDIA Docs][6])

Compile for B200 explicitly:

```bash
-gencode arch=compute_100,code=sm_100
-gencode arch=compute_100,code=compute_100
```

If you use architecture-specific features, check the installed toolkit with:

```bash
nvcc --list-gpu-arch
nvcc --list-gpu-code
```

and only add `sm_100a`/architecture-specific targets when the toolchain and challenge environment actually support them.

### PyTorch

Use **PyTorch 2.12** as the Python-facing layer if the environment allows. PyTorch 2.12 was released in May 2026, and its release notes include CUDA/linalg backend improvements; its packaging discussion says CUDA 13.0 remains the stable PyPI CUDA variant while CUDA 13.2 is introduced experimentally in the 2.12 cycle, so I would align with the contest image rather than assuming every CUDA 13.3 combination is packaged as a prebuilt wheel. ([PyTorch][7]) ([PyTorch Developer Mailing List][8])

Use `torch.utils.cpp_extension` for development, but prefer an ahead-of-time built extension for serious benchmarking. PyTorch’s extension docs support C++ and CUDA extension building and JIT loading, but final tournament runs should avoid hidden JIT/build noise. ([PyTorch Documentation][9])

### NVIDIA libraries

Use these in this order:

```text
cuSOLVER:
  large-n fallback and correctness oracle

cuSolverDx:
  n=32, n=176, maybe panel experiments

cuBLASLt:
  first implementation of batched/grouped WY updates

CUTLASS/CuTe:
  implicit-V and fused update kernels for n=512/1024

CUB/CCCL:
  reductions, scans, prefix-sum compaction, classification

CUDA Graphs:
  repeated panel/update workflows and candidate benchmarking
```

CUTLASS is essential for the B200-specific custom update path because its Blackwell SM100 docs expose `tcgen05.mma` support for Tensor Core GEMM paths, including TF32 and other formats; for this QR task, that matters mainly for the trailing update, not Householder panel generation. ([NVIDIA Docs][10]) CUDA Graphs are worth using because they let you define a repeated workflow once and relaunch it with reduced CPU overhead, which matters if your QR path has many panel kernels. ([NVIDIA Docs][11])

### Profiling and debugging

Use:

```text
Nsight Compute 2026.2
Nsight Systems
compute-sanitizer
cuda-gdb only for rare correctness bugs
nvdisasm
cuobjdump
ncu CLI in automated profile mode
```

Nsight Compute 2026.2 supports CUDA 13.3, CUDA Tile profiling, CUDA graph node profiling, and improved SASS/stall visualizations. ([NVIDIA Developer][12])

Before paying for a long B200 run, verify that your provider allows the profiling you need:

```bash
ncu --query-metrics
ncu --set speedOfLight ./your_smoke_binary
```

If performance counters are blocked, you can still benchmark, but serious kernel tuning becomes much harder.

## Recommended implementation plan

### Phase 0: off-B200 infrastructure

Target: no B200 access.

Deliverables:

```text
benchmark generator
official-style checker
candidate result schema
candidate registry
build cache
failure taxonomy
per-case timing database
plotting/dashboard
```

Use:

```text
Python + PyTorch + pytest
DuckDB or SQLite for results
Polars/Pandas for analysis
Jinja2 or simple templates for kernel variants
Ninja/CMake for builds
```

### Phase 1: off-B200 semantic feasibility

Target: no B200 access.

Implement slow/prototype versions of:

```text
UpperTriangularShortcut
Pow2ColumnGauge
PanelPow2ColumnGauge
ExactZeroReflectorPrune
EarlyStopAfterPanel(k)
SpeculateFastThenFallback
```

The output of this phase should be a table like:

```text
case, n, batch, median stop k, p90 stop k, shortcut rate, gauge helps?, fallback rate
```

If early stop does not trigger on the hard cases, kill it quickly. If it does trigger, promote it to CUDA.

### Phase 2: baseline native extension

Target: can be developed on H100/H200/RTX/CPU build boxes.

Deliver:

```text
catqr.geqrf(A) -> (H, tau)

Backends:
  cuSOLVER fallback
  cuSolverDx small path
  shortcut kernels
  pow2 scale/unscale kernels
  active-index compaction
```

At this point, you should already pass the full correctness suite on whatever CUDA GPU you have, even if the performance is not meaningful.

### Phase 3: first B200 session

Budget: `5–10` B200 GPU-hours.

Goals:

```text
validate environment
measure vendor baselines
measure semantic fast-path overhead
verify correctness under real contest driver/toolkit
collect first ncu profiles
identify top 2 bottleneck shapes
```

Stop if basic correctness is broken. Do not start deep tuning until correctness is boring.

### Phase 4: B200 tournament

Budget: `40–100` B200 GPU-hours.

Search over:

```text
n ∈ {352, 512, 1024}
nb ∈ {16, 32, 48, 64}
presentation ∈ {identity, pow2 gauge, early stop, shortcut, speculate+fallback}
update ∈ {cuBLASLt, CUTLASS materialized-V, CUTLASS implicit-V}
panel ∈ {custom panel, cuSolverDx panel, hybrid}
routing ∈ {whole batch, per-matrix compacted sub-batch}
```

Use adaptive racing:

```text
smoke correctness
short benchmark
keep top K per shape
full correctness
full benchmark
profile only the survivors
```

### Phase 5: top-three push

Budget: `40–80` more B200 GPU-hours.

This phase is about narrowing:

```text
remove slow fast-path checks
specialize by n
specialize by case/property
capture CUDA graphs
freeze workspaces
remove allocations
inspect SASS for register spills
reduce launch count
replace materialized-V if implicit-V wins
```

If you are within 10–15% of top three, this is where profiling matters. If you are still 2× away, go back to semantic work reduction, not tile polishing.

## What I would not use initially

I would not use:

```text
full Lean/mathlib compiler
MLIR/NVVM backend
raw PTX generator
RL policy training
Triton as the primary backend
CUDA Tile as the primary backend
```

Triton and CUDA Tile are useful for prototypes, but for this challenge I would keep the production path in CUDA C++/CUTLASS/cuBLASLt/cuSOLVER. MLIR/NVVM is overkill unless you already have that compiler infrastructure; the MLIR NVGPU/NVVM dialects are relevant for a serious compiler backend, but not for a short-run leaderboard implementation. ([mlir.llvm.org][13]) ([mlir.llvm.org][14])

## My practical recommendation

Start now without B200. Aim to arrive at your first B200 session with:

```text
1. a passing cuSOLVER/cuSolverDx baseline;
2. a deterministic benchmark harness;
3. semantic probes already analyzed;
4. candidate generation automated;
5. at least 20 candidate variants buildable;
6. zero dependency on interactive debugging.
```

Then reserve **at least 100 single-GPU-equivalent B200 hours** if top three is the target. If you can only get 20–30 hours, focus narrowly on semantic work reduction and vendor dispatch; do not attempt a full custom CUTLASS kernel search.

[1]: https://www.coreweave.com/pricing "CoreWeave Cloud Pricing | CoreWeave"
[2]: https://docs.nvidia.com/cuda/cusolverdx/get_started/functions/geqrf.html "QR Factorization — cuSOLVERDx"
[3]: https://developer.nvidia.com/cuda-gpus?from=20423&from_column=20423 "CUDA GPU Compute Capability | NVIDIA Developer"
[4]: https://docs.nvidia.com/cuda/archive/12.8.1/blackwell-tuning-guide/index.html "1. NVIDIA Blackwell Tuning Guide — Blackwell Tuning Guide 12.8 documentation"
[5]: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html?highlight=ptx&utm_source=chatgpt.com "CUDA Toolkit 13.3 - Release Notes — Release Notes 13.3 documentation"
[6]: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html "CUDA Toolkit 13.3 - Release Notes — Release Notes 13.3 documentation"
[7]: https://pytorch.org/blog/pytorch-2-12-release-blog/ "PyTorch 2.12 Release Blog – PyTorch"
[8]: https://dev-discuss.pytorch.org/t/introducing-cuda-13-2-and-deprecating-cuda-12-8-release-2-12/3337 "Introducing CUDA 13.2 and Deprecating CUDA 12.8 (Release 2.12) - release/packaging - PyTorch Developer Mailing List"
[9]: https://docs.pytorch.org/docs/stable/cpp_extension.html "torch.utils.cpp_extension — PyTorch 2.12 documentation"
[10]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html "Blackwell SM100 GEMMs — NVIDIA CUTLASS Documentation"
[11]: https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/04-special-topics/cuda-graphs.html "4.2. CUDA Graphs — CUDA Programming Guide"
[12]: https://developer.nvidia.com/tools-overview/nsight-compute/get-started "Getting Started with Nsight Compute | NVIDIA Developer"
[13]: https://mlir.llvm.org/docs/Dialects/NVGPU/ "'nvgpu' Dialect - MLIR"
[14]: https://mlir.llvm.org/docs/Dialects/NVVMDialect/ "'nvvm' Dialect - MLIR"
