# qr_v2 L4 Handoff

Generated: 2026-06-18

This file summarizes the current project state and the next steps for continuing on the Linux x86_64 NVIDIA L4 host.

## Current Project Shape

This is a minimal `uv` Python app in a Flox environment.

- Package name: `gpumode`
- CLI entry point: `qr-v2 = "gpumode.qr_v2.cli:main"`
- Source layout: flat package under `src/gpumode/qr_v2/`
- Python requirement: `>=3.13`
- All generated experiment data should live under `./data/`
- `uv.toml` was removed; uv config lives in `pyproject.toml`
- Caches are configured away from the source tree:
  - Ruff cache: `.flox/cache/ruff`
  - Python bytecode cache: `.flox/cache/python` via `PYTHONPYCACHEPREFIX`

Pinned runtime dependencies:

- `torch==2.12.1`
- `numpy==2.4.6`
- `polars==1.41.2`
- `jinja2==3.1.6`
- `rich==15.0.0`

Pinned dev dependencies:

- `ruff==0.15.17`
- `ty==0.0.50`
- `lightning-sdk==2026.6.8.post0`

The Lightning CLI is installed as a local project/dev dependency through uv, not globally.

## Flox State

Current `.flox/env/manifest.toml` installs:

- `python313 == python3-3.13.13`
- `uv == 0.11.19`
- `gh == 2.94.0`
- `age == 1.3.1`
- `sops == 3.13.1`
- `yq-go == 4.53.3`

Current hook:

```sh
export PYTHONPATH="$FLOX_ENV_PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="$FLOX_ENV_CACHE/python"
```

Important uv config in `pyproject.toml`:

```toml
[tool.uv]
package = true
python-downloads = "never"
python-preference = "only-system"
```

This means uv must use the Flox Python and must not download its own interpreter.

## Implemented qr_v2 Modules

`src/gpumode/qr_v2/specs.py`

- Official smoke/test/benchmark spec definitions.

`src/gpumode/qr_v2/inputs.py`

- Official-style input generation.

`src/gpumode/qr_v2/check.py`

- Compact Householder QR checking helpers.

`src/gpumode/qr_v2/baseline.py`

- Torch `geqrf` baseline runner.

`src/gpumode/qr_v2/probes.py`

- Structure probes for generated inputs.

`src/gpumode/qr_v2/jsonl.py`

- Small JSONL writer helpers.

`src/gpumode/qr_v2/summary.py`

- Polars summaries and Rich/TSV output.

`src/gpumode/qr_v2/render.py`

- Jinja2 CUDA stub rendering.
- Writes artifacts under `data/qr_v2/renders/<suite>/...`.
- Writes render manifest at `data/qr_v2/renders/<suite>/manifest.jsonl`.
- Includes `verify_render_manifest`.

`src/gpumode/qr_v2/compile_plan.py`

- Converts render manifests into future nvcc compile jobs.
- Writes compile plan at `data/qr_v2/compile-plans/<suite>/manifest.jsonl`.
- Future compiled outputs are planned under `data/qr_v2/compiled/<suite>/...`.
- Includes `verify_compile_plan_manifest`.

`src/gpumode/qr_v2/run_plan.py`

- Converts compile plans into future run jobs.
- Writes run plan at `data/qr_v2/run-plans/<suite>/manifest.jsonl`.
- Future run/check results are planned under `data/qr_v2/run-results/<suite>/...`.
- Includes `verify_run_plan_manifest`.
- Does not require compiled `.so` files to exist yet, so it works on macOS.

`src/gpumode/qr_v2/cli.py`

- Uses `argparse` plus `rich`.
- Current commands:
  - `baseline`
  - `probe`
  - `summary`
  - `render`
  - `verify-renders`
  - `plan-compile`
  - `verify-compile-plan`
  - `plan-run`
  - `verify-run-plan`

We intentionally did not add `pytest` yet. The current flow is CLI/manifests/JSONL first.

## Current Pipeline

The macOS-side planning pipeline is:

```text
render
verify-renders
plan-compile
verify-compile-plan
plan-run
verify-run-plan
```

For smoke data, the current generated files are under:

```text
data/qr_v2/renders/smoke/
data/qr_v2/compile-plans/smoke/manifest.jsonl
data/qr_v2/run-plans/smoke/manifest.jsonl
```

The rendered CUDA files are stubs. They are useful for validating the planning/compile machinery, but they are not the real QR kernel yet.

## Verification Commands Used

These passed on macOS before moving to the L4 work:

```sh
uv run qr-v2 verify-renders --suite smoke
uv run qr-v2 verify-compile-plan --suite smoke
uv run qr-v2 verify-run-plan --suite smoke
uv run ruff check .
uv run ty check
uv sync --check
env -u PYTHONPATH uv run python -m gpumode.qr_v2.cli verify-run-plan --suite smoke
```

No stray `__pycache__`, `.ruff_cache`, or egg-info directories were left outside `.flox/cache`.

## L4 Host Notes

Initial L4 info from `l4.txt` / manual commands:

```text
uname -m: x86_64
GPU: NVIDIA L4
Driver Version: 580.159.03
nvidia-smi CUDA Version: 13.0
```

Flox is now confirmed to expose the right Python and uv when checked with:

```sh
flox activate -d . -c 'type -a python python3 uv; python --version; uv --version'
```

Expected output includes:

```text
python is .../.flox/run/x86_64-linux.gpumode.dev/bin/python
Python 3.13.13
uv 0.11.19
```

There was one uv interpreter discovery problem on L4:

```text
uv sync
error: No interpreter found for Python 3.13 in managed installations
hint: A managed Python download is available for Python 3.13, but Python downloads are set to 'never'
```

If that happens, force uv to use Flox Python:

```sh
flox activate -d .
echo "$FLOX_ENV"
"$FLOX_ENV/bin/python" --version
"$FLOX_ENV/bin/uv" sync --python "$FLOX_ENV/bin/python"
```

If this works, make it sticky by adding this to the Flox hook:

```sh
export UV_PYTHON="$FLOX_ENV/bin/python"
```

Do not pin `.python-version` to an absolute `.flox/run/...` path unless it is kept machine-local and not committed.

## CUDA/Flox Dependencies For L4

The NVIDIA driver comes from the host, not Flox. Check it with:

```sh
nvidia-smi
```

Flox package names verified from package metadata:

```sh
flox install -d . \
  -i cuda_nvcc cudaPackages.cuda_nvcc@12.9.86 \
  -i cuda_cudart cudaPackages.cuda_cudart@12.9.79 \
  -i cuda_cccl cudaPackages.cuda_cccl@12.9.27 \
  -i gcc gcc13@13.4.0
```

Then reactivate and check:

```sh
flox activate -d .
nvcc --version
nvcc --list-gpu-arch | grep compute_89
gcc --version
```

For L4, target `sm_89`. Do not use the B200/Blackwell `sm_100` notes from `docs/plan.md` for this L4 phase.

Optional later tooling:

```sh
flox install -d . \
  -i cuda_cuobjdump cudaPackages.cuda_cuobjdump@12.9.82 \
  -i cuda_nvdisasm cudaPackages.cuda_nvdisasm@12.9.88 \
  -i binutils binutils
```

Defer profiler/tooling packages until compile/run works.

## Immediate L4 Commands

From the L4 repo:

```sh
cd /teamspace/studios/this_studio/gpumode
flox activate -d .

python --version
uv --version
uv sync --python "$FLOX_ENV/bin/python"

uv run qr-v2 verify-renders --suite smoke
uv run qr-v2 verify-compile-plan --suite smoke
uv run qr-v2 verify-run-plan --suite smoke
```

If uv still does not pick the right interpreter, try:

```sh
UV_PYTHON="$FLOX_ENV/bin/python" uv sync
UV_PYTHON="$FLOX_ENV/bin/python" uv run python --version
```

## Next Implementation Step

Add `qr-v2 compile-one`.

Suggested behavior:

```sh
uv run qr-v2 compile-one --suite smoke --index 0
```

It should:

- Read exactly one row from `data/qr_v2/compile-plans/<suite>/manifest.jsonl`.
- Run that row's `compile_argv`.
- Ensure output paths stay under `data/qr_v2/`.
- Create parent directories for the planned `.so`.
- Measure duration.
- Capture return code.
- Capture bounded stdout/stderr snippets.
- Record `nvcc --version`.
- On success, record output `.so` bytes and SHA256.
- Write a compile result JSONL record under `data/qr_v2/compile-results/<suite>/...`.

Keep the failure path useful on any machine:

- Missing compile plan should emit a JSONL-style error and return nonzero.
- Missing `nvcc` should emit a compile result record with nonzero status.
- A failed compile should preserve stderr in the result record.

After `compile-one` works:

1. Add a compile-result verifier.
2. Add `qr-v2 run-one` that consumes one run-plan row and one compiled `.so`.
3. Write run/check result JSONL under `data/qr_v2/run-results/<suite>/...`.
4. Only then start replacing the CUDA stub body with the real QR kernel.

## Gotchas

- `from __future__ import annotations` was removed because we are on Python 3.13 and were not relying on it.
- The CLI uses `argparse` plus `rich`; Typer is not installed.
- `structlog` and `msgspec` were discussed and intentionally not added.
- `pytest` was discussed and intentionally not added yet.
- Use JSONL plus Polars for data, not extra Python data dependencies.
- Keep all generated data under `./data/`.
- Keep project tooling local to uv/Flox; do not install Lightning or Python packages globally.
