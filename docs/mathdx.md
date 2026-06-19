# MathDx/cuSolverDx With Flox

Flox has CUDA 13.0 toolkit/library packages for this repo, but the Flox catalog used here does not currently expose NVIDIA MathDx/cuSolverDx under `mathdx`, `cusolverdx`, `commonDx`, or `cublasDx` package names.

The Flox-aligned path is to package MathDx as a Flox/Nix package with a pinned source hash, then consume that package output. Do not depend on an ad hoc local unpack directory or an activation hook that points at a cache.

This repo now defines that package at `.flox/pkgs/mathdx.nix`. Flox requires package files under `.flox/pkgs/` to be tracked by git, at least with `git add -N`, before `flox build` can evaluate them.

Official references:

- Flox CUDA package workflow: https://flox.dev/blog/get-nvidia-cuda-stacks-that-travel-across-your-sdlc-with-flox/
- Flox Nix expression builds: https://flox.dev/docs/concepts/nix-expression-builds
- Flox build/publish workflow: https://flox.dev/docs/tutorials/build-and-publish
- cuSolverDx installation: https://docs.nvidia.com/cuda/cusolverdx/get_started/installation.html
- cuSolverDx GEQRF: https://docs.nvidia.com/cuda/cusolverdx/get_started/functions/geqrf.html
- cuSolverDx download page: https://developer.nvidia.com/cusolverdx-downloads
- NVIDIA samples: https://github.com/NVIDIA/CUDALibrarySamples/tree/main/MathDx/cuSolverDx

## Package Shape

The package fetches NVIDIA's MathDx tarball with a fixed hash and installs the SDK into `$out`:

```nix
{ fetchurl, lib, stdenvNoCC }:

stdenvNoCC.mkDerivation rec {
  pname = "nvidia-mathdx";
  version = "26.03.0-cuda13";

  src = fetchurl {
    url = "https://developer.nvidia.com/downloads/compute/cuSOLVERDx/redist/cuSOLVERDx/cuda13/nvidia-mathdx-26.03.0-cuda13.tar.gz";
    hash = "sha256-lcJ0QFEC9qJj956wREe9dTn2bOZJd5vY4iX8uMxxWU0=";
  };

  sourceRoot = ".";
  dontBuild = true;

  installPhase = ''
    runHook preInstall

    mathdx_include="$(find . -path '*/include/cusolverdx.hpp' -print -quit)"
    if [ -z "$mathdx_include" ]; then
      echo "cusolverdx.hpp not found in MathDx archive" >&2
      exit 1
    fi

    mathdx_root="$(dirname "$(dirname "$mathdx_include")")"
    mkdir -p "$out"
    cp -R "$mathdx_root"/. "$out"/

    runHook postInstall
  '';

  meta = with lib; {
    description = "NVIDIA MathDx SDK containing cuSolverDx";
    homepage = "https://developer.nvidia.com/cusolverdx-downloads";
    license = licenses.unfreeRedistributable or licenses.unfree;
    platforms = platforms.linux;
  };
}
```

Then build it:

```sh
flox build mathdx
```

For local development, `qr-v2 mathdx-info` detects the Flox build result at `result-mathdx`. For team/repeated use, publish the package to a private catalog and install it into the environment. Once installed, MathDx should be visible under `$FLOX_ENV/include` and `$FLOX_ENV/lib`.

## Hash And License

NVIDIA's download page states that downloading/using MathDx is subject to NVIDIA's software license. Do not fetch the tarball implicitly from automation unless the user has explicitly accepted those terms.

The pinned `fetchurl` hash currently used by `.flox/pkgs/mathdx.nix` is:

```text
sha256-lcJ0QFEC9qJj956wREe9dTn2bOZJd5vY4iX8uMxxWU0=
```

If the MathDx source URL or version changes, set the hash to an empty string after accepting NVIDIA's license, run `flox build mathdx`, and copy the SRI hash from the fixed-output hash mismatch.

## Check Detection

```sh
flox activate -- uv run qr-v2 mathdx-info --stdout
```

When MathDx is visible, this reports the root, include path, library path, and compile args. `plan-compile` adds NVIDIA's required `-dlto`, the MathDx include path, and either `libcusolverdx.fatbin` or `-lcusolverdx` only for rendered artifacts that include `cusolverdx.hpp`.

On the current H200/Verda host after `flox build mathdx`, detection reports:

```json
{"compile_args":["-dlto","-Iresult-mathdx/include","-Iresult-mathdx/external/cutlass/include","-Lresult-mathdx/lib","-lcusolverdx"],"event":"mathdx_info","found":true,"include_dir":"result-mathdx/include","lib_dir":"result-mathdx/lib","library_kind":"static","library_path":"result-mathdx/lib/libcusolverdx.a","root":"result-mathdx"}
```

## Candidate Slice

The `cusolverdx` render candidate uses cuSolverDx for `n=32` and `n=176` rows, where block GEQRF can plausibly fit in shared memory, and falls back to the existing host cuSOLVER template for larger rows.

```sh
flox activate -- uv run qr-v2 render --suite tests --limit 3 --candidate cusolverdx
flox activate -- uv run qr-v2 verify-renders --suite tests
flox activate -- uv run qr-v2 plan-compile --suite tests --gpu-arch sm_90
flox activate -- uv run qr-v2 compile --suite tests --timeout-seconds 180
flox activate -- uv run qr-v2 plan-run --suite tests
flox activate -- uv run qr-v2 run --suite tests --limit 3
```

Use `sm_90` on the current H200/Verda host. The first three official `tests` rows now compile and pass checked execution for the `cusolverdx` candidate: `n=32` and `n=176` use cuSolverDx GEQRF, while `n=352` uses the existing cuSOLVER fallback.

On B200, first confirm the target with:

```sh
flox activate -- nvcc --list-gpu-arch
```
