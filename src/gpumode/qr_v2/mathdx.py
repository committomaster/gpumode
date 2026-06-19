"""MathDx/cuSolverDx package discovery helpers."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class MathDxDiscovery:
    root: Path
    include_dir: Path
    lib_dir: Path
    library_path: Path
    library_kind: str

    def to_record(self) -> dict[str, str]:
        record = asdict(self)
        return {key: value.as_posix() if isinstance(value, Path) else str(value) for key, value in record.items()}


def _looks_like_mathdx_root(path: Path) -> bool:
    return (path / "include" / "cusolverdx.hpp").is_file() and (path / "lib").is_dir()


def _version_dirs(root: Path) -> list[Path]:
    mathdx_root = root / "nvidia" / "mathdx"
    if not mathdx_root.is_dir():
        return []
    return sorted((path for path in mathdx_root.iterdir() if path.is_dir()), key=lambda path: path.name, reverse=True)


def _candidate_paths(*, project_root: Path, environ: Mapping[str, str]) -> list[Path]:
    paths: list[Path] = []
    flox_env = environ.get("FLOX_ENV")
    if flox_env:
        paths.append(Path(flox_env))
    paths.append(project_root / "result-mathdx")
    return paths


def discover_mathdx(
    *,
    project_root: Path = Path("."),
    environ: Mapping[str, str] | None = None,
) -> MathDxDiscovery | None:
    env = os.environ if environ is None else environ
    for candidate in _candidate_paths(project_root=project_root, environ=env):
        for root in [candidate, *_version_dirs(candidate)]:
            if not _looks_like_mathdx_root(root):
                continue
            lib_dir = root / "lib"
            fatbin = lib_dir / "libcusolverdx.fatbin"
            static_library = lib_dir / "libcusolverdx.a"
            if static_library.is_file():
                return MathDxDiscovery(
                    root=root,
                    include_dir=root / "include",
                    lib_dir=lib_dir,
                    library_path=static_library,
                    library_kind="static",
                )
            if fatbin.is_file():
                return MathDxDiscovery(
                    root=root,
                    include_dir=root / "include",
                    lib_dir=lib_dir,
                    library_path=fatbin,
                    library_kind="fatbin",
                )
    return None


def mathdx_compile_args(discovery: MathDxDiscovery) -> list[str]:
    args = ["-dlto", f"-I{discovery.include_dir.as_posix()}"]
    cutlass_include = discovery.root / "external" / "cutlass" / "include"
    if cutlass_include.is_dir():
        args.append(f"-I{cutlass_include.as_posix()}")
    if discovery.library_kind == "fatbin":
        args.append(discovery.library_path.as_posix())
    else:
        args.extend([f"-L{discovery.lib_dir.as_posix()}", "-lcusolverdx"])
    return args
