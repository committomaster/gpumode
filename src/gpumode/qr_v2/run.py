"""Run execution for compiled qr_v2 CUDA artifacts."""

import ctypes
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from gpumode.qr_v2.check import check_compact_qr
from gpumode.qr_v2.inputs import generate_input
from gpumode.qr_v2.jsonl import write_jsonl


RUN_RESULT_VERSION = "run_result_v1"
DEFAULT_SNIPPET_CHARS = 4000
DEFAULT_THREADS_PER_BLOCK = 128
PROFILE_TEMPLATE_ID = "cuda_geqr2_parallel_profile_v1"
PROFILE_COUNTER_FIELDS = (
    "total_cycles",
    "init_cycles",
    "norm_cycles",
    "scale_cycles",
    "dot_cycles",
    "update_cycles",
    "active_reflectors",
    "inactive_reflectors",
    "trailing_columns",
)
PROFILE_PHASE_FIELDS = (
    "init_cycles",
    "norm_cycles",
    "scale_cycles",
    "dot_cycles",
    "update_cycles",
)


@dataclass(frozen=True, slots=True)
class RunExecutionResult:
    result_path: Path
    records: list[dict[str, object]]

    @property
    def ok(self) -> bool:
        return all(bool(record.get("ok")) for record in self.records)


@dataclass(frozen=True, slots=True)
class RunResultVerificationIssue:
    code: str
    message: str
    run_result_path: str | None = None
    compiled_artifact_path: str | None = None
    line_number: int | None = None

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "code": self.code,
            "message": self.message,
        }
        if self.run_result_path is not None:
            record["run_result_path"] = self.run_result_path
        if self.compiled_artifact_path is not None:
            record["compiled_artifact_path"] = self.compiled_artifact_path
        if self.line_number is not None:
            record["line_number"] = self.line_number
        return record


@dataclass(frozen=True, slots=True)
class RunResultVerificationResult:
    manifest_path: Path
    records_checked: int
    result_files_checked: int
    artifacts_checked: int
    issues: list[RunResultVerificationIssue]

    @property
    def ok(self) -> bool:
        return not self.issues


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"{path} contains a non-object JSONL row")
            records.append(record)
    return records


def _required_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"run plan row is missing string field {field!r}")
    return value


def _required_int(record: dict[str, Any], field: str) -> int:
    value = record.get(field)
    if not isinstance(value, int):
        raise ValueError(f"run plan row is missing integer field {field!r}")
    return value


def _is_relative_under(path_text: str, prefix: tuple[str, ...]) -> bool:
    path = Path(path_text)
    return not path.is_absolute() and path.parts[: len(prefix)] == prefix


def _check_execution_path(path_text: str, *, field: str, prefix: tuple[str, ...]) -> None:
    if not _is_relative_under(path_text, prefix):
        joined = "/".join(prefix)
        raise ValueError(f"{field} must be relative and stay under {joined}/")


def _bounded_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n[... omitted {omitted} chars ...]"


def _tensor_sha256(tensor: torch.Tensor) -> str:
    content = tensor.detach().cpu().contiguous().numpy().tobytes()
    return _sha256_bytes(content)


def _uses_device_profile(plan_record: dict[str, Any] | None, launcher_entry_point: str | None) -> bool:
    if launcher_entry_point is not None and launcher_entry_point.endswith("_launch_profile"):
        return True
    if plan_record is None:
        return False
    return plan_record.get("template_id") == PROFILE_TEMPLATE_ID


def _profile_cycle_summary(profile: torch.Tensor) -> dict[str, object]:
    cpu_profile = profile.detach().cpu().contiguous()
    rows = [[int(value) for value in row] for row in cpu_profile.tolist()]
    if not rows:
        return {
            "profile_source": "device_clock64",
            "profile_field_names": list(PROFILE_COUNTER_FIELDS),
            "profile_block_count": 0,
            "profile_sha256": _tensor_sha256(cpu_profile),
            "profile_cycles_sum": {field: 0 for field in PROFILE_COUNTER_FIELDS},
            "profile_cycles_mean": {field: 0.0 for field in PROFILE_COUNTER_FIELDS},
            "profile_cycles_max": {field: 0 for field in PROFILE_COUNTER_FIELDS},
            "profile_cycles_min": {field: 0 for field in PROFILE_COUNTER_FIELDS},
            "profile_measured_phase_cycles_sum": 0,
            "profile_phase_cycle_percent": {field: 0.0 for field in PROFILE_PHASE_FIELDS},
            "profile_worst_matrix": -1,
            "profile_worst_matrix_cycles": {field: 0 for field in PROFILE_COUNTER_FIELDS},
        }

    columns = list(zip(*rows, strict=True))
    sums = [sum(column) for column in columns]
    means = [sum(column) / len(column) for column in columns]
    maxima = [max(column) for column in columns]
    minima = [min(column) for column in columns]
    worst_matrix = max(range(len(rows)), key=lambda index: rows[index][0])

    cycle_sum = {field: sums[index] for index, field in enumerate(PROFILE_COUNTER_FIELDS)}
    cycle_mean = {field: means[index] for index, field in enumerate(PROFILE_COUNTER_FIELDS)}
    cycle_max = {field: maxima[index] for index, field in enumerate(PROFILE_COUNTER_FIELDS)}
    cycle_min = {field: minima[index] for index, field in enumerate(PROFILE_COUNTER_FIELDS)}
    worst_cycles = {field: rows[worst_matrix][index] for index, field in enumerate(PROFILE_COUNTER_FIELDS)}

    measured_phase_sum = sum(cycle_sum[field] for field in PROFILE_PHASE_FIELDS)
    if measured_phase_sum > 0:
        phase_percent = {field: 100.0 * cycle_sum[field] / measured_phase_sum for field in PROFILE_PHASE_FIELDS}
    else:
        phase_percent = {field: 0.0 for field in PROFILE_PHASE_FIELDS}

    return {
        "profile_source": "device_clock64",
        "profile_field_names": list(PROFILE_COUNTER_FIELDS),
        "profile_block_count": len(rows),
        "profile_sha256": _tensor_sha256(cpu_profile),
        "profile_cycles_sum": cycle_sum,
        "profile_cycles_mean": cycle_mean,
        "profile_cycles_max": cycle_max,
        "profile_cycles_min": cycle_min,
        "profile_measured_phase_cycles_sum": int(measured_phase_sum),
        "profile_phase_cycle_percent": phase_percent,
        "profile_worst_matrix": worst_matrix,
        "profile_worst_matrix_cycles": worst_cycles,
    }


def _cuda_device_index(device: str | None) -> int:
    if device is None:
        return 0
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("run-one currently requires a CUDA device")
    return 0 if resolved.index is None else int(resolved.index)


def _result_path_from_artifact(*, suite: str, compiled_artifact_path: str) -> Path:
    artifact = Path(compiled_artifact_path)
    candidate_id = artifact.parent.name if artifact.parent.name else "unknown_candidate"
    return Path("data") / "qr_v2" / "run-results" / suite / candidate_id / artifact.with_suffix(".jsonl").name


def _write_result(result_path: Path, records: list[dict[str, object]]) -> RunExecutionResult:
    write_jsonl(result_path, records)
    return RunExecutionResult(result_path=result_path, records=records)


def _base_record(
    *,
    suite: str,
    plan_record: dict[str, Any] | None,
    run_plan_manifest_path: Path | None,
    plan_index: int,
    result_path: Path,
    compiled_artifact_path: str | None,
    entry_point: str | None,
    launcher_entry_point: str | None,
    batch: int | None,
    n: int | None,
    cond: int | None,
    seed: int | None,
    case: str | None,
    threads_per_block: int,
    max_snippet_chars: int,
) -> dict[str, object]:
    record: dict[str, object] = {
        "event": "run_result",
        "result_version": RUN_RESULT_VERSION,
        "suite": suite,
        "plan_index": plan_index,
        "run_result_path": result_path.as_posix(),
        "check_result_path": result_path.as_posix(),
        "ok": False,
        "passed": False,
        "status": "input_error",
        "returncode": 1,
        "elapsed_ns": 0,
        "stdout_snippet": "",
        "stderr_snippet": "",
        "max_snippet_chars": max_snippet_chars,
    }
    if run_plan_manifest_path is not None:
        record["run_plan_manifest_path"] = run_plan_manifest_path.as_posix()
    if compiled_artifact_path is not None:
        record["compiled_artifact_path"] = compiled_artifact_path
    if entry_point is not None:
        record["entry_point"] = entry_point
    if launcher_entry_point is not None:
        record["launcher_entry_point"] = launcher_entry_point
    if batch is not None:
        record["batch"] = batch
    if n is not None:
        record["n"] = n
    if cond is not None:
        record["cond"] = cond
    if seed is not None:
        record["seed"] = seed
    if case is not None:
        record["case"] = case
    record["threads_per_block"] = threads_per_block

    if plan_record is None:
        return record

    for field in (
        "compile_plan_index",
        "compile_plan_version",
        "plan_version",
        "target_backend",
        "runner",
        "runner_family",
        "runner_cwd",
        "compiled_artifact_kind",
        "source_artifact_path",
        "source_sha256",
        "candidate_id",
        "template_id",
        "tile_size",
        "unroll",
    ):
        if field in plan_record:
            value = plan_record[field]
            if isinstance(value, (str, int, bool)):
                record[field] = value
    return record


def _input_error_record(
    *,
    suite: str,
    run_plan_manifest_path: Path | None,
    plan_index: int,
    result_path: Path,
    code: str,
    message: str,
    max_snippet_chars: int,
) -> dict[str, object]:
    record = _base_record(
        suite=suite,
        plan_record=None,
        run_plan_manifest_path=run_plan_manifest_path,
        plan_index=plan_index,
        result_path=result_path,
        compiled_artifact_path=None,
        entry_point=None,
        launcher_entry_point=None,
        batch=None,
        n=None,
        cond=None,
        seed=None,
        case=None,
        threads_per_block=DEFAULT_THREADS_PER_BLOCK,
        max_snippet_chars=max_snippet_chars,
    )
    record.update({"status": code, "stderr_snippet": _bounded_text(message, max_chars=max_snippet_chars)})
    return record


def _record_artifact_metadata(record: dict[str, object], compiled_artifact_path: str) -> None:
    artifact = Path(compiled_artifact_path)
    record["compiled_artifact_exists"] = artifact.exists()
    if artifact.exists():
        content = artifact.read_bytes()
        record["compiled_artifact_bytes"] = len(content)
        record["compiled_artifact_sha256"] = _sha256_bytes(content)


def _run_kernel_record(
    *,
    suite: str,
    plan_record: dict[str, Any] | None,
    run_plan_manifest_path: Path | None,
    plan_index: int,
    result_path: Path,
    compiled_artifact_path: str,
    entry_point: str,
    launcher_entry_point: str | None,
    batch: int,
    n: int,
    cond: int,
    seed: int,
    case: str,
    threads_per_block: int,
    device: str | None,
    max_snippet_chars: int,
) -> dict[str, object]:
    _check_execution_path(compiled_artifact_path, field="compiled_artifact_path", prefix=("data", "qr_v2", "compiled"))
    _check_execution_path(result_path.as_posix(), field="run_result_path", prefix=("data", "qr_v2", "run-results"))

    collect_device_profile = _uses_device_profile(plan_record, launcher_entry_point)
    launcher_name = launcher_entry_point or (
        f"{entry_point}_launch_profile" if collect_device_profile else f"{entry_point}_launch"
    )
    record = _base_record(
        suite=suite,
        plan_record=plan_record,
        run_plan_manifest_path=run_plan_manifest_path,
        plan_index=plan_index,
        result_path=result_path,
        compiled_artifact_path=compiled_artifact_path,
        entry_point=entry_point,
        launcher_entry_point=launcher_name,
        batch=batch,
        n=n,
        cond=cond,
        seed=seed,
        case=case,
        threads_per_block=threads_per_block,
        max_snippet_chars=max_snippet_chars,
    )
    _record_artifact_metadata(record, compiled_artifact_path)

    artifact = Path(compiled_artifact_path)
    if not artifact.exists():
        record.update(
            {
                "status": "missing_compiled_artifact",
                "stderr_snippet": f"missing compiled artifact: {compiled_artifact_path}",
            }
        )
        return record

    if not torch.cuda.is_available():
        record.update(
            {
                "status": "cuda_unavailable",
                "stderr_snippet": "torch.cuda.is_available() is false",
            }
        )
        return record

    start_ns = time.perf_counter_ns()
    try:
        device_index = _cuda_device_index(device)
        torch.cuda.set_device(device_index)
        resolved_device = torch.device("cuda", device_index)
        record["cuda_device_index"] = device_index
        record["cuda_device_name"] = torch.cuda.get_device_name(device_index)

        a = generate_input(batch=batch, n=n, cond=cond, seed=seed, case=case, device=resolved_device)
        h = torch.zeros_like(a)
        tau = torch.zeros((batch, n), device=resolved_device, dtype=torch.float32)
        record["input_sha256"] = _tensor_sha256(a)

        library = ctypes.CDLL(artifact.as_posix())
        try:
            launcher = getattr(library, launcher_name)
        except AttributeError:
            elapsed_ns = time.perf_counter_ns() - start_ns
            record.update(
                {
                    "status": "missing_launcher",
                    "returncode": 127,
                    "elapsed_ns": elapsed_ns,
                    "elapsed_ms": elapsed_ns / 1_000_000.0,
                    "stderr_snippet": f"missing launcher symbol: {launcher_name}",
                }
            )
            return record

        profile = None
        if collect_device_profile:
            profile = torch.zeros(
                (batch, len(PROFILE_COUNTER_FIELDS)),
                device=resolved_device,
                dtype=torch.uint64,
            )
            launcher.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            launcher.restype = ctypes.c_int
            returncode = int(
                launcher(
                    ctypes.c_void_p(a.data_ptr()),
                    ctypes.c_void_p(h.data_ptr()),
                    ctypes.c_void_p(tau.data_ptr()),
                    ctypes.c_void_p(profile.data_ptr()),
                    ctypes.c_int(batch),
                    ctypes.c_int(n),
                    ctypes.c_int(threads_per_block),
                    ctypes.c_int(len(PROFILE_COUNTER_FIELDS)),
                )
            )
        else:
            launcher.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            launcher.restype = ctypes.c_int
            returncode = int(
                launcher(
                    ctypes.c_void_p(a.data_ptr()),
                    ctypes.c_void_p(h.data_ptr()),
                    ctypes.c_void_p(tau.data_ptr()),
                    ctypes.c_int(batch),
                    ctypes.c_int(n),
                    ctypes.c_int(threads_per_block),
                )
            )
        elapsed_ns = time.perf_counter_ns() - start_ns
        record.update(
            {
                "returncode": returncode,
                "elapsed_ns": elapsed_ns,
                "elapsed_ms": elapsed_ns / 1_000_000.0,
                "output_h_sha256": _tensor_sha256(h),
                "output_tau_sha256": _tensor_sha256(tau),
            }
        )
        if returncode == 0 and profile is not None:
            record.update(_profile_cycle_summary(profile))
        if returncode != 0:
            record.update(
                {
                    "status": "launch_failed",
                    "stderr_snippet": f"launcher returned CUDA error code {returncode}",
                }
            )
            return record

        check_result = check_compact_qr(a, h, tau)
        record.update(asdict(check_result))
        record.update(
            {
                "ok": True,
                "passed": check_result.passed,
                "status": "checked",
                "stdout_snippet": "",
                "stderr_snippet": "" if check_result.passed else "QR check completed but did not pass",
            }
        )
        return record
    except Exception as error:
        elapsed_ns = time.perf_counter_ns() - start_ns
        record.update(
            {
                "ok": False,
                "passed": False,
                "status": "run_error",
                "returncode": 1,
                "elapsed_ns": elapsed_ns,
                "elapsed_ms": elapsed_ns / 1_000_000.0,
                "stderr_snippet": _bounded_text(f"{type(error).__name__}: {error}", max_chars=max_snippet_chars),
            }
        )
        return record


def run_one(
    *,
    suite: str,
    compiled_artifact_path: Path,
    entry_point: str,
    batch: int,
    n: int,
    cond: int,
    seed: int,
    case: str,
    result_path: Path | None = None,
    launcher_entry_point: str | None = None,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    device: str | None = None,
    max_snippet_chars: int = DEFAULT_SNIPPET_CHARS,
) -> RunExecutionResult:
    compiled_artifact_text = compiled_artifact_path.as_posix()
    resolved_result_path = result_path or _result_path_from_artifact(
        suite=suite,
        compiled_artifact_path=compiled_artifact_text,
    )
    try:
        record = _run_kernel_record(
            suite=suite,
            plan_record=None,
            run_plan_manifest_path=None,
            plan_index=0,
            result_path=resolved_result_path,
            compiled_artifact_path=compiled_artifact_text,
            entry_point=entry_point,
            launcher_entry_point=launcher_entry_point,
            batch=batch,
            n=n,
            cond=cond,
            seed=seed,
            case=case,
            threads_per_block=threads_per_block,
            device=device,
            max_snippet_chars=max_snippet_chars,
        )
    except (OSError, ValueError) as error:
        record = _input_error_record(
            suite=suite,
            run_plan_manifest_path=None,
            plan_index=0,
            result_path=resolved_result_path,
            code="run_input_error",
            message=str(error),
            max_snippet_chars=max_snippet_chars,
        )
    return _write_result(resolved_result_path, [record])


def run_one_from_plan_manifest(
    *,
    suite: str,
    run_plan_manifest_path: Path,
    index: int,
    result_path: Path | None = None,
    device: str | None = None,
    max_snippet_chars: int = DEFAULT_SNIPPET_CHARS,
) -> RunExecutionResult:
    fallback_result_path = result_path or (Path("data") / "qr_v2" / "run-results" / suite / "manifest.jsonl")
    if not run_plan_manifest_path.exists():
        record = _input_error_record(
            suite=suite,
            run_plan_manifest_path=run_plan_manifest_path,
            plan_index=index,
            result_path=fallback_result_path,
            code="missing_run_plan",
            message=f"missing run plan: {run_plan_manifest_path}",
            max_snippet_chars=max_snippet_chars,
        )
        return _write_result(fallback_result_path, [record])

    try:
        plan_records = _read_jsonl_objects(run_plan_manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        record = _input_error_record(
            suite=suite,
            run_plan_manifest_path=run_plan_manifest_path,
            plan_index=index,
            result_path=fallback_result_path,
            code="invalid_run_plan",
            message=str(error),
            max_snippet_chars=max_snippet_chars,
        )
        return _write_result(fallback_result_path, [record])

    if index < 0 or index >= len(plan_records):
        record = _input_error_record(
            suite=suite,
            run_plan_manifest_path=run_plan_manifest_path,
            plan_index=index,
            result_path=fallback_result_path,
            code="run_plan_index_out_of_range",
            message=f"run plan index {index} is outside 0..{len(plan_records) - 1}",
            max_snippet_chars=max_snippet_chars,
        )
        return _write_result(fallback_result_path, [record])

    plan_record = plan_records[index]
    try:
        row_result_path = result_path or Path(_required_string(plan_record, "run_result_path"))
        record = _run_kernel_record(
            suite=suite,
            plan_record=plan_record,
            run_plan_manifest_path=run_plan_manifest_path,
            plan_index=_required_int(plan_record, "plan_index"),
            result_path=row_result_path,
            compiled_artifact_path=_required_string(plan_record, "compiled_artifact_path"),
            entry_point=_required_string(plan_record, "entry_point"),
            launcher_entry_point=plan_record.get("launcher_entry_point")
            if isinstance(plan_record.get("launcher_entry_point"), str)
            else None,
            batch=_required_int(plan_record, "batch"),
            n=_required_int(plan_record, "n"),
            cond=_required_int(plan_record, "cond"),
            seed=_required_int(plan_record, "seed"),
            case=_required_string(plan_record, "case"),
            threads_per_block=_required_int(plan_record, "threads_per_block"),
            device=device,
            max_snippet_chars=max_snippet_chars,
        )
    except (OSError, ValueError) as error:
        record = _input_error_record(
            suite=suite,
            run_plan_manifest_path=run_plan_manifest_path,
            plan_index=index,
            result_path=fallback_result_path,
            code="run_input_error",
            message=str(error),
            max_snippet_chars=max_snippet_chars,
        )
        return _write_result(fallback_result_path, [record])
    return _write_result(Path(str(record["run_result_path"])), [record])



def run_from_plan_manifest(
    *,
    suite: str,
    run_plan_manifest_path: Path,
    out_dir: Path,
    limit: int | None = None,
    device: str | None = None,
    max_snippet_chars: int = DEFAULT_SNIPPET_CHARS,
) -> RunExecutionResult:
    manifest_path = out_dir / "manifest.jsonl"
    if limit is not None and limit < 0:
        record = _input_error_record(
            suite=suite,
            run_plan_manifest_path=run_plan_manifest_path,
            plan_index=0,
            result_path=manifest_path,
            code="invalid_limit",
            message="limit must be non-negative",
            max_snippet_chars=max_snippet_chars,
        )
        return _write_result(manifest_path, [record])

    if not run_plan_manifest_path.exists():
        record = _input_error_record(
            suite=suite,
            run_plan_manifest_path=run_plan_manifest_path,
            plan_index=0,
            result_path=manifest_path,
            code="missing_run_plan",
            message=f"missing run plan: {run_plan_manifest_path}",
            max_snippet_chars=max_snippet_chars,
        )
        return _write_result(manifest_path, [record])

    try:
        plan_records = _read_jsonl_objects(run_plan_manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        record = _input_error_record(
            suite=suite,
            run_plan_manifest_path=run_plan_manifest_path,
            plan_index=0,
            result_path=manifest_path,
            code="invalid_run_plan",
            message=str(error),
            max_snippet_chars=max_snippet_chars,
        )
        return _write_result(manifest_path, [record])

    selected_records = plan_records[:limit] if limit is not None else plan_records
    records: list[dict[str, object]] = []
    for selected_index, plan_record in enumerate(selected_records):
        try:
            row_result_path = Path(_required_string(plan_record, "run_result_path"))
            record = _run_kernel_record(
                suite=suite,
                plan_record=plan_record,
                run_plan_manifest_path=run_plan_manifest_path,
                plan_index=_required_int(plan_record, "plan_index"),
                result_path=row_result_path,
                compiled_artifact_path=_required_string(plan_record, "compiled_artifact_path"),
                entry_point=_required_string(plan_record, "entry_point"),
                launcher_entry_point=plan_record.get("launcher_entry_point")
                if isinstance(plan_record.get("launcher_entry_point"), str)
                else None,
                batch=_required_int(plan_record, "batch"),
                n=_required_int(plan_record, "n"),
                cond=_required_int(plan_record, "cond"),
                seed=_required_int(plan_record, "seed"),
                case=_required_string(plan_record, "case"),
                threads_per_block=_required_int(plan_record, "threads_per_block"),
                device=device,
                max_snippet_chars=max_snippet_chars,
            )
        except (OSError, ValueError) as error:
            raw_plan_index = plan_record.get("plan_index")
            plan_index = raw_plan_index if isinstance(raw_plan_index, int) else selected_index
            row_result_path = manifest_path
            if isinstance(plan_record.get("run_result_path"), str):
                row_result_path = Path(str(plan_record["run_result_path"]))
            record = _input_error_record(
                suite=suite,
                run_plan_manifest_path=run_plan_manifest_path,
                plan_index=plan_index,
                result_path=row_result_path,
                code="run_input_error",
                message=str(error),
                max_snippet_chars=max_snippet_chars,
            )

        records.append(record)
        row_result_path = Path(str(record["run_result_path"]))
        if row_result_path != manifest_path:
            write_jsonl(row_result_path, [record])

    return _write_result(manifest_path, records)


def _load_result_records(manifest_path: Path) -> tuple[list[dict[str, Any]], list[RunResultVerificationIssue]]:
    if not manifest_path.exists():
        return [], [RunResultVerificationIssue(code="missing_manifest", message=f"missing manifest: {manifest_path}")]

    records: list[dict[str, Any]] = []
    issues: list[RunResultVerificationIssue] = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as error:
                issues.append(
                    RunResultVerificationIssue(
                        code="invalid_json",
                        message=f"invalid JSON: {error.msg}",
                        line_number=line_number,
                    )
                )
                continue
            if not isinstance(parsed, dict):
                issues.append(
                    RunResultVerificationIssue(
                        code="invalid_record",
                        message="manifest row is not a JSON object",
                        line_number=line_number,
                    )
                )
                continue
            parsed["_line_number"] = line_number
            records.append(parsed)
    return records, issues


def _result_string_field(
    record: dict[str, Any],
    field: str,
    issues: list[RunResultVerificationIssue],
    *,
    run_result_path: str | None = None,
    compiled_artifact_path: str | None = None,
) -> str | None:
    value = record.get(field)
    if isinstance(value, str) and value:
        return value
    issues.append(
        RunResultVerificationIssue(
            code=f"invalid_{field}",
            message=f"missing or invalid {field!r}",
            run_result_path=run_result_path,
            compiled_artifact_path=compiled_artifact_path,
            line_number=int(record["_line_number"]),
        )
    )
    return None


def _result_int_field(
    record: dict[str, Any],
    field: str,
    issues: list[RunResultVerificationIssue],
    *,
    run_result_path: str | None = None,
    compiled_artifact_path: str | None = None,
) -> int | None:
    value = record.get(field)
    if isinstance(value, int):
        return value
    issues.append(
        RunResultVerificationIssue(
            code=f"invalid_{field}",
            message=f"missing or invalid {field!r}",
            run_result_path=run_result_path,
            compiled_artifact_path=compiled_artifact_path,
            line_number=int(record["_line_number"]),
        )
    )
    return None


def _result_bool_field(
    record: dict[str, Any],
    field: str,
    issues: list[RunResultVerificationIssue],
    *,
    run_result_path: str | None = None,
    compiled_artifact_path: str | None = None,
) -> bool | None:
    value = record.get(field)
    if isinstance(value, bool):
        return value
    issues.append(
        RunResultVerificationIssue(
            code=f"invalid_{field}",
            message=f"missing or invalid {field!r}",
            run_result_path=run_result_path,
            compiled_artifact_path=compiled_artifact_path,
            line_number=int(record["_line_number"]),
        )
    )
    return None


def _result_number_field(
    record: dict[str, Any],
    field: str,
    issues: list[RunResultVerificationIssue],
    *,
    run_result_path: str | None = None,
    compiled_artifact_path: str | None = None,
) -> float | None:
    value = record.get(field)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    issues.append(
        RunResultVerificationIssue(
            code=f"invalid_{field}",
            message=f"missing or invalid {field!r}",
            run_result_path=run_result_path,
            compiled_artifact_path=compiled_artifact_path,
            line_number=int(record["_line_number"]),
        )
    )
    return None


def _check_relative_result_path(
    path_text: str,
    issues: list[RunResultVerificationIssue],
    *,
    code_prefix: str,
    expected_parts_prefix: tuple[str, ...],
    line_number: int,
    run_result_path: str | None = None,
    compiled_artifact_path: str | None = None,
) -> bool:
    if _is_relative_under(path_text, expected_parts_prefix):
        return True
    issues.append(
        RunResultVerificationIssue(
            code=f"{code_prefix}_outside_data",
            message=f"{code_prefix} must be relative and stay under {'/'.join(expected_parts_prefix)}/",
            run_result_path=run_result_path,
            compiled_artifact_path=compiled_artifact_path,
            line_number=line_number,
        )
    )
    return False


def _check_sha256_value(
    value: str | None,
    issues: list[RunResultVerificationIssue],
    *,
    field: str,
    line_number: int,
    run_result_path: str | None,
    compiled_artifact_path: str | None,
) -> None:
    if value is None:
        return
    if len(value) == 64 and all(char in "0123456789abcdef" for char in value):
        return
    issues.append(
        RunResultVerificationIssue(
            code=f"invalid_{field}",
            message=f"{field} must be a lowercase SHA256 hex digest",
            run_result_path=run_result_path,
            compiled_artifact_path=compiled_artifact_path,
            line_number=line_number,
        )
    )


def _check_per_row_result_file(
    *,
    manifest_path: Path,
    record: dict[str, Any],
    run_result_path: str,
    compiled_artifact_path: str | None,
    issues: list[RunResultVerificationIssue],
    line_number: int,
) -> bool:
    path = Path(run_result_path)
    if path == manifest_path:
        return False
    if not path.exists():
        issues.append(
            RunResultVerificationIssue(
                code="missing_run_result_file",
                message="run_result_path does not exist",
                run_result_path=run_result_path,
                compiled_artifact_path=compiled_artifact_path,
                line_number=line_number,
            )
        )
        return False

    try:
        row_records = _read_jsonl_objects(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        issues.append(
            RunResultVerificationIssue(
                code="invalid_run_result_file",
                message=str(error),
                run_result_path=run_result_path,
                compiled_artifact_path=compiled_artifact_path,
                line_number=line_number,
            )
        )
        return False
    if len(row_records) != 1:
        issues.append(
            RunResultVerificationIssue(
                code="invalid_run_result_file_record_count",
                message="run_result_path must contain exactly one JSONL record",
                run_result_path=run_result_path,
                compiled_artifact_path=compiled_artifact_path,
                line_number=line_number,
            )
        )
        return False

    row_record = row_records[0]
    for field in ("event", "plan_index", "run_result_path", "status", "ok", "passed"):
        if row_record.get(field) == record.get(field):
            continue
        issues.append(
            RunResultVerificationIssue(
                code="run_result_file_mismatch",
                message=f"per-row file field {field!r} does not match aggregate manifest",
                run_result_path=run_result_path,
                compiled_artifact_path=compiled_artifact_path,
                line_number=line_number,
            )
        )
        return False
    return True


def verify_run_result_manifest(manifest_path: Path) -> RunResultVerificationResult:
    records, issues = _load_result_records(manifest_path)
    seen_paths: set[str] = set()
    result_files_checked = 0
    artifacts_checked = 0

    for record in records:
        line_number = int(record["_line_number"])
        if record.get("event") != "run_result":
            issues.append(
                RunResultVerificationIssue(
                    code="invalid_event",
                    message="manifest row event is not 'run_result'",
                    line_number=line_number,
                )
            )

        run_result_path = _result_string_field(record, "run_result_path", issues)
        check_result_path = _result_string_field(record, "check_result_path", issues, run_result_path=run_result_path)
        compiled_artifact_path = _result_string_field(record, "compiled_artifact_path", issues, run_result_path=run_result_path)
        ok = _result_bool_field(
            record,
            "ok",
            issues,
            run_result_path=run_result_path,
            compiled_artifact_path=compiled_artifact_path,
        )
        _result_bool_field(
            record,
            "passed",
            issues,
            run_result_path=run_result_path,
            compiled_artifact_path=compiled_artifact_path,
        )
        result_version = _result_string_field(
            record,
            "result_version",
            issues,
            run_result_path=run_result_path,
            compiled_artifact_path=compiled_artifact_path,
        )
        status = _result_string_field(
            record,
            "status",
            issues,
            run_result_path=run_result_path,
            compiled_artifact_path=compiled_artifact_path,
        )
        for field in ("suite", "entry_point", "launcher_entry_point", "case"):
            _result_string_field(
                record,
                field,
                issues,
                run_result_path=run_result_path,
                compiled_artifact_path=compiled_artifact_path,
            )
        for field in ("plan_index", "batch", "n", "cond", "seed", "threads_per_block", "returncode", "elapsed_ns"):
            _result_int_field(
                record,
                field,
                issues,
                run_result_path=run_result_path,
                compiled_artifact_path=compiled_artifact_path,
            )

        if result_version is not None and result_version != RUN_RESULT_VERSION:
            issues.append(
                RunResultVerificationIssue(
                    code="invalid_result_version",
                    message=f"result_version must be {RUN_RESULT_VERSION!r}",
                    run_result_path=run_result_path,
                    compiled_artifact_path=compiled_artifact_path,
                    line_number=line_number,
                )
            )

        if run_result_path is not None:
            if run_result_path in seen_paths:
                issues.append(
                    RunResultVerificationIssue(
                        code="duplicate_run_result_path",
                        message="duplicate run_result_path in run results",
                        run_result_path=run_result_path,
                        compiled_artifact_path=compiled_artifact_path,
                        line_number=line_number,
                    )
                )
            seen_paths.add(run_result_path)
            if _check_relative_result_path(
                run_result_path,
                issues,
                code_prefix="run_result_path",
                expected_parts_prefix=("data", "qr_v2", "run-results"),
                line_number=line_number,
                run_result_path=run_result_path,
                compiled_artifact_path=compiled_artifact_path,
            ) and _check_per_row_result_file(
                manifest_path=manifest_path,
                record=record,
                run_result_path=run_result_path,
                compiled_artifact_path=compiled_artifact_path,
                issues=issues,
                line_number=line_number,
            ):
                result_files_checked += 1

        if check_result_path is not None:
            _check_relative_result_path(
                check_result_path,
                issues,
                code_prefix="check_result_path",
                expected_parts_prefix=("data", "qr_v2", "run-results"),
                line_number=line_number,
                run_result_path=run_result_path,
                compiled_artifact_path=compiled_artifact_path,
            )
            if run_result_path is not None and check_result_path != run_result_path:
                issues.append(
                    RunResultVerificationIssue(
                        code="check_result_path_mismatch",
                        message="check_result_path must match run_result_path for combined records",
                        run_result_path=run_result_path,
                        compiled_artifact_path=compiled_artifact_path,
                        line_number=line_number,
                    )
                )

        if compiled_artifact_path is not None and _check_relative_result_path(
            compiled_artifact_path,
            issues,
            code_prefix="compiled_artifact_path",
            expected_parts_prefix=("data", "qr_v2", "compiled"),
            line_number=line_number,
            run_result_path=run_result_path,
            compiled_artifact_path=compiled_artifact_path,
        ):
            artifact = Path(compiled_artifact_path)
            if not artifact.exists():
                issues.append(
                    RunResultVerificationIssue(
                        code="missing_compiled_artifact",
                        message="compiled artifact does not exist",
                        run_result_path=run_result_path,
                        compiled_artifact_path=compiled_artifact_path,
                        line_number=line_number,
                    )
                )
            else:
                artifacts_checked += 1
                content = artifact.read_bytes()
                expected_bytes = record.get("compiled_artifact_bytes")
                if isinstance(expected_bytes, int) and expected_bytes != len(content):
                    issues.append(
                        RunResultVerificationIssue(
                            code="compiled_artifact_bytes_mismatch",
                            message=f"expected {expected_bytes} bytes, found {len(content)}",
                            run_result_path=run_result_path,
                            compiled_artifact_path=compiled_artifact_path,
                            line_number=line_number,
                        )
                    )
                expected_sha256 = record.get("compiled_artifact_sha256")
                if isinstance(expected_sha256, str) and expected_sha256 != _sha256_bytes(content):
                    issues.append(
                        RunResultVerificationIssue(
                            code="compiled_artifact_sha256_mismatch",
                            message=f"expected {expected_sha256}, found {_sha256_bytes(content)}",
                            run_result_path=run_result_path,
                            compiled_artifact_path=compiled_artifact_path,
                            line_number=line_number,
                        )
                    )

        for field in ("compiled_artifact_sha256", "input_sha256", "output_h_sha256", "output_tau_sha256"):
            value = _result_string_field(
                record,
                field,
                issues,
                run_result_path=run_result_path,
                compiled_artifact_path=compiled_artifact_path,
            )
            _check_sha256_value(
                value,
                issues,
                field=field,
                line_number=line_number,
                run_result_path=run_result_path,
                compiled_artifact_path=compiled_artifact_path,
            )

        if ok or status == "checked":
            for field in (
                "factor_abs",
                "factor_rel",
                "factor_scaled",
                "factor_threshold",
                "orth_abs",
                "orth_rel",
                "orth_scaled",
                "orth_threshold",
                "lower_abs",
                "lower_rel",
                "lower_scaled",
                "lower_threshold",
                "reconstruction_abs",
                "reconstruction_scaled",
            ):
                _result_number_field(
                    record,
                    field,
                    issues,
                    run_result_path=run_result_path,
                    compiled_artifact_path=compiled_artifact_path,
                )
            _result_int_field(
                record,
                "worst_batch",
                issues,
                run_result_path=run_result_path,
                compiled_artifact_path=compiled_artifact_path,
            )

    return RunResultVerificationResult(
        manifest_path=manifest_path,
        records_checked=len(records),
        result_files_checked=result_files_checked,
        artifacts_checked=artifacts_checked,
        issues=issues,
    )
