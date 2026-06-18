"""Compile execution for qr_v2 CUDA artifacts."""

import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpumode.qr_v2.jsonl import write_jsonl


COMPILE_RESULT_VERSION = "compile_result_v1"
DEFAULT_SNIPPET_CHARS = 4000


@dataclass(frozen=True, slots=True)
class CompileExecutionResult:
    manifest_path: Path
    records: list[dict[str, object]]

    @property
    def ok(self) -> bool:
        return all(bool(record.get("ok")) for record in self.records)


@dataclass(frozen=True, slots=True)
class CompileResultVerificationIssue:
    code: str
    message: str
    output_path: str | None = None
    artifact_path: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    line_number: int | None = None

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "code": self.code,
            "message": self.message,
        }
        if self.output_path is not None:
            record["output_path"] = self.output_path
        if self.artifact_path is not None:
            record["artifact_path"] = self.artifact_path
        if self.stdout_path is not None:
            record["stdout_path"] = self.stdout_path
        if self.stderr_path is not None:
            record["stderr_path"] = self.stderr_path
        if self.line_number is not None:
            record["line_number"] = self.line_number
        return record


@dataclass(frozen=True, slots=True)
class CompileResultVerificationResult:
    manifest_path: Path
    records_checked: int
    outputs_checked: int
    logs_checked: int
    issues: list[CompileResultVerificationIssue]

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


def _load_result_records(manifest_path: Path) -> tuple[list[dict[str, Any]], list[CompileResultVerificationIssue]]:
    if not manifest_path.exists():
        return [], [CompileResultVerificationIssue(code="missing_manifest", message=f"missing manifest: {manifest_path}")]

    records: list[dict[str, Any]] = []
    issues: list[CompileResultVerificationIssue] = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as error:
                issues.append(
                    CompileResultVerificationIssue(
                        code="invalid_json",
                        message=f"invalid JSON: {error.msg}",
                        line_number=line_number,
                    )
                )
                continue
            if not isinstance(parsed, dict):
                issues.append(
                    CompileResultVerificationIssue(
                        code="invalid_record",
                        message="manifest row is not a JSON object",
                        line_number=line_number,
                    )
                )
                continue
            parsed["_line_number"] = line_number
            records.append(parsed)
    return records, issues


def _required_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"compile plan row is missing string field {field!r}")
    return value


def _required_int(record: dict[str, Any], field: str) -> int:
    value = record.get(field)
    if not isinstance(value, int):
        raise ValueError(f"compile plan row is missing integer field {field!r}")
    return value


def _argv_field(record: dict[str, Any]) -> list[str]:
    value = record.get("compile_argv")
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError("compile plan row is missing valid 'compile_argv'")
    return [str(item) for item in value]


def _is_relative_under(path_text: str, prefix: tuple[str, ...]) -> bool:
    path = Path(path_text)
    return not path.is_absolute() and path.parts[: len(prefix)] == prefix


def _check_execution_path(path_text: str, *, field: str, prefix: tuple[str, ...]) -> None:
    if not _is_relative_under(path_text, prefix):
        joined = "/".join(prefix)
        raise ValueError(f"{field} must be relative and stay under {joined}/")


def _log_paths(*, suite: str, candidate_id: str, output_path: str, result_root: Path) -> tuple[Path, Path]:
    stem = Path(output_path).stem
    log_dir = result_root / suite / "logs" / candidate_id
    return log_dir / f"{stem}.stdout.txt", log_dir / f"{stem}.stderr.txt"


def _bounded_text(content: bytes, *, max_chars: int) -> str:
    text = content.decode("utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n[... omitted {omitted} chars ...]"


def _resolve_executable(command: str) -> str | None:
    if "/" in command:
        path = Path(command)
        return command if path.exists() else None
    return shutil.which(command)


def _run_version_command(compiler_path: str | None, *, max_snippet_chars: int) -> dict[str, object]:
    if compiler_path is None:
        return {
            "nvcc_version_returncode": 127,
            "nvcc_version_stdout": "",
            "nvcc_version_stderr": "nvcc not found on PATH",
        }

    completed = subprocess.run(
        [compiler_path, "--version"],
        check=False,
        capture_output=True,
    )
    return {
        "nvcc_version_returncode": completed.returncode,
        "nvcc_version_stdout": _bounded_text(completed.stdout, max_chars=max_snippet_chars),
        "nvcc_version_stderr": _bounded_text(completed.stderr, max_chars=max_snippet_chars),
    }


def _base_result_record(
    *,
    suite: str,
    plan_record: dict[str, Any] | None,
    compile_plan_manifest_path: Path,
    plan_index: int,
    out_dir: Path,
    max_snippet_chars: int,
) -> dict[str, object]:
    if plan_record is None:
        return {
            "event": "compile_result",
            "result_version": COMPILE_RESULT_VERSION,
            "suite": suite,
            "plan_index": plan_index,
            "compile_plan_manifest_path": compile_plan_manifest_path.as_posix(),
            "ok": False,
            "status": "input_error",
            "returncode": 1,
            "elapsed_ns": 0,
            "stdout_snippet": "",
            "stderr_snippet": "",
            "max_snippet_chars": max_snippet_chars,
        }

    candidate_id = _required_string(plan_record, "candidate_id")
    output_path = _required_string(plan_record, "output_path")
    stdout_path, stderr_path = _log_paths(
        suite=suite,
        candidate_id=candidate_id,
        output_path=output_path,
        result_root=out_dir.parent,
    )

    return {
        "event": "compile_result",
        "result_version": COMPILE_RESULT_VERSION,
        "suite": suite,
        "plan_index": _required_int(plan_record, "plan_index"),
        "compile_plan_index": _required_int(plan_record, "plan_index"),
        "compile_plan_version": _required_string(plan_record, "plan_version"),
        "compile_plan_manifest_path": compile_plan_manifest_path.as_posix(),
        "target_backend": _required_string(plan_record, "target_backend"),
        "compiler": _required_string(plan_record, "compiler"),
        "compiler_family": _required_string(plan_record, "compiler_family"),
        "compile_cwd": _required_string(plan_record, "compile_cwd"),
        "compile_argv": _argv_field(plan_record),
        "output_kind": _required_string(plan_record, "output_kind"),
        "output_path": output_path,
        "artifact_path": _required_string(plan_record, "artifact_path"),
        "artifact_kind": _required_string(plan_record, "artifact_kind"),
        "source_bytes": _required_int(plan_record, "source_bytes"),
        "source_sha256": _required_string(plan_record, "source_sha256"),
        "candidate_id": candidate_id,
        "template_id": _required_string(plan_record, "template_id"),
        "entry_point": _required_string(plan_record, "entry_point"),
        "renderer_version": _required_string(plan_record, "renderer_version"),
        "batch": _required_int(plan_record, "batch"),
        "n": _required_int(plan_record, "n"),
        "cond": _required_int(plan_record, "cond"),
        "seed": _required_int(plan_record, "seed"),
        "case": _required_string(plan_record, "case"),
        "threads_per_block": _required_int(plan_record, "threads_per_block"),
        "tile_size": _required_int(plan_record, "tile_size"),
        "unroll": _required_int(plan_record, "unroll"),
        "stdout_path": stdout_path.as_posix(),
        "stderr_path": stderr_path.as_posix(),
        "max_snippet_chars": max_snippet_chars,
    }


def _input_error_record(
    *,
    suite: str,
    compile_plan_manifest_path: Path,
    plan_index: int,
    out_dir: Path,
    code: str,
    message: str,
    max_snippet_chars: int,
) -> dict[str, object]:
    record = _base_result_record(
        suite=suite,
        plan_record=None,
        compile_plan_manifest_path=compile_plan_manifest_path,
        plan_index=plan_index,
        out_dir=out_dir,
        max_snippet_chars=max_snippet_chars,
    )
    record.update(
        {
            "status": code,
            "stderr_snippet": message,
        }
    )
    return record


def _compile_plan_record(
    *,
    suite: str,
    plan_record: dict[str, Any],
    compile_plan_manifest_path: Path,
    out_dir: Path,
    timeout_seconds: float | None,
    max_snippet_chars: int,
) -> dict[str, object]:
    argv = _argv_field(plan_record)
    artifact_path = _required_string(plan_record, "artifact_path")
    output_path = _required_string(plan_record, "output_path")
    _check_execution_path(artifact_path, field="artifact_path", prefix=("data",))
    _check_execution_path(output_path, field="output_path", prefix=("data", "qr_v2", "compiled"))

    record = _base_result_record(
        suite=suite,
        plan_record=plan_record,
        compile_plan_manifest_path=compile_plan_manifest_path,
        plan_index=_required_int(plan_record, "plan_index"),
        out_dir=out_dir,
        max_snippet_chars=max_snippet_chars,
    )

    compiler_path = _resolve_executable(argv[0])
    record["compiler_resolved_path"] = compiler_path or ""
    record.update(_run_version_command(compiler_path, max_snippet_chars=max_snippet_chars))

    stdout_path = Path(str(record["stdout_path"]))
    stderr_path = Path(str(record["stderr_path"]))
    _check_execution_path(stdout_path.as_posix(), field="stdout_path", prefix=("data", "qr_v2", "compile-results"))
    _check_execution_path(stderr_path.as_posix(), field="stderr_path", prefix=("data", "qr_v2", "compile-results"))
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    if compiler_path is None:
        stdout = b""
        stderr = b"nvcc not found on PATH"
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        record.update(
            {
                "ok": False,
                "status": "missing_compiler",
                "returncode": 127,
                "elapsed_ns": 0,
                "stdout_bytes": 0,
                "stderr_bytes": len(stderr),
                "stdout_snippet": "",
                "stderr_snippet": _bounded_text(stderr, max_chars=max_snippet_chars),
                "output_exists": False,
            }
        )
        return record

    start_ns = time.perf_counter_ns()
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
        )
        elapsed_ns = time.perf_counter_ns() - start_ns
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        status = "ok" if returncode == 0 else "failed"
    except subprocess.TimeoutExpired as error:
        elapsed_ns = time.perf_counter_ns() - start_ns
        returncode = -1
        stdout = error.output or b""
        stderr = error.stderr or b""
        timeout_message = f"compile timed out after {timeout_seconds} seconds".encode()
        stderr = stderr + (b"\n" if stderr else b"") + timeout_message
        status = "timeout"

    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)

    ok = returncode == 0 and output.exists()
    record.update(
        {
            "ok": ok,
            "status": status if ok or status != "ok" else "missing_output",
            "returncode": returncode,
            "elapsed_ns": elapsed_ns,
            "elapsed_ms": elapsed_ns / 1_000_000.0,
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
            "stdout_snippet": _bounded_text(stdout, max_chars=max_snippet_chars),
            "stderr_snippet": _bounded_text(stderr, max_chars=max_snippet_chars),
            "output_exists": output.exists(),
        }
    )

    if output.exists():
        content = output.read_bytes()
        record.update(
            {
                "output_bytes": len(content),
                "output_sha256": _sha256_bytes(content),
            }
        )

    return record


def _write_result(out_dir: Path, records: list[dict[str, object]]) -> CompileExecutionResult:
    manifest_path = out_dir / "manifest.jsonl"
    write_jsonl(manifest_path, records)
    return CompileExecutionResult(manifest_path=manifest_path, records=records)


def compile_one_from_plan_manifest(
    *,
    suite: str,
    compile_plan_manifest_path: Path,
    index: int,
    out_dir: Path,
    timeout_seconds: float | None = None,
    max_snippet_chars: int = DEFAULT_SNIPPET_CHARS,
) -> CompileExecutionResult:
    if not compile_plan_manifest_path.exists():
        record = _input_error_record(
            suite=suite,
            compile_plan_manifest_path=compile_plan_manifest_path,
            plan_index=index,
            out_dir=out_dir,
            code="missing_compile_plan",
            message=f"missing compile plan: {compile_plan_manifest_path}",
            max_snippet_chars=max_snippet_chars,
        )
        return _write_result(out_dir, [record])

    try:
        plan_records = _read_jsonl_objects(compile_plan_manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        record = _input_error_record(
            suite=suite,
            compile_plan_manifest_path=compile_plan_manifest_path,
            plan_index=index,
            out_dir=out_dir,
            code="invalid_compile_plan",
            message=str(error),
            max_snippet_chars=max_snippet_chars,
        )
        return _write_result(out_dir, [record])

    if index < 0 or index >= len(plan_records):
        record = _input_error_record(
            suite=suite,
            compile_plan_manifest_path=compile_plan_manifest_path,
            plan_index=index,
            out_dir=out_dir,
            code="compile_plan_index_out_of_range",
            message=f"compile plan index {index} is outside 0..{len(plan_records) - 1}",
            max_snippet_chars=max_snippet_chars,
        )
        return _write_result(out_dir, [record])

    try:
        record = _compile_plan_record(
            suite=suite,
            plan_record=plan_records[index],
            compile_plan_manifest_path=compile_plan_manifest_path,
            out_dir=out_dir,
            timeout_seconds=timeout_seconds,
            max_snippet_chars=max_snippet_chars,
        )
    except (OSError, ValueError) as error:
        record = _input_error_record(
            suite=suite,
            compile_plan_manifest_path=compile_plan_manifest_path,
            plan_index=index,
            out_dir=out_dir,
            code="compile_input_error",
            message=str(error),
            max_snippet_chars=max_snippet_chars,
        )
    return _write_result(out_dir, [record])


def compile_from_plan_manifest(
    *,
    suite: str,
    compile_plan_manifest_path: Path,
    out_dir: Path,
    limit: int | None = None,
    timeout_seconds: float | None = None,
    max_snippet_chars: int = DEFAULT_SNIPPET_CHARS,
) -> CompileExecutionResult:
    if not compile_plan_manifest_path.exists():
        record = _input_error_record(
            suite=suite,
            compile_plan_manifest_path=compile_plan_manifest_path,
            plan_index=0,
            out_dir=out_dir,
            code="missing_compile_plan",
            message=f"missing compile plan: {compile_plan_manifest_path}",
            max_snippet_chars=max_snippet_chars,
        )
        return _write_result(out_dir, [record])

    try:
        plan_records = _read_jsonl_objects(compile_plan_manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        record = _input_error_record(
            suite=suite,
            compile_plan_manifest_path=compile_plan_manifest_path,
            plan_index=0,
            out_dir=out_dir,
            code="invalid_compile_plan",
            message=str(error),
            max_snippet_chars=max_snippet_chars,
        )
        return _write_result(out_dir, [record])

    selected_records = plan_records[:limit] if limit is not None else plan_records
    records: list[dict[str, object]] = []
    for plan_record in selected_records:
        try:
            records.append(
                _compile_plan_record(
                    suite=suite,
                    plan_record=plan_record,
                    compile_plan_manifest_path=compile_plan_manifest_path,
                    out_dir=out_dir,
                    timeout_seconds=timeout_seconds,
                    max_snippet_chars=max_snippet_chars,
                )
            )
        except (OSError, ValueError) as error:
            raw_plan_index = plan_record.get("plan_index")
            plan_index = raw_plan_index if isinstance(raw_plan_index, int) else len(records)
            records.append(
                _input_error_record(
                    suite=suite,
                    compile_plan_manifest_path=compile_plan_manifest_path,
                    plan_index=plan_index,
                    out_dir=out_dir,
                    code="compile_input_error",
                    message=str(error),
                    max_snippet_chars=max_snippet_chars,
                )
            )
    return _write_result(out_dir, records)


def _result_string_field(
    record: dict[str, Any],
    field: str,
    issues: list[CompileResultVerificationIssue],
    *,
    output_path: str | None = None,
    artifact_path: str | None = None,
    stdout_path: str | None = None,
    stderr_path: str | None = None,
) -> str | None:
    value = record.get(field)
    if isinstance(value, str) and value:
        return value
    issues.append(
        CompileResultVerificationIssue(
            code=f"invalid_{field}",
            message=f"missing or invalid {field!r}",
            output_path=output_path,
            artifact_path=artifact_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            line_number=int(record["_line_number"]),
        )
    )
    return None


def _result_int_field(
    record: dict[str, Any],
    field: str,
    issues: list[CompileResultVerificationIssue],
    *,
    output_path: str | None = None,
    artifact_path: str | None = None,
    stdout_path: str | None = None,
    stderr_path: str | None = None,
) -> int | None:
    value = record.get(field)
    if isinstance(value, int):
        return value
    issues.append(
        CompileResultVerificationIssue(
            code=f"invalid_{field}",
            message=f"missing or invalid {field!r}",
            output_path=output_path,
            artifact_path=artifact_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            line_number=int(record["_line_number"]),
        )
    )
    return None


def _result_bool_field(
    record: dict[str, Any],
    field: str,
    issues: list[CompileResultVerificationIssue],
    *,
    output_path: str | None = None,
    artifact_path: str | None = None,
    stdout_path: str | None = None,
    stderr_path: str | None = None,
) -> bool | None:
    value = record.get(field)
    if isinstance(value, bool):
        return value
    issues.append(
        CompileResultVerificationIssue(
            code=f"invalid_{field}",
            message=f"missing or invalid {field!r}",
            output_path=output_path,
            artifact_path=artifact_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            line_number=int(record["_line_number"]),
        )
    )
    return None


def _check_relative_data_path(
    path_text: str,
    issues: list[CompileResultVerificationIssue],
    *,
    code_prefix: str,
    expected_parts_prefix: tuple[str, ...],
    line_number: int,
    output_path: str | None = None,
    artifact_path: str | None = None,
    stdout_path: str | None = None,
    stderr_path: str | None = None,
) -> bool:
    if _is_relative_under(path_text, expected_parts_prefix):
        return True
    issues.append(
        CompileResultVerificationIssue(
            code=f"{code_prefix}_outside_data",
            message=f"{code_prefix} must be relative and stay under {'/'.join(expected_parts_prefix)}/",
            output_path=output_path,
            artifact_path=artifact_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            line_number=line_number,
        )
    )
    return False


def verify_compile_result_manifest(manifest_path: Path) -> CompileResultVerificationResult:
    records, issues = _load_result_records(manifest_path)
    outputs_checked = 0
    logs_checked = 0

    for record in records:
        line_number = int(record["_line_number"])
        if record.get("event") != "compile_result":
            issues.append(
                CompileResultVerificationIssue(
                    code="invalid_event",
                    message="manifest row event is not 'compile_result'",
                    line_number=line_number,
                )
            )

        output_path = _result_string_field(record, "output_path", issues)
        artifact_path = _result_string_field(record, "artifact_path", issues, output_path=output_path)
        stdout_path = _result_string_field(
            record,
            "stdout_path",
            issues,
            output_path=output_path,
            artifact_path=artifact_path,
        )
        stderr_path = _result_string_field(
            record,
            "stderr_path",
            issues,
            output_path=output_path,
            artifact_path=artifact_path,
            stdout_path=stdout_path,
        )
        ok = _result_bool_field(
            record,
            "ok",
            issues,
            output_path=output_path,
            artifact_path=artifact_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        _result_string_field(
            record,
            "status",
            issues,
            output_path=output_path,
            artifact_path=artifact_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        _result_int_field(
            record,
            "returncode",
            issues,
            output_path=output_path,
            artifact_path=artifact_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        _result_int_field(
            record,
            "elapsed_ns",
            issues,
            output_path=output_path,
            artifact_path=artifact_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

        if output_path is not None:
            _check_relative_data_path(
                output_path,
                issues,
                code_prefix="output_path",
                expected_parts_prefix=("data", "qr_v2", "compiled"),
                line_number=line_number,
                output_path=output_path,
                artifact_path=artifact_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        if artifact_path is not None:
            _check_relative_data_path(
                artifact_path,
                issues,
                code_prefix="artifact_path",
                expected_parts_prefix=("data",),
                line_number=line_number,
                output_path=output_path,
                artifact_path=artifact_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        for field, path_text in (("stdout_path", stdout_path), ("stderr_path", stderr_path)):
            if path_text is None:
                continue
            if not _check_relative_data_path(
                path_text,
                issues,
                code_prefix=field,
                expected_parts_prefix=("data", "qr_v2", "compile-results"),
                line_number=line_number,
                output_path=output_path,
                artifact_path=artifact_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            ):
                continue
            path = Path(path_text)
            if not path.exists():
                issues.append(
                    CompileResultVerificationIssue(
                        code=f"missing_{field}",
                        message=f"{field} does not exist",
                        output_path=output_path,
                        artifact_path=artifact_path,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        line_number=line_number,
                    )
                )
                continue
            logs_checked += 1
            expected_bytes = record.get(field.replace("_path", "_bytes"))
            if isinstance(expected_bytes, int) and path.stat().st_size != expected_bytes:
                issues.append(
                    CompileResultVerificationIssue(
                        code=f"{field}_bytes_mismatch",
                        message=f"expected {expected_bytes} bytes, found {path.stat().st_size}",
                        output_path=output_path,
                        artifact_path=artifact_path,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        line_number=line_number,
                    )
                )

        if not ok or output_path is None:
            continue
        output = Path(output_path)
        if not output.exists():
            issues.append(
                CompileResultVerificationIssue(
                    code="missing_output",
                    message="compiled output does not exist",
                    output_path=output_path,
                    artifact_path=artifact_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    line_number=line_number,
                )
            )
            continue
        content = output.read_bytes()
        outputs_checked += 1
        expected_bytes = record.get("output_bytes")
        if isinstance(expected_bytes, int) and len(content) != expected_bytes:
            issues.append(
                CompileResultVerificationIssue(
                    code="output_bytes_mismatch",
                    message=f"expected {expected_bytes} bytes, found {len(content)}",
                    output_path=output_path,
                    artifact_path=artifact_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    line_number=line_number,
                )
            )
        expected_sha256 = record.get("output_sha256")
        actual_sha256 = _sha256_bytes(content)
        if isinstance(expected_sha256, str) and expected_sha256 != actual_sha256:
            issues.append(
                CompileResultVerificationIssue(
                    code="output_sha256_mismatch",
                    message=f"expected {expected_sha256}, found {actual_sha256}",
                    output_path=output_path,
                    artifact_path=artifact_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    line_number=line_number,
                )
            )

    return CompileResultVerificationResult(
        manifest_path=manifest_path,
        records_checked=len(records),
        outputs_checked=outputs_checked,
        logs_checked=logs_checked,
        issues=issues,
    )
