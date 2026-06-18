"""Run planning for compiled qr_v2 candidates."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpumode.qr_v2.jsonl import write_jsonl


RUN_PLAN_VERSION = "run_plan_v1"


@dataclass(frozen=True, slots=True)
class RunPlanResult:
    manifest_path: Path
    records: list[dict[str, object]]

    @property
    def result_paths(self) -> list[Path]:
        return [Path(str(record["run_result_path"])) for record in self.records]


@dataclass(frozen=True, slots=True)
class RunPlanVerificationIssue:
    code: str
    message: str
    compiled_artifact_path: str | None = None
    run_result_path: str | None = None
    check_result_path: str | None = None
    source_artifact_path: str | None = None
    line_number: int | None = None

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "code": self.code,
            "message": self.message,
        }
        if self.compiled_artifact_path is not None:
            record["compiled_artifact_path"] = self.compiled_artifact_path
        if self.run_result_path is not None:
            record["run_result_path"] = self.run_result_path
        if self.check_result_path is not None:
            record["check_result_path"] = self.check_result_path
        if self.source_artifact_path is not None:
            record["source_artifact_path"] = self.source_artifact_path
        if self.line_number is not None:
            record["line_number"] = self.line_number
        return record


@dataclass(frozen=True, slots=True)
class RunPlanVerificationResult:
    manifest_path: Path
    records_checked: int
    sources_checked: int
    issues: list[RunPlanVerificationIssue]

    @property
    def ok(self) -> bool:
        return not self.issues


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
        raise ValueError(f"compile plan row is missing string field {field!r}")
    return value


def _required_int(record: dict[str, Any], field: str) -> int:
    value = record.get(field)
    if not isinstance(value, int):
        raise ValueError(f"compile plan row is missing integer field {field!r}")
    return value


def _run_result_path(*, suite: str, candidate_id: str, compiled_artifact_path: str, output_root: Path) -> Path:
    stem = Path(compiled_artifact_path).with_suffix(".jsonl").name
    return output_root / suite / candidate_id / stem


def _runner_argv(
    *,
    compiled_artifact_path: str,
    entry_point: str,
    batch: int,
    n: int,
    cond: int,
    seed: int,
    case: str,
    run_result_path: Path,
) -> list[str]:
    return [
        "qr-v2",
        "run-one",
        "--compiled-artifact",
        compiled_artifact_path,
        "--entry-point",
        entry_point,
        "--batch",
        str(batch),
        "--n",
        str(n),
        "--cond",
        str(cond),
        "--seed",
        str(seed),
        "--case",
        case,
        "--out",
        run_result_path.as_posix(),
    ]


def plan_run_from_compile_plan_manifest(
    *,
    suite: str,
    compile_plan_manifest_path: Path,
    out_dir: Path,
    result_root: Path,
) -> RunPlanResult:
    compile_records = _read_jsonl_objects(compile_plan_manifest_path)
    records: list[dict[str, object]] = []

    for index, compile_record in enumerate(compile_records):
        candidate_id = _required_string(compile_record, "candidate_id")
        compiled_artifact_path = _required_string(compile_record, "output_path")
        entry_point = _required_string(compile_record, "entry_point")
        batch = _required_int(compile_record, "batch")
        n = _required_int(compile_record, "n")
        cond = _required_int(compile_record, "cond")
        seed = _required_int(compile_record, "seed")
        case = _required_string(compile_record, "case")
        run_result_path = _run_result_path(
            suite=suite,
            candidate_id=candidate_id,
            compiled_artifact_path=compiled_artifact_path,
            output_root=result_root,
        )

        records.append(
            {
                "event": "run_plan",
                "plan_version": RUN_PLAN_VERSION,
                "suite": suite,
                "plan_index": index,
                "compile_plan_index": _required_int(compile_record, "plan_index"),
                "compile_plan_version": _required_string(compile_record, "plan_version"),
                "target_backend": _required_string(compile_record, "target_backend"),
                "runner": "qr-v2 run-one",
                "runner_family": "local_cuda_shared_object",
                "runner_cwd": ".",
                "runner_argv": _runner_argv(
                    compiled_artifact_path=compiled_artifact_path,
                    entry_point=entry_point,
                    batch=batch,
                    n=n,
                    cond=cond,
                    seed=seed,
                    case=case,
                    run_result_path=run_result_path,
                ),
                "run_result_path": run_result_path.as_posix(),
                "check_result_path": run_result_path.as_posix(),
                "compiled_artifact_path": compiled_artifact_path,
                "compiled_artifact_kind": _required_string(compile_record, "output_kind"),
                "source_artifact_path": _required_string(compile_record, "artifact_path"),
                "source_sha256": _required_string(compile_record, "source_sha256"),
                "candidate_id": candidate_id,
                "template_id": _required_string(compile_record, "template_id"),
                "entry_point": entry_point,
                "batch": batch,
                "n": n,
                "cond": cond,
                "seed": seed,
                "case": case,
                "threads_per_block": _required_int(compile_record, "threads_per_block"),
                "tile_size": _required_int(compile_record, "tile_size"),
                "unroll": _required_int(compile_record, "unroll"),
            }
        )

    manifest_path = out_dir / "manifest.jsonl"
    write_jsonl(manifest_path, records)
    return RunPlanResult(manifest_path=manifest_path, records=records)


def _load_plan_records(manifest_path: Path) -> tuple[list[dict[str, Any]], list[RunPlanVerificationIssue]]:
    if not manifest_path.exists():
        return [], [RunPlanVerificationIssue(code="missing_manifest", message=f"missing manifest: {manifest_path}")]

    records: list[dict[str, Any]] = []
    issues: list[RunPlanVerificationIssue] = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as error:
                issues.append(
                    RunPlanVerificationIssue(
                        code="invalid_json",
                        message=f"invalid JSON: {error.msg}",
                        line_number=line_number,
                    )
                )
                continue
            if not isinstance(parsed, dict):
                issues.append(
                    RunPlanVerificationIssue(
                        code="invalid_record",
                        message="manifest row is not a JSON object",
                        line_number=line_number,
                    )
                )
                continue
            parsed["_line_number"] = line_number
            records.append(parsed)
    return records, issues


def _plan_string_field(
    record: dict[str, Any],
    field: str,
    issues: list[RunPlanVerificationIssue],
    *,
    compiled_artifact_path: str | None = None,
    run_result_path: str | None = None,
    check_result_path: str | None = None,
    source_artifact_path: str | None = None,
) -> str | None:
    value = record.get(field)
    if isinstance(value, str) and value:
        return value
    issues.append(
        RunPlanVerificationIssue(
            code=f"invalid_{field}",
            message=f"missing or invalid {field!r}",
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
            line_number=int(record["_line_number"]),
        )
    )
    return None


def _plan_int_field(
    record: dict[str, Any],
    field: str,
    issues: list[RunPlanVerificationIssue],
    *,
    compiled_artifact_path: str | None = None,
    run_result_path: str | None = None,
    check_result_path: str | None = None,
    source_artifact_path: str | None = None,
) -> int | None:
    value = record.get(field)
    if isinstance(value, int):
        return value
    issues.append(
        RunPlanVerificationIssue(
            code=f"invalid_{field}",
            message=f"missing or invalid {field!r}",
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
            line_number=int(record["_line_number"]),
        )
    )
    return None


def _plan_argv_field(
    record: dict[str, Any],
    issues: list[RunPlanVerificationIssue],
    *,
    compiled_artifact_path: str | None,
    run_result_path: str | None,
    check_result_path: str | None,
    source_artifact_path: str | None,
) -> list[str] | None:
    value = record.get("runner_argv")
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return [str(item) for item in value]
    issues.append(
        RunPlanVerificationIssue(
            code="invalid_runner_argv",
            message="missing or invalid 'runner_argv'",
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
            line_number=int(record["_line_number"]),
        )
    )
    return None


def _check_relative_data_path(
    path_text: str,
    issues: list[RunPlanVerificationIssue],
    *,
    code_prefix: str,
    expected_parts_prefix: tuple[str, ...],
    line_number: int,
    compiled_artifact_path: str | None = None,
    run_result_path: str | None = None,
    check_result_path: str | None = None,
    source_artifact_path: str | None = None,
) -> bool:
    path = Path(path_text)
    if path.is_absolute():
        issues.append(
            RunPlanVerificationIssue(
                code=f"absolute_{code_prefix}",
                message=f"{code_prefix} must be relative",
                compiled_artifact_path=compiled_artifact_path,
                run_result_path=run_result_path,
                check_result_path=check_result_path,
                source_artifact_path=source_artifact_path,
                line_number=line_number,
            )
        )
        return False
    if path.parts[: len(expected_parts_prefix)] != expected_parts_prefix:
        issues.append(
            RunPlanVerificationIssue(
                code=f"{code_prefix}_outside_data",
                message=f"{code_prefix} must stay under {'/'.join(expected_parts_prefix)}/",
                compiled_artifact_path=compiled_artifact_path,
                run_result_path=run_result_path,
                check_result_path=check_result_path,
                source_artifact_path=source_artifact_path,
                line_number=line_number,
            )
        )
        return False
    return True


def _check_string_value(
    *,
    actual: str | None,
    expected: str,
    code: str,
    message: str,
    issues: list[RunPlanVerificationIssue],
    line_number: int,
    compiled_artifact_path: str | None,
    run_result_path: str | None,
    check_result_path: str | None,
    source_artifact_path: str | None,
) -> None:
    if actual is None or actual == expected:
        return
    issues.append(
        RunPlanVerificationIssue(
            code=code,
            message=message,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
            line_number=line_number,
        )
    )


def _check_path_suffix(
    *,
    path_text: str | None,
    suffix: str,
    field: str,
    issues: list[RunPlanVerificationIssue],
    line_number: int,
    compiled_artifact_path: str | None,
    run_result_path: str | None,
    check_result_path: str | None,
    source_artifact_path: str | None,
) -> None:
    if path_text is None or Path(path_text).suffix == suffix:
        return
    issues.append(
        RunPlanVerificationIssue(
            code=f"invalid_{field}_suffix",
            message=f"{field} must end with {suffix}",
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
            line_number=line_number,
        )
    )


def _argv_value_matches(argv: list[str], flag: str, expected: str) -> bool:
    try:
        index = argv.index(flag)
    except ValueError:
        return False
    return index + 1 < len(argv) and argv[index + 1] == expected


def _check_runner_argv(
    *,
    runner_argv: list[str] | None,
    compiled_artifact_path: str | None,
    run_result_path: str | None,
    check_result_path: str | None,
    source_artifact_path: str | None,
    entry_point: str | None,
    batch: int | None,
    n: int | None,
    cond: int | None,
    seed: int | None,
    case: str | None,
    issues: list[RunPlanVerificationIssue],
    line_number: int,
) -> None:
    if runner_argv is None:
        return

    if runner_argv[:2] != ["qr-v2", "run-one"]:
        issues.append(
            RunPlanVerificationIssue(
                code="invalid_runner_argv_command",
                message="runner_argv must start with 'qr-v2 run-one'",
                compiled_artifact_path=compiled_artifact_path,
                run_result_path=run_result_path,
                check_result_path=check_result_path,
                source_artifact_path=source_artifact_path,
                line_number=line_number,
            )
        )

    expected_flags: list[tuple[str, str | None]] = [
        ("--compiled-artifact", compiled_artifact_path),
        ("--entry-point", entry_point),
        ("--batch", None if batch is None else str(batch)),
        ("--n", None if n is None else str(n)),
        ("--cond", None if cond is None else str(cond)),
        ("--seed", None if seed is None else str(seed)),
        ("--case", case),
        ("--out", run_result_path),
    ]
    for flag, expected in expected_flags:
        if expected is None or _argv_value_matches(runner_argv, flag, expected):
            continue
        issues.append(
            RunPlanVerificationIssue(
                code="runner_argv_missing_value",
                message=f"runner_argv does not include {flag} {expected}",
                compiled_artifact_path=compiled_artifact_path,
                run_result_path=run_result_path,
                check_result_path=check_result_path,
                source_artifact_path=source_artifact_path,
                line_number=line_number,
            )
        )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def verify_run_plan_manifest(manifest_path: Path) -> RunPlanVerificationResult:
    records, issues = _load_plan_records(manifest_path)
    seen_run_paths: set[str] = set()
    seen_check_paths: set[str] = set()
    sources_checked = 0

    for record in records:
        line_number = int(record["_line_number"])
        if record.get("event") != "run_plan":
            issues.append(
                RunPlanVerificationIssue(
                    code="invalid_event",
                    message="manifest row event is not 'run_plan'",
                    line_number=line_number,
                )
            )

        compiled_artifact_path = _plan_string_field(record, "compiled_artifact_path", issues)
        run_result_path = _plan_string_field(
            record,
            "run_result_path",
            issues,
            compiled_artifact_path=compiled_artifact_path,
        )
        check_result_path = _plan_string_field(
            record,
            "check_result_path",
            issues,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
        )
        source_artifact_path = _plan_string_field(
            record,
            "source_artifact_path",
            issues,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
        )
        source_sha256 = _plan_string_field(
            record,
            "source_sha256",
            issues,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )
        runner_argv = _plan_argv_field(
            record,
            issues,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )

        plan_version = _plan_string_field(
            record,
            "plan_version",
            issues,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )
        suite = _plan_string_field(
            record,
            "suite",
            issues,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )
        runner = _plan_string_field(
            record,
            "runner",
            issues,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )
        runner_family = _plan_string_field(
            record,
            "runner_family",
            issues,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )
        runner_cwd = _plan_string_field(
            record,
            "runner_cwd",
            issues,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )
        compiled_artifact_kind = _plan_string_field(
            record,
            "compiled_artifact_kind",
            issues,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )
        entry_point = _plan_string_field(
            record,
            "entry_point",
            issues,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )
        case = _plan_string_field(
            record,
            "case",
            issues,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )

        for field in (
            "plan_index",
            "compile_plan_index",
            "batch",
            "n",
            "cond",
            "seed",
            "threads_per_block",
            "tile_size",
            "unroll",
        ):
            _plan_int_field(
                record,
                field,
                issues,
                compiled_artifact_path=compiled_artifact_path,
                run_result_path=run_result_path,
                check_result_path=check_result_path,
                source_artifact_path=source_artifact_path,
            )

        batch = record.get("batch") if isinstance(record.get("batch"), int) else None
        n = record.get("n") if isinstance(record.get("n"), int) else None
        cond = record.get("cond") if isinstance(record.get("cond"), int) else None
        seed = record.get("seed") if isinstance(record.get("seed"), int) else None

        _check_string_value(
            actual=plan_version,
            expected=RUN_PLAN_VERSION,
            code="invalid_plan_version",
            message=f"plan_version must be {RUN_PLAN_VERSION!r}",
            issues=issues,
            line_number=line_number,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )
        if suite is not None and suite not in {"smoke", "tests", "benchmarks"}:
            issues.append(
                RunPlanVerificationIssue(
                    code="invalid_suite",
                    message="suite must be one of smoke, tests, benchmarks",
                    compiled_artifact_path=compiled_artifact_path,
                    run_result_path=run_result_path,
                    check_result_path=check_result_path,
                    source_artifact_path=source_artifact_path,
                    line_number=line_number,
                )
            )
        _check_string_value(
            actual=runner,
            expected="qr-v2 run-one",
            code="invalid_runner",
            message="runner must be 'qr-v2 run-one'",
            issues=issues,
            line_number=line_number,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )
        _check_string_value(
            actual=runner_family,
            expected="local_cuda_shared_object",
            code="invalid_runner_family",
            message="runner_family must be 'local_cuda_shared_object'",
            issues=issues,
            line_number=line_number,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )
        _check_string_value(
            actual=runner_cwd,
            expected=".",
            code="invalid_runner_cwd",
            message="runner_cwd must be '.'",
            issues=issues,
            line_number=line_number,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )
        _check_string_value(
            actual=compiled_artifact_kind,
            expected="cuda_shared_object",
            code="invalid_compiled_artifact_kind",
            message="compiled_artifact_kind must be 'cuda_shared_object'",
            issues=issues,
            line_number=line_number,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )

        _check_path_suffix(
            path_text=compiled_artifact_path,
            suffix=".so",
            field="compiled_artifact_path",
            issues=issues,
            line_number=line_number,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )
        _check_path_suffix(
            path_text=run_result_path,
            suffix=".jsonl",
            field="run_result_path",
            issues=issues,
            line_number=line_number,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )
        _check_path_suffix(
            path_text=check_result_path,
            suffix=".jsonl",
            field="check_result_path",
            issues=issues,
            line_number=line_number,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        )

        if compiled_artifact_path is not None:
            _check_relative_data_path(
                compiled_artifact_path,
                issues,
                code_prefix="compiled_artifact_path",
                expected_parts_prefix=("data", "qr_v2", "compiled"),
                line_number=line_number,
                compiled_artifact_path=compiled_artifact_path,
                run_result_path=run_result_path,
                check_result_path=check_result_path,
                source_artifact_path=source_artifact_path,
            )

        for field, path_text, seen_paths in (
            ("run_result_path", run_result_path, seen_run_paths),
            ("check_result_path", check_result_path, seen_check_paths),
        ):
            if path_text is None:
                continue
            if path_text in seen_paths:
                issues.append(
                    RunPlanVerificationIssue(
                        code=f"duplicate_{field}",
                        message=f"duplicate {field} in run plan",
                        compiled_artifact_path=compiled_artifact_path,
                        run_result_path=run_result_path,
                        check_result_path=check_result_path,
                        source_artifact_path=source_artifact_path,
                        line_number=line_number,
                    )
                )
            seen_paths.add(path_text)
            _check_relative_data_path(
                path_text,
                issues,
                code_prefix=field,
                expected_parts_prefix=("data", "qr_v2", "run-results"),
                line_number=line_number,
                compiled_artifact_path=compiled_artifact_path,
                run_result_path=run_result_path,
                check_result_path=check_result_path,
                source_artifact_path=source_artifact_path,
            )

        _check_runner_argv(
            runner_argv=runner_argv,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
            entry_point=entry_point,
            batch=batch,
            n=n,
            cond=cond,
            seed=seed,
            case=case,
            issues=issues,
            line_number=line_number,
        )

        if source_artifact_path is None:
            continue
        if not _check_relative_data_path(
            source_artifact_path,
            issues,
            code_prefix="source_artifact_path",
            expected_parts_prefix=("data",),
            line_number=line_number,
            compiled_artifact_path=compiled_artifact_path,
            run_result_path=run_result_path,
            check_result_path=check_result_path,
            source_artifact_path=source_artifact_path,
        ):
            continue

        source_path = Path(source_artifact_path)
        if not source_path.exists():
            issues.append(
                RunPlanVerificationIssue(
                    code="missing_source_artifact",
                    message="source artifact does not exist",
                    compiled_artifact_path=compiled_artifact_path,
                    run_result_path=run_result_path,
                    check_result_path=check_result_path,
                    source_artifact_path=source_artifact_path,
                    line_number=line_number,
                )
            )
            continue

        sources_checked += 1
        actual_sha256 = _sha256_bytes(source_path.read_bytes())
        if source_sha256 is not None and actual_sha256 != source_sha256:
            issues.append(
                RunPlanVerificationIssue(
                    code="source_sha256_mismatch",
                    message=f"expected {source_sha256}, found {actual_sha256}",
                    compiled_artifact_path=compiled_artifact_path,
                    run_result_path=run_result_path,
                    check_result_path=check_result_path,
                    source_artifact_path=source_artifact_path,
                    line_number=line_number,
                )
            )

    return RunPlanVerificationResult(
        manifest_path=manifest_path,
        records_checked=len(records),
        sources_checked=sources_checked,
        issues=issues,
    )
