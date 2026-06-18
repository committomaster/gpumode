"""Compile planning for rendered qr_v2 artifacts."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpumode.qr_v2.jsonl import write_jsonl


COMPILE_PLAN_VERSION = "compile_plan_v1"


@dataclass(frozen=True, slots=True)
class CompilePlanResult:
    manifest_path: Path
    records: list[dict[str, object]]

    @property
    def output_paths(self) -> list[Path]:
        return [Path(str(record["output_path"])) for record in self.records]


@dataclass(frozen=True, slots=True)
class CompilePlanVerificationIssue:
    code: str
    message: str
    artifact_path: str | None = None
    output_path: str | None = None
    line_number: int | None = None

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "code": self.code,
            "message": self.message,
        }
        if self.artifact_path is not None:
            record["artifact_path"] = self.artifact_path
        if self.output_path is not None:
            record["output_path"] = self.output_path
        if self.line_number is not None:
            record["line_number"] = self.line_number
        return record


@dataclass(frozen=True, slots=True)
class CompilePlanVerificationResult:
    manifest_path: Path
    records_checked: int
    sources_checked: int
    issues: list[CompilePlanVerificationIssue]

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
        raise ValueError(f"render manifest row is missing string field {field!r}")
    return value


def _required_int(record: dict[str, Any], field: str) -> int:
    value = record.get(field)
    if not isinstance(value, int):
        raise ValueError(f"render manifest row is missing integer field {field!r}")
    return value


def _compile_output_path(*, suite: str, candidate_id: str, artifact_path: str, output_root: Path) -> Path:
    stem = Path(artifact_path).with_suffix(".so").name
    return output_root / suite / candidate_id / stem


def _compile_argv(*, artifact_path: str, output_path: Path, gpu_arch: str) -> list[str]:
    return [
        "nvcc",
        f"-arch={gpu_arch}",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        "-O3",
        "--std=c++17",
        "-o",
        output_path.as_posix(),
        artifact_path,
    ]


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def plan_compile_from_render_manifest(
    *,
    suite: str,
    render_manifest_path: Path,
    out_dir: Path,
    output_root: Path,
    target_backend: str = "cuda",
    gpu_arch: str = "sm_89",
) -> CompilePlanResult:
    render_records = _read_jsonl_objects(render_manifest_path)
    records: list[dict[str, object]] = []

    for index, render_record in enumerate(render_records):
        artifact_path = _required_string(render_record, "artifact_path")
        candidate_id = _required_string(render_record, "candidate_id")
        output_path = _compile_output_path(
            suite=suite,
            candidate_id=candidate_id,
            artifact_path=artifact_path,
            output_root=output_root,
        )

        records.append(
            {
                "event": "compile_plan",
                "plan_version": COMPILE_PLAN_VERSION,
                "suite": suite,
                "plan_index": index,
                "target_backend": target_backend,
                "compiler": "nvcc",
                "compiler_family": "cuda",
                "gpu_arch": gpu_arch,
                "compile_cwd": ".",
                "compile_argv": _compile_argv(
                    artifact_path=artifact_path,
                    output_path=output_path,
                    gpu_arch=gpu_arch,
                ),
                "output_kind": "cuda_shared_object",
                "output_path": output_path.as_posix(),
                "artifact_path": artifact_path,
                "artifact_kind": _required_string(render_record, "artifact_kind"),
                "source_bytes": _required_int(render_record, "artifact_bytes"),
                "source_sha256": _required_string(render_record, "artifact_sha256"),
                "candidate_id": candidate_id,
                "template_id": _required_string(render_record, "template_id"),
                "entry_point": _required_string(render_record, "entry_point"),
                "renderer_version": _required_string(render_record, "renderer_version"),
                "batch": _required_int(render_record, "batch"),
                "n": _required_int(render_record, "n"),
                "cond": _required_int(render_record, "cond"),
                "seed": _required_int(render_record, "seed"),
                "case": _required_string(render_record, "case"),
                "threads_per_block": _required_int(render_record, "threads_per_block"),
                "tile_size": _required_int(render_record, "tile_size"),
                "unroll": _required_int(render_record, "unroll"),
            }
        )

    manifest_path = out_dir / "manifest.jsonl"
    write_jsonl(manifest_path, records)
    return CompilePlanResult(manifest_path=manifest_path, records=records)


def _load_plan_records(manifest_path: Path) -> tuple[list[dict[str, Any]], list[CompilePlanVerificationIssue]]:
    if not manifest_path.exists():
        return [], [CompilePlanVerificationIssue(code="missing_manifest", message=f"missing manifest: {manifest_path}")]

    records: list[dict[str, Any]] = []
    issues: list[CompilePlanVerificationIssue] = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as error:
                issues.append(
                    CompilePlanVerificationIssue(
                        code="invalid_json",
                        message=f"invalid JSON: {error.msg}",
                        line_number=line_number,
                    )
                )
                continue
            if not isinstance(parsed, dict):
                issues.append(
                    CompilePlanVerificationIssue(
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
    issues: list[CompilePlanVerificationIssue],
    *,
    artifact_path: str | None = None,
    output_path: str | None = None,
) -> str | None:
    value = record.get(field)
    if isinstance(value, str) and value:
        return value
    issues.append(
        CompilePlanVerificationIssue(
            code=f"invalid_{field}",
            message=f"missing or invalid {field!r}",
            artifact_path=artifact_path,
            output_path=output_path,
            line_number=int(record["_line_number"]),
        )
    )
    return None


def _plan_int_field(
    record: dict[str, Any],
    field: str,
    issues: list[CompilePlanVerificationIssue],
    *,
    artifact_path: str | None = None,
    output_path: str | None = None,
) -> int | None:
    value = record.get(field)
    if isinstance(value, int):
        return value
    issues.append(
        CompilePlanVerificationIssue(
            code=f"invalid_{field}",
            message=f"missing or invalid {field!r}",
            artifact_path=artifact_path,
            output_path=output_path,
            line_number=int(record["_line_number"]),
        )
    )
    return None


def _plan_argv_field(
    record: dict[str, Any],
    issues: list[CompilePlanVerificationIssue],
    *,
    artifact_path: str | None,
    output_path: str | None,
) -> list[str] | None:
    value = record.get("compile_argv")
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return [str(item) for item in value]
    issues.append(
        CompilePlanVerificationIssue(
            code="invalid_compile_argv",
            message="missing or invalid 'compile_argv'",
            artifact_path=artifact_path,
            output_path=output_path,
            line_number=int(record["_line_number"]),
        )
    )
    return None


def _check_relative_data_path(
    path_text: str,
    issues: list[CompilePlanVerificationIssue],
    *,
    code_prefix: str,
    expected_parts_prefix: tuple[str, ...],
    line_number: int,
    artifact_path: str | None = None,
    output_path: str | None = None,
) -> bool:
    path = Path(path_text)
    if path.is_absolute():
        issues.append(
            CompilePlanVerificationIssue(
                code=f"absolute_{code_prefix}",
                message=f"{code_prefix} must be relative",
                artifact_path=artifact_path,
                output_path=output_path,
                line_number=line_number,
            )
        )
        return False
    if path.parts[: len(expected_parts_prefix)] != expected_parts_prefix:
        issues.append(
            CompilePlanVerificationIssue(
                code=f"{code_prefix}_outside_data",
                message=f"{code_prefix} must stay under {'/'.join(expected_parts_prefix)}/",
                artifact_path=artifact_path,
                output_path=output_path,
                line_number=line_number,
            )
        )
        return False
    return True


def verify_compile_plan_manifest(manifest_path: Path) -> CompilePlanVerificationResult:
    records, issues = _load_plan_records(manifest_path)
    seen_outputs: set[str] = set()
    sources_checked = 0

    for record in records:
        line_number = int(record["_line_number"])
        if record.get("event") != "compile_plan":
            issues.append(
                CompilePlanVerificationIssue(
                    code="invalid_event",
                    message="manifest row event is not 'compile_plan'",
                    line_number=line_number,
                )
            )

        artifact_path = _plan_string_field(record, "artifact_path", issues)
        output_path = _plan_string_field(record, "output_path", issues, artifact_path=artifact_path)
        source_sha256 = _plan_string_field(
            record,
            "source_sha256",
            issues,
            artifact_path=artifact_path,
            output_path=output_path,
        )
        source_bytes = _plan_int_field(
            record,
            "source_bytes",
            issues,
            artifact_path=artifact_path,
            output_path=output_path,
        )
        compile_argv = _plan_argv_field(
            record,
            issues,
            artifact_path=artifact_path,
            output_path=output_path,
        )

        if output_path is not None:
            if output_path in seen_outputs:
                issues.append(
                    CompilePlanVerificationIssue(
                        code="duplicate_output_path",
                        message="duplicate output path in compile plan",
                        artifact_path=artifact_path,
                        output_path=output_path,
                        line_number=line_number,
                    )
                )
            seen_outputs.add(output_path)
            _check_relative_data_path(
                output_path,
                issues,
                code_prefix="output_path",
                expected_parts_prefix=("data", "qr_v2", "compiled"),
                line_number=line_number,
                artifact_path=artifact_path,
                output_path=output_path,
            )

        if compile_argv is not None:
            if artifact_path is not None and artifact_path not in compile_argv:
                issues.append(
                    CompilePlanVerificationIssue(
                        code="compile_argv_missing_source",
                        message="compile_argv does not include artifact_path",
                        artifact_path=artifact_path,
                        output_path=output_path,
                        line_number=line_number,
                    )
                )
            if output_path is not None and output_path not in compile_argv:
                issues.append(
                    CompilePlanVerificationIssue(
                        code="compile_argv_missing_output",
                        message="compile_argv does not include output_path",
                        artifact_path=artifact_path,
                        output_path=output_path,
                        line_number=line_number,
                    )
                )

        if artifact_path is None:
            continue
        if not _check_relative_data_path(
            artifact_path,
            issues,
            code_prefix="artifact_path",
            expected_parts_prefix=("data",),
            line_number=line_number,
            artifact_path=artifact_path,
            output_path=output_path,
        ):
            continue

        source_path = Path(artifact_path)
        if not source_path.exists():
            issues.append(
                CompilePlanVerificationIssue(
                    code="missing_artifact",
                    message="source artifact does not exist",
                    artifact_path=artifact_path,
                    output_path=output_path,
                    line_number=line_number,
                )
            )
            continue

        content = source_path.read_bytes()
        sources_checked += 1
        actual_bytes = len(content)
        if source_bytes is not None and actual_bytes != source_bytes:
            issues.append(
                CompilePlanVerificationIssue(
                    code="source_bytes_mismatch",
                    message=f"expected {source_bytes} bytes, found {actual_bytes}",
                    artifact_path=artifact_path,
                    output_path=output_path,
                    line_number=line_number,
                )
            )

        actual_sha256 = _sha256_bytes(content)
        if source_sha256 is not None and actual_sha256 != source_sha256:
            issues.append(
                CompilePlanVerificationIssue(
                    code="source_sha256_mismatch",
                    message=f"expected {source_sha256}, found {actual_sha256}",
                    artifact_path=artifact_path,
                    output_path=output_path,
                    line_number=line_number,
                )
            )

    return CompilePlanVerificationResult(
        manifest_path=manifest_path,
        records_checked=len(records),
        sources_checked=sources_checked,
        issues=issues,
    )
