"""Command-line helpers for local qr_v2 work."""

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

from rich.console import Console
from rich.table import Table

from gpumode.qr_v2.baseline import get_spec_suite, run_torch_geqrf_baseline
from gpumode.qr_v2.compile import (
    CompileExecutionResult,
    CompileResultVerificationResult,
    compile_from_plan_manifest,
    compile_one_from_plan_manifest,
    verify_compile_result_manifest,
)
from gpumode.qr_v2.compile_plan import (
    CompilePlanResult,
    CompilePlanVerificationResult,
    plan_compile_from_render_manifest,
    verify_compile_plan_manifest,
)
from gpumode.qr_v2.jsonl import append_jsonl, dumps_record, write_jsonl
from gpumode.qr_v2.probes import run_structure_probe
from gpumode.qr_v2.render import KERNEL_CONFIGS, RenderSuiteResult, RenderVerificationResult, render_suite, verify_render_manifest
from gpumode.qr_v2.run import (
    RunExecutionResult,
    RunResultVerificationResult,
    run_from_plan_manifest,
    run_one,
    run_one_from_plan_manifest,
    verify_run_result_manifest,
)
from gpumode.qr_v2.run_plan import (
    RunPlanResult,
    RunPlanVerificationResult,
    plan_run_from_compile_plan_manifest,
    verify_run_plan_manifest,
)
from gpumode.qr_v2.specs import QrV2Spec
from gpumode.qr_v2.summary import (
    render_rich_summary_section,
    render_summary_section,
    summarize_baseline,
    summarize_probes,
)


DATA_DIR = Path("data")


def _filtered_specs(specs: Iterable[QrV2Spec], *, limit: int | None, max_n: int | None) -> list[QrV2Spec]:
    filtered = [spec for spec in specs if max_n is None or spec.n <= max_n]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def _data_output_path(out: str | None, *, suite: str, kind: str) -> Path:
    path = Path(out) if out is not None else Path("qr_v2") / kind / f"{suite}.jsonl"
    if path.is_absolute():
        raise ValueError("--out must be relative so generated data stays under ./data/")
    if path.parts[:1] == ("data",):
        return path
    return DATA_DIR / path


def _data_directory_path(out: str | None, *, suite: str, kind: str) -> Path:
    path = Path(out) if out is not None else Path("qr_v2") / kind / suite
    if path.is_absolute():
        raise ValueError("--out must be relative so generated data stays under ./data/")
    if path.parts[:1] == ("data",):
        return path
    return DATA_DIR / path


def _write_records(args: argparse.Namespace, records: list[dict[str, object]], *, kind: str) -> None:
    if args.stdout:
        for record in records:
            print(dumps_record(record))
    elif args.append:
        out = _data_output_path(args.out, suite=args.suite, kind=kind)
        for record in records:
            append_jsonl(out, record)
    else:
        write_jsonl(_data_output_path(args.out, suite=args.suite, kind=kind), records)


def _run_baseline(args: argparse.Namespace) -> int:
    specs = _filtered_specs(get_spec_suite(args.suite), limit=args.limit, max_n=args.max_n)
    records = [run_torch_geqrf_baseline(spec, device=args.device) for spec in specs]
    _write_records(args, records, kind="baseline")

    return 0 if all(bool(record["passed"]) for record in records) else 1


def _run_probe(args: argparse.Namespace) -> int:
    specs = _filtered_specs(get_spec_suite(args.suite), limit=args.limit, max_n=args.max_n)
    records = [
        run_structure_probe(
            spec,
            device=args.device,
            early_stop_max_n=args.early_stop_max_n,
            rank_probe_max_n=args.rank_probe_max_n,
        )
        for spec in specs
    ]
    _write_records(args, records, kind="probes")
    return 0


def _render_report(result: RenderSuiteResult) -> None:
    artifact_count = len(result.records)
    if not sys.stdout.isatty():
        print(f"rendered {artifact_count} artifacts")
        print(f"manifest: {result.manifest_path}")
        return

    table = Table(title="render", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("artifacts", str(artifact_count))
    table.add_row("manifest", result.manifest_path.as_posix())
    if artifact_count:
        table.add_row("candidate", str(result.records[0]["candidate_id"]))
        table.add_row("template", str(result.records[0]["template_id"]))
    Console().print(table)


def _run_render(args: argparse.Namespace) -> int:
    specs = _filtered_specs(get_spec_suite(args.suite), limit=args.limit, max_n=args.max_n)
    result = render_suite(
        suite=args.suite,
        specs=specs,
        out_dir=_data_directory_path(args.out, suite=args.suite, kind="renders"),
        config=KERNEL_CONFIGS[args.candidate],
    )

    if args.stdout:
        for record in result.records:
            print(dumps_record(record))
    else:
        _render_report(result)
    return 0


def _verify_report(result: RenderVerificationResult) -> None:
    if not sys.stdout.isatty():
        print("render manifest ok" if result.ok else "render manifest failed")
        print(f"manifest: {result.manifest_path}")
        print(f"records: {result.records_checked}")
        print(f"artifacts: {result.artifacts_checked}")
        print(f"issues: {len(result.issues)}")
        for issue in result.issues:
            print(dumps_record(issue.to_record()))
        return

    table = Table(title="verify-renders", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("status", "[green]pass[/green]" if result.ok else "[bold red]fail[/bold red]")
    table.add_row("manifest", result.manifest_path.as_posix())
    table.add_row("records", str(result.records_checked))
    table.add_row("artifacts", str(result.artifacts_checked))
    table.add_row("issues", str(len(result.issues)))
    Console().print(table)

    if result.issues:
        issues = Table(title="render manifest issues")
        issues.add_column("code", style="bold red")
        issues.add_column("line", justify="right")
        issues.add_column("artifact")
        issues.add_column("message")
        for issue in result.issues:
            issues.add_row(
                issue.code,
                "" if issue.line_number is None else str(issue.line_number),
                issue.artifact_path or "",
                issue.message,
            )
        Console().print(issues)


def _run_verify_renders(args: argparse.Namespace) -> int:
    manifest_path = _data_output_path(args.manifest, suite=args.suite, kind="renders")
    if args.manifest is None:
        manifest_path = _data_directory_path(None, suite=args.suite, kind="renders") / "manifest.jsonl"

    result = verify_render_manifest(manifest_path)
    if args.stdout:
        for issue in result.issues:
            print(dumps_record(issue.to_record()))
    else:
        _verify_report(result)
    return 0 if result.ok else 1


def _compile_plan_report(result: CompilePlanResult) -> None:
    plan_count = len(result.records)
    if not sys.stdout.isatty():
        print(f"planned {plan_count} compile jobs")
        print(f"manifest: {result.manifest_path}")
        if plan_count:
            print(f"backend: {result.records[0]['target_backend']}")
            print(f"compiler: {result.records[0]['compiler']}")
            print(f"gpu_arch: {result.records[0]['gpu_arch']}")
        return

    table = Table(title="compile plan", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("jobs", str(plan_count))
    table.add_row("manifest", result.manifest_path.as_posix())
    if plan_count:
        table.add_row("backend", str(result.records[0]["target_backend"]))
        table.add_row("compiler", str(result.records[0]["compiler"]))
        table.add_row("gpu_arch", str(result.records[0]["gpu_arch"]))
    Console().print(table)


def _run_plan_compile(args: argparse.Namespace) -> int:
    render_manifest_path = _data_directory_path(None, suite=args.suite, kind="renders") / "manifest.jsonl"
    if args.render_manifest is not None:
        render_manifest_path = _data_output_path(args.render_manifest, suite=args.suite, kind="renders")

    verification = verify_render_manifest(render_manifest_path)
    if not verification.ok:
        if args.stdout:
            for issue in verification.issues:
                print(dumps_record(issue.to_record()))
        else:
            _verify_report(verification)
        return 1

    result = plan_compile_from_render_manifest(
        suite=args.suite,
        render_manifest_path=render_manifest_path,
        out_dir=_data_directory_path(args.out, suite=args.suite, kind="compile-plans"),
        output_root=DATA_DIR / "qr_v2" / "compiled",
        gpu_arch=args.gpu_arch,
    )
    if args.stdout:
        for record in result.records:
            print(dumps_record(record))
    else:
        _compile_plan_report(result)
    return 0


def _verify_compile_plan_report(result: CompilePlanVerificationResult) -> None:
    if not sys.stdout.isatty():
        print("compile plan ok" if result.ok else "compile plan failed")
        print(f"manifest: {result.manifest_path}")
        print(f"records: {result.records_checked}")
        print(f"sources: {result.sources_checked}")
        print(f"issues: {len(result.issues)}")
        for issue in result.issues:
            print(dumps_record(issue.to_record()))
        return

    table = Table(title="verify-compile-plan", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("status", "[green]pass[/green]" if result.ok else "[bold red]fail[/bold red]")
    table.add_row("manifest", result.manifest_path.as_posix())
    table.add_row("records", str(result.records_checked))
    table.add_row("sources", str(result.sources_checked))
    table.add_row("issues", str(len(result.issues)))
    Console().print(table)

    if result.issues:
        issues = Table(title="compile plan issues")
        issues.add_column("code", style="bold red")
        issues.add_column("line", justify="right")
        issues.add_column("artifact")
        issues.add_column("output")
        issues.add_column("message")
        for issue in result.issues:
            issues.add_row(
                issue.code,
                "" if issue.line_number is None else str(issue.line_number),
                issue.artifact_path or "",
                issue.output_path or "",
                issue.message,
            )
        Console().print(issues)


def _run_verify_compile_plan(args: argparse.Namespace) -> int:
    manifest_path = _data_directory_path(None, suite=args.suite, kind="compile-plans") / "manifest.jsonl"
    if args.manifest is not None:
        manifest_path = _data_output_path(args.manifest, suite=args.suite, kind="compile-plans")

    result = verify_compile_plan_manifest(manifest_path)
    if args.stdout:
        for issue in result.issues:
            print(dumps_record(issue.to_record()))
    else:
        _verify_compile_plan_report(result)
    return 0 if result.ok else 1


def _compile_result_report(result: CompileExecutionResult) -> None:
    result_count = len(result.records)
    ok_count = sum(1 for record in result.records if bool(record.get("ok")))
    failed_count = result_count - ok_count
    if not sys.stdout.isatty():
        print(f"compiled {ok_count}/{result_count} jobs")
        print(f"manifest: {result.manifest_path}")
        print(f"failed: {failed_count}")
        if result_count == 1:
            record = result.records[0]
            print(f"status: {record.get('status', '')}")
            if record.get("output_path"):
                print(f"output: {record['output_path']}")
        return

    table = Table(title="compile", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("status", "[green]pass[/green]" if result.ok else "[bold red]fail[/bold red]")
    table.add_row("manifest", result.manifest_path.as_posix())
    table.add_row("jobs", str(result_count))
    table.add_row("passed", str(ok_count))
    table.add_row("failed", str(failed_count))
    if result_count == 1:
        record = result.records[0]
        table.add_row("result", str(record.get("status", "")))
        if record.get("output_path"):
            table.add_row("output", str(record["output_path"]))
    Console().print(table)


def _run_compile_one(args: argparse.Namespace) -> int:
    compile_plan_path = _data_directory_path(None, suite=args.suite, kind="compile-plans") / "manifest.jsonl"
    if args.compile_plan is not None:
        compile_plan_path = _data_output_path(args.compile_plan, suite=args.suite, kind="compile-plans")

    result = compile_one_from_plan_manifest(
        suite=args.suite,
        compile_plan_manifest_path=compile_plan_path,
        index=args.index,
        out_dir=_data_directory_path(args.out, suite=args.suite, kind="compile-results"),
        timeout_seconds=args.timeout_seconds,
        max_snippet_chars=args.max_snippet_chars,
    )
    if args.stdout:
        for record in result.records:
            print(dumps_record(record))
    else:
        _compile_result_report(result)
    return 0 if result.ok else 1


def _run_compile(args: argparse.Namespace) -> int:
    compile_plan_path = _data_directory_path(None, suite=args.suite, kind="compile-plans") / "manifest.jsonl"
    if args.compile_plan is not None:
        compile_plan_path = _data_output_path(args.compile_plan, suite=args.suite, kind="compile-plans")

    result = compile_from_plan_manifest(
        suite=args.suite,
        compile_plan_manifest_path=compile_plan_path,
        out_dir=_data_directory_path(args.out, suite=args.suite, kind="compile-results"),
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
        max_snippet_chars=args.max_snippet_chars,
    )
    if args.stdout:
        for record in result.records:
            print(dumps_record(record))
    else:
        _compile_result_report(result)
    return 0 if result.ok else 1


def _verify_compile_result_report(result: CompileResultVerificationResult) -> None:
    if not sys.stdout.isatty():
        print("compile results ok" if result.ok else "compile results failed")
        print(f"manifest: {result.manifest_path}")
        print(f"records: {result.records_checked}")
        print(f"outputs: {result.outputs_checked}")
        print(f"logs: {result.logs_checked}")
        print(f"issues: {len(result.issues)}")
        for issue in result.issues:
            print(dumps_record(issue.to_record()))
        return

    table = Table(title="verify-compile-results", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("status", "[green]pass[/green]" if result.ok else "[bold red]fail[/bold red]")
    table.add_row("manifest", result.manifest_path.as_posix())
    table.add_row("records", str(result.records_checked))
    table.add_row("outputs", str(result.outputs_checked))
    table.add_row("logs", str(result.logs_checked))
    table.add_row("issues", str(len(result.issues)))
    Console().print(table)

    if result.issues:
        issues = Table(title="compile result issues")
        issues.add_column("code", style="bold red")
        issues.add_column("line", justify="right")
        issues.add_column("output")
        issues.add_column("message")
        for issue in result.issues:
            issues.add_row(
                issue.code,
                "" if issue.line_number is None else str(issue.line_number),
                issue.output_path or "",
                issue.message,
            )
        Console().print(issues)


def _run_verify_compile_results(args: argparse.Namespace) -> int:
    manifest_path = _data_directory_path(None, suite=args.suite, kind="compile-results") / "manifest.jsonl"
    if args.manifest is not None:
        manifest_path = _data_output_path(args.manifest, suite=args.suite, kind="compile-results")

    result = verify_compile_result_manifest(manifest_path)
    if args.stdout:
        for issue in result.issues:
            print(dumps_record(issue.to_record()))
    else:
        _verify_compile_result_report(result)
    return 0 if result.ok else 1


def _run_result_report(result: RunExecutionResult) -> None:
    result_count = len(result.records)
    ok_count = sum(1 for record in result.records if bool(record.get("ok")))
    failed_count = result_count - ok_count
    if not sys.stdout.isatty():
        print(f"ran {ok_count}/{result_count} jobs")
        print(f"manifest: {result.result_path}")
        print(f"failed: {failed_count}")
        if result_count == 1:
            record = result.records[0]
            print(f"status: {record.get('status', '')}")
            print(f"passed: {record.get('passed', False)}")
            if record.get("run_result_path"):
                print(f"result: {record['run_result_path']}")
        return

    table = Table(title="run", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("status", "[green]pass[/green]" if result.ok else "[bold red]fail[/bold red]")
    table.add_row("manifest", result.result_path.as_posix())
    table.add_row("jobs", str(result_count))
    table.add_row("completed", str(ok_count))
    table.add_row("failed", str(failed_count))
    if result_count == 1:
        record = result.records[0]
        table.add_row("result", str(record.get("status", "")))
        table.add_row("passed", str(record.get("passed", False)))
        if record.get("run_result_path"):
            table.add_row("output", str(record["run_result_path"]))
    Console().print(table)


def _run_run_one(args: argparse.Namespace) -> int:
    result_path = None if args.out is None else _data_output_path(args.out, suite=args.suite, kind="run-results")

    if args.compiled_artifact is not None:
        missing = [
            name
            for name in ("entry_point", "batch", "n", "cond", "seed", "case")
            if getattr(args, name) is None
        ]
        if missing:
            print(f"missing direct run-one args: {', '.join('--' + name.replace('_', '-') for name in missing)}", file=sys.stderr)
            return 2
        assert args.entry_point is not None
        assert args.batch is not None
        assert args.n is not None
        assert args.cond is not None
        assert args.seed is not None
        assert args.case is not None
        result = run_one(
            suite=args.suite,
            compiled_artifact_path=Path(args.compiled_artifact),
            entry_point=args.entry_point,
            launcher_entry_point=args.launcher_entry_point,
            batch=args.batch,
            n=args.n,
            cond=args.cond,
            seed=args.seed,
            case=args.case,
            threads_per_block=args.threads_per_block,
            result_path=result_path,
            device=args.device,
            max_snippet_chars=args.max_snippet_chars,
        )
    else:
        run_plan_path = _data_directory_path(None, suite=args.suite, kind="run-plans") / "manifest.jsonl"
        if args.run_plan is not None:
            run_plan_path = _data_output_path(args.run_plan, suite=args.suite, kind="run-plans")
        result = run_one_from_plan_manifest(
            suite=args.suite,
            run_plan_manifest_path=run_plan_path,
            index=args.index,
            result_path=result_path,
            device=args.device,
            max_snippet_chars=args.max_snippet_chars,
        )

    if args.stdout:
        for record in result.records:
            print(dumps_record(record))
    else:
        _run_result_report(result)
    return 0 if result.ok else 1


def _run_run(args: argparse.Namespace) -> int:
    run_plan_path = _data_directory_path(None, suite=args.suite, kind="run-plans") / "manifest.jsonl"
    if args.run_plan is not None:
        run_plan_path = _data_output_path(args.run_plan, suite=args.suite, kind="run-plans")

    result = run_from_plan_manifest(
        suite=args.suite,
        run_plan_manifest_path=run_plan_path,
        out_dir=_data_directory_path(args.out, suite=args.suite, kind="run-results"),
        limit=args.limit,
        device=args.device,
        max_snippet_chars=args.max_snippet_chars,
    )
    if args.stdout:
        for record in result.records:
            print(dumps_record(record))
    else:
        _run_result_report(result)
    return 0 if result.ok else 1


def _verify_run_result_report(result: RunResultVerificationResult) -> None:
    if not sys.stdout.isatty():
        print("run results ok" if result.ok else "run results failed")
        print(f"manifest: {result.manifest_path}")
        print(f"records: {result.records_checked}")
        print(f"result_files: {result.result_files_checked}")
        print(f"artifacts: {result.artifacts_checked}")
        print(f"issues: {len(result.issues)}")
        for issue in result.issues:
            print(dumps_record(issue.to_record()))
        return

    table = Table(title="verify-run-results", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("status", "[green]pass[/green]" if result.ok else "[bold red]fail[/bold red]")
    table.add_row("manifest", result.manifest_path.as_posix())
    table.add_row("records", str(result.records_checked))
    table.add_row("result files", str(result.result_files_checked))
    table.add_row("artifacts", str(result.artifacts_checked))
    table.add_row("issues", str(len(result.issues)))
    Console().print(table)

    if result.issues:
        issues = Table(title="run result issues")
        issues.add_column("code", style="bold red")
        issues.add_column("line", justify="right")
        issues.add_column("result")
        issues.add_column("message")
        for issue in result.issues:
            issues.add_row(
                issue.code,
                "" if issue.line_number is None else str(issue.line_number),
                issue.run_result_path or "",
                issue.message,
            )
        Console().print(issues)


def _run_verify_run_results(args: argparse.Namespace) -> int:
    manifest_path = _data_directory_path(None, suite=args.suite, kind="run-results") / "manifest.jsonl"
    if args.manifest is not None:
        manifest_path = _data_output_path(args.manifest, suite=args.suite, kind="run-results")

    result = verify_run_result_manifest(manifest_path)
    if args.stdout:
        for issue in result.issues:
            print(dumps_record(issue.to_record()))
    else:
        _verify_run_result_report(result)
    return 0 if result.ok else 1


def _run_plan_report(result: RunPlanResult) -> None:
    plan_count = len(result.records)
    if not sys.stdout.isatty():
        print(f"planned {plan_count} run jobs")
        print(f"manifest: {result.manifest_path}")
        if plan_count:
            print(f"runner: {result.records[0]['runner']}")
        return

    table = Table(title="run plan", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("jobs", str(plan_count))
    table.add_row("manifest", result.manifest_path.as_posix())
    if plan_count:
        table.add_row("runner", str(result.records[0]["runner"]))
    Console().print(table)


def _run_plan_run(args: argparse.Namespace) -> int:
    compile_plan_path = _data_directory_path(None, suite=args.suite, kind="compile-plans") / "manifest.jsonl"
    if args.compile_plan is not None:
        compile_plan_path = _data_output_path(args.compile_plan, suite=args.suite, kind="compile-plans")

    verification = verify_compile_plan_manifest(compile_plan_path)
    if not verification.ok:
        if args.stdout:
            for issue in verification.issues:
                print(dumps_record(issue.to_record()))
        else:
            _verify_compile_plan_report(verification)
        return 1

    result = plan_run_from_compile_plan_manifest(
        suite=args.suite,
        compile_plan_manifest_path=compile_plan_path,
        out_dir=_data_directory_path(args.out, suite=args.suite, kind="run-plans"),
        result_root=DATA_DIR / "qr_v2" / "run-results",
    )
    if args.stdout:
        for record in result.records:
            print(dumps_record(record))
    else:
        _run_plan_report(result)
    return 0


def _verify_run_plan_report(result: RunPlanVerificationResult) -> None:
    if not sys.stdout.isatty():
        print("run plan ok" if result.ok else "run plan failed")
        print(f"manifest: {result.manifest_path}")
        print(f"records: {result.records_checked}")
        print(f"sources: {result.sources_checked}")
        print(f"issues: {len(result.issues)}")
        for issue in result.issues:
            print(dumps_record(issue.to_record()))
        return

    table = Table(title="verify-run-plan", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("status", "[green]pass[/green]" if result.ok else "[bold red]fail[/bold red]")
    table.add_row("manifest", result.manifest_path.as_posix())
    table.add_row("records", str(result.records_checked))
    table.add_row("sources", str(result.sources_checked))
    table.add_row("issues", str(len(result.issues)))
    Console().print(table)

    if result.issues:
        issues = Table(title="run plan issues")
        issues.add_column("code", style="bold red")
        issues.add_column("line", justify="right")
        issues.add_column("compiled")
        issues.add_column("result")
        issues.add_column("message")
        for issue in result.issues:
            issues.add_row(
                issue.code,
                "" if issue.line_number is None else str(issue.line_number),
                issue.compiled_artifact_path or "",
                issue.run_result_path or "",
                issue.message,
            )
        Console().print(issues)


def _run_verify_run_plan(args: argparse.Namespace) -> int:
    manifest_path = _data_directory_path(None, suite=args.suite, kind="run-plans") / "manifest.jsonl"
    if args.manifest is not None:
        manifest_path = _data_output_path(args.manifest, suite=args.suite, kind="run-plans")

    result = verify_run_plan_manifest(manifest_path)
    if args.stdout:
        for issue in result.issues:
            print(dumps_record(issue.to_record()))
    else:
        _verify_run_plan_report(result)
    return 0 if result.ok else 1


def _use_rich_output(output_format: str) -> bool:
    return output_format == "rich" or (output_format == "auto" and sys.stdout.isatty())


def _run_summary(args: argparse.Namespace) -> int:
    sections = []

    if args.kind in ("all", "baseline"):
        baseline_path = _data_output_path(
            args.baseline_in,
            suite=args.suite,
            kind="baseline",
        )
        baseline = summarize_baseline(baseline_path)
        sections.append(("baseline", baseline_path, baseline))

    if args.kind in ("all", "probes"):
        probe_path = _data_output_path(args.probe_in, suite=args.suite, kind="probes")
        probes = summarize_probes(probe_path)
        sections.append(("probes", probe_path, probes))

    if _use_rich_output(args.format):
        console = Console()
        for index, (title, path, frame) in enumerate(sections):
            if index:
                console.print()
            console.print(render_rich_summary_section(title, path, frame))
        return 0

    print("\n\n".join(render_summary_section(title, path, frame) for title, path, frame in sections))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gpumode.qr_v2.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="run the torch.geqrf baseline and emit JSONL")
    baseline.add_argument("--suite", choices=("smoke", "tests", "benchmarks"), default="smoke")
    baseline.add_argument("--device", default=None)
    baseline.add_argument("--out", default=None)
    baseline.add_argument("--stdout", action="store_true")
    baseline.add_argument("--append", action="store_true")
    baseline.add_argument("--limit", type=int, default=None)
    baseline.add_argument("--max-n", type=int, default=None)
    baseline.set_defaults(func=_run_baseline)

    probe = subparsers.add_parser("probe", help="run structure probes and emit JSONL")
    probe.add_argument("--suite", choices=("smoke", "tests", "benchmarks"), default="smoke")
    probe.add_argument("--device", default=None)
    probe.add_argument("--out", default=None)
    probe.add_argument("--stdout", action="store_true")
    probe.add_argument("--append", action="store_true")
    probe.add_argument("--limit", type=int, default=None)
    probe.add_argument("--max-n", type=int, default=None)
    probe.add_argument("--early-stop-max-n", type=int, default=512)
    probe.add_argument("--rank-probe-max-n", type=int, default=512)
    probe.set_defaults(func=_run_probe)

    render = subparsers.add_parser("render", help="render deterministic qr_v2 candidate artifacts")
    render.add_argument("--suite", choices=("smoke", "tests", "benchmarks"), default="smoke")
    render.add_argument("--out", default=None)
    render.add_argument("--stdout", action="store_true")
    render.add_argument("--limit", type=int, default=None)
    render.add_argument("--max-n", type=int, default=None)
    render.add_argument("--candidate", choices=tuple(KERNEL_CONFIGS), default="serial")
    render.set_defaults(func=_run_render)

    verify_renders = subparsers.add_parser("verify-renders", help="verify rendered qr_v2 artifact manifests")
    verify_renders.add_argument("--suite", choices=("smoke", "tests", "benchmarks"), default="smoke")
    verify_renders.add_argument("--manifest", default=None)
    verify_renders.add_argument("--stdout", action="store_true")
    verify_renders.set_defaults(func=_run_verify_renders)

    plan_compile = subparsers.add_parser("plan-compile", help="plan future compile jobs from rendered artifacts")
    plan_compile.add_argument("--suite", choices=("smoke", "tests", "benchmarks"), default="smoke")
    plan_compile.add_argument("--render-manifest", default=None)
    plan_compile.add_argument("--out", default=None)
    plan_compile.add_argument("--stdout", action="store_true")
    plan_compile.add_argument("--gpu-arch", default="sm_89")
    plan_compile.set_defaults(func=_run_plan_compile)

    verify_compile_plan = subparsers.add_parser("verify-compile-plan", help="verify qr_v2 compile-plan manifests")
    verify_compile_plan.add_argument("--suite", choices=("smoke", "tests", "benchmarks"), default="smoke")
    verify_compile_plan.add_argument("--manifest", default=None)
    verify_compile_plan.add_argument("--stdout", action="store_true")
    verify_compile_plan.set_defaults(func=_run_verify_compile_plan)

    compile_one = subparsers.add_parser("compile-one", help="compile one qr_v2 compile-plan row")
    compile_one.add_argument("--suite", choices=("smoke", "tests", "benchmarks"), default="smoke")
    compile_one.add_argument("--compile-plan", default=None)
    compile_one.add_argument("--index", type=int, default=0)
    compile_one.add_argument("--out", default=None)
    compile_one.add_argument("--stdout", action="store_true")
    compile_one.add_argument("--timeout-seconds", type=float, default=None)
    compile_one.add_argument("--max-snippet-chars", type=int, default=4000)
    compile_one.set_defaults(func=_run_compile_one)

    compile_cmd = subparsers.add_parser("compile", help="compile qr_v2 compile-plan rows")
    compile_cmd.add_argument("--suite", choices=("smoke", "tests", "benchmarks"), default="smoke")
    compile_cmd.add_argument("--compile-plan", default=None)
    compile_cmd.add_argument("--out", default=None)
    compile_cmd.add_argument("--stdout", action="store_true")
    compile_cmd.add_argument("--limit", type=int, default=None)
    compile_cmd.add_argument("--timeout-seconds", type=float, default=None)
    compile_cmd.add_argument("--max-snippet-chars", type=int, default=4000)
    compile_cmd.set_defaults(func=_run_compile)

    verify_compile_results = subparsers.add_parser(
        "verify-compile-results",
        help="verify qr_v2 compile-result manifests",
    )
    verify_compile_results.add_argument("--suite", choices=("smoke", "tests", "benchmarks"), default="smoke")
    verify_compile_results.add_argument("--manifest", default=None)
    verify_compile_results.add_argument("--stdout", action="store_true")
    verify_compile_results.set_defaults(func=_run_verify_compile_results)

    run_one_cmd = subparsers.add_parser("run-one", help="run one qr_v2 run-plan row or compiled artifact")
    run_one_cmd.add_argument("--suite", choices=("smoke", "tests", "benchmarks"), default="smoke")
    run_one_cmd.add_argument("--run-plan", default=None)
    run_one_cmd.add_argument("--index", type=int, default=0)
    run_one_cmd.add_argument("--compiled-artifact", default=None)
    run_one_cmd.add_argument("--entry-point", default=None)
    run_one_cmd.add_argument("--launcher-entry-point", default=None)
    run_one_cmd.add_argument("--batch", type=int, default=None)
    run_one_cmd.add_argument("--n", type=int, default=None)
    run_one_cmd.add_argument("--cond", type=int, default=None)
    run_one_cmd.add_argument("--seed", type=int, default=None)
    run_one_cmd.add_argument("--case", default=None)
    run_one_cmd.add_argument("--threads-per-block", type=int, default=128)
    run_one_cmd.add_argument("--device", default=None)
    run_one_cmd.add_argument("--out", default=None)
    run_one_cmd.add_argument("--stdout", action="store_true")
    run_one_cmd.add_argument("--max-snippet-chars", type=int, default=4000)
    run_one_cmd.set_defaults(func=_run_run_one)

    run_cmd = subparsers.add_parser("run", help="run qr_v2 run-plan rows")
    run_cmd.add_argument("--suite", choices=("smoke", "tests", "benchmarks"), default="smoke")
    run_cmd.add_argument("--run-plan", default=None)
    run_cmd.add_argument("--out", default=None)
    run_cmd.add_argument("--stdout", action="store_true")
    run_cmd.add_argument("--limit", type=int, default=None)
    run_cmd.add_argument("--device", default=None)
    run_cmd.add_argument("--max-snippet-chars", type=int, default=4000)
    run_cmd.set_defaults(func=_run_run)

    verify_run_results = subparsers.add_parser(
        "verify-run-results",
        help="verify qr_v2 run-result manifests",
    )
    verify_run_results.add_argument("--suite", choices=("smoke", "tests", "benchmarks"), default="smoke")
    verify_run_results.add_argument("--manifest", default=None)
    verify_run_results.add_argument("--stdout", action="store_true")
    verify_run_results.set_defaults(func=_run_verify_run_results)

    plan_run = subparsers.add_parser("plan-run", help="plan future run jobs from compile plans")
    plan_run.add_argument("--suite", choices=("smoke", "tests", "benchmarks"), default="smoke")
    plan_run.add_argument("--compile-plan", default=None)
    plan_run.add_argument("--out", default=None)
    plan_run.add_argument("--stdout", action="store_true")
    plan_run.set_defaults(func=_run_plan_run)

    verify_run_plan = subparsers.add_parser("verify-run-plan", help="verify qr_v2 run-plan manifests")
    verify_run_plan.add_argument("--suite", choices=("smoke", "tests", "benchmarks"), default="smoke")
    verify_run_plan.add_argument("--manifest", default=None)
    verify_run_plan.add_argument("--stdout", action="store_true")
    verify_run_plan.set_defaults(func=_run_verify_run_plan)

    summary = subparsers.add_parser("summary", help="summarize qr_v2 JSONL data")
    summary.add_argument("--suite", choices=("smoke", "tests", "benchmarks"), default="smoke")
    summary.add_argument("--kind", choices=("all", "baseline", "probes"), default="all")
    summary.add_argument("--format", choices=("auto", "rich", "tsv"), default="auto")
    summary.add_argument("--baseline-in", default=None)
    summary.add_argument("--probe-in", default=None)
    summary.set_defaults(func=_run_summary)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
