"""Polars summaries for qr_v2 JSONL records."""

import math
from pathlib import Path
from typing import Any

import polars as pl
from rich import box
from rich.table import Table
from rich.text import Text


_BASELINE_GROUP_COLUMNS = [
    "candidate_id",
    "device",
    "dtype",
    "case",
    "n",
    "batch",
    "cond",
]

_PROBE_GROUP_COLUMNS = [
    "probe_id",
    "device",
    "dtype",
    "case",
    "n",
    "batch",
    "cond",
]

_RICH_COLUMNS = {
    "baseline": [
        "case",
        "shape",
        "cond",
        "runs",
        "failures",
        "ok",
        "p50_us",
        "factor",
        "orth",
        "lower",
        "recon",
    ],
    "probes": [
        "case",
        "shape",
        "cond",
        "records",
        "upper_rate",
        "approx_upper",
        "stop50",
        "stop90",
        "stop_n4",
        "saved",
        "speedup",
        "zero_tau",
        "rank_frac",
        "surv_n4",
        "col_span50",
    ],
}

_RIGHT_ALIGNED_COLUMNS = {
    "n",
    "batch",
    "cond",
    "runs",
    "failures",
    "records",
    "p50_us",
    "mean_us",
    "factor",
    "orth",
    "lower",
    "recon",
    "upper_mean",
    "upper_max",
    "upper_rate",
    "approx_upper",
    "stop50",
    "stop90",
    "stop_n4",
    "stop_n2",
    "saved",
    "speedup",
    "zero_tau",
    "zero_suffix",
    "rank50",
    "rank_frac",
    "surv_n4",
    "surv_n2",
    "zero_cols",
    "tiny_cols",
    "col_span",
    "col_span50",
    "row_span",
    "row_span50",
    "norm1_mean",
}

_BUDGET_COLUMNS = {"factor", "orth", "lower", "recon"}

_CONTEXT_COLUMNS = {
    "baseline": ["candidate", "dev", "dtype"],
    "probes": ["probe", "dev", "dtype"],
}

_HEADER_LABELS = {
    "cond": "c",
    "records": "rows",
    "failures": "fail",
    "ok": "ok",
    "p50_us": "p50",
    "factor": "fact",
    "lower": "low",
    "recon": "rec",
    "upper_mean": "upper avg",
    "upper_max": "upper max",
    "upper_rate": "upper",
    "approx_upper": "approx",
    "stop50": "stop50",
    "stop90": "stop90",
    "stop_n4": "<=n/4",
    "stop_n2": "<=n/2",
    "saved": "saved",
    "speedup": "ceil",
    "zero_tau": "tau0",
    "zero_suffix": "tau0 tail",
    "rank50": "rank50",
    "rank_frac": "rank/n",
    "surv_n4": "surv>n/4",
    "surv_n2": "surv>n/2",
    "zero_cols": "zero",
    "tiny_cols": "tiny",
    "col_span": "col span",
    "col_span50": "col p50",
    "row_span": "row span",
    "row_span50": "row p50",
}


def _read_event_records(path: Path, event: str) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    records = pl.read_ndjson(path)
    if "event" not in records.columns:
        raise ValueError(f"{path} does not contain an event column")

    records = records.filter(pl.col("event") == event)
    if records.is_empty():
        raise ValueError(f"{path} has no {event!r} records")
    return records


def _round_existing_columns(
    frame: pl.DataFrame,
    columns: list[str],
    digits: int,
) -> pl.DataFrame:
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return frame
    return frame.with_columns(pl.col(existing).round(digits))


def _with_missing_columns(frame: pl.DataFrame, defaults: dict[str, object]) -> pl.DataFrame:
    for column, default in defaults.items():
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(default).alias(column))
    return frame


def summarize_baseline(path: Path) -> pl.DataFrame:
    """Summarize torch/candidate baseline JSONL records."""

    records = _read_event_records(path, "baseline_result")
    summary = records.group_by(_BASELINE_GROUP_COLUMNS, maintain_order=True).agg(
        pl.len().alias("runs"),
        (pl.len() - pl.col("passed").sum()).alias("failures"),
        pl.col("passed").all().alias("all_passed"),
        pl.col("elapsed_us").median().alias("elapsed_us_p50"),
        pl.col("elapsed_us").mean().alias("elapsed_us_mean"),
        pl.col("factor_scaled").max().alias("factor_scaled_max"),
        pl.col("orth_scaled").max().alias("orth_scaled_max"),
        pl.col("lower_scaled").max().alias("lower_scaled_max"),
        pl.col("reconstruction_scaled").max().alias("reconstruction_scaled_max"),
    )

    summary = _round_existing_columns(
        summary,
        [
            "elapsed_us_p50",
            "elapsed_us_mean",
            "factor_scaled_max",
            "orth_scaled_max",
            "lower_scaled_max",
            "reconstruction_scaled_max",
        ],
        4,
    )
    return summary.rename(
        {
            "candidate_id": "candidate",
            "device": "dev",
            "all_passed": "ok",
            "elapsed_us_p50": "p50_us",
            "elapsed_us_mean": "mean_us",
            "factor_scaled_max": "factor",
            "orth_scaled_max": "orth",
            "lower_scaled_max": "lower",
            "reconstruction_scaled_max": "recon",
        }
    )


def summarize_probes(path: Path) -> pl.DataFrame:
    """Summarize semantic probe JSONL records."""

    records = _read_event_records(path, "probe_result")
    records = _with_missing_columns(
        records,
        {
            "upper_triangular_rate": None,
            "approx_upper_triangular_rate": None,
            "early_stop_evaluated": False,
            "early_stop_median_k": None,
            "early_stop_p90_k": None,
            "early_stop_n4_rate": None,
            "early_stop_n2_rate": None,
            "early_stop_work_saved_mean": None,
            "estimated_speedup_ceiling": None,
            "zero_tau_reflector_rate_mean": None,
            "zero_tau_suffix_median": None,
            "rank_probe_evaluated": False,
            "effective_rank_median": None,
            "effective_rank_fraction_median": None,
            "compaction_survivor_fraction_n4": None,
            "compaction_survivor_fraction_n2": None,
            "column_l1_span_median": None,
            "row_l1_span_median": None,
        },
    ).with_columns(
        pl.col("upper_triangular_ratio").list.mean().alias("upper_ratio_mean"),
        pl.col("upper_triangular_ratio").list.max().alias("upper_ratio_max"),
        pl.col("zero_column_count").list.max().alias("zero_cols_max"),
        pl.col("tiny_column_count").list.max().alias("tiny_cols_max"),
        pl.col("column_l1_span").list.max().alias("column_span_max"),
        pl.col("row_l1_span").list.max().alias("row_span_max"),
        pl.col("matrix_l1").list.mean().alias("matrix_l1_mean"),
    )
    summary = records.group_by(_PROBE_GROUP_COLUMNS, maintain_order=True).agg(
        pl.len().alias("records"),
        pl.col("upper_ratio_mean").mean().alias("upper_ratio_mean"),
        pl.col("upper_ratio_max").max().alias("upper_ratio_max"),
        pl.col("upper_triangular_rate").mean().alias("upper_triangular_rate"),
        pl.col("approx_upper_triangular_rate").mean().alias("approx_upper_triangular_rate"),
        pl.col("early_stop_evaluated").cast(pl.Int64).sum().alias("early_stop_records"),
        pl.col("early_stop_median_k").median().alias("early_stop_median_k"),
        pl.col("early_stop_p90_k").median().alias("early_stop_p90_k"),
        pl.col("early_stop_n4_rate").mean().alias("early_stop_n4_rate"),
        pl.col("early_stop_n2_rate").mean().alias("early_stop_n2_rate"),
        pl.col("early_stop_work_saved_mean").mean().alias("early_stop_work_saved_mean"),
        pl.col("estimated_speedup_ceiling").mean().alias("estimated_speedup_ceiling"),
        pl.col("zero_tau_reflector_rate_mean").mean().alias("zero_tau_reflector_rate_mean"),
        pl.col("zero_tau_suffix_median").median().alias("zero_tau_suffix_median"),
        pl.col("rank_probe_evaluated").cast(pl.Int64).sum().alias("rank_probe_records"),
        pl.col("effective_rank_median").median().alias("effective_rank_median"),
        pl.col("effective_rank_fraction_median").median().alias("effective_rank_fraction_median"),
        pl.col("compaction_survivor_fraction_n4").mean().alias("compaction_survivor_fraction_n4"),
        pl.col("compaction_survivor_fraction_n2").mean().alias("compaction_survivor_fraction_n2"),
        pl.col("zero_cols_max").max().alias("zero_cols_max"),
        pl.col("tiny_cols_max").max().alias("tiny_cols_max"),
        pl.col("column_span_max").max().alias("column_span_max"),
        pl.col("column_l1_span_median").median().alias("column_l1_span_median"),
        pl.col("row_span_max").max().alias("row_span_max"),
        pl.col("row_l1_span_median").median().alias("row_l1_span_median"),
        pl.col("matrix_l1_mean").mean().alias("matrix_l1_mean"),
    )

    summary = _round_existing_columns(
        summary,
        [
            "upper_ratio_mean",
            "upper_ratio_max",
            "upper_triangular_rate",
            "approx_upper_triangular_rate",
            "early_stop_median_k",
            "early_stop_p90_k",
            "early_stop_n4_rate",
            "early_stop_n2_rate",
            "early_stop_work_saved_mean",
            "estimated_speedup_ceiling",
            "zero_tau_reflector_rate_mean",
            "zero_tau_suffix_median",
            "effective_rank_median",
            "effective_rank_fraction_median",
            "compaction_survivor_fraction_n4",
            "compaction_survivor_fraction_n2",
            "column_span_max",
            "column_l1_span_median",
            "row_span_max",
            "row_l1_span_median",
            "matrix_l1_mean",
        ],
        4,
    )
    return summary.rename(
        {
            "probe_id": "probe",
            "device": "dev",
            "upper_ratio_mean": "upper_mean",
            "upper_ratio_max": "upper_max",
            "upper_triangular_rate": "upper_rate",
            "approx_upper_triangular_rate": "approx_upper",
            "early_stop_median_k": "stop50",
            "early_stop_p90_k": "stop90",
            "early_stop_n4_rate": "stop_n4",
            "early_stop_n2_rate": "stop_n2",
            "early_stop_work_saved_mean": "saved",
            "estimated_speedup_ceiling": "speedup",
            "zero_tau_reflector_rate_mean": "zero_tau",
            "zero_tau_suffix_median": "zero_suffix",
            "effective_rank_median": "rank50",
            "effective_rank_fraction_median": "rank_frac",
            "compaction_survivor_fraction_n4": "surv_n4",
            "compaction_survivor_fraction_n2": "surv_n2",
            "zero_cols_max": "zero_cols",
            "tiny_cols_max": "tiny_cols",
            "column_span_max": "col_span",
            "column_l1_span_median": "col_span50",
            "row_span_max": "row_span",
            "row_l1_span_median": "row_span50",
            "matrix_l1_mean": "norm1_mean",
        }
    )


def render_table(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "(no rows)"

    return frame.write_csv(separator="\t").rstrip()


def render_summary_section(title: str, path: Path, frame: pl.DataFrame) -> str:
    return f"{title}\nsource: {path}\n{render_table(frame)}"


def _rich_columns(title: str, frame: pl.DataFrame) -> list[str]:
    preferred = _RICH_COLUMNS.get(title, frame.columns).copy()
    context_columns = _CONTEXT_COLUMNS.get(title, [])
    for column in reversed(context_columns):
        if column in frame.columns and frame.get_column(column).n_unique() > 1:
            preferred.insert(1, column)
    return [column for column in preferred if column in frame.columns]


def _with_shape_column(frame: pl.DataFrame) -> pl.DataFrame:
    if "batch" not in frame.columns or "n" not in frame.columns:
        return frame

    return frame.with_columns(
        pl.concat_str(
            [
                pl.col("batch").cast(pl.String),
                pl.lit("x"),
                pl.col("n").cast(pl.String),
                pl.lit("x"),
                pl.col("n").cast(pl.String),
            ]
        ).alias("shape")
    )


def _rich_caption(title: str, path: Path, frame: pl.DataFrame) -> str:
    lines = [f"source: {path}"]
    labels = []
    for column in _CONTEXT_COLUMNS.get(title, []):
        if column in frame.columns and frame.get_column(column).n_unique() == 1:
            labels.append(f"{column}={frame.get_column(column).item(0)}")
    if labels:
        lines.append(", ".join(labels))
    return "\n".join(lines)


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return str(value)
    if abs(value) >= 1_000:
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    return f"{value:.4g}"


def _format_elapsed_us(value: float) -> str:
    if abs(value) >= 1_000:
        return f"{value / 1_000:.3g}ms"
    return f"{value:.3g}us"


def _rich_cell(column: str, value: Any) -> str | Text:
    if value is None:
        return Text("-", style="dim")

    if column == "ok":
        return Text("pass" if bool(value) else "fail", style="green" if bool(value) else "bold red")

    if column == "failures":
        failures = int(value)
        return Text(str(failures), style="bold red" if failures else "dim")

    if column in _BUDGET_COLUMNS:
        scaled = float(value)
        if scaled > 1:
            style = "bold red"
        elif scaled > 0.5:
            style = "yellow"
        else:
            style = "green"
        return Text(_format_number(scaled), style=style)

    if column == "p50_us":
        return _format_elapsed_us(float(value))

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return _format_number(value)
    return str(value)


def render_rich_summary_section(title: str, path: Path, frame: pl.DataFrame) -> Table:
    frame = _with_shape_column(frame)
    table = Table(
        title=title,
        caption=_rich_caption(title, path, frame),
        box=box.SIMPLE,
        caption_style="dim",
        collapse_padding=True,
        header_style="bold",
        pad_edge=False,
        show_edge=False,
        title_style="bold",
    )

    if frame.is_empty():
        table.add_column("status")
        table.add_row(Text("no rows", style="dim"))
        return table

    columns = _rich_columns(title, frame)
    for column in columns:
        table.add_column(
            _HEADER_LABELS.get(column, column),
            justify="right" if column in _RIGHT_ALIGNED_COLUMNS else "left",
            no_wrap=column in {"candidate", "dev", "dtype", "probe", "case"},
        )

    for row in frame.select(columns).iter_rows(named=True):
        table.add_row(*[_rich_cell(column, row[column]) for column in columns])

    return table
