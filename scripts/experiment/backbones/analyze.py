from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from scripts.experiment.backbones import backbones_of, methods_of, runs
from scripts.experiment.lib import layout, tables
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.runs import RunSpec

GROUP_OF: dict[str, tuple[str, str]] = {
    "IN-1K": ("imagenet_standard", "farood"),
    "near": ("imagenet_openood", "nearood"),
    "far": ("imagenet_openood", "farood"),
}
GROUP_HEADERS: dict[str, str] = {
    "IN-1K": "IN-1K",
    "near": "Near",
    "far": "Far",
}
TABLE_FILES: dict[str, str] = {
    "FPR@95": "backbones",
    "AUROC": "backbones_auroc",
}
LONG_FILE = "backbones_long.csv"
MISSING_FILE = "missing_runs.csv"
CSV_COLUMNS: tuple[str, ...] = (
    "method",
    "method_name",
    "backbone",
    "backbone_name",
    "group",
    "benchmark",
    "row",
    "metric",
    "mean",
    "std",
    "n_seeds",
    "seeds",
    "missing",
)


def groups_for(
    benchmarks: Sequence[str],
) -> list[str]:

    names = [
        group
        for group, (benchmark, _) in GROUP_OF.items()
        if benchmark in benchmarks
    ]

    return names


def columns_for(
    backbones: Sequence[str],
    groups: Sequence[str],
    metric: str,
) -> list[tables.Column]:

    rows = [
        tables.Column(
            GROUP_OF[group][0],
            GROUP_OF[group][1],
            metric,
            backbone=backbone,
            header=f"{layout.BACKBONE_NAMES[backbone]} {GROUP_HEADERS[group]}",
        )
        for backbone in backbones
        for group in groups
    ]

    return rows


def table_frame(
    summary: pd.DataFrame,
    methods: Sequence[str],
    backbones: Sequence[str],
    groups: Sequence[str],
    metric: str,
) -> pd.DataFrame:

    seed_columns = [column for column in summary.columns if column.startswith("seed_")]
    records: list[dict[str, object]] = []

    for method in methods:
        for backbone in backbones:
            for group in groups:
                benchmark, row = GROUP_OF[group]
                hit = tables.lookup(
                    summary,
                    tables.Row(method, backbone=backbone),
                    tables.Column(benchmark, row, metric, backbone=backbone),
                )

                record: dict[str, object] = {
                    "method": method,
                    "method_name": tables.method_name(method),
                    "backbone": backbone,
                    "backbone_name": layout.BACKBONE_NAMES[backbone],
                    "group": group,
                    "benchmark": benchmark,
                    "row": row,
                    "metric": metric,
                    "mean": float("nan") if hit is None else float(hit["mean"]),
                    "std": float("nan") if hit is None else float(hit["std"]),
                    "n_seeds": 0 if hit is None else int(hit["n_seeds"]),
                    "seeds": "" if hit is None else str(hit["seeds"]),
                    "missing": "" if hit is None else str(hit["missing"]),
                }

                for column in seed_columns:
                    record[column] = float("nan") if hit is None else float(hit[column])

                records.append(record)

    return pd.DataFrame.from_records(records, columns=[*CSV_COLUMNS, *seed_columns])


def long_frame(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    frame = summary.copy()
    reverse = {value: key for key, value in GROUP_OF.items()}
    frame.insert(1, "method_name", [tables.method_name(m) for m in frame["method"]])
    frame.insert(
        frame.columns.get_loc("backbone") + 1,
        "backbone_name",
        [layout.BACKBONE_NAMES.get(b, b) for b in frame["backbone"]],
    )

    frame.insert(
        frame.columns.get_loc("row") + 1,
        "group",
        [reverse.get((b, r), "") for b, r in zip(frame["benchmark"], frame["row"])],
    )

    return frame


def missing_frame(
    specs: Sequence[RunSpec],
) -> pd.DataFrame:

    records = [{
            "method": spec.method,
            "benchmark": spec.benchmark,
            "seed": spec.seed,
            "backbone": spec.backbone,
            "dir": layout.relative(spec.dir()),
        } for spec in specs if not spec.recorded()]
    return pd.DataFrame.from_records(records, columns=["method", "benchmark", "seed", "backbone", "dir"])


def report_missing(
    specs: Sequence[RunSpec],
    missing: pd.DataFrame,
) -> list[str]:

    lines = [f"backbones: {len(specs) - len(missing)}/{len(specs)} runs recorded, {len(missing)} missing"]

    if missing.empty:
        return lines

    for backbone in dict.fromkeys(spec.backbone for spec in specs):
        part = missing[missing["backbone"] == backbone]
        total = sum(spec.backbone == backbone for spec in specs)

        if part.empty:
            continue

        methods = ",".join(dict.fromkeys(part["method"]))
        lines.append(f"  {layout.BACKBONE_NAMES[backbone]}: {len(part)}/{total} missing ({methods})")

    lines.append(f"  the missing runs are listed in tables/{MISSING_FILE}; `plan` prints their commands")
    return lines


def analyze(
    ctx: Context,
) -> list[Path]:

    specs = runs(ctx)
    if not specs:
        raise ValueError("the backbones plan is empty: check --benchmarks, --seeds, --opt backbones= / methods=")

    methods = list(methods_of(ctx))
    backbones = list(backbones_of(ctx))
    groups = groups_for(ctx.benchmarks)
    summary = tables.summarize(specs, metrics=tables.DEFAULT_METRICS)
    missing = missing_frame(specs)
    for line in report_missing(specs=specs, missing=missing):
        print(line)

    ctx.notes["recorded_runs"] = len(specs) - len(missing)
    ctx.notes["missing_runs"] = list(missing["dir"])
    ctx.notes["backbones"] = backbones
    ctx.notes["methods"] = methods
    ctx.notes["std"] = "sample (ddof=1) over the recorded seeds"

    written: list[Path] = []
    rows = [tables.Row(method) for method in methods]

    for metric, stem in TABLE_FILES.items():
        columns = columns_for(backbones=backbones, groups=groups, metric=metric)
        text = tables.markdown_table(summary, columns=columns, rows=rows)
        written.append(tables.write_markdown(text, ctx.tables / f"{stem}.md"))
        written.append(
            tables.write_csv(
                table_frame(summary, methods, backbones, groups, metric),
                ctx.tables / f"{stem}.csv",
            ),
        )

    written.append(tables.write_csv(long_frame(summary), ctx.tables / LONG_FILE))
    written.append(tables.write_csv(missing, ctx.tables / MISSING_FILE))
    return written
