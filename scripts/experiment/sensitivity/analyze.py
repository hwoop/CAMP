from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from scripts.experiment.lib import layout, tables
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.runs import RunSpec
from scripts.experiment.sensitivity import (MAIN_LABEL, TABLE_LONG, TABLE_RAW, Arm, Axis, Spec, arms_for, expand,
    selected_axes, spec_for)

METRICS: tuple[str, ...] = tables.DEFAULT_METRICS
KEY_COLUMNS: tuple[str, ...] = ("axis", "value", "label", "group", "benchmark", "row", "metric")
VALUE_COLUMNS: tuple[str, ...] = ("mean", "std", "n_seeds", "seeds", "missing")


def value_name(
    axis: Axis,
    value: float,
    default: bool,
) -> str:
    return f"{axis.symbol} = {axis.format(value)}" + (" (default)" if default else "")


def _lookup(
    frame: pd.DataFrame,
    arm: Arm,
    benchmark: str,
    row: str,
    metric: str,
) -> pd.Series | None:

    labels = frame["label"].isna() if arm.label is None else frame["label"] == arm.label
    hit = frame[(frame["method"] == arm.method) & labels & (frame["benchmark"] == benchmark) & (frame["row"] == row)
        & (frame["metric"] == metric)]
    return None if hit.empty else hit.iloc[0]


def long_table(
    frame: pd.DataFrame,
    spec: Spec,
    axes: Sequence[str],
    seeds: Sequence[int],
    metrics: Sequence[str] = METRICS,
) -> pd.DataFrame:

    seed_columns = [f"seed_{seed}" for seed in sorted(seeds)]
    records: list[dict[str, object]] = []

    for axis_name in axes:
        axis = spec.axes[axis_name]

        for value in spec.values_of(axis_name):
            arm = spec.arm_at(axis_name, value)

            for group, (benchmark, row) in spec.groups.items():
                for metric in metrics:
                    record: dict[str, object] = {
                        "axis": axis_name,
                        "value": value,
                        "label": MAIN_LABEL if arm.label is None else arm.label,
                        "group": group,
                        "benchmark": benchmark,
                        "row": row,
                        "metric": metric,
                    }

                    hit = _lookup(
                        frame=frame,
                        arm=arm,
                        benchmark=benchmark,
                        row=row,
                        metric=metric
                    )
                    if hit is None:
                        record.update(
                            {
                                "mean": math.nan,
                                "std": math.nan,
                                "n_seeds": 0,
                                "seeds": "",
                                "missing": ",".join(str(s) for s in sorted(seeds)),
                            },
                        )

                        record.update({column: math.nan for column in seed_columns})
                    else:
                        record.update({
                            "mean": float(hit["mean"]),
                            "std": float(hit["std"]),
                            "n_seeds": int(hit["n_seeds"]),
                            "seeds": str(hit["seeds"]),
                            "missing": str(hit["missing"]),
                        })

                        record.update({
                            column: float(hit[column]) if column in hit.index else math.nan
                            for column in seed_columns
                        })

                    records.append(record)

    return pd.DataFrame.from_records(records, columns=[*KEY_COLUMNS, *VALUE_COLUMNS, *seed_columns])


def markdown_for_axis(
    frame: pd.DataFrame,
    spec: Spec,
    axis_name: str,
    seeds: Sequence[int],
    metrics: Sequence[str] = METRICS,
) -> str:

    axis = spec.axes[axis_name]
    rows = [
        tables.Row(
            spec.method,
            spec.arm_at(axis_name, value).label,
            name=value_name(
                axis=axis,
                value=value,
                default=math.isclose(a=value, b=axis.default, rel_tol=1e-9)
            ),
        ) for value in spec.values_of(axis_name)
    ]

    group_titles = {
        "IN-1K": "ImageNet-1K",
        "near": "Near-OOD",
        "far": "Far-OOD",
    }
    columns = [
        tables.Column(benchmark, row, metric, header=f"{group_titles.get(group, group)} {metric}")
        for group, (benchmark, row) in spec.groups.items() for metric in metrics
    ]
    title = (
        f"{axis.title} {axis.symbol} (default {axis.format(axis.default)}); CAMP, ViT-B/16, "
        f"mean (sample std) over seeds {', '.join(str(s) for s in seeds)}"
    )
    batch_size = spec.main.overrides.get("dataset.test.batch_size")
    if batch_size is not None:
        title += f"; batch size {batch_size}"

    return tables.markdown_table(frame, columns=columns, rows=rows, title=title, row_header=axis.title)


def missing_summary(
    specs: Sequence[RunSpec],
) -> dict[str, list[str]]:

    summary: dict[str, list[str]] = {}

    for spec in specs:
        if not spec.recorded():
            summary.setdefault(spec.label or "main", []).append(f"{spec.benchmark}-s{spec.seed}")

    return summary


def analyze(
    ctx: Context,
) -> list[Path]:

    spec = spec_for(ctx)
    axes = selected_axes(ctx=ctx, spec=spec)
    specs = expand(arms_for(ctx=ctx, spec=spec), ctx.benchmarks, ctx.seeds, ctx.backbone)
    frame = tables.summarize(specs, metrics=tables.METRICS)
    recorded = sum(spec_.recorded() for spec_ in specs)
    missing = missing_summary(specs)

    ctx.notes.update({
        "runs_recorded": recorded,
        "runs_total": len(specs),
        "axes": list(axes),
        "missing_runs": [layout.relative(s.dir()) for s in specs if not s.recorded()],
        "missing_by_arm": {label: len(runs_) for label, runs_ in missing.items()},
    })

    if frame.empty:
        raise RuntimeError(
            f"sensitivity: none of the {len(specs)} runs is recorded (nothing to analyse); "
            "`python -m scripts.experiment sensitivity plan` lists them",
        )

    print(f"[sensitivity] {recorded}/{len(specs)} runs recorded" + ("" if missing else "; nothing missing"))
    per_arm = len(ctx.benchmarks) * len(ctx.seeds)

    for label, runs_ in missing.items():
        detail = "all" if len(runs_) == per_arm else ", ".join(runs_)
        print(f"[sensitivity] missing {label}: {len(runs_)}/{per_arm} runs ({detail})")

    long = long_table(frame=frame, spec=spec, axes=axes, seeds=ctx.seeds)
    written = [tables.write_csv(long, ctx.tables / TABLE_LONG), tables.write_csv(frame, ctx.tables / TABLE_RAW)]

    for axis_name in axes:
        text = markdown_for_axis(frame=frame, spec=spec, axis_name=axis_name, seeds=ctx.seeds)
        written.append(tables.write_markdown(text, ctx.tables / f"{spec.axes[axis_name].table}.md"))

    return written
