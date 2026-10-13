from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from scripts.experiment.lib import layout, tables
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.runs import RunSpec

if TYPE_CHECKING:
    from scripts.experiment.ablation import Arm

GROUP_COLUMNS: dict[str, tuple[str, str]] = {
    "IN-1K": ("imagenet_standard", "farood"),
    "near": ("imagenet_openood", "nearood"),
    "far": ("imagenet_openood", "farood"),
}
GROUP_HEADERS: dict[str, str] = {
    "IN-1K": "ImageNet-1K",
    "near": "OpenOOD near-OOD",
    "far": "OpenOOD far-OOD",
}
METRICS: tuple[str, ...] = tables.DEFAULT_METRICS
TABLE_STEM = "ablation"
LEAVES_STEM = "ablation_leaves"


def groups_for(
    benchmarks: Sequence[str],
) -> dict[str, tuple[str, str]]:

    table = {
        group: pair
        for group, pair in GROUP_COLUMNS.items()
        if pair[0] in benchmarks
    }

    return table


def table_columns(
    benchmarks: Sequence[str],
    metrics: Sequence[str] = METRICS,
    backbone: str = "vitb16",
) -> list[tables.Column]:

    rows = [
        tables.Column(benchmark, row, metric, header=f"{GROUP_HEADERS[group]} {metric}", backbone=backbone)
        for group, (benchmark, row) in groups_for(benchmarks).items()
        for metric in metrics
    ]

    return rows


def table_rows(
    arms: Sequence[Arm],
    backbone: str = "vitb16",
) -> list[tables.Row]:

    rows = [
        tables.Row(arm.method, arm.label, name=arm.name, backbone=backbone)
        for arm in arms
    ]

    return rows


def variant_names(
    arms: Sequence[Arm],
) -> dict[tuple[str, str | None], str]:

    table = {
        (arm.method, arm.label): arm.name
        for arm in arms
    }

    return table


def with_variant(
    frame: pd.DataFrame,
    arms: Sequence[Arm],
) -> pd.DataFrame:

    names = variant_names(arms)
    order = {key: index for index, key in enumerate(names)}
    part = frame.copy()
    keys = [(method, None if pd.isna(label) else label) for method, label in zip(part["method"], part["label"])]
    part.insert(0, "variant", [names.get(key, tables.label_name(*key)) for key in keys])
    part["_arm"] = [order.get(key, len(order)) for key in keys]
    return part


def group_table(
    frame: pd.DataFrame,
    arms: Sequence[Arm],
    benchmarks: Sequence[str],
) -> pd.DataFrame:

    groups = groups_for(benchmarks)
    part = with_variant(frame=tables.group_rows(frame, groups), arms=arms)
    group_order = {group: index for index, group in enumerate(groups)}
    metric_order = {metric: index for index, metric in enumerate(METRICS)}
    part["_group"] = [group_order[g] for g in part["group"]]
    part["_metric"] = [metric_order.get(m, len(metric_order)) for m in part["metric"]]
    part = part.sort_values(["_arm", "_group", "_metric"], kind="stable")
    part = part.drop(columns=["_arm", "_group", "_metric"])
    front = ["variant", "group"]
    return part[front + [c for c in part.columns if c not in front]].reset_index(drop=True)


def leaf_table(
    frame: pd.DataFrame,
    arms: Sequence[Arm],
) -> pd.DataFrame:

    part = with_variant(frame=frame[frame["kind"] == "leaf"], arms=arms)
    part = part.sort_values(["_arm"], kind="stable").drop(columns=["_arm"])
    part.insert(1, "leaf", [tables.leaf_name(row) for row in part["row"]])
    return part.reset_index(drop=True)


def missing_report(
    specs: Sequence[RunSpec],
) -> tuple[list[str], list[str]]:

    recorded = [layout.relative(spec.dir()) for spec in specs if spec.recorded()]
    missing = [layout.relative(spec.dir()) for spec in specs if not spec.recorded()]
    return recorded, missing


def write_tables(
    ctx: Context,
    specs: Sequence[RunSpec],
    arms: Sequence[Arm],
) -> list[Path]:

    recorded, missing = missing_report(specs)
    ctx.notes["runs_recorded"] = len(recorded)
    ctx.notes["runs_missing"] = missing

    if not recorded:
        raise RuntimeError(
            f"ablation: none of the {len(specs)} runs is recorded; run `python -m scripts.experiment "
            f"ablation plan` for the commands (first missing: {missing[0]})",
        )

    print(f"[ablation] {len(recorded)}/{len(specs)} runs recorded" + (f", {len(missing)} missing:" if missing else ""))

    for path in missing:
        print(f"[missing] {path}")

    frame = tables.summarize(specs, metrics=METRICS)
    columns = table_columns(ctx.benchmarks, backbone=ctx.backbone)
    rows = table_rows(arms, backbone=ctx.backbone)
    markdown = tables.markdown_table(frame, columns=columns, rows=rows, row_header="Variant")

    if missing:
        markdown += (f"\n{len(missing)} of {len(specs)} runs are not recorded; the `seeds` column lists the seeds each "
            "row averages and the missing ones.\n")

    written = [
        tables.write_csv(group_table(frame=frame, arms=arms, benchmarks=ctx.benchmarks), ctx.tables / f"{TABLE_STEM}.csv"),
        tables.write_markdown(markdown, ctx.tables / f"{TABLE_STEM}.md"),
        tables.write_csv(leaf_table(frame=frame, arms=arms), ctx.tables / f"{LEAVES_STEM}.csv"),
    ]
    return written
