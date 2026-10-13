from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from scripts.experiment.complementarity import PLUGINS, plugin_label, plugin_of
from scripts.experiment.lib import layout, tables
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.runs import RunSpec

METRICS: tuple[str, ...] = ("FPR@95", "AUROC")

GROUP_MAP: dict[str, tuple[str, str]] = {
    "IN-1K": ("imagenet_standard", "farood"),
    "near": ("imagenet_openood", "nearood"),
    "far": ("imagenet_openood", "farood"),
}
GROUP_HEADERS: dict[str, str] = {
    "IN-1K": "ImageNet-1K",
    "near": "OpenOOD near-OOD",
    "far": "OpenOOD far-OOD",
}

ROLE_CAMP = "camp"
ROLE_ALONE = "alone"
ROLE_PLUGIN = "+ CAMP"
PLUGIN_ROW_NAME = "+ CAMP"
BLOCK_TITLES: dict[str, str] = {
    layout.ZERO_SHOT: "Zero-shot methods",
    layout.TEST_TIME: "Test-time methods",
}

TABLE_STEM = "complementarity"
LEAVES_NAME = "complementarity_leaves.csv"


def role_of(
    method: str,
    label: str | None,
) -> str:

    if plugin_of(label) is not None:
        return ROLE_PLUGIN

    return ROLE_CAMP if method == "camp" else ROLE_ALONE


def detector_of(
    method: str,
    label: str | None,
) -> str:

    plugin = plugin_of(label)
    return method if plugin is None else plugin


def annotate(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    out = frame.copy()
    pairs = list(zip(out["method"], out["label"]))
    out.insert(0, "detector", [detector_of(method=m, label=label) for m, label in pairs])
    out.insert(1, "role", [role_of(method=m, label=label) for m, label in pairs])
    return out


def groups_present(
    benchmarks: Sequence[str],
) -> list[str]:

    names = [
        group
        for group, (benchmark, _) in GROUP_MAP.items()
        if benchmark in benchmarks
    ]

    return names


def table_columns(
    groups: Sequence[str],
    metrics: Sequence[str] = METRICS,
) -> list[tables.Column]:

    columns: list[tables.Column] = []

    for group in groups:
        benchmark, row = GROUP_MAP[group]

        for metric in metrics:
            columns.append(tables.Column(benchmark, row, metric, header=f"{GROUP_HEADERS[group]} {metric}"))

    return columns


def table_rows() -> list[tuple[tables.Row, str | None]]:

    rows: list[tuple[tables.Row, str | None]] = [(tables.Row("camp"), None)]
    seen_kinds: set[str] = set()

    for method in tables.method_order(PLUGINS):
        kind = layout.METHODS[method].kind
        title = BLOCK_TITLES[kind] if kind not in seen_kinds else None
        seen_kinds.add(kind)
        rows.append((tables.Row(method), title))
        rows.append((tables.Row("camp", label=plugin_label(method), name=PLUGIN_ROW_NAME), None))

    return rows


def insert_block_titles(
    text: str,
    titles: Sequence[str | None],
    width: int,
) -> str:

    lines = text.rstrip("\n").split("\n")
    separator = next((i for i, line in enumerate(lines) if line.startswith("|---")), None)
    if separator is None:
        return text

    head, body = lines[:separator + 1], lines[separator + 1:]
    out = list(head)

    for index, line in enumerate(body):
        title = titles[index] if index < len(titles) else None
        if title is not None:
            out.append("| *" + title + "* |" + " |" * (width - 1))

        out.append(line)

    return "\n".join(out) + "\n"


def markdown(
    frame: pd.DataFrame,
    groups: Sequence[str],
) -> str:

    rows_and_titles = table_rows()
    rows = [row for row, _ in rows_and_titles]
    titles = [title for _, title in rows_and_titles]
    columns = table_columns(groups)
    text = tables.markdown_table(frame, columns=columns, rows=rows)
    return insert_block_titles(text=text, titles=titles, width=len(columns) + 2)


def table_frame(
    frame: pd.DataFrame,
    groups: Sequence[str],
) -> pd.DataFrame:

    mapping = {group: GROUP_MAP[group] for group in groups}
    part = tables.group_rows(frame, mapping)
    part = part[part["metric"].isin(METRICS)]
    columns = ["detector", "role", "group", *[c for c in part.columns if c not in ("detector", "role", "group")]]
    return part[columns].reset_index(drop=True)


def missing_runs(
    specs: Sequence[RunSpec],
) -> list[RunSpec]:

    run_specs = [
        spec
        for spec in specs
        if not spec.recorded()
    ]

    return run_specs


def describe_missing(
    specs: Sequence[RunSpec],
) -> list[str]:

    seeds: dict[tuple[str, str | None, str], list[int]] = {}

    for spec in specs:
        seeds.setdefault((spec.method, spec.label, spec.benchmark), []).append(spec.seed)

    lines: list[str] = []

    for (method, label, benchmark), missing in seeds.items():
        name = method if label is None else f"{method} {label}"
        lines.append(f"{name} on {benchmark}: seeds {','.join(str(s) for s in sorted(missing))}")

    return lines


def analyze(
    ctx: Context,
    specs: Sequence[RunSpec],
) -> list[Path]:

    frame = tables.summarize(specs, metrics=tables.METRICS)
    missing = missing_runs(specs)
    ctx.notes["runs_recorded"] = len(specs) - len(missing)
    ctx.notes["runs_total"] = len(specs)
    ctx.notes["missing_runs"] = [layout.relative(spec.dir()) for spec in missing]

    if frame.empty:
        raise FileNotFoundError(
            f"no run of the complementarity item is recorded under {layout.relative(layout.RESULTS)} "
            f"(benchmarks {', '.join(ctx.benchmarks)}, seeds {', '.join(map(str, ctx.seeds))}); "
            "run `python -m scripts.experiment complementarity plan` for the commands",
        )

    groups = groups_present(ctx.benchmarks)
    annotated = annotate(frame)
    written = [
        tables.write_csv(annotated, ctx.tables / LEAVES_NAME),
        tables.write_csv(table_frame(annotated, groups), ctx.tables / f"{TABLE_STEM}.csv"),
        tables.write_markdown(markdown(frame, groups), ctx.tables / f"{TABLE_STEM}.md"),
    ]
    print(f"{len(specs) - len(missing)}/{len(specs)} runs recorded")

    if missing:
        print(f"missing {len(missing)} runs (cells left empty, seeds listed in the tables and manifest.json):")

        for line in describe_missing(missing):
            print(f"  {line}")

    return written


__all__ = [
    "GROUP_HEADERS",
    "GROUP_MAP",
    "LEAVES_NAME",
    "METRICS",
    "ROLE_ALONE",
    "ROLE_CAMP",
    "ROLE_PLUGIN",
    "TABLE_STEM",
    "analyze",
    "annotate",
    "describe_missing",
    "detector_of",
    "groups_present",
    "insert_block_titles",
    "markdown",
    "missing_runs",
    "role_of",
    "table_columns",
    "table_frame",
    "table_rows",
]
