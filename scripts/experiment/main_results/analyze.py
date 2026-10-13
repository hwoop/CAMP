from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.experiment.lib import layout, tables
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.runs import RunSpec
from scripts.experiment.lib.tables import Column, Row

METRICS: tuple[str, ...] = tables.DEFAULT_METRICS
BLOCKS: tuple[tuple[str, str], ...] = ((layout.ZERO_SHOT, "Zero-shot methods"), (layout.TEST_TIME, "Test-time methods"))
SHORT_BENCHMARK: dict[str, str] = {
    "cifar10_openood": "CIFAR-10",
    "cifar100_openood": "CIFAR-100",
    "imagenet_standard": "ImageNet-1K",
    "imagenet_openood": "OpenOOD ImageNet-1K",
}
CSV_KEYS: tuple[str, ...] = (
    "method",
    "method_name",
    "kind",
    "label",
    "benchmark",
    "row",
    "row_name",
    "column",
    "metric",
)


@dataclass(frozen=True)
class Section:

    columns: tuple[Column, ...]
    heading: str | None = None


@dataclass(frozen=True)
class TableSpec:

    stem: str
    benchmarks: tuple[str, ...]
    sections: tuple[Section, ...]

    @property
    def columns(self) -> list[Column]:

        columns = [
            column
            for section in self.sections
            for column in section.columns
        ]

        return columns


def leaf_columns(
    benchmark: str,
    group: str,
    average_header: str,
    metrics: Sequence[str] = METRICS,
) -> tuple[Column, ...]:

    columns = [Column(benchmark, leaf, metric) for leaf in layout.LEAVES[benchmark][group] for metric in metrics]
    columns += [Column(benchmark, group, metric, header=f"{average_header} {metric}") for metric in metrics]
    return tuple(columns)


def group_columns(
    benchmark: str,
    metrics: Sequence[str] = METRICS,
) -> tuple[Column, ...]:

    return tuple(Column(
            benchmark,
            group,
            metric,
            header=f"{SHORT_BENCHMARK[benchmark]} {layout.GROUPS[group]} {metric}",
        )
        for group in layout.LEAVES[benchmark] for metric in metrics)


def table_specs() -> list[TableSpec]:

    per_dataset_sections = {
        benchmark: tuple(Section(leaf_columns(benchmark=benchmark, group=group, average_header="Average"), heading=layout.GROUPS[group])
            for group in layout.LEAVES[benchmark])
        for benchmark in ("cifar10_openood", "cifar100_openood")
    }

    return [
        TableSpec(
            "imagenet_standard",
            ("imagenet_standard",),
            (Section(leaf_columns(benchmark="imagenet_standard", group="farood", average_header="Average")),),
        ),
        TableSpec(
            "imagenet_openood",
            ("imagenet_openood",),
            (Section(
                    leaf_columns(benchmark="imagenet_openood", group="nearood", average_header="Near-OOD Average")
                    + leaf_columns(benchmark="imagenet_openood", group="farood", average_header="Far-OOD Average"),
                ),),
        ),
        TableSpec(
            "cifar",
            ("cifar10_openood", "cifar100_openood"),
            (Section(group_columns("cifar10_openood") + group_columns("cifar100_openood")),),
        ),
        TableSpec(
            "cifar10_openood_per_dataset",
            ("cifar10_openood",),
            per_dataset_sections["cifar10_openood"],
        ),
        TableSpec(
            "cifar100_openood_per_dataset",
            ("cifar100_openood",),
            per_dataset_sections["cifar100_openood"],
        ),
    ]


def method_rows(
    methods: Sequence[str],
) -> list[Row]:

    rows = [
        Row(method)
        for method in tables.method_order(methods)
    ]

    return rows


def blocked_markdown(
    frame: pd.DataFrame,
    columns: Sequence[Column],
    methods: Sequence[str],
    title: str | None = None,
) -> str:

    rows = method_rows(methods)
    text = tables.markdown_table(frame, columns=columns, rows=rows, title=None)
    lines = text.rstrip("\n").split("\n")
    header, body = lines[:2], lines[2:]
    width = header[0].count("|") - 1
    out: list[str] = []

    for kind, heading in BLOCKS:
        block = [line for row, line in zip(rows, body) if layout.METHODS[row.method].kind == kind]
        if block:
            out.append(f"| *{heading}* |" + " |" * (width - 1))
            out += block

    result = "\n".join([*header, *out]) + "\n"
    return (f"### {title}\n\n" if title else "") + result


def markdown_document(
    spec: TableSpec,
    frame: pd.DataFrame,
    methods: Sequence[str],
) -> str:

    parts: list[str] = []

    for section in spec.sections:
        parts.append(blocked_markdown(frame=frame, columns=section.columns, methods=methods, title=section.heading))

    return "\n".join(parts)


def table_csv(
    frame: pd.DataFrame,
    columns: Sequence[Column],
    methods: Sequence[str],
    seeds: Sequence[int],
) -> pd.DataFrame:

    seed_columns = [f"seed_{seed}" for seed in seeds]
    records: list[dict[str, object]] = []

    for method in tables.method_order(methods):
        row = Row(method)

        for column in columns:
            hit = tables.lookup(frame, row, column)
            record: dict[str, object] = {
                "method": method,
                "method_name": tables.method_name(method),
                "kind": layout.METHODS[method].kind,
                "label": None,
                "benchmark": column.benchmark,
                "row": column.row,
                "row_name": tables.row_name(column.benchmark, column.row),
                "column": column.title,
                "metric": column.metric,
            }

            if hit is None:
                record.update(
                    {
                        "mean": math.nan,
                        "std": math.nan,
                        "n_seeds": 0,
                        "seeds": "",
                        "missing": ",".join(map(str, seeds)),
                    },
                )

                record.update({name: math.nan for name in seed_columns})
            else:
                record.update(
                    {
                        "mean": float(hit["mean"]),
                        "std": float(hit["std"]),
                        "n_seeds": int(hit["n_seeds"]),
                        "seeds": str(hit["seeds"]),
                        "missing": str(hit["missing"]),
                    },
                )
                record.update({name: float(hit[name]) if name in hit.index else math.nan for name in seed_columns})

            records.append(record)

    return pd.DataFrame.from_records(records, columns=[*CSV_KEYS, *tables.VALUE_COLUMNS, *seed_columns])


def missing_summary(
    specs: Sequence[RunSpec],
) -> list[str]:

    missing: dict[tuple[str, str], list[int]] = {}

    for spec in specs:
        if not spec.recorded():
            missing.setdefault((spec.method, spec.benchmark), []).append(spec.seed)

    names = [
        f"{method} {benchmark}: missing seeds {','.join(map(str, sorted(seeds)))}"
        for (method, benchmark), seeds in sorted(missing.items())
    ]

    return names


def analyze(
    ctx: Context,
) -> list[Path]:

    from scripts.experiment.main_results import methods_of, runs

    specs = runs(ctx)
    methods = methods_of(ctx)
    frame = tables.summarize(specs, metrics=METRICS)
    recorded = [spec for spec in specs if spec.recorded()]

    if frame.empty:
        raise FileNotFoundError(
            f"no run of {ctx.item.name} is recorded under {layout.relative(layout.RESULTS)} "
            f"({len(specs)} runs wanted; see `python -m scripts.experiment {ctx.item.name} plan`)",
        )

    written: list[Path] = [tables.write_csv(frame, ctx.tables / "summary_long.csv")]

    for spec in table_specs():
        in_scope = [b for b in spec.benchmarks if b in ctx.benchmarks]
        if not in_scope:
            continue

        sections = tuple(Section(tuple(c for c in section.columns if c.benchmark in in_scope), section.heading)
            for section in spec.sections)

        sections = tuple(section for section in sections if section.columns)
        scoped = TableSpec(spec.stem, tuple(in_scope), sections)
        written.append(
            tables.write_markdown(
                markdown_document(spec=scoped, frame=frame, methods=methods),
                ctx.tables / f"{spec.stem}.md",
            ),
        )

        written.append(
            tables.write_csv(
                table_csv(frame=frame, columns=scoped.columns, methods=methods, seeds=ctx.seeds),
                ctx.tables / f"{spec.stem}.csv",
            ),
        )

    missing = missing_summary(specs)
    ctx.notes.update(
        {
            "std": "sample (ddof=1) over the recorded seeds",
            "runs_recorded": len(recorded), "runs_wanted": len(specs),
            "missing_runs": [layout.relative(spec.dir()) for spec in specs if not spec.recorded()],
        },
    )

    print(
        f"[{ctx.item.name}] {len(recorded)}/{len(specs)} runs recorded"
        + ("" if not missing else f"; {len(specs) - len(recorded)} missing:"),
    )

    for line in missing:
        print(f"  {line}")

    return written
