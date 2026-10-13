"""
Build the item tables from recorded ``ood.csv`` files, 
with per-seed values, mean and sample standard deviation, 
and Markdown and CSV writers.
"""
from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.experiment.lib import layout
from scripts.experiment.lib.runs import RunSpec

METRICS: tuple[str, ...] = ("FPR@95", "AUROC", "AUPR_IN", "AUPR_OUT", "ACC")
DEFAULT_METRICS: tuple[str, ...] = ("FPR@95", "AUROC")
LOWER_IS_BETTER: frozenset[str] = frozenset({"FPR@95"})
GROUPS: dict[str, str] = layout.GROUPS
ARROWS: dict[str, str] = {metric: ("↓" if metric in LOWER_IS_BETTER else "↑") for metric in METRICS}
KEY_COLUMNS: tuple[str, ...] = ("method", "label", "benchmark", "backbone", "row", "kind", "metric")
VALUE_COLUMNS: tuple[str, ...] = ("mean", "std", "n_seeds", "seeds", "missing")
MISSING_CELL = "–"


def read_ood_csv(
    path: Path,
) -> dict[str, dict[str, float]]:

    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    table: dict[str, dict[str, float]] = {}

    for row in rows:
        table[row["dataset"]] = {metric: float(row[metric]) for metric in METRICS if metric in row}

    return table


def leaf_name(
    leaf: str,
) -> str:

    if leaf in GROUPS:
        return GROUPS[leaf]

    return layout.LEAF_NAMES.get(leaf, leaf)


def method_name(
    method: str,
) -> str:
    return layout.METHODS[method].name if method in layout.METHODS else method


def row_name(
    benchmark: str,
    row: str,
) -> str:

    if benchmark == "imagenet_standard" and row == "farood":
        return "Average"

    return leaf_name(row)


def method_order(
    methods: Iterable[str],
) -> list[str]:

    seen = list(dict.fromkeys(methods))
    known = [key for key in layout.METHODS if key in seen]
    return known + [key for key in seen if key not in layout.METHODS]


def label_name(
    method: str,
    label: str | None,
) -> str:
    return method_name(method) if label is None else f"{method_name(method)} {label}"


def sample_std(
    values: Sequence[float],
) -> float:

    if len(values) < 2:
        return math.nan

    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def summarize(
    runs: Iterable[RunSpec],
    metrics: Sequence[str] = DEFAULT_METRICS,
    results: Path | None = None,
) -> pd.DataFrame:

    groups: dict[tuple[str, str | None, str, str], list[RunSpec]] = {}

    for spec in runs:
        groups.setdefault((spec.method, spec.label, spec.benchmark, spec.backbone), []).append(spec)

    all_seeds = sorted({spec.seed for specs in groups.values() for spec in specs})
    records: list[dict[str, object]] = []

    for (method, label, benchmark, backbone), specs in groups.items():
        tables = {spec.seed: read_ood_csv(spec.ood_csv(results)) for spec in specs if spec.recorded(results)}
        if not tables:
            continue

        recorded_seeds = sorted(tables)
        missing_seeds = sorted({spec.seed for spec in specs} - set(tables))
        rows = list(dict.fromkeys(name for table in tables.values() for name in table))
        for row in rows:
            for metric in metrics:
                values = {seed: tables[seed][row][metric] for seed in recorded_seeds
                    if row in tables[seed] and metric in tables[seed][row]}
                series = list(values.values())
                record: dict[str, object] = {
                    "method": method,
                    "label": label,
                    "benchmark": benchmark,
                    "backbone": backbone,
                    "row": row,
                    "kind": "group" if row in GROUPS else "leaf",
                    "metric": metric,
                    "mean": sum(series) / len(series) if series else math.nan,
                    "std": sample_std(series),
                    "n_seeds": len(series),
                    "seeds": ",".join(str(s) for s in values),
                    "missing": ",".join(str(s) for s in missing_seeds),
                }

                for seed in all_seeds:
                    record[f"seed_{seed}"] = values.get(seed, math.nan)

                records.append(record)

    columns = [*KEY_COLUMNS, *VALUE_COLUMNS, *(f"seed_{seed}" for seed in all_seeds)]
    frame = pd.DataFrame.from_records(records, columns=columns)
    frame["label"] = frame["label"].astype(object).where(frame["label"].notna(), None)
    return frame


def wide(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    keys = ["method", "label", "benchmark", "backbone", "row", "kind", "seeds", "missing"]

    if frame.empty:
        return pd.DataFrame(columns=keys)

    metrics = list(dict.fromkeys(frame["metric"]))
    base = frame.drop_duplicates(keys)[keys].reset_index(drop=True)

    for metric in metrics:
        part = frame[frame["metric"] == metric].set_index(keys[:5])
        base[f"{metric} mean"] = [part["mean"].get(tuple(k), math.nan) for k in base[keys[:5]].itertuples(index=False)]
        base[f"{metric} std"] = [part["std"].get(tuple(k), math.nan) for k in base[keys[:5]].itertuples(index=False)]

    return base


@dataclass(frozen=True)
class Column:

    benchmark: str
    row: str
    metric: str
    header: str | None = None
    backbone: str = "vitb16"

    @property
    def title(self) -> str:

        if self.header is not None:
            return self.header
        else:
            return f"{row_name(benchmark=self.benchmark, row=self.row)} {self.metric}"


@dataclass(frozen=True)
class Row:

    method: str
    label: str | None = None
    name: str | None = None
    backbone: str = "vitb16"

    @property
    def title(self) -> str:
        return self.name if self.name is not None else label_name(method=self.method, label=self.label)


def default_columns(
    frame: pd.DataFrame,
    benchmark: str | None = None,
    metrics: Sequence[str] | None = None,
    groups_only: bool = False,
    backbone: str = "vitb16",
) -> list[Column]:

    part = frame if benchmark is None else frame[frame["benchmark"] == benchmark]
    part = part[part["backbone"] == backbone]

    if groups_only:
        part = part[part["kind"] == "group"]

    wanted = list(metrics) if metrics is not None else list(dict.fromkeys(part["metric"]))
    columns: list[Column] = []

    for bench, row in dict.fromkeys(zip(part["benchmark"], part["row"])):
        for metric in wanted:
            columns.append(Column(bench, row, metric, backbone=backbone))

    return columns


def default_rows(
    frame: pd.DataFrame,
    backbone: str = "vitb16",
) -> list[Row]:

    pairs = list(dict.fromkeys(zip(frame["method"], frame["label"])))
    ordered = [(method, label) for method in method_order(m for m, _ in pairs)
        for (m, label) in pairs if m == method]
    rows = [
        Row(method, label, backbone=backbone)
        for method, label in ordered
    ]

    return rows


def best_marks(
    values: Sequence[float],
    lower: bool,
) -> dict[int, str]:

    finite = sorted({v for v in values if not math.isnan(v)}, reverse=not lower)
    marks: dict[int, str] = {}

    for index, value in enumerate(values):
        if math.isnan(value):
            continue

        if value == finite[0]:
            marks[index] = "best"
        elif len(finite) > 1 and value == finite[1]:
            marks[index] = "second"

    return marks


def cell(
    mean: float,
    std: float,
    decimals: int = 2,
) -> str:

    if math.isnan(mean):
        return MISSING_CELL

    if math.isnan(std):
        return f"{mean:.{decimals}f}"

    return f"{mean:.{decimals}f} ({std:.{decimals}f})"


def decorate(
    text: str,
    mark: str | None,
) -> str:

    if mark == "best":
        return f"**{text}**"

    if mark == "second":
        return f"<u>{text}</u>"

    return text


def lookup(
    frame: pd.DataFrame,
    row: Row,
    column: Column,
) -> pd.Series | None:

    labels = frame["label"].isna() if row.label is None else frame["label"] == row.label
    hit = frame[(frame["method"] == row.method) & labels & (frame["benchmark"] == column.benchmark)
        & (frame["backbone"] == column.backbone) & (frame["row"] == column.row)
        & (frame["metric"] == column.metric)]
    return None if hit.empty else hit.iloc[0]


def markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[Column] | None = None,
    rows: Sequence[Row] | None = None,
    title: str | None = None,
    seeds_column: bool = True,
    decimals: int = 2,
    mark_best: bool = True,
    row_header: str = "Method",
) -> str:

    columns = list(columns) if columns is not None else default_columns(frame)
    rows = list(rows) if rows is not None else default_rows(frame)
    if not rows or not columns:
        return (f"### {title}\n\n" if title else "") + "no run recorded\n"

    records = [[lookup(frame=frame, row=row, column=column) for column in columns] for row in rows]
    marks: list[dict[int, str]] = []

    for j, column in enumerate(columns):
        values = [float(records[i][j]["mean"]) if records[i][j] is not None else math.nan for i in range(len(rows))]
        marks.append(best_marks(values=values, lower=column.metric in LOWER_IS_BETTER) if mark_best else {})

    headers = [row_header] + (["seeds"] if seeds_column else [])
    headers += [f"{c.title}{ARROWS.get(c.metric, '')}" for c in columns]
    lines: list[str] = []

    if title:
        lines += [f"### {title}", ""]

    lines += ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]

    for i, row in enumerate(rows):
        cells = [row.title]

        if seeds_column:
            present = [r for r in records[i] if r is not None]
            if present:
                seeds = max((str(r["seeds"]) for r in present), key=len)
                absent = max((str(r["missing"]) for r in present), key=len)
                cells.append(seeds + (f" (missing {absent})" if absent else ""))
            else:
                cells.append(MISSING_CELL)

        for j, record in enumerate(records[i]):
            if record is None:
                cells.append(MISSING_CELL)
                continue

            cells.append(decorate(text=cell(mean=float(record["mean"]), std=float(record["std"]), decimals=decimals), mark=marks[j].get(i)))

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def write_csv(
    frame: pd.DataFrame,
    path: Path,
    float_format: str = "%.4f",
) -> Path:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format=float_format, encoding="utf-8")
    return path


def write_markdown(
    text: str,
    path: Path,
) -> Path:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def group_rows(
    frame: pd.DataFrame,
    mapping: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:

    reverse = {value: key for key, value in mapping.items()}
    keys = list(zip(frame["benchmark"], frame["row"]))
    part = frame[[key in reverse for key in keys]].copy()
    part["group"] = [reverse[(b, r)] for b, r in zip(part["benchmark"], part["row"])]
    return part
