from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experiment.lib import config_io, layout, tables
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.runs import RunSpec, dataset_configs

TABLE_STEM = "stream_composition"
CROSS_CHECK_STEM = "cross_check"
SPEC_PATH: Path = Path(__file__).resolve().parent / "spec.yml"
NO_GROUP = "–"
GROUP_SHORT: dict[str, str] = {
    "nearood": "Near",
    "farood": "Far",
}
MATCH = "match"
MISMATCH = "mismatch"
NO_SCORES = "no scores"
TABLE_COLUMNS: tuple[str, ...] = (
    "benchmark",
    "benchmark_name",
    "group",
    "group_name",
    "leaf",
    "ood_dataset",
    "id_samples",
    "ood_samples",
    "stream_samples",
    "ood_ratio",
    "id_imglist",
    "ood_imglist",
)
CROSS_CHECK_COLUMNS: tuple[str, ...] = (
    "benchmark",
    "leaf",
    "ood_dataset",
    "seed",
    "run_dir",
    "status",
    "expected_id",
    "expected_ood",
    "recorded_id",
    "recorded_ood",
    "recorded_samples",
)


@dataclass(frozen=True)
class StreamDefinition:

    benchmark: str
    group: str
    leaf: str
    id_imglist: Path
    ood_imglist: Path


def imglist_length(
    path: Path,
) -> int:

    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        return len(handle.readlines())


def _resolve(
    root: Path,
    text: str,
) -> Path:

    path = Path(text)
    return path if path.is_absolute() else Path(root) / path


def merged_dataset_tree(
    benchmark: str,
    configs_root: Path | None = None,
) -> dict[str, Any]:

    base = layout.ROOT if configs_root is None else Path(configs_root)
    tree: dict[str, Any] = {}

    for relative in dataset_configs(benchmark):
        tree = config_io.merge_trees(tree, config_io.load_yaml(base / relative))

    return tree


def stream_definitions(
    benchmark: str,
    root: Path | None = None,
    configs_root: Path | None = None,
) -> list[StreamDefinition]:

    base = layout.ROOT if root is None else Path(root)
    tree = merged_dataset_tree(benchmark=benchmark, configs_root=configs_root)
    id_imglist = _resolve(root=base, text=str(tree["dataset"]["test"]["imglist_pth"]))
    ood = tree["ood_dataset"]
    definitions: list[StreamDefinition] = []

    for group in ood["split_names"]:
        block = ood[group]
        for leaf in block["datasets"]:
            definitions.append(
                StreamDefinition(
                    benchmark,
                    str(group),
                    str(leaf),
                    id_imglist,
                    _resolve(root=base, text=str(block[leaf]["imglist_pth"])),
                ),
            )

    return definitions


def group_name(
    benchmark: str,
    group: str,
) -> str:

    if benchmark == "imagenet_standard":
        return NO_GROUP

    return GROUP_SHORT.get(group, group)


def ood_ratio(
    id_samples: float,
    ood_samples: float,
) -> float:

    if math.isnan(id_samples) or math.isnan(ood_samples) or id_samples + ood_samples == 0:
        return math.nan

    return 100.0 * ood_samples / (id_samples + ood_samples)


def compose(
    benchmarks: Iterable[str],
    root: Path | None = None,
    configs_root: Path | None = None,
) -> pd.DataFrame:

    lengths: dict[Path, float] = {}

    def length(
        path: Path,
    ) -> float:

        if path not in lengths:
            lengths[path] = float(imglist_length(path)) if path.is_file() else math.nan

        return lengths[path]

    records: list[dict[str, Any]] = []

    for benchmark in benchmarks:
        for stream in stream_definitions(benchmark=benchmark, root=root, configs_root=configs_root):
            id_samples = length(stream.id_imglist)
            ood_samples = length(stream.ood_imglist)
            total = id_samples + ood_samples
            records.append(
                {
                    "benchmark": benchmark,
                    "benchmark_name": layout.BENCHMARK_NAMES[benchmark],
                    "group": stream.group,
                    "group_name": group_name(benchmark=benchmark, group=stream.group),
                    "leaf": stream.leaf,
                    "ood_dataset": tables.leaf_name(stream.leaf),
                    "id_samples": id_samples,
                    "ood_samples": ood_samples,
                    "stream_samples": total,
                    "ood_ratio": ood_ratio(id_samples=id_samples, ood_samples=ood_samples),
                    "id_imglist": layout.relative(stream.id_imglist),
                    "ood_imglist": layout.relative(stream.ood_imglist),
                },
            )

    return pd.DataFrame.from_records(records, columns=list(TABLE_COLUMNS))


def missing_imglists(
    frame: pd.DataFrame,
) -> list[str]:

    missing: list[str] = []

    for column, count in (("id_imglist", "id_samples"), ("ood_imglist", "ood_samples")):
        for path, value in zip(frame[column], frame[count]):
            if math.isnan(float(value)) and path not in missing:
                missing.append(str(path))

    return missing


def cross_check_runs(
    ctx: Context,
) -> list[RunSpec]:

    spec = config_io.load_yaml(SPEC_PATH)["cross_check"]
    run_specs = [
        RunSpec(str(spec["method"]), benchmark, seed, backbone=ctx.backbone, label=spec.get("label"))
        for benchmark in ctx.benchmarks
        for seed in ctx.seeds
    ]

    return run_specs


def recorded_label_counts(
    npz_path: Path,
) -> tuple[int, int]:

    with np.load(npz_path) as data:
        labels = np.asarray(data["label"])

    return int((labels >= 0).sum()), int((labels < 0).sum())


def cross_check(
    frame: pd.DataFrame,
    specs: Sequence[RunSpec],
) -> pd.DataFrame:

    records: list[dict[str, Any]] = []

    for spec in specs:
        part = frame[frame["benchmark"] == spec.benchmark]
        for row in part.itertuples(index=False):
            npz = spec.dir() / "scores" / f"{row.leaf}.npz"
            record: dict[str, Any] = {
                "benchmark": spec.benchmark,
                "leaf": row.leaf,
                "ood_dataset": row.ood_dataset,
                "seed": spec.seed,
                "run_dir": layout.relative(spec.dir()),
                "status": NO_SCORES,
                "expected_id": row.id_samples,
                "expected_ood": row.ood_samples,
                "recorded_id": math.nan,
                "recorded_ood": math.nan,
                "recorded_samples": math.nan,
            }

            if npz.is_file():
                recorded_id, recorded_ood = recorded_label_counts(npz)
                agree = recorded_id == row.id_samples and recorded_ood == row.ood_samples
                record.update(
                    recorded_id=recorded_id,
                    recorded_ood=recorded_ood,
                    recorded_samples=recorded_id + recorded_ood,
                    status=MATCH if agree else MISMATCH,
                )

            records.append(record)

    return pd.DataFrame.from_records(records, columns=list(CROSS_CHECK_COLUMNS))


def _count(
    value: float,
) -> str:
    return tables.MISSING_CELL if math.isnan(float(value)) else f"{int(value):,}"


def _ratio(
    value: float,
) -> str:
    return tables.MISSING_CELL if math.isnan(float(value)) else f"{float(value):.1f}"


def table_markdown(
    frame: pd.DataFrame,
    checks: pd.DataFrame | None = None,
) -> str:

    headers = ["Benchmark", "Group", "OOD dataset", "ID samples", "OOD samples", "OOD ratio (%)"]

    if checks is not None:
        headers.append("Recorded streams")

    lines = [f"### the composition table. Composition of the test streams of the {frame['benchmark'].nunique()} benchmarks", "",
        "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    previous: str | None = None

    for row in frame.itertuples(index=False):
        name = row.benchmark_name if row.benchmark != previous else ""
        previous = row.benchmark
        cells = [
            name,
            row.group_name,
            row.ood_dataset,
            _count(row.id_samples),
            _count(row.ood_samples),
            _ratio(row.ood_ratio),
        ]

        if checks is not None:
            cells.append(_recorded_cell(checks=checks, benchmark=row.benchmark, leaf=row.leaf))

        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "OOD ratio = OOD samples / (ID + OOD samples). Counts are the lengths of the image lists in "
        "`data/benchmark_imglist/` as the data loader reads them.",
    ]
    missing = missing_imglists(frame)
    if missing:
        lines += ["", "Missing image lists (counts shown as " + tables.MISSING_CELL + "): " + ", ".join(missing)]

    return "\n".join(lines) + "\n"


def _recorded_cell(
    checks: pd.DataFrame,
    benchmark: str,
    leaf: str,
) -> str:

    part = checks[(checks["benchmark"] == benchmark) & (checks["leaf"] == leaf) & (checks["status"] != NO_SCORES)]
    if part.empty:
        return tables.MISSING_CELL

    seeds = ",".join(str(seed) for seed in part["seed"])
    bad = part[part["status"] == MISMATCH]
    if bad.empty:
        return f"seeds {seeds} match"

    return f"MISMATCH (seeds {','.join(str(seed) for seed in bad['seed'])})"


def cross_check_markdown(
    checks: pd.DataFrame,
    specs: Sequence[RunSpec],
) -> str:

    recorded = checks[checks["status"] != NO_SCORES]
    mismatches = checks[checks["status"] == MISMATCH]
    absent = [layout.relative(spec.dir()) for spec in specs
        if checks[(checks["run_dir"] == layout.relative(spec.dir())) & (checks["status"] != NO_SCORES)].empty]
    lines = ["### Cross-check of the composition table against the recorded CAMP runs", "",
        f"- runs consulted: {len(specs)}; with recorded scores: {len(specs) - len(absent)}",
        f"- streams compared: {len(recorded)}; matches: {len(recorded) - len(mismatches)}; "
        f"mismatches: {len(mismatches)}"]

    if absent:
        lines += ["", "Runs without `scores/<leaf>.npz` (not recorded on this machine):"]
        lines += [f"- {path}" for path in absent]

    if not mismatches.empty:
        lines += [
            "",
            "| Benchmark | OOD dataset | seed | ID (imglist) | ID (recorded) | OOD (imglist) "
            "| OOD (recorded) |",
            "|---|---|---|---|---|---|---|",
        ]

        for row in mismatches.itertuples(index=False):
            lines.append(
                f"| {layout.BENCHMARK_NAMES[row.benchmark]} | {row.ood_dataset} | {row.seed} | "
                f"{_count(row.expected_id)} | {_count(row.recorded_id)} | {_count(row.expected_ood)} | "
                f"{_count(row.recorded_ood)} |",
            )
    elif not recorded.empty:
        lines += ["", "Every recorded stream has the ID and OOD counts of its image lists."]

    return "\n".join(lines) + "\n"


def summary(
    frame: pd.DataFrame,
    checks: pd.DataFrame,
    specs: Sequence[RunSpec],
) -> dict[str, Any]:

    recorded = checks[checks["status"] != NO_SCORES]
    mismatches = checks[checks["status"] == MISMATCH]
    consulted = [layout.relative(spec.dir()) for spec in specs]
    with_scores = sorted(set(recorded["run_dir"]))
    return {
        "imglists": sorted(set(frame["id_imglist"]) | set(frame["ood_imglist"])),
        "missing_imglists": missing_imglists(frame),
        "cross_check_runs": consulted,
        "cross_check_runs_without_scores": [path for path in consulted if path not in with_scores],
        "cross_check_streams": int(len(recorded)),
        "cross_check_mismatches": [
            {
                "benchmark": row.benchmark,
                "leaf": row.leaf,
                "seed": int(row.seed),
                "expected": [_nan_to_none(row.expected_id), _nan_to_none(row.expected_ood)],
                "recorded": [int(row.recorded_id), int(row.recorded_ood)],
            }
            for row in mismatches.itertuples(index=False)],
    }


def _nan_to_none(
    value: float,
) -> int | None:
    return None if math.isnan(float(value)) else int(value)


def analyze(
    ctx: Context,
) -> list[Path]:

    frame = compose(ctx.benchmarks, root=ctx.root)
    if frame.empty or bool(frame[["id_samples", "ood_samples"]].isna().all().all()):
        raise FileNotFoundError(
            "no image list of the requested benchmarks exists under "
            f"{layout.relative(ctx.root / 'data' / 'benchmark_imglist')}; nothing to tabulate",
        )

    specs = cross_check_runs(ctx)
    checks = cross_check(frame=frame, specs=specs)
    notes = summary(frame, checks, specs)
    ctx.notes.update(notes)
    written = [
        tables.write_csv(frame, ctx.tables / f"{TABLE_STEM}.csv"),
        tables.write_markdown(table_markdown(frame=frame, checks=checks), ctx.tables / f"{TABLE_STEM}.md"),
        tables.write_csv(checks, ctx.tables / f"{CROSS_CHECK_STEM}.csv"),
        tables.write_markdown(cross_check_markdown(checks=checks, specs=specs), ctx.tables / f"{CROSS_CHECK_STEM}.md"),
    ]
    print(
        f"stream_composition: {len(frame)} streams on {frame['benchmark'].nunique()} benchmarks; "
        f"cross-check: {notes['cross_check_streams']} recorded streams, "
        f"{len(notes['cross_check_mismatches'])} mismatches, "
        f"{len(notes['cross_check_runs_without_scores'])}/{len(specs)} runs without recorded scores"
        + (f"; missing image lists: {', '.join(notes['missing_imglists'])}" if notes["missing_imglists"] else ""),
    )

    return written


def load_table(
    ctx: Context,
) -> pd.DataFrame:

    path = ctx.tables / f"{TABLE_STEM}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{layout.relative(path)} is missing; run `analyze` first")

    return pd.read_csv(path)


__all__ = [
    "CROSS_CHECK_COLUMNS",
    "CROSS_CHECK_STEM",
    "MATCH",
    "MISMATCH",
    "NO_SCORES",
    "StreamDefinition",
    "TABLE_COLUMNS",
    "TABLE_STEM",
    "analyze",
    "compose",
    "cross_check",
    "cross_check_markdown",
    "cross_check_runs",
    "group_name",
    "imglist_length",
    "load_table",
    "merged_dataset_tree",
    "missing_imglists",
    "ood_ratio",
    "recorded_label_counts",
    "stream_definitions",
    "summary",
    "table_markdown",
]
