from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experiment.lib import config_io, layout, tables
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.runs import RunSpec, command

LABEL = "WUB_batch1"
OVERLAYS: tuple[str, ...] = ("configs/diagnostics/camp_samples.yml",)
STANDARD_BATCH_SIZE = 256
OVERRIDES: dict[str, Any] = {
    "dataset.test.batch_size": 1,
    "ood_dataset.batch_size": 1,
    "cache.image.extraction_batch_size": STANDARD_BATCH_SIZE,
    "postprocessor.postprocessor_args.camp.min_samples": 2,
}
DEFAULT_N_MIN: tuple[int, ...] = tuple(2 ** k for k in range(1, 14))
DEFAULT_WARMUP = 512
REFERENCE = "mcm"
METRICS: tuple[str, ...] = ("FPR@95", "AUROC")
GROUP_LEAVES: dict[str, tuple[str, str]] = {
    "IN-1K": ("imagenet_standard", "farood"),
    "near": ("imagenet_openood", "nearood"),
    "far": ("imagenet_openood", "farood"),
}
TABLE_STREAMS = "warmup_streams.csv"
TABLE_SUMMARY = "warmup.csv"
TABLE_MD = "warmup.md"
STREAM_COLUMNS: tuple[str, ...] = (
    "benchmark",
    "seed",
    "leaf",
    "group",
    "n_min",
    "FPR@95",
    "AUROC",
    "mcm_FPR@95",
    "mcm_AUROC",
    "n_samples",
    "n_ood",
    "first_repeat",
    "run_min_samples",
    "run_batch_size",
    "semantic_weight",
    "check_max_abs_diff",
)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "group",
    "n_min",
    "metric",
    "camp_mean",
    "camp_std",
    "mcm_mean",
    "mcm_std",
    "gain_mean",
    "gain_std",
    "default_mean",
    "default_std",
    "n_seeds",
    "seeds",
    "missing",
    "leaves",
    "partial",
)


def camp_run(
    benchmark: str,
    seed: int,
    backbone: str = "vitb16",
) -> RunSpec:
    return RunSpec("camp", benchmark, seed, backbone=backbone, label=LABEL, overrides=dict(OVERRIDES), overlays=OVERLAYS)


def item_runs(
    ctx: Context,
) -> list[RunSpec]:

    own = [camp_run(benchmark=b, seed=s, backbone=ctx.backbone) for b in ctx.benchmarks for s in ctx.seeds]
    mcm = [RunSpec(REFERENCE, b, s, backbone=ctx.backbone) for b in ctx.benchmarks for s in ctx.seeds]
    default = [RunSpec("camp", b, s, backbone=ctx.backbone) for b in ctx.benchmarks for s in ctx.seeds]
    return own + mcm + default


def n_min_values(
    ctx: Context,
) -> list[int]:

    values = sorted({int(v) for v in ctx.option_list("n_min", [str(v) for v in DEFAULT_N_MIN])})
    if not values or values[0] < 2:
        raise ValueError("n_min values must be integers >= 2 (the window needs two samples to standardize)")

    return values


def run_settings(
    run_dir: Path,
) -> dict[str, Any]:

    tree = config_io.load_yaml(run_dir / "config.yml", python_tags=True)
    args = tree["postprocessor"]["postprocessor_args"]
    return {
        "batch_size": int(tree["dataset"]["test"]["batch_size"]),
        "min_samples": int(args["camp"]["min_samples"]),
        "window_size": int(args["camp"]["window_size"]),
        "semantic_weight": float(args.get("fusion", {}).get("semantic_weight", 0.0)),
    }


def sample_scores_path(
    run_dir: Path,
    leaf: str,
) -> Path:

    folder = run_dir / "camp_diagnostics" / leaf

    for name in ("sample_scores.csv", "sample_scores.parquet"):
        if (folder / name).is_file():
            return folder / name

    raise FileNotFoundError(
        f"{layout.relative(folder)} holds no sample_scores table; the run must be made with "
        f"{OVERLAYS[0]} appended (diagnostics.level samples or above)",
    )


def read_sample_scores(
    run_dir: Path,
    leaf: str,
) -> pd.DataFrame:

    path = sample_scores_path(run_dir=run_dir, leaf=leaf)
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        try:
            frame = pd.read_parquet(path)
        except ImportError as error:
            raise RuntimeError(f"{layout.relative(path)} needs pyarrow to be read (pip install pyarrow)") from error

    needed = ["order_idx", "S_additive_axis", "zS", "zR", "final", "pred"]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError(f"{layout.relative(path)} lacks the columns {missing}")

    return frame.sort_values("order_idx").reset_index(drop=True)


def complete_leaves(
    spec: RunSpec,
) -> tuple[list[str], list[str]]:

    run_dir = spec.dir()
    complete, missing = [], []

    for leaves in layout.LEAVES[spec.benchmark].values():
        for leaf in leaves:
            has_scores = (run_dir / "scores" / f"{leaf}.npz").is_file()
            has_samples = any((run_dir / "camp_diagnostics" / leaf / f"sample_scores{ext}").is_file()
                for ext in (".csv", ".parquet"))

            (complete if has_scores and has_samples else missing).append(leaf)

    return complete, missing


def recorded_scores(
    run_dir: Path,
    leaf: str,
) -> dict[str, np.ndarray]:

    path = run_dir / "scores" / f"{leaf}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{layout.relative(path)} is missing")

    with np.load(path) as data:
        table = {
            key: np.asarray(data[key])
            for key in ("pred", "conf", "label")
        }

        return table


def compose(
    semantic_logit: np.ndarray,
    z_semantic: np.ndarray,
    z_camp: np.ndarray,
    n_min: int,
    semantic_weight: float = 0.0,
) -> np.ndarray:

    score = np.asarray(semantic_logit, dtype=np.float64).copy()
    zs = np.asarray(z_semantic, dtype=np.float64)
    zr = np.asarray(z_camp, dtype=np.float64)
    has_zs = np.isfinite(zs)
    score[has_zs] = zs[has_zs]
    position = np.arange(score.size)
    after = (position >= n_min) & np.isfinite(zr)

    if semantic_weight == 0.0:
        score[after] = zr[after]
    else:
        score[after] = semantic_weight * zs[after] + zr[after]

    return score


def first_repeat(
    pred: np.ndarray,
) -> int:

    seen: set[int] = set()

    for position, value in enumerate(np.asarray(pred).tolist()):
        if value in seen:
            return position

        seen.add(value)

    return -1


def evaluate(
    score: np.ndarray,
    label: np.ndarray,
) -> tuple[float, float]:

    from openood.evaluators.metrics import auc_and_fpr_recall

    auroc, _, _, fpr = auc_and_fpr_recall(conf=np.asarray(score, dtype=np.float64), label=np.asarray(label), tpr_th=0.95)
    return float(fpr) * 100.0, float(auroc) * 100.0


def stream_rows(
    spec: RunSpec,
    n_mins: Sequence[int],
    leaves: Sequence[str] | None = None,
    reference: Mapping[str, Mapping[str, float]] | None = None,
) -> list[dict[str, Any]]:

    run_dir = spec.dir()
    settings = run_settings(run_dir)
    reference = reference or {}
    rows: list[dict[str, Any]] = []

    for group, group_leaves in layout.LEAVES[spec.benchmark].items():
        for leaf in group_leaves:
            if leaves is not None and leaf not in leaves:
                continue

            frame = read_sample_scores(run_dir=run_dir, leaf=leaf)
            recorded = recorded_scores(run_dir, leaf)
            if len(frame) != recorded["label"].size:
                raise ValueError(
                    f"{spec.dir().name}/{leaf}: {len(frame)} recorded samples but {recorded['label'].size} "
                    "scores; the recorder and the evaluator disagree on the stream",
                )

            if "true_class" in frame.columns:
                label = frame["true_class"].to_numpy(dtype=np.int64)
            else:
                label = recorded["label"]

            s = frame["S_additive_axis"].to_numpy(dtype=np.float64)
            zs = frame["zS"].to_numpy(dtype=np.float64)
            zr = frame["zR"].to_numpy(dtype=np.float64)
            own = compose(s, zs, zr, settings["min_samples"], settings["semantic_weight"])
            check = float(np.max(np.abs(own - recorded["conf"].astype(np.float64)))) if own.size else 0.0
            repeat = first_repeat(frame["pred"].to_numpy())
            mcm_leaf = reference.get(leaf, {})

            for n_min in n_mins:
                fpr, auroc = evaluate(score=compose(s, zs, zr, n_min, settings["semantic_weight"]), label=label)
                rows.append(
                    {
                        "benchmark": spec.benchmark,
                        "seed": spec.seed,
                        "leaf": leaf,
                        "group": group,
                        "n_min": int(n_min),
                        "FPR@95": fpr,
                        "AUROC": auroc,
                        "mcm_FPR@95": float(mcm_leaf.get("FPR@95", math.nan)),
                        "mcm_AUROC": float(mcm_leaf.get("AUROC", math.nan)),
                        "n_samples": int(label.size),
                        "n_ood": int((label == -1).sum()),
                        "first_repeat": repeat,
                        "run_min_samples": settings["min_samples"],
                        "run_batch_size": settings["batch_size"],
                        "semantic_weight": settings["semantic_weight"],
                        "check_max_abs_diff": check,
                    },
                )

    return rows


def leaf_metrics(
    specs: Sequence[RunSpec],
) -> dict[tuple[str, int], dict[str, dict[str, float]]]:

    out: dict[tuple[str, int], dict[str, dict[str, float]]] = {}

    for spec in specs:
        if not spec.recorded():
            continue

        values = tables.read_ood_csv(spec.ood_csv())
        leaves = [leaf for group in layout.LEAVES[spec.benchmark].values() for leaf in group]
        out[(spec.benchmark, spec.seed)] = {leaf: {metric: float(values[leaf][metric]) for metric in METRICS}
            for leaf in leaves if leaf in values}

    return out


def _group_mean(
    by_leaf: Mapping[str, Mapping[str, float]],
    leaves: Sequence[str],
    metric: str,
) -> float | None:

    if not all(leaf in by_leaf for leaf in leaves):
        return None

    return float(np.mean([by_leaf[leaf][metric] for leaf in leaves]))


def _mean_std(
    values: Sequence[float],
) -> tuple[float, float]:

    if not values:
        return math.nan, math.nan

    return float(np.mean(values)), tables.sample_std(list(values))


def summarize(
    streams: pd.DataFrame,
    mcm: Mapping[tuple[str, int], Mapping[str, Mapping[str, float]]],
    default: Mapping[tuple[str, int], Mapping[str, Mapping[str, float]]],
    requested_seeds: Sequence[int],
    *,
    partial: bool = False,
) -> pd.DataFrame:

    records: list[dict[str, Any]] = []

    if streams.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    for group, (benchmark, row) in GROUP_LEAVES.items():
        all_leaves = list(layout.LEAVES[benchmark][row])
        part = streams[(streams["benchmark"] == benchmark) & streams["leaf"].isin(all_leaves)]
        if part.empty:
            continue

        for n_min in sorted(set(part["n_min"])):
            for metric in METRICS:
                camp: dict[int, float] = {}
                mcm_values: dict[int, float] = {}
                default_values: dict[int, float] = {}
                used: dict[int, list[str]] = {}

                for seed, rows in part[part["n_min"] == n_min].groupby("seed"):
                    present = sorted(set(rows["leaf"]), key=all_leaves.index)
                    if set(present) != set(all_leaves) and not partial:
                        continue

                    seed = int(seed)
                    camp[seed] = float(rows[metric].mean())
                    used[seed] = present
                    reference = _group_mean(by_leaf=mcm.get((benchmark, seed), {}), leaves=present, metric=metric)
                    if reference is not None:
                        mcm_values[seed] = reference

                    base = _group_mean(by_leaf=default.get((benchmark, seed), {}), leaves=present, metric=metric)
                    if base is not None:
                        default_values[seed] = base

                seeds = sorted(camp)
                gains = [(mcm_values[s] - camp[s]) if metric == "FPR@95" else (camp[s] - mcm_values[s])
                    for s in seeds if s in mcm_values]
                camp_mean, camp_std = _mean_std([camp[s] for s in seeds])
                mcm_mean, mcm_std = _mean_std([mcm_values[s] for s in seeds if s in mcm_values])
                gain_mean, gain_std = _mean_std(gains)
                default_mean, default_std = _mean_std([default_values[s] for s in seeds if s in default_values])
                leaves_used = sorted(set().union(*used.values()), key=all_leaves.index) if used else []
                records.append(
                    {
                        "group": group,
                        "n_min": int(n_min),
                        "metric": metric,
                        "camp_mean": camp_mean,
                        "camp_std": camp_std,
                        "mcm_mean": mcm_mean,
                        "mcm_std": mcm_std,
                        "gain_mean": gain_mean,
                        "gain_std": gain_std,
                        "default_mean": default_mean,
                        "default_std": default_std,
                        "n_seeds": len(seeds),
                        "seeds": ",".join(str(s) for s in seeds),
                        "missing": ",".join(str(s) for s in requested_seeds if s not in camp),
                        "leaves": ",".join(leaves_used),
                        "partial": any(set(used[s]) != set(all_leaves) for s in seeds),
                    },
                )

    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def markdown(
    summary: pd.DataFrame,
) -> str:

    if summary.empty:
        return "### Warm-up length at batch size 1\n\nno batch-1 CAMP run recorded\n"

    lines = ["### Warm-up length at batch size 1 (whole-stream metrics, CAMP vs MCM)", ""]
    groups = [g for g in GROUP_LEAVES if g in set(summary["group"])]
    names = {
        "IN-1K": "ImageNet-1K",
        "near": "Near-OOD",
        "far": "Far-OOD",
    }

    for metric in METRICS:
        part = summary[summary["metric"] == metric]
        if part.empty:
            continue

        arrow = "↓" if metric == "FPR@95" else "↑"
        header = f"| N_min | seeds | " + " | ".join(f"{names[g]} {metric}{arrow} | {names[g]} gain over MCM" for g in groups) + " |"
        lines += [f"#### {metric}", "", header, "|" + "---|" * (2 + 2 * len(groups))]

        for n_min in sorted(set(part["n_min"])):
            cells = [f"N_min = {n_min:,}"]
            rows = part[part["n_min"] == n_min]
            seeds = rows["seeds"].iloc[0] if not rows.empty else ""
            missing = rows["missing"].iloc[0] if not rows.empty else ""
            cells.append(f"{seeds} (missing {missing})" if missing else (seeds or tables.MISSING_CELL))

            for g in groups:
                hit = rows[rows["group"] == g]
                if hit.empty or math.isnan(float(hit["camp_mean"].iloc[0])):
                    cells += [tables.MISSING_CELL, tables.MISSING_CELL]
                    continue

                r = hit.iloc[0]
                cells.append(tables.cell(float(r["camp_mean"]), float(r["camp_std"])))
                cells.append(
                    tables.cell(float(r["gain_mean"]), float(r["gain_std"]))
                    if not math.isnan(float(r["gain_mean"])) else tables.MISSING_CELL,
                )

            lines.append("| " + " | ".join(cells) + " |")

        mcm_cells = [f"{names[g]}: " + (tables.cell(
                    float(part[part['group'] == g]['mcm_mean'].iloc[0]),
                    float(part[part['group'] == g]['mcm_std'].iloc[0]),
                )
                if not part[part["group"] == g].empty
                and not math.isnan(float(part[part["group"] == g]["mcm_mean"].iloc[0]))
                else tables.MISSING_CELL) for g in groups]

        lines += ["", f"MCM reference ({metric}): " + "; ".join(mcm_cells), ""]

    lines.append(
        "Gains are MCM − CAMP for FPR@95 and CAMP − MCM for AUROC (percentage points; positive = CAMP better). "
        "`default_mean` in `warmup.csv` is the CAMP main run at batch size 256 and N_min = 512.",
    )

    return "\n".join(lines) + "\n"


def analyze(
    ctx: Context,
) -> list[Path]:

    n_mins = n_min_values(ctx)
    partial = str(ctx.option("partial", "0")).lower() in ("1", "true", "yes")
    own = [camp_run(benchmark=b, seed=s, backbone=ctx.backbone) for b in ctx.benchmarks for s in ctx.seeds]
    missing = [spec for spec in own if not spec.recorded()]
    usable: list[tuple[RunSpec, list[str] | None]] = []
    incomplete: dict[str, list[str]] = {}
    partial_runs: list[str] = []

    for spec in own:
        if not spec.recorded():
            continue

        present, absent = complete_leaves(spec)
        if not absent:
            usable.append((spec, None))
        else:
            incomplete[layout.relative(spec.dir())] = absent
            if partial and present:
                usable.append((spec, present))
                partial_runs.append(layout.relative(spec.dir()))

    complete = [spec for spec, leaves in usable if leaves is None]
    print(
        f"[warmup_batch1] {len(complete)}/{len(own)} batch-1 runs complete"
        + (f", {len(partial_runs)} read partially" if partial else "") + f"; N_min = {n_mins}",
    )

    for spec in missing:
        print(f"  missing {layout.relative(spec.dir())}")

    for directory, absent in incomplete.items():
        if directory in partial_runs:
            print(f"  partial {directory}: no scores / sample table for {absent}; the other leaves are read")
        else:
            print(
                f"  incomplete {directory}: no scores / sample table for {absent} (the evaluator starts ood.csv after "
                "its first stream, so an interrupted run counts as recorded; delete the directory to run it again, "
                "or pass --opt partial=1 to read its finished leaves)",
            )

    if not usable:
        example = " ".join(command(own[0], python=ctx.python))
        raise FileNotFoundError(
            f"none of the {len(own)} batch-1 CAMP runs is complete, e.g. {layout.relative(own[0].dir())}; "
            f"launch them with `python -m scripts.experiment warmup_batch1 run`, e.g.\n  {example}",
        )

    mcm_specs = [RunSpec(REFERENCE, b, s, backbone=ctx.backbone) for b in ctx.benchmarks for s in ctx.seeds]
    default_specs = [RunSpec("camp", b, s, backbone=ctx.backbone) for b in ctx.benchmarks for s in ctx.seeds]
    mcm = leaf_metrics(mcm_specs)
    default = leaf_metrics(default_specs)
    rows: list[dict[str, Any]] = []

    for spec, leaves in usable:
        print(f"[warmup_batch1] reading {layout.relative(spec.dir())}" + (f" (leaves {leaves})" if leaves else ""))
        rows += stream_rows(spec, n_mins, leaves=leaves, reference=mcm.get((spec.benchmark, spec.seed)))

    streams = pd.DataFrame.from_records(rows, columns=STREAM_COLUMNS)
    summary = summarize(streams, mcm, default, ctx.seeds, partial=partial)
    written = [
        tables.write_csv(streams, ctx.tables / TABLE_STREAMS),
        tables.write_csv(summary, ctx.tables / TABLE_SUMMARY),
        tables.write_markdown(markdown(summary), ctx.tables / TABLE_MD),
    ]

    if not streams.empty:
        checks = streams.groupby(["benchmark", "seed", "leaf"])["check_max_abs_diff"].first()
    else:
        checks = None

    ctx.notes.update(
        {
            "runs_complete": len(complete), "runs_total": len(own), "partial": partial,
            "partial_runs": partial_runs,
            "missing_runs": [layout.relative(spec.dir()) for spec in missing],
            "incomplete_runs": incomplete,
            "mcm_missing": [layout.relative(spec.dir()) for spec in mcm_specs if not spec.recorded()],
            "n_min": n_mins,
            "check_max_abs_diff": float(checks.max()) if checks is not None and len(checks) else None,
            "first_repeat": {f"{b}-s{s}/{leaf}": int(v) for (b, s, leaf), v in
                (streams.groupby(["benchmark", "seed", "leaf"])["first_repeat"].first().items()
                    if not streams.empty else [])},
        },
    )

    return written


__all__ = [
    "DEFAULT_N_MIN",
    "DEFAULT_WARMUP",
    "GROUP_LEAVES",
    "LABEL",
    "METRICS",
    "OVERLAYS",
    "OVERRIDES",
    "REFERENCE",
    "STREAM_COLUMNS",
    "SUMMARY_COLUMNS",
    "TABLE_MD",
    "TABLE_STREAMS",
    "TABLE_SUMMARY",
    "analyze",
    "camp_run",
    "complete_leaves",
    "compose",
    "evaluate",
    "first_repeat",
    "item_runs",
    "markdown",
    "n_min_values",
    "read_sample_scores",
    "leaf_metrics",
    "recorded_scores",
    "run_settings",
    "sample_scores_path",
    "stream_rows",
    "summarize",
]
