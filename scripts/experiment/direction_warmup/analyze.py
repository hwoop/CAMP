from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experiment.lib import layout, tables
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.runs import RunSpec, command

CACHE_VERSION = 1
DEFAULT_WARMUP = 512
DEFAULT_WINDOW = 8192
DENSE_UNTIL = 32
METRICS: tuple[str, ...] = ("AUROC", "FPR@95")
GROUP_LEAVES: dict[str, tuple[str, str]] = {
    "IN-1K": ("imagenet_standard", "farood"),
    "near": ("imagenet_openood", "nearood"),
    "far": ("imagenet_openood", "farood"),
}
TABLE_STREAMS = "direction_warmup_streams.csv"
TABLE_SUMMARY = "direction_warmup.csv"
TABLE_CROSSING = "crossing.csv"
TABLE_MD = "direction_warmup.md"
STREAM_COLUMNS: tuple[str, ...] = (
    "benchmark",
    "seed",
    "leaf",
    "group",
    "t",
    "AUROC",
    "FPR@95",
    "classes_seen",
    "mcm_AUROC",
    "mcm_FPR@95",
    "first_repeat",
    "window_size",
    "n_samples",
    "n_ood",
)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "group",
    "t",
    "metric",
    "camp_mean",
    "camp_std",
    "mcm_mean",
    "mcm_std",
    "gain_mean",
    "gain_std",
    "n_seeds",
    "seeds",
    "missing",
)
CROSSING_COLUMNS: tuple[str, ...] = ("benchmark", "seed", "leaf", "group", "metric", "mcm", "first_t", "stable_t")


def item_runs(
    ctx: Context,
) -> list[RunSpec]:

    run_specs = [
        RunSpec("camp", b, s, backbone=ctx.backbone)
        for b in ctx.benchmarks
        for s in ctx.seeds
    ]

    return run_specs


def default_grid(
    window_size: int = DEFAULT_WINDOW,
) -> list[int]:

    dense = list(range(2, min(DENSE_UNTIL, window_size) + 1))
    steps = int(math.ceil(4 * math.log2(max(window_size, DENSE_UNTIL) / DENSE_UNTIL)))
    geometric = {int(round(DENSE_UNTIL * 2 ** (k / 4))) for k in range(1, steps + 1)}
    return sorted({t for t in dense + list(geometric) if 2 <= t <= window_size} | {window_size})


def grid_for(
    ctx: Context,
    window_size: int,
) -> list[int]:

    values = ctx.option_list("t")
    if not values:
        return default_grid(window_size)

    grid = sorted({int(v) for v in values})
    if grid[0] < 2:
        raise ValueError("t values must be integers >= 2 (the window needs two samples)")

    values_ = [
        t
        for t in grid
        if t <= window_size
    ]

    return values_


def grid_tag(
    grid: Sequence[int],
) -> str:
    return hashlib.sha1(",".join(str(t) for t in grid).encode("utf-8")).hexdigest()[:8]


def wanted_streams(
    ctx: Context,
    benchmark: str,
) -> list[str]:

    leaves = [leaf for members in layout.LEAVES[benchmark].values() for leaf in members]
    selected = ctx.option_list("leaves", leaves)
    unknown = [leaf for leaf in selected if leaf not in leaves]
    if unknown:
        raise KeyError(f"unknown stream(s) {unknown} for {benchmark}; one of {leaves}")

    return selected


def cache_path(
    cache: Path,
    spec: RunSpec,
    leaf: str,
    tag: str,
) -> Path:

    run_dir = spec.dir()
    return Path(cache) / f"direction_v{CACHE_VERSION}_{tag}__{run_dir.parent.name}__{run_dir.name}__{leaf}.csv"


def evaluate(
    score: np.ndarray,
    label: np.ndarray,
) -> tuple[float, float]:

    from openood.evaluators.metrics import auc_and_fpr_recall

    auroc, _, _, fpr = auc_and_fpr_recall(conf=np.asarray(score, dtype=np.float64), label=np.asarray(label), tpr_th=0.95)
    return float(fpr) * 100.0, float(auroc) * 100.0


def first_repeat(
    pred: np.ndarray,
) -> int:

    seen: set[int] = set()

    for position, value in enumerate(np.asarray(pred).tolist()):
        if value in seen:
            return position

        seen.add(value)

    return -1


def stream_curve(
    run: Any,
    features: Any,
    stream: Any,
    grid: Sequence[int],
) -> pd.DataFrame:

    import torch

    from scripts.experiment.lib import geometry, replay

    replay.verify_stream(stream)
    camp = run.policy().camp
    args = {
        "eps": float(camp.shift_eps),
        "class_num": int(features.class_num),
        "covariance": camp.covariance,
        "id_reference": camp.id_reference,
        "weighting": camp.weighting,
    }
    labels = np.asarray(stream.labels)
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        F = stream.features.to(features.device)
        msp, pred, _ = geometry.zero_shot(F, features.prototypes.to(F.device), features.logit_scale.to(F.device))
        weights = msp.to(torch.float64)
        classes = pred.to(torch.long)
        mcm_fpr, mcm_auroc = evaluate(score=msp.detach().cpu().numpy(), label=labels)
        repeat = first_repeat(pred.detach().cpu().numpy())
        F64 = F.to(torch.float64)

        for t in grid:
            if t > len(stream):
                break

            g = geometry.camp_geometry(F[:t], weights[:t], classes[:t], **args)
            score = -(F64 @ g.u)
            fpr, auroc = evaluate(score=score.detach().cpu().numpy(), label=labels)
            rows.append(
                {
                    "benchmark": run.benchmark,
                    "seed": int(run.seed),
                    "leaf": str(stream.leaf),
                    "group": str(stream.group),
                    "t": int(t),
                    "AUROC": auroc,
                    "FPR@95": fpr,
                    "classes_seen": int(g.k),
                    "mcm_AUROC": mcm_auroc,
                    "mcm_FPR@95": mcm_fpr,
                    "first_repeat": repeat,
                    "window_size": int(camp.window_size),
                    "n_samples": int(labels.size),
                    "n_ood": int((labels < 0).sum()),
                },
            )

    return pd.DataFrame.from_records(rows, columns=STREAM_COLUMNS)


def _mean_std(
    values: Sequence[float],
) -> tuple[float, float]:

    if not values:
        return math.nan, math.nan

    return float(np.mean(values)), tables.sample_std(list(values))


def summarize(
    streams: pd.DataFrame,
    requested_seeds: Sequence[int],
) -> pd.DataFrame:

    records: list[dict[str, Any]] = []

    if streams.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    for group, (benchmark, row) in GROUP_LEAVES.items():
        leaves = set(layout.LEAVES[benchmark][row])
        part = streams[(streams["benchmark"] == benchmark) & streams["leaf"].isin(leaves)]
        if part.empty:
            continue

        for t in sorted(set(part["t"])):
            for metric in METRICS:
                camp: dict[int, float] = {}
                mcm: dict[int, float] = {}

                for seed, rows in part[part["t"] == t].groupby("seed"):
                    if set(rows["leaf"]) != leaves:
                        continue

                    camp[int(seed)] = float(rows[metric].mean())
                    mcm[int(seed)] = float(rows[f"mcm_{metric}"].mean())

                seeds = sorted(camp)
                gains = [(camp[s] - mcm[s]) if metric == "AUROC" else (mcm[s] - camp[s]) for s in seeds]
                camp_mean, camp_std = _mean_std([camp[s] for s in seeds])
                mcm_mean, mcm_std = _mean_std([mcm[s] for s in seeds])
                gain_mean, gain_std = _mean_std(gains)
                records.append(
                    {
                        "group": group,
                        "t": int(t),
                        "metric": metric,
                        "camp_mean": camp_mean,
                        "camp_std": camp_std,
                        "mcm_mean": mcm_mean,
                        "mcm_std": mcm_std,
                        "gain_mean": gain_mean,
                        "gain_std": gain_std,
                        "n_seeds": len(seeds),
                        "seeds": ",".join(str(s) for s in seeds),
                        "missing": ",".join(str(s) for s in requested_seeds if s not in camp),
                    },
                )

    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def crossings(
    streams: pd.DataFrame,
) -> pd.DataFrame:

    records: list[dict[str, Any]] = []

    if streams.empty:
        return pd.DataFrame(columns=CROSSING_COLUMNS)

    for (benchmark, seed, leaf), part in streams.groupby(["benchmark", "seed", "leaf"]):
        part = part.sort_values("t")

        for metric in METRICS:
            reference = float(part[f"mcm_{metric}"].iloc[0])
            values = part[metric].to_numpy(dtype=float)
            better = values >= reference if metric == "AUROC" else values <= reference
            ts = part["t"].to_numpy(dtype=int)
            first = int(ts[np.argmax(better)]) if better.any() else -1
            stable = -1

            for i in range(len(ts)):
                if better[i:].all():
                    stable = int(ts[i])
                    break

            records.append(
                {
                    "benchmark": benchmark,
                    "seed": int(seed),
                    "leaf": leaf,
                    "group": str(part["group"].iloc[0]),
                    "metric": metric,
                    "mcm": reference,
                    "first_t": first,
                    "stable_t": stable,
                },
            )

    return pd.DataFrame.from_records(records, columns=CROSSING_COLUMNS)


def markdown(
    summary: pd.DataFrame,
    crossing: pd.DataFrame,
) -> str:

    if summary.empty:
        return "### Direction from the first t samples, scored on the whole stream\n\nno recorded CAMP run\n"

    names = {
        "IN-1K": "ImageNet-1K",
        "near": "Near-OOD",
        "far": "Far-OOD",
    }
    shown = [t for t in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192) if t in set(summary["t"])]
    lines = ["### Direction from the first t samples, scored on the whole stream (CAMP, MCM in parentheses)", ""]

    for metric in METRICS:
        part = summary[summary["metric"] == metric]
        if part.empty:
            continue

        arrow = "↑" if metric == "AUROC" else "↓"
        lines += [f"#### {metric}{arrow}", "", "| group | seeds | " + " | ".join(f"t = {t:,}" for t in shown) + " |",
            "|" + "---|" * (2 + len(shown))]

        for group in GROUP_LEAVES:
            rows = part[part["group"] == group]
            if rows.empty:
                continue

            cells = [names[group], str(rows["seeds"].iloc[0])]

            for t in shown:
                hit = rows[rows["t"] == t]
                if hit.empty or math.isnan(float(hit["camp_mean"].iloc[0])):
                    cells.append(tables.MISSING_CELL)
                else:
                    cells.append(f"{float(hit['camp_mean'].iloc[0]):.2f} ({float(hit['mcm_mean'].iloc[0]):.2f})")

            lines.append("| " + " | ".join(cells) + " |")

        lines.append("")

    if not crossing.empty:
        lines += ["#### Window fill from which CAMP is at least as good as MCM on the whole stream", "",
            "| group | metric | streams x seeds | first_t: median / max | stable_t: median / max |", "|---|---|---|---|---|"]

        for group, (benchmark, row) in GROUP_LEAVES.items():
            leaves = set(layout.LEAVES[benchmark][row])

            for metric in METRICS:
                part = crossing[(crossing["benchmark"] == benchmark) & crossing["leaf"].isin(leaves)
                    & (crossing["metric"] == metric)]
                if part.empty:
                    continue

                def text(
                    column: str,
                ) -> str:

                    values = part[column].to_numpy(dtype=int)
                    if (values < 0).any():
                        return f"{int(np.median(values[values >= 0])) if (values >= 0).any() else '–'} / never on {int((values < 0).sum())} stream(s)"

                    return f"{int(np.median(values)):,} / {int(values.max()):,}"

                lines.append(f"| {names[group]} | {metric} | {len(part)} | {text('first_t')} | {text('stable_t')} |")

        lines.append("")

    lines.append(
        "`first_t` is the first grid point at which CAMP's whole-stream metric matches MCM's, `stable_t` the point "
        "from which every later grid point does. The grid is every integer to 32, then quarter octaves to 8,192.",
    )

    return "\n".join(lines) + "\n"


def analyze(
    ctx: Context,
) -> list[Path]:

    specs = item_runs(ctx)
    recorded = [spec for spec in specs if spec.recorded()]
    missing = [spec for spec in specs if not spec.recorded()]

    if not recorded:
        example = " ".join(command(specs[0], python=ctx.python))
        raise FileNotFoundError(
            f"none of the {len(specs)} CAMP main runs is recorded, e.g. {layout.relative(specs[0].dir())}; "
            f"run them first (`python -m scripts.experiment main_results run`), e.g.\n  {example}",
        )

    print(f"[direction_warmup] {len(recorded)}/{len(specs)} CAMP main runs recorded")

    for spec in missing:
        print(f"  missing {layout.relative(spec.dir())}")

    frames: list[pd.DataFrame] = []
    replayed: list[str] = []
    cached_streams: list[str] = []
    grid: list[int] | None = None

    for spec in recorded:
        from scripts.experiment.lib import replay

        run = replay.RecordedRun(spec.dir())
        window_size = int(run.policy().camp.window_size)
        grid = grid_for(ctx=ctx, window_size=window_size)
        tag = grid_tag(grid)
        leaves = wanted_streams(ctx=ctx, benchmark=spec.benchmark)
        paths = {leaf: cache_path(ctx.cache, spec, leaf, tag) for leaf in leaves}
        for leaf, path in paths.items():
            if path.is_file():
                frames.append(pd.read_csv(path))
                cached_streams.append(f"{spec.dir().name}/{leaf}")

        todo = [leaf for leaf, path in paths.items() if not path.is_file()]
        if not todo:
            continue

        if ctx.dry_run:
            print(f"[dry-run] would replay {layout.relative(spec.dir())} streams {todo}")
            continue

        if str(ctx.device).startswith("cpu"):
            raise RuntimeError("the replay needs a CUDA device (the released CLIP network runs on CUDA); pass --device cuda")

        features = replay.features_of(run, ctx.device)
        for stream in replay.streams(run, features, leaves=todo):
            print(f"[replay] {layout.relative(spec.dir())} stream {stream.name} ({len(stream)} samples, {len(grid)} window fills)")
            frame = stream_curve(run=run, features=features, stream=stream, grid=grid)
            frame.to_csv(paths[str(stream.leaf)], index=False, encoding="utf-8")
            frames.append(frame)
            replayed.append(f"{spec.dir().name}/{stream.leaf}")

    streams = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=STREAM_COLUMNS)
    summary = summarize(streams, ctx.seeds)
    crossing = crossings(streams)
    written = [
        tables.write_csv(streams, ctx.tables / TABLE_STREAMS),
        tables.write_csv(summary, ctx.tables / TABLE_SUMMARY),
        tables.write_csv(crossing, ctx.tables / TABLE_CROSSING),
        tables.write_markdown(markdown(summary, crossing), ctx.tables / TABLE_MD),
    ]
    ctx.notes.update(
        {
            "runs_recorded": len(recorded), "runs_total": len(specs),
            "missing_runs": [layout.relative(spec.dir()) for spec in missing],
            "grid": grid or [], "streams_replayed": replayed, "streams_cached": cached_streams,
            "first_repeat": {f"{b}-s{s}/{leaf}": int(v) for (b, s, leaf), v in
                (streams.groupby(["benchmark", "seed", "leaf"])["first_repeat"].first().items()
                    if not streams.empty else [])},
        },
    )

    return written


__all__ = [
    "CACHE_VERSION",
    "CROSSING_COLUMNS",
    "DEFAULT_WARMUP",
    "DEFAULT_WINDOW",
    "GROUP_LEAVES",
    "METRICS",
    "STREAM_COLUMNS",
    "SUMMARY_COLUMNS",
    "TABLE_CROSSING",
    "TABLE_MD",
    "TABLE_STREAMS",
    "TABLE_SUMMARY",
    "analyze",
    "cache_path",
    "crossings",
    "default_grid",
    "evaluate",
    "first_repeat",
    "grid_for",
    "grid_tag",
    "item_runs",
    "markdown",
    "stream_curve",
    "summarize",
    "wanted_streams",
]
