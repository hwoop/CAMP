"""
Tables of the id_oracle item: CAMP's whole-stream detection when the ID mean and
covariance of the window equation are replaced by oracles while the stream mean
stays the window's, batch by batch.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from scripts.experiment.lib import geometry, layout, tables
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.runs import RunSpec

if TYPE_CHECKING:
    import torch

    from openood.postprocessors.camp.config import CAMPPolicy
    from scripts.experiment.lib.replay import RecordedRun, ReplayBatch, RunFeatures, Stream

CACHE_VERSION = 1
REPLAY_TOLERANCE = 1e-3
RUNTIME_TOLERANCE = 1e-4
METRICS: tuple[str, ...] = ("AUROC", "FPR@95")

# An arm names where the three quantities of the window equation u = (Sigma + eps I)^-1 (mu_stream - mu_ID) come from.
#   mu_ID / Sigma:  camp - CAMP's own window estimate (confidence-weighted, class-balanced pseudo-classes)
#                   win  - the window's true ID rows with their true classes, weight 1 (no contamination, finite window)
#                   full - every ID row of the stream with its true class, weight 1 (the population ID statistics)
#   mu_stream:      window - the plain mean of the window (what CAMP observes)
#                   ood    - the population OOD mean of the stream in place of the stream mean (the rank-one ceiling)
# The warm-up batches keep the runtime's score in every arm, and every arm standardizes its projection against the
# same window rows, so only the direction differs between arms.
ARMS: tuple[str, ...] = ("camp", "mu_win", "sigma_win", "win", "mu_full", "sigma_full", "full", "ceiling")
ARM_SOURCES: dict[str, tuple[str, str, str]] = {
    "camp": ("camp", "camp", "window"),
    "mu_win": ("win", "camp", "window"),
    "sigma_win": ("camp", "win", "window"),
    "win": ("win", "win", "window"),
    "mu_full": ("full", "camp", "window"),
    "sigma_full": ("camp", "full", "window"),
    "full": ("full", "full", "window"),
    "ceiling": ("full", "full", "ood"),
}
ARM_NAMES: dict[str, str] = {
    "camp": "CAMP (window estimate)",
    "mu_win": "μ_ID ← window oracle",
    "sigma_win": "Σ_ID ← window oracle",
    "win": "μ_ID, Σ_ID ← window oracle",
    "mu_full": "μ_ID ← full-ID oracle",
    "sigma_full": "Σ_ID ← full-ID oracle",
    "full": "μ_ID, Σ_ID ← full-ID oracle",
    "ceiling": "rank-one ceiling (full-ID oracle, population OOD mean)",
}
STREAM_COLUMNS: tuple[str, ...] = (
    "benchmark",
    "label",
    "seed",
    "stream",
    "leaf",
    "group",
    "arm",
    *METRICS,
    "n",
    "n_ood",
    "ood_ratio",
    "batches_scored",
    "cos_ceiling",
    "cos_camp",
    "runtime_max_abs_diff",
    "replay_max_abs_diff",
)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "benchmark",
    "label",
    "arm",
    "row",
    "kind",
    "metric",
    "mean",
    "std",
    "n_seeds",
    "seeds",
)


def policy_args(
    policy: CAMPPolicy,
    class_num: int,
) -> dict[str, Any]:

    return {
        "eps": float(policy.camp.shift_eps),
        "class_num": int(class_num),
        "covariance": policy.camp.covariance,
        "id_reference": policy.camp.id_reference,
    }


def unit(
    vector: torch.Tensor,
) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-12)


def direction(
    operator: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:

    import torch

    return unit(torch.linalg.solve(operator, delta))


def stream_table(
    spec: RunSpec,
    run: RecordedRun,
    features: RunFeatures,
    stream: Stream,
    *,
    args: dict[str, Any],
) -> pd.DataFrame:

    import torch

    from openood.postprocessors.camp.scores import moments, projection_score
    from scripts.experiment.lib import replay

    replay.verify_stream(stream)
    policy = run.policy()
    if float(policy.fusion.semantic_weight) != 0.0:
        raise ValueError(f"{layout.relative(spec.dir())}: the item rescores z(R) alone; it needs semantic_weight 0.")

    device = features.device
    feats = stream.features.to(device)
    labels = np.asarray(stream.labels, dtype=np.int64)
    labels_t = torch.as_tensor(labels, device=device)
    is_id = labels >= 0
    id_index = torch.as_tensor(np.flatnonzero(is_id), device=device, dtype=torch.long)
    ood_index = torch.as_tensor(np.flatnonzero(~is_id), device=device, dtype=torch.long)
    n = len(stream)
    dim = int(feats.shape[1])

    with torch.no_grad():
        zero = torch.zeros(dim, dtype=torch.float64, device=device)
        full = geometry.oracle_geometry(feats[id_index], labels_t[id_index], mu_stream=zero, **args)
        mu_ood = feats[ood_index].to(torch.float64).mean(0)
        u_ceiling = direction(full.operator, mu_ood - full.mu_id)

    scores = {arm: np.full(n, np.nan, dtype=np.float64) for arm in ARMS}
    cos_ceiling: dict[str, list[float]] = {arm: [] for arm in ARMS}
    cos_camp: dict[str, list[float]] = {arm: [] for arm in ARMS}
    state = {"runtime_diff": 0.0, "scored": 0}

    def every_batch(
        batch: ReplayBatch,
    ) -> None:

        if batch.geometry is None or batch.rows is None:
            for arm in ARMS:
                scores[arm][batch.start:batch.stop] = batch.score

            return

        with torch.no_grad():
            camp = batch.geometry
            window_rows = batch.rows[0].to(device)
            window_labels = labels_t[batch.window.start:batch.window.stop]
            id_mask = window_labels >= 0
            win = geometry.oracle_geometry(window_rows[id_mask], window_labels[id_mask], mu_stream=camp.mu_stream, **args)
            mu = {"camp": camp.mu_id, "win": win.mu_id, "full": full.mu_id}
            op = {"camp": camp.operator, "win": win.operator, "full": full.operator}
            rows = feats[batch.start:batch.stop]

            for arm in ARMS:
                mu_source, op_source, stream_source = ARM_SOURCES[arm]
                if stream_source == "ood":
                    u = u_ceiling
                else:
                    u = direction(op[op_source], camp.mu_stream - mu[mu_source])

                m, v = moments(projection_score(window_rows, u))
                scores[arm][batch.start:batch.stop] = geometry.camp_score(rows, u, m, v).cpu().numpy()
                cos_ceiling[arm].append(geometry.cosine(u, u_ceiling))
                cos_camp[arm].append(geometry.cosine(u, camp.u))

            difference = float(np.max(np.abs(scores["camp"][batch.start:batch.stop] - batch.score)))
            state["runtime_diff"] = max(state["runtime_diff"], difference)
            state["scored"] += 1

    replay.replay_camp(stream, features, every_batch=every_batch)
    replay_diff = replay.check_against_recorded(stream, scores["camp"], tolerance=REPLAY_TOLERANCE)
    if state["runtime_diff"] > RUNTIME_TOLERANCE:
        raise RuntimeError(
            f"stream {stream.name!r}: the item's camp arm differs from the runtime score by {state['runtime_diff']:.3g}; "
            "the scoring path does not reproduce CAMP.",
        )

    records: list[dict[str, Any]] = []

    for evaluation in stream.evaluations:
        positions = np.asarray(evaluation.positions)
        eval_is_id = labels[positions] >= 0
        n_eval = int(positions.size)
        n_ood = int((~eval_is_id).sum())

        for arm in ARMS:
            values = scores[arm][positions]
            records.append(
                {
                    "benchmark": spec.benchmark,
                    "label": spec.label,
                    "seed": spec.seed,
                    "stream": stream.name,
                    "leaf": evaluation.leaf,
                    "group": evaluation.group,
                    "arm": arm,
                    "AUROC": 100.0 * geometry.auroc(values, eval_is_id),
                    "FPR@95": 100.0 * geometry.fpr95(values, ~eval_is_id),
                    "n": n_eval,
                    "n_ood": n_ood,
                    "ood_ratio": n_ood / max(n_eval, 1),
                    "batches_scored": state["scored"],
                    "cos_ceiling": float(np.mean(cos_ceiling[arm])) if cos_ceiling[arm] else math.nan,
                    "cos_camp": float(np.mean(cos_camp[arm])) if cos_camp[arm] else math.nan,
                    "runtime_max_abs_diff": state["runtime_diff"],
                    "replay_max_abs_diff": replay_diff,
                },
            )

    return pd.DataFrame.from_records(records, columns=list(STREAM_COLUMNS))


def cache_path(
    cache: Path,
    spec: RunSpec,
    stream: str,
) -> Path:

    run_dir = spec.dir()
    return Path(cache) / f"streams_v{CACHE_VERSION}__{run_dir.parent.name}__{run_dir.name}__{stream}.csv"


def wanted_streams(
    ctx: Context,
    benchmark: str,
) -> list[str]:

    leaves = [leaf for members in layout.LEAVES[benchmark].values() for leaf in members]
    selected = ctx.option_list("streams", leaves)
    unknown = [leaf for leaf in selected if leaf not in leaves]
    if unknown:
        raise KeyError(f"unknown stream(s) {unknown} for {benchmark}; one of {leaves}")

    return selected


def read_cached(
    path: Path,
) -> pd.DataFrame:

    frame = pd.read_csv(path)
    missing = [column for column in STREAM_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is not a stream table of this item: it lacks {missing}.")

    frame["label"] = frame["label"].astype(object).where(frame["label"].notna(), None)
    return frame


def run_tables(
    spec: RunSpec,
    ctx: Context,
) -> list[pd.DataFrame]:

    streams = wanted_streams(ctx=ctx, benchmark=spec.benchmark)
    cached = {name: cache_path(cache=ctx.cache, spec=spec, stream=name) for name in streams}
    frames = [read_cached(path) for name, path in cached.items() if path.is_file()]
    todo = [name for name, path in cached.items() if not path.is_file()]
    if not todo:
        return frames

    from scripts.experiment.lib import replay

    run = replay.RecordedRun(spec.dir())
    features = replay.features_of(run, ctx.device)
    args = policy_args(policy=run.policy(), class_num=features.class_num)

    for stream in replay.streams(run, features, leaves=todo):
        print(f"[replay] {layout.relative(spec.dir())} stream {stream.name} ({len(stream)} samples)", flush=True)
        frame = stream_table(spec=spec, run=run, features=features, stream=stream, args=args)
        frame.to_csv(cached[stream.name], index=False, encoding="utf-8")
        frames.append(frame)

    return frames


def with_groups(
    streams: pd.DataFrame,
) -> pd.DataFrame:

    leaves = streams.copy()
    leaves["row"] = leaves["leaf"]
    leaves["kind"] = "leaf"
    keys = ["benchmark", "label", "seed", "arm", "group"]
    grouped = streams.groupby(keys, sort=False, dropna=False)[list(METRICS) + ["cos_ceiling", "cos_camp"]].mean().reset_index()
    grouped["row"] = grouped["group"]
    grouped["kind"] = "group"
    columns = ["benchmark", "label", "seed", "arm", "group", "row", "kind", *METRICS, "cos_ceiling", "cos_camp"]
    return pd.concat([leaves[columns], grouped[columns]], ignore_index=True)


def summarize(
    streams: pd.DataFrame,
) -> pd.DataFrame:

    rows = with_groups(streams)
    records: list[dict[str, Any]] = []
    keys = ["benchmark", "label", "arm", "row", "kind"]

    for (benchmark, label, arm, row, kind), part in rows.groupby(keys, sort=False, dropna=False):
        seeds = sorted(int(s) for s in part["seed"])

        for metric in (*METRICS, "cos_ceiling", "cos_camp"):
            values = [float(v) for v in part[metric] if np.isfinite(v)]
            records.append(
                {
                    "benchmark": benchmark,
                    "label": None if pd.isna(label) else label,
                    "arm": arm,
                    "row": row,
                    "kind": kind,
                    "metric": metric,
                    "mean": sum(values) / len(values) if values else math.nan,
                    "std": tables.sample_std(values),
                    "n_seeds": len(values),
                    "seeds": ",".join(str(s) for s in seeds),
                },
            )

    frame = pd.DataFrame.from_records(records, columns=list(SUMMARY_COLUMNS))
    order = {arm: i for i, arm in enumerate(ARMS)}
    frame["_order"] = frame["arm"].map(lambda a: order.get(a, len(order)))
    return frame.sort_values(["benchmark", "label", "_order"], kind="stable").drop(columns="_order").reset_index(drop=True)


def row_order(
    benchmark: str,
    present: Sequence[str],
) -> list[str]:

    ordered: list[str] = []

    for group, leaves in layout.LEAVES[benchmark].items():
        ordered += [leaf for leaf in leaves if leaf in present]
        if group in present:
            ordered.append(group)

    return ordered + [row for row in present if row not in ordered]


def _section_title(
    benchmark: str,
    label: str | None,
) -> str:

    name = layout.BENCHMARK_NAMES.get(benchmark, benchmark)
    return name if label is None else f"{name}, arm {label}"


def arm_table(
    part: pd.DataFrame,
    benchmark: str,
    metrics: Sequence[str],
    decimals: int = 2,
    mark_best: bool = True,
) -> list[str]:

    rows = row_order(benchmark, list(dict.fromkeys(part["row"])))
    arms = [arm for arm in ARMS if arm in set(part["arm"])]
    columns = [(row, metric) for row in rows for metric in metrics]
    headers = ["Arm", "seeds"] + [
        f"{tables.row_name(benchmark=benchmark, row=row)} {metric}{tables.ARROWS.get(metric, '')}" for row, metric in columns
    ]
    cells: dict[tuple[str, str, str], tuple[float, float]] = {}
    seeds: dict[str, str] = {}

    for record in part.itertuples(index=False):
        cells[(record.arm, record.row, record.metric)] = (float(record.mean), float(record.std))
        seeds[record.arm] = str(record.seeds)

    marks: list[dict[int, str]] = []

    for row, metric in columns:
        values = [cells.get((arm, row, metric), (math.nan, math.nan))[0] for arm in arms]
        lower = metric in tables.LOWER_IS_BETTER
        marks.append(tables.best_marks(values=values, lower=lower) if mark_best and any(np.isfinite(values)) else {})

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]

    for i, arm in enumerate(arms):
        text = [ARM_NAMES.get(arm, arm), seeds.get(arm, tables.MISSING_CELL)]

        for j, (row, metric) in enumerate(columns):
            mean, std = cells.get((arm, row, metric), (math.nan, math.nan))
            text.append(tables.decorate(text=tables.cell(mean=mean, std=std, decimals=decimals), mark=marks[j].get(i)))

        lines.append("| " + " | ".join(text) + " |")

    return lines


def markdown(
    summary: pd.DataFrame,
) -> str:

    if summary.empty:
        return "no stream replayed\n"

    lines: list[str] = []
    sections = list(dict.fromkeys(zip(summary["benchmark"], summary["label"])))

    for benchmark, label in sections:
        label_value = None if pd.isna(label) else label
        mask = (summary["benchmark"] == benchmark) & (summary["label"].isna() if label_value is None else summary["label"] == label_value)
        part = summary[mask]
        lines += [f"### {_section_title(benchmark, label_value)}", ""]
        lines += [
            "Whole-stream detection with the window equation's ID mean / covariance replaced by oracles. The stream mean "
            "stays the window's plain mean except in the ceiling row, which uses the population OOD mean. Warm-up "
            "batches keep the runtime's score; every arm standardizes against the same window rows.",
            "",
        ]
        lines += arm_table(part[part["metric"].isin(METRICS)], benchmark=benchmark, metrics=METRICS)
        lines += ["", "Mean cosine of each arm's direction to the rank-one ceiling direction over the scored batches "
            "(`cos_ceiling`) and to CAMP's own direction (`cos_camp`).", ""]
        lines += arm_table(
            part[part["metric"].isin(("cos_ceiling", "cos_camp"))],
            benchmark=benchmark,
            metrics=("cos_ceiling", "cos_camp"),
            decimals=3,
            mark_best=False,
        )
        lines.append("")

    lines += [
        "Arms: `camp` is CAMP's own window estimate; `win` replaces μ_ID / Σ_ID with the window's true ID rows (true "
        "classes, weight 1: no contamination, finite window); `full` with every ID row of the stream (the population "
        "ID statistics); the ceiling fixes the direction to (Σ_full + εI)⁻¹(μ_OOD − μ_ID,full). FPR@95 follows the "
        "OpenOOD evaluator.",
        "",
    ]

    return "\n".join(lines)


def replay_check(
    streams: pd.DataFrame,
    specs: Sequence[RunSpec],
) -> pd.DataFrame:

    recorded: dict[tuple[str, str | None, int], dict[str, dict[str, float]]] = {}

    for spec in specs:
        if spec.recorded():
            recorded[(spec.benchmark, spec.label, spec.seed)] = tables.read_ood_csv(spec.ood_csv())

    part = streams[streams["arm"] == "camp"]
    records: list[dict[str, Any]] = []

    for record in part.itertuples(index=False):
        label = None if pd.isna(record.label) else record.label
        table = recorded.get((record.benchmark, label, int(record.seed)), {})
        row = table.get(record.leaf, {})
        records.append(
            {
                "benchmark": record.benchmark,
                "label": label,
                "seed": int(record.seed),
                "leaf": record.leaf,
                "AUROC": record.AUROC,
                "AUROC_recorded": row.get("AUROC", math.nan),
                "FPR@95": record[STREAM_COLUMNS.index("FPR@95")],
                "FPR@95_recorded": row.get("FPR@95", math.nan),
                "runtime_max_abs_diff": record.runtime_max_abs_diff,
                "replay_max_abs_diff": record.replay_max_abs_diff,
                "within_tolerance": record.replay_max_abs_diff <= REPLAY_TOLERANCE,
            },
        )

    return pd.DataFrame.from_records(records)


def analyze(
    ctx: Context,
) -> list[Path]:

    from scripts.experiment.id_oracle import runs as item_runs

    specs = item_runs(ctx)
    recorded = [spec for spec in specs if spec.recorded()]
    missing = [layout.relative(spec.dir()) for spec in specs if not spec.recorded()]
    ctx.notes["missing_runs"] = missing

    if not recorded:
        raise FileNotFoundError("no recorded run for " + ", ".join(missing))

    frames: list[pd.DataFrame] = []

    for spec in recorded:
        frames += run_tables(spec=spec, ctx=ctx)

    streams = pd.concat(frames, ignore_index=True)
    summary = summarize(streams)
    check = replay_check(streams, specs)
    ctx.notes["replay_max_abs_diff"] = float(check["replay_max_abs_diff"].max()) if not check.empty else None
    ctx.notes["runtime_max_abs_diff"] = float(check["runtime_max_abs_diff"].max()) if not check.empty else None
    ctx.notes["replay_tolerance"] = REPLAY_TOLERANCE
    ctx.notes["arms"] = list(ARMS)
    written = [
        tables.write_csv(streams, ctx.tables / "streams.csv", float_format="%.6f"),
        tables.write_csv(summary, ctx.tables / "summary.csv", float_format="%.6f"),
        tables.write_markdown(markdown(summary), ctx.tables / "id_oracle.md"),
        tables.write_csv(check, ctx.tables / "replay_check.csv", float_format="%.6f"),
    ]

    if missing:
        print("missing runs (not in the tables): " + ", ".join(missing))

    return written


__all__ = [
    "ARMS",
    "ARM_NAMES",
    "ARM_SOURCES",
    "CACHE_VERSION",
    "METRICS",
    "REPLAY_TOLERANCE",
    "STREAM_COLUMNS",
    "SUMMARY_COLUMNS",
    "analyze",
    "arm_table",
    "cache_path",
    "direction",
    "markdown",
    "policy_args",
    "replay_check",
    "row_order",
    "run_tables",
    "stream_table",
    "summarize",
    "wanted_streams",
    "with_groups",
]
