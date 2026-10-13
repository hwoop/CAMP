from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from scripts.experiment.lib import config_io, geometry, layout, tables
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.runs import RunSpec

if TYPE_CHECKING:
    import torch

    from openood.postprocessors.camp.config import CAMPPolicy
    from scripts.experiment.lib.replay import RecordedRun, ReplayBatch, RunFeatures, Stream

ESTIMATORS: tuple[str, ...] = ("camp", "lower_bound", "tanl_memory")
ESTIMATOR_NAMES: dict[str, str] = {
    "camp": "CAMP (Proposed)",
    "lower_bound": "Window oracle",
    "tanl_memory": "TANL ID memory",
}
METRICS: tuple[str, ...] = ("mu_mahalanobis", "sigma_whitened_rel_error", "cos_to_oracle")
METRIC_NAMES: dict[str, str] = {
    "mu_mahalanobis": "ID mean error (Mahalanobis)",
    "sigma_whitened_rel_error": "ID covariance error (whitened, relative)",
    "cos_to_oracle": "Cosine similarity to the oracle direction",
}
POINTS: tuple[int, ...] = (512, 8192)
LAST_POINT = "last"
DEFAULT_GRID_STEP = 256
DEFAULT_TANL_MEMORY = 300
TRAJECTORY_VERSION = 3
TRAJECTORY_COLUMNS: tuple[str, ...] = (
    "benchmark",
    "seed",
    "stream",
    "group",
    "leaf",
    "batch",
    "order_idx",
    "window_count",
    "estimator",
    "n_rows",
    "k",
    *METRICS,
    "replay_max_abs_diff",
)
CURVE_KEYS: tuple[str, ...] = ("benchmark", "estimator", "order_idx")
REPLAY_TOLERANCE = 1e-3


@dataclass(frozen=True, eq=False)
class Oracle:

    mu: torch.Tensor
    sigma: torch.Tensor
    operator: torch.Tensor
    whitener: torch.Tensor
    whitened_norm: float
    eps: float

    def direction(
        self,
        mu_stream: torch.Tensor,
    ) -> torch.Tensor:

        import torch

        delta = mu_stream.to(self.mu.device, torch.float64) - self.mu
        r = torch.linalg.solve(self.operator, delta)
        return r / r.norm().clamp_min(1e-12)


    def compare(
        self,
        estimate: geometry.Geometry,
    ) -> dict[str, float]:

        import torch

        sigma = estimate.sigma.to(self.whitener.device, torch.float64)
        whitened = self.whitener @ (sigma - self.sigma) @ self.whitener
        return {
            "mu_mahalanobis": geometry.mahalanobis(estimate.mu_id, self.mu, self.operator),
            "sigma_whitened_rel_error": float(whitened.norm()) / max(self.whitened_norm, 1e-12),
            "cos_to_oracle": geometry.cosine(estimate.u, self.direction(estimate.mu_stream)),
        }


def oracle_of(
    features_id: torch.Tensor,
    labels_id: torch.Tensor,
    *,
    eps: float,
    class_num: int,
    covariance: str = "full",
    id_reference: str = "classwise",
) -> Oracle:

    import torch

    zero = torch.zeros(int(features_id.shape[1]), dtype=torch.float64, device=features_id.device)
    base = geometry.oracle_geometry(
        features_id,
        labels_id,
        mu_stream=zero,
        eps=eps,
        class_num=class_num,
        covariance=covariance,
        id_reference=id_reference,
    )

    whitener = geometry.inverse_sqrt(base.operator)
    return Oracle(
        mu=base.mu_id,
        sigma=base.sigma,
        operator=base.operator,
        whitener=whitener,
        whitened_norm=float((whitener @ base.sigma @ whitener).norm()),
        eps=float(eps),
    )


def nan_metrics() -> dict[str, float]:

    table = {
        metric: math.nan
        for metric in METRICS
    }

    return table


def tanl_memory_positions(
    id_admit: np.ndarray,
    start: int,
    memory_size: int = DEFAULT_TANL_MEMORY,
) -> np.ndarray:

    admitted = np.flatnonzero(np.asarray(id_admit[:start], dtype=bool))
    return admitted[-int(memory_size):] if memory_size > 0 else admitted[:0]


def estimate_rows(
    rows: torch.Tensor,
    weights: torch.Tensor,
    classes: torch.Tensor,
    *,
    mu_stream: torch.Tensor,
    policy_args: dict[str, Any],
    weighting: str,
) -> geometry.Geometry | None:

    if int(rows.shape[0]) == 0:
        return None

    try:
        return geometry.camp_geometry(rows, weights, classes, mu_stream=mu_stream, weighting=weighting, **policy_args)
    except ValueError:
        return None


def batch_estimates(
    batch: ReplayBatch,
    *,
    labels: np.ndarray,
    pseudo: torch.Tensor,
    features: torch.Tensor,
    id_admit: np.ndarray,
    oracle: Oracle,
    policy_args: dict[str, Any],
    memory_size: int = DEFAULT_TANL_MEMORY,
) -> list[dict[str, Any]]:

    import torch

    if batch.geometry is None or batch.rows is None:
        raise ValueError(f"batch {batch.index} was scored during the warm-up: no window geometry to compare.")

    camp = batch.geometry
    mu_stream = camp.mu_stream
    window_positions = np.arange(batch.window.start, batch.window.stop)
    window_labels = torch.as_tensor(labels[window_positions], device=features.device)
    id_mask = window_labels >= 0
    window_rows = batch.rows[0].to(features.device)
    lower = estimate_rows(
        rows=window_rows[id_mask],
        weights=torch.ones(
            int(id_mask.sum()),
            dtype=torch.float64,
            device=features.device,
        ),
        classes=window_labels[id_mask],
        mu_stream=mu_stream,
        policy_args=policy_args,
        weighting="uniform",
    )

    memory = tanl_memory_positions(id_admit=id_admit, start=batch.start, memory_size=memory_size)
    memory_index = torch.as_tensor(memory, device=features.device, dtype=torch.long)
    tanl = estimate_rows(
        rows=features[memory_index],
        weights=torch.ones(len(memory), dtype=torch.float64, device=features.device),
        classes=pseudo[memory_index],
        mu_stream=mu_stream,
        policy_args={**policy_args, "id_reference": "global"},
        weighting="uniform",
    )

    estimates: list[tuple[str, geometry.Geometry | None, int]] = [
        ("camp", camp, int(batch.rows[0].shape[0])),
        ("lower_bound", lower, int(id_mask.sum())),
        ("tanl_memory", tanl, int(len(memory))),
    ]
    rows: list[dict[str, Any]] = []

    for name, estimate, n_rows in estimates:
        metrics = oracle.compare(estimate) if estimate is not None else nan_metrics()
        rows.append(
            {
                "batch": batch.index,
                "order_idx": batch.start,
                "window_count": batch.window.stop - batch.window.start,
                "estimator": name,
                "n_rows": n_rows,
                "k": 0 if estimate is None else estimate.k,
                **metrics,
            },
        )

    return rows


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


def tanl_memory_size(
    root: Path | None = None,
) -> int:

    from scripts.experiment.lib.replay import default_config_path

    defaults = config_io.load_yaml(default_config_path(method="tanl", root=root))
    args = dict((defaults.get("postprocessor") or {}).get("postprocessor_args") or {})
    return int(args.get("memleng", DEFAULT_TANL_MEMORY))


def trajectory_path(
    cache: Path,
    spec: RunSpec,
    stream: str,
) -> Path:

    run_dir = spec.dir()
    return Path(cache) / f"trajectory_v{TRAJECTORY_VERSION}__{run_dir.parent.name}__{run_dir.name}__{stream}.csv"


def stream_trajectory(
    spec: RunSpec,
    run: RecordedRun,
    features: RunFeatures,
    stream: Stream,
    *,
    trace_dir: Path,
    memory_size: int,
) -> pd.DataFrame:

    import torch

    from scripts.experiment.lib import replay

    replay.verify_stream(stream)
    policy = run.policy()
    args = policy_args(policy=policy, class_num=features.class_num)
    device = features.device
    feats = stream.features.to(device)
    labels = np.asarray(stream.labels, dtype=np.int64)

    with torch.no_grad():
        _, pseudo, _ = geometry.zero_shot(feats, features.prototypes.to(device), features.logit_scale.to(device))
        id_index = torch.as_tensor(np.flatnonzero(labels >= 0), device=device, dtype=torch.long)
        oracle = oracle_of(
            features_id=feats[id_index],
            labels_id=torch.as_tensor(labels[labels >= 0], device=device),
            eps=args["eps"],
            class_num=args["class_num"],
            covariance=args["covariance"],
            id_reference=args["id_reference"],
        )

    trace = replay.trace_tanl(features, stream, cache_dir=trace_dir)
    id_admit = np.asarray(trace.id_admit, dtype=bool)
    records: list[dict[str, Any]] = []

    def every_batch(
        batch: replay.ReplayBatch,
    ) -> None:

        if batch.geometry is None:
            return

        with torch.no_grad():
            records.extend(
                batch_estimates(
                    batch,
                    labels=labels,
                    pseudo=pseudo,
                    features=feats,
                    id_admit=id_admit,
                    oracle=oracle,
                    policy_args=args,
                    memory_size=memory_size,
                ),
            )

    result = replay.replay_camp(stream, features, every_batch=every_batch)
    difference = replay.check_against_recorded(stream, result.score, tolerance=REPLAY_TOLERANCE)
    frame = pd.DataFrame.from_records(records)
    frame.insert(0, "benchmark", spec.benchmark)
    frame.insert(1, "seed", spec.seed)
    frame.insert(2, "stream", stream.name)
    frame.insert(3, "group", stream.group)
    frame.insert(4, "leaf", stream.leaf)
    frame["replay_max_abs_diff"] = difference
    return frame[list(TRAJECTORY_COLUMNS)]


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


def read_trajectory(
    path: Path,
) -> pd.DataFrame:

    frame = pd.read_csv(path)
    missing = [column for column in TRAJECTORY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is not a trajectory table: it lacks {missing}.")

    return frame


def run_trajectories(
    spec: RunSpec,
    ctx: Context,
    *,
    trace_dir: Path,
    memory_size: int,
) -> list[pd.DataFrame]:

    streams = wanted_streams(ctx=ctx, benchmark=spec.benchmark)
    cached = {name: trajectory_path(cache=ctx.cache, spec=spec, stream=name) for name in streams}
    frames = [read_trajectory(path) for name, path in cached.items() if path.is_file()]
    todo = [name for name, path in cached.items() if not path.is_file()]
    if not todo:
        return frames

    from scripts.experiment.lib import replay

    run = replay.RecordedRun(spec.dir())
    features = replay.features_of(run, ctx.device)
    for stream in replay.streams(run, features, leaves=todo):
        print(f"[replay] {layout.relative(spec.dir())} stream {stream.name} ({len(stream)} samples)")
        frame = stream_trajectory(spec=spec, run=run, features=features, stream=stream, trace_dir=trace_dir, memory_size=memory_size)
        frame.to_csv(cached[stream.name], index=False, encoding="utf-8")
        frames.append(frame)

    return frames


def interpolate(
    order_idx: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:

    finite = np.isfinite(values)
    x = np.asarray(order_idx, dtype=np.float64)[finite]
    y = np.asarray(values, dtype=np.float64)[finite]
    out = np.full(shape=grid.shape, fill_value=np.nan)

    if x.size == 0:
        return out

    inside = (grid >= x.min()) & (grid <= x.max())
    out[inside] = np.interp(x=grid[inside], xp=x, fp=y)
    return out


def common_grid(
    trajectories: pd.DataFrame,
    step: int = DEFAULT_GRID_STEP,
) -> np.ndarray:

    first = int(trajectories["order_idx"].min())
    last = int(trajectories["order_idx"].max())
    return np.arange(first, last + 1, int(step), dtype=np.int64)


def seed_curves(
    trajectories: pd.DataFrame,
    grid: np.ndarray,
) -> pd.DataFrame:

    records: list[dict[str, Any]] = []
    keys = ["benchmark", "seed", "estimator"]
    for (benchmark, seed, estimator), part in trajectories.groupby(keys, sort=True):
        stacks = {metric: [] for metric in METRICS}

        for _, stream_part in part.groupby("stream", sort=True):
            ordered = stream_part.sort_values("order_idx")

            for metric in METRICS:
                stacks[metric].append(interpolate(order_idx=ordered["order_idx"].to_numpy(), values=ordered[metric].to_numpy(), grid=grid))

        stacked = {metric: np.vstack(stack) for metric, stack in stacks.items()}
        n_streams = np.isfinite(stacked[METRICS[0]]).sum(axis=0)

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            means = {metric: np.nanmean(values, axis=0) for metric, values in stacked.items()}

        for i, point in enumerate(grid):
            if n_streams[i] == 0:
                continue

            records.append(
                {
                    "benchmark": benchmark,
                    "seed": int(seed),
                    "estimator": estimator,
                    "order_idx": int(point),
                    "n_streams": int(n_streams[i]),
                    **{metric: float(means[metric][i]) for metric in METRICS},
                },
            )

    return pd.DataFrame.from_records(
        records,
        columns=[
            "benchmark",
            "seed",
            "estimator",
            "order_idx",
            "n_streams",
            *METRICS,
        ],
    )


def curves_over_seeds(
    seeds: pd.DataFrame,
) -> pd.DataFrame:

    records: list[dict[str, Any]] = []

    for (benchmark, estimator, point), part in seeds.groupby(list(CURVE_KEYS), sort=True):
        record: dict[str, Any] = {
            "benchmark": benchmark,
            "estimator": estimator,
            "order_idx": int(point),
            "n_seeds": int(len(part)),
            "seeds": ",".join(str(s) for s in sorted(part["seed"])),
            "n_streams": int(part["n_streams"].min()),
        }

        for metric in METRICS:
            values = [float(v) for v in part[metric] if np.isfinite(v)]
            record[f"{metric}_mean"] = sum(values) / len(values) if values else math.nan
            record[f"{metric}_std"] = tables.sample_std(values)

        records.append(record)

    columns = [
        *CURVE_KEYS,
        "n_seeds",
        "seeds",
        "n_streams",
        *(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")),
    ]
    frame = pd.DataFrame.from_records(records, columns=columns)
    order = {name: i for i, name in enumerate(ESTIMATORS)}
    frame["_order"] = frame["estimator"].map(lambda e: order.get(e, len(order)))
    return frame.sort_values(["benchmark", "_order", "order_idx"]).drop(columns="_order").reset_index(drop=True)


def last_common_point(
    curves: pd.DataFrame,
    benchmark: str,
) -> int | None:

    part = curves[curves["benchmark"] == benchmark]
    if part.empty:
        return None

    full = part[(part["n_streams"] == part["n_streams"].max()) & (part["n_seeds"] == part["n_seeds"].max())]
    return None if full.empty else int(full["order_idx"].max())


def points_table(
    curves: pd.DataFrame,
    points: Sequence[int] = POINTS,
) -> pd.DataFrame:

    records: list[dict[str, Any]] = []

    for benchmark in dict.fromkeys(curves["benchmark"]):
        named: list[tuple[str, int]] = [(str(p), int(p)) for p in points]
        last = last_common_point(curves=curves, benchmark=benchmark)
        if last is not None:
            named.append((LAST_POINT, last))

        part = curves[curves["benchmark"] == benchmark]

        for name, point in named:
            rows = part[part["order_idx"] == point]
            for _, row in rows.iterrows():
                for metric in METRICS:
                    records.append(
                        {
                            "benchmark": benchmark,
                            "estimator": row["estimator"],
                            "point": name,
                            "order_idx": point,
                            "metric": metric,
                            "mean": float(row[f"{metric}_mean"]),
                            "std": float(row[f"{metric}_std"]),
                            "n_seeds": int(row["n_seeds"]),
                            "seeds": row["seeds"],
                            "n_streams": int(row["n_streams"]),
                        },
                    )

    return pd.DataFrame.from_records(
        records,
        columns=[
            "benchmark",
            "estimator",
            "point",
            "order_idx",
            "metric",
            "mean",
            "std",
            "n_seeds",
            "seeds",
            "n_streams",
        ],
    )


def points_markdown(
    points: pd.DataFrame,
    decimals: int = 3,
) -> str:

    if points.empty:
        return "no trajectory recorded\n"

    lines: list[str] = []

    for benchmark in dict.fromkeys(points["benchmark"]):
        part = points[points["benchmark"] == benchmark]
        named = list(dict.fromkeys(zip(part["point"], part["order_idx"])))
        headers = ["Estimate", "seeds"] + [f"{METRIC_NAMES[m]} @ {name} ({idx:,})" for m in METRICS
            for name, idx in named]
        lines += [f"### {layout.BENCHMARK_NAMES.get(benchmark, benchmark)}", "",
            "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]

        for estimator in [e for e in ESTIMATORS if e in set(part["estimator"])]:
            rows = part[part["estimator"] == estimator]
            cells = [ESTIMATOR_NAMES.get(estimator, estimator), str(rows["seeds"].iloc[0])]

            for metric in METRICS:
                for name, _ in named:
                    hit = rows[(rows["metric"] == metric) & (rows["point"] == name)]
                    cells.append(
                        tables.MISSING_CELL if hit.empty else
                        tables.cell(float(hit["mean"].iloc[0]), float(hit["std"].iloc[0]), decimals),
                    )

            lines.append("| " + " | ".join(cells) + " |")

        lines.append("")

    return "\n".join(lines)


def replay_check(
    trajectories: pd.DataFrame,
) -> pd.DataFrame:

    keys = ["benchmark", "seed", "stream"]
    part = trajectories[trajectories["estimator"] == "camp"]
    grouped = part.groupby(keys, sort=True).agg(
        batches=("batch", "count"),
        last_order_idx=("order_idx", "max"),
        max_abs_diff=("replay_max_abs_diff", "max"),
    ).reset_index()

    grouped["within_tolerance"] = grouped["max_abs_diff"] <= REPLAY_TOLERANCE
    return grouped


def reduce(
    trajectories: pd.DataFrame,
    grid_step: int = DEFAULT_GRID_STEP,
) -> tuple[pd.DataFrame, pd.DataFrame,
    pd.DataFrame]:

    grid = common_grid(trajectories=trajectories, step=grid_step)
    seeds = seed_curves(trajectories=trajectories, grid=grid)
    curves = curves_over_seeds(seeds)
    return curves, seeds, points_table(curves)


def analyze(
    ctx: Context,
) -> list[Path]:

    from scripts.experiment.id_approximation import runs as item_runs

    specs = item_runs(ctx)
    recorded = [spec for spec in specs if spec.recorded()]
    missing = [layout.relative(spec.dir()) for spec in specs if not spec.recorded()]
    ctx.notes["missing_runs"] = missing

    if not recorded:
        raise FileNotFoundError("no recorded run for " + ", ".join(missing))

    grid_step = int(ctx.option("grid", str(DEFAULT_GRID_STEP)) or DEFAULT_GRID_STEP)
    trace_option = ctx.option("trace_dir")
    trace_dir = ctx.cache if trace_option is None else Path(trace_option)
    memory_size = tanl_memory_size(ctx.root)
    frames: list[pd.DataFrame] = []

    for spec in recorded:
        frames += run_trajectories(spec=spec, ctx=ctx, trace_dir=trace_dir, memory_size=memory_size)

    trajectories = pd.concat(frames, ignore_index=True)
    curves, seeds, points = reduce(trajectories=trajectories, grid_step=grid_step)
    check = replay_check(trajectories)
    ctx.notes["replay_max_abs_diff"] = float(check["max_abs_diff"].max()) if not check.empty else None
    ctx.notes["replay_tolerance"] = REPLAY_TOLERANCE
    ctx.notes["streams"] = [f"{b}/s{s}/{st}" for b, s, st in zip(check["benchmark"], check["seed"], check["stream"])]
    ctx.notes["tanl_memory_size"] = memory_size
    ctx.notes["grid_step"] = grid_step
    ctx.notes["trace_dir"] = layout.relative(trace_dir)
    written = [
        tables.write_csv(curves, ctx.tables / "curves.csv", float_format="%.6f"),
        tables.write_csv(seeds, ctx.tables / "curves_seeds.csv", float_format="%.6f"),
        tables.write_csv(points, ctx.tables / "points.csv", float_format="%.6f"),
        tables.write_markdown(points_markdown(points), ctx.tables / "points.md"),
        tables.write_csv(check, ctx.tables / "replay_check.csv", float_format="%.3e"),
    ]

    if missing:
        print("missing runs (not in the tables): " + ", ".join(missing))

    return written


__all__ = [
    "DEFAULT_GRID_STEP",
    "DEFAULT_TANL_MEMORY",
    "ESTIMATORS",
    "ESTIMATOR_NAMES",
    "METRICS",
    "METRIC_NAMES",
    "Oracle",
    "POINTS",
    "TRAJECTORY_COLUMNS",
    "analyze",
    "batch_estimates",
    "common_grid",
    "curves_over_seeds",
    "interpolate",
    "oracle_of",
    "points_markdown",
    "points_table",
    "reduce",
    "replay_check",
    "run_trajectories",
    "seed_curves",
    "stream_trajectory",
    "tanl_memory_positions",
    "tanl_memory_size",
    "trajectory_path",
    "wanted_streams",
]
