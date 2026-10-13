from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment.lib import layout, tables
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.runs import RunSpec

GRID: np.ndarray = np.linspace(start=-6.0, stop=16.0, num=881)
DIRECTIONS: tuple[str, ...] = ("before", "after")
DIRECTION_NAMES: dict[str, str] = {
    "before": "Before adjustment",
    "after": "After adjustment",
}
GAPS_COLUMNS: tuple[str, ...] = (
    "bench",
    "seed",
    "leaf",
    "group",
    "direction",
    "gap_id_std",
    "auroc",
    "delta_norm",
    "replay_max_abs_diff",
)
DENSITY_COLUMNS: tuple[str, ...] = ("bench", "leaf", "group", "direction", "z", "id", "ood", "n_seeds", "seeds")
GAPS_CSV = "gaps.csv"
GAPS_MD = "gaps.md"
DENSITIES_CSV = "densities.csv"
DENSITIES_PER_SEED_CSV = "densities_per_seed.csv"
REPLAY_TOLERANCE = 1e-3


@dataclass(frozen=True, eq=False)
class StreamProjection:

    leaf: str
    group: str
    is_ood: np.ndarray
    before: np.ndarray
    after: np.ndarray
    replay_max_abs_diff: float
    delta_norm: float

    def projection(
        self,
        direction: str,
    ) -> np.ndarray:

        if direction not in DIRECTIONS:
            raise KeyError(f"direction must be one of {list(DIRECTIONS)}, got {direction!r}")

        return self.before if direction == "before" else self.after


Replayer = Callable[[RunSpec, Context], list[StreamProjection]]


def replay_run(
    spec: RunSpec,
    ctx: Context,
) -> list[StreamProjection]:

    from scripts.experiment.lib import geometry, replay

    run = replay.RecordedRun(spec.dir())
    features = replay.features_of(run, ctx.device)
    results: list[StreamProjection] = []

    for stream in replay.streams(run, features):
        started = time.perf_counter()
        replay.verify_stream(stream)
        replayed = replay.replay_camp(stream, features)
        if replayed.final is None:
            raise RuntimeError(
                f"{layout.relative(run.dir)} / {stream.name}: the stream never left CAMP's warm-up, "
                "so no window geometry exists.",
            )

        diff = replay.check_against_recorded(stream, replayed.score, tolerance=REPLAY_TOLERANCE)
        rows = stream.features.to(replayed.final.u.device)
        before = geometry.project(rows, replayed.final.e).cpu().numpy()
        after = geometry.project(rows, replayed.final.u).cpu().numpy()
        results.append(
            StreamProjection(
                leaf=str(stream.leaf),
                group=str(stream.group),
                is_ood=stream.is_ood.copy(),
                before=before,
                after=after,
                replay_max_abs_diff=float(diff),
                delta_norm=float(replayed.final.delta.norm()),
            ),
        )

        print(
            f"[replay] {layout.relative(run.dir)} {stream.name}: {len(stream)} samples, max abs diff to recorded "
            f"conf {diff:.2e}, {time.perf_counter() - started:.1f} s",
        )

    return results


def cache_path(
    ctx: Context,
    spec: RunSpec,
    leaf: str,
) -> Path:
    return ctx.cache / f"projections__{spec.dir().parent.name}__{spec.dir().name}__{leaf}.npz"


def save_projection(
    path: Path,
    projection: StreamProjection,
) -> Path:

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        leaf=np.asarray(projection.leaf),
        group=np.asarray(projection.group),
        is_ood=projection.is_ood.astype(bool),
        before=projection.before.astype(np.float64),
        after=projection.after.astype(np.float64),
        replay_max_abs_diff=np.asarray(projection.replay_max_abs_diff, dtype=np.float64),
        delta_norm=np.asarray(projection.delta_norm, dtype=np.float64),
    )

    return path


def load_projection(
    path: Path,
) -> StreamProjection:

    with np.load(path) as data:
        return StreamProjection(
            leaf=str(data["leaf"]),
            group=str(data["group"]),
            is_ood=data["is_ood"].astype(bool),
            before=data["before"].astype(np.float64),
            after=data["after"].astype(np.float64),
            replay_max_abs_diff=float(data["replay_max_abs_diff"]),
            delta_norm=float(data["delta_norm"]),
        )


def expected_leaves(
    benchmark: str,
) -> list[tuple[str, str]]:

    rows = [
        (group, leaf)
        for group, leaves in layout.LEAVES[benchmark].items()
        for leaf in leaves
    ]

    return rows


def projections_of(
    spec: RunSpec,
    ctx: Context,
    replayer: Replayer | None = None,
) -> tuple[list[StreamProjection], bool]:

    paths = {leaf: cache_path(ctx, spec, leaf) for _, leaf in expected_leaves(spec.benchmark)}
    if all(path.is_file() for path in paths.values()):
        return [load_projection(paths[leaf]) for _, leaf in expected_leaves(spec.benchmark)], True

    projections = (replay_run if replayer is None else replayer)(spec, ctx)
    for projection in projections:
        save_projection(path=cache_path(ctx, spec, projection.leaf), projection=projection)

    return projections, False


def standardize(
    projection: np.ndarray,
    is_ood: np.ndarray,
) -> np.ndarray:

    identity = np.asarray(projection, dtype=np.float64)[~np.asarray(is_ood, dtype=bool)]
    if identity.size == 0:
        raise ValueError("a stream without ID samples cannot be standardized.")

    return (np.asarray(projection, dtype=np.float64) - identity.mean()) / max(float(identity.std()), 1e-12)


def gap(
    z: np.ndarray,
    is_ood: np.ndarray,
) -> float:

    is_ood = np.asarray(is_ood, dtype=bool)
    if not is_ood.any():
        raise ValueError("a stream without OOD samples has no gap.")

    return float(np.asarray(z, dtype=np.float64)[is_ood].mean())


def auroc_percent(
    projection: np.ndarray,
    is_ood: np.ndarray,
) -> float:

    from scripts.experiment.lib.geometry import auroc

    return 100.0 * auroc(score=-np.asarray(projection, dtype=np.float64), is_id=~np.asarray(is_ood, dtype=bool))


def kde(
    z: np.ndarray,
    grid: np.ndarray = GRID,
) -> np.ndarray:

    from scipy.stats import gaussian_kde

    values = np.asarray(z, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2 or float(values.std()) == 0.0:
        raise ValueError("a kernel density needs at least two distinct finite values.")

    return np.asarray(gaussian_kde(values, bw_method="silverman")(np.asarray(grid, dtype=np.float64)))


def stream_rows(
    spec: RunSpec,
    projection: StreamProjection,
) -> tuple[list[dict[str, object]], pd.DataFrame]:

    records: list[dict[str, object]] = []
    densities: list[pd.DataFrame] = []

    for direction in DIRECTIONS:
        values = projection.projection(direction)
        z = standardize(values, projection.is_ood)
        records.append(
            {
                "bench": spec.benchmark,
                "seed": spec.seed,
                "leaf": projection.leaf,
                "group": projection.group,
                "direction": direction,
                "gap_id_std": gap(z=z, is_ood=projection.is_ood),
                "auroc": auroc_percent(projection=values, is_ood=projection.is_ood),
                "delta_norm": projection.delta_norm,
                "replay_max_abs_diff": projection.replay_max_abs_diff,
            },
        )

        densities.append(
            pd.DataFrame(
                {
                    "bench": spec.benchmark,
                    "seed": spec.seed,
                    "leaf": projection.leaf,
                    "group": projection.group,
                    "direction": direction,
                    "z": GRID,
                    "id": kde(z[~projection.is_ood]),
                    "ood": kde(z[projection.is_ood]),
                },
            ),
        )

    return records, pd.concat(densities, ignore_index=True)


def average_densities(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:

    keys = ["bench", "leaf", "group", "direction", "z"]
    grouped = per_seed.groupby(keys, sort=False)
    mean = grouped[["id", "ood"]].mean().reset_index()
    seeds = grouped["seed"].agg(lambda s: ",".join(str(v) for v in sorted(set(int(x) for x in s)))).reset_index()
    counts = grouped["seed"].nunique().reset_index(name="n_seeds")
    frame = mean.merge(counts, on=keys).merge(seeds.rename(columns={"seed": "seeds"}), on=keys)
    return frame[list(DENSITY_COLUMNS)]


def _summary_cell(
    values: Sequence[float],
    decimals: int,
) -> str:

    series = [float(v) for v in values if not math.isnan(float(v))]
    if not series:
        return tables.MISSING_CELL

    return tables.cell(sum(series) / len(series), tables.sample_std(series), decimals)


def gaps_markdown(
    gaps: pd.DataFrame,
    missing: Sequence[RunSpec] = (),
) -> str:

    lines: list[str] = []

    for bench in dict.fromkeys(gaps["bench"]):
        part = gaps[gaps["bench"] == bench]
        absent = sorted({spec.seed for spec in missing if spec.benchmark == bench})
        lines += [f"### {layout.BENCHMARK_NAMES.get(bench, bench)}: gap between the OOD and ID means (ID standard "
            "deviations) and AUROC (%) along the mean difference before and after the covariance adjustment", "",
            "| OOD set | seeds | Gap before | Gap after | AUROC before↑ | AUROC after↑ |",
            "|---|---|---|---|---|---|"]
        rows: list[tuple[str, pd.DataFrame]] = []

        for group, leaves in layout.LEAVES[bench].items():
            for leaf in leaves:
                rows.append((tables.leaf_name(leaf), part[part["leaf"] == leaf]))

            members = part[part["group"] == group]
            if not members.empty:
                per_seed = members.groupby(["seed", "direction"], sort=False)[["gap_id_std", "auroc"]].mean()
                rows.append((f"{tables.leaf_name(group)} (mean)", per_seed.reset_index()))

        for name, frame in rows:
            if frame.empty:
                lines.append(f"| {name} | {tables.MISSING_CELL} | " + " | ".join([tables.MISSING_CELL] * 4) + " |")
                continue

            seeds = ",".join(str(s) for s in sorted(set(int(s) for s in frame["seed"])))
            seeds += f" (missing {','.join(map(str, absent))})" if absent else ""
            cells = [_summary_cell(values=frame[frame["direction"] == d]["gap_id_std"].tolist(), decimals=2) for d in DIRECTIONS]
            cells += [_summary_cell(values=frame[frame["direction"] == d]["auroc"].tolist(), decimals=2) for d in DIRECTIONS]
            lines.append(f"| {name} | {seeds} | " + " | ".join(cells) + " |")

        lines.append("")

    return "\n".join(lines)


def analyze(
    ctx: Context,
    specs: Iterable[RunSpec],
    replayer: Replayer | None = None,
) -> list[Path]:

    specs = list(specs)
    recorded = [spec for spec in specs if spec.recorded()]
    missing = [spec for spec in specs if not spec.recorded()]

    if not recorded:
        wanted = ", ".join(layout.relative(spec.dir()) for spec in specs) or "none requested"
        raise FileNotFoundError(f"no recorded run for {ctx.item.name}: {wanted}")

    records: list[dict[str, object]] = []
    per_seed: list[pd.DataFrame] = []
    diffs: dict[str, dict[str, float]] = {}
    from_cache: list[str] = []

    for spec in recorded:
        projections, cached = projections_of(spec=spec, ctx=ctx, replayer=replayer)
        if cached:
            from_cache.append(layout.relative(spec.dir()))

        for projection in projections:
            rows, densities = stream_rows(spec, projection)
            records += rows
            per_seed.append(densities)
            diffs.setdefault(layout.relative(spec.dir()), {})[projection.leaf] = projection.replay_max_abs_diff

    gaps = pd.DataFrame.from_records(records, columns=list(GAPS_COLUMNS))
    densities_per_seed = pd.concat(per_seed, ignore_index=True)
    written = [
        tables.write_csv(gaps, ctx.tables / GAPS_CSV, float_format="%.6f"),
        tables.write_markdown(gaps_markdown(gaps=gaps, missing=missing), ctx.tables / GAPS_MD),
        tables.write_csv(average_densities(densities_per_seed), ctx.tables / DENSITIES_CSV, float_format="%.8g"),
    ]
    tables.write_csv(densities_per_seed, ctx.cache / DENSITIES_PER_SEED_CSV, float_format="%.8g")
    overall = max(v for leaves in diffs.values() for v in leaves.values())
    ctx.notes.update(
        {
            "replay_max_abs_diff": overall,
            "replay_max_abs_diff_per_stream": diffs,
            "replay_tolerance": REPLAY_TOLERANCE,
            "replay_within_tolerance": bool(overall <= REPLAY_TOLERANCE),
            "missing_runs": [layout.relative(spec.dir()) for spec in missing],
            "projections_from_cache": from_cache,
            "grid": {"start": float(GRID[0]), "stop": float(GRID[-1]), "points": int(GRID.size)},
        },
    )

    for spec in missing:
        print(f"[missing] {layout.relative(spec.ood_csv())}")

    print(f"[replay] max abs diff between the replayed CAMP score and the recorded conf: {overall:.2e}")
    return written
