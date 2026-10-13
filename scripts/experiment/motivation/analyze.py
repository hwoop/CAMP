from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

from scripts.experiment.lib import layout, replay, tables
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.runs import RunSpec

if TYPE_CHECKING:
    from scripts.experiment.lib.replay import RunFeatures, Stream

DEFAULT_STREAMS: tuple[str, ...] = ("inaturalist",)
MEAN_NAMES: dict[str, str] = {
    "true_id_mean": "True ID mean (all ID test samples)",
    "camp_id_mean": "CAMP ID mean (confidence-weighted, every window sample)",
    "hard_selection_mean": "Hard selection (window samples AdaNeg admitted as ID)",
    "stream_mean": "Stream mean (every window sample)",
}
MEANS_COLUMNS: tuple[str, ...] = (
    "benchmark",
    "seed",
    "stream",
    "group",
    "name",
    "display",
    "n_samples",
    "n_ood",
    "distance_to_true_id_mean",
    "pc1",
    "pc2",
)
POINTS_COLUMNS: tuple[str, ...] = ("pc1", "pc2", "is_ood", "admitted_id")
PCA_COLUMNS: tuple[str, ...] = ("benchmark", "seed", "stream", "component", "explained_variance_ratio")


@dataclass(frozen=True, eq=False)
class WindowReplay:

    benchmark: str
    seed: int
    stream: str
    group: str
    window_features: np.ndarray
    window_is_ood: np.ndarray
    window_admitted: np.ndarray
    id_features: np.ndarray
    mu_id_camp: np.ndarray
    mu_stream: np.ndarray
    score_max_abs_diff: float
    n_batches: int

    @property
    def window_size(self) -> int:
        return int(self.window_features.shape[0])


def run_key(
    benchmark: str,
    seed: int,
) -> str:
    return f"{benchmark}-s{seed}"


def points_path(
    cache: Path,
    benchmark: str,
    seed: int,
    stream: str,
) -> Path:
    return Path(cache) / f"window_points__{run_key(benchmark=benchmark, seed=seed)}__{stream}.csv"


def replay_window(
    stream: Stream,
    features: RunFeatures,
    cache_dir: Path,
) -> WindowReplay:

    import torch

    run = features.run
    replay.verify_stream(stream)
    result = replay.replay_camp(stream, features)
    diff = replay.check_against_recorded(stream, result.score)

    if result.final is None:
        raise replay.StreamMismatch(f"stream {stream.name!r} never left CAMP's warm-up: no window to analyse.")

    trace = replay.trace_adaneg(features, stream, cache_dir=cache_dir)
    rows = stream.features.detach().cpu().to(torch.float64).numpy()
    window = result.final_window
    window_features = rows[window]
    mu_stream = result.final.mu_stream.detach().cpu().numpy()
    if not np.allclose(a=mu_stream, b=window_features.mean(axis=0), atol=1e-6):
        raise replay.StreamMismatch(
            f"stream {stream.name!r}: CAMP's window mean differs from the mean of the "
            f"replayed window rows; the window positions do not match CAMP's rows.",
        )

    return WindowReplay(
        benchmark=run.benchmark,
        seed=run.seed,
        stream=str(stream.name),
        group=str(stream.group),
        window_features=window_features,
        window_is_ood=stream.is_ood[window].astype(bool),
        window_admitted=np.asarray(trace.id_admit, dtype=bool)[window],
        id_features=rows[~stream.is_ood],
        mu_id_camp=result.final.mu_id.detach().cpu().numpy(),
        mu_stream=mu_stream,
        score_max_abs_diff=float(diff),
        n_batches=int(result.batches),
    )


def pca_axes(
    points: np.ndarray,
    orient_first: np.ndarray | None = None,
    orient_second: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        raise ValueError(f"PCA needs at least 3 rows and 2 columns, got {points.shape}.")

    center = points.mean(axis=0)
    centered = points - center
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular ** 2
    ratio = variance[:2] / max(float(variance.sum()), 1e-300)
    components = vt[:2].T.copy()

    for index, orient in enumerate((orient_first, orient_second)):
        column = components[:, index]

        if orient is not None:
            sign = np.sign(float(np.asarray(orient, dtype=np.float64) @ column)) or 1.0
        else:
            sign = np.sign(column[int(np.argmax(np.abs(column)))]) or 1.0

        components[:, index] = sign * column

    return center, components, ratio


def summarize_window(
    window: WindowReplay,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    features = np.asarray(window.window_features, dtype=np.float64)
    is_ood = np.asarray(window.window_is_ood, dtype=bool)
    admitted = np.asarray(window.window_admitted, dtype=bool)
    true_id = np.asarray(window.id_features, dtype=np.float64).mean(axis=0)
    hard = features[admitted].mean(axis=0) if admitted.any() else np.full(shape=features.shape[1], fill_value=np.nan)
    means = {
        "true_id_mean": (true_id, int(window.id_features.shape[0]), 0),
        "camp_id_mean": (np.asarray(window.mu_id_camp, dtype=np.float64), int(features.shape[0]), int(is_ood.sum())),
        "hard_selection_mean": (hard, int(admitted.sum()), int((admitted & is_ood).sum())),
        "stream_mean": (np.asarray(window.mu_stream, dtype=np.float64), int(features.shape[0]), int(is_ood.sum())),
    }
    ood_direction = features[is_ood].mean(axis=0) - true_id if is_ood.any() else None
    center, components, ratio = pca_axes(
        features,
        orient_first=means["stream_mean"][0] - true_id,
        orient_second=ood_direction,
    )

    def coordinates(
        vector: np.ndarray,
    ) -> np.ndarray:
        return (vector - center) @ components

    records: list[dict[str, object]] = []

    for name, (vector, count, ood_count) in means.items():
        pc = coordinates(vector) if np.all(np.isfinite(vector)) else np.array([np.nan, np.nan])
        records.append(
            {
                "benchmark": window.benchmark,
                "seed": window.seed,
                "stream": window.stream,
                "group": window.group,
                "name": name,
                "display": MEAN_NAMES[name],
                "n_samples": count,
                "n_ood": ood_count,
                "distance_to_true_id_mean": float(np.linalg.norm(vector - true_id)) if count else np.nan,
                "pc1": float(pc[0]),
                "pc2": float(pc[1]),
            },
        )

    projected = coordinates(features)
    points = pd.DataFrame(
        {
            "pc1": projected[:, 0],
            "pc2": projected[:, 1],
            "is_ood": is_ood.astype(int),
            "admitted_id": admitted.astype(int),
        },
        columns=list(POINTS_COLUMNS),
    )

    pca = pd.DataFrame(
        [{
                "benchmark": window.benchmark,
                "seed": window.seed,
                "stream": window.stream,
                "component": index + 1,
                "explained_variance_ratio": float(ratio[index]),
            }
            for index in range(2)],
        columns=list(PCA_COLUMNS),
    )

    return pd.DataFrame.from_records(records, columns=list(MEANS_COLUMNS)), points, pca


def means_markdown(
    means: pd.DataFrame,
    pca: pd.DataFrame,
) -> str:

    lines: list[str] = []
    keys = ["benchmark", "seed", "stream"]
    for (benchmark, seed, stream), part in means.groupby(keys, sort=False):
        name = tables.leaf_name(str(stream))
        lines += [f"### {layout.BENCHMARK_NAMES.get(str(benchmark), str(benchmark))}, seed {seed}, {name} stream: "
            f"the last {int(part.loc[part['name'] == 'stream_mean', 'n_samples'].iloc[0]):,} samples", ""]
        ratios = pca[(pca["benchmark"] == benchmark) & (pca["seed"] == seed) & (pca["stream"] == stream)]
        if not ratios.empty:
            explained = ", ".join(f"PC{int(r.component)} {100 * float(r.explained_variance_ratio):.1f} %"
                for r in ratios.itertuples())

            lines += [f"Explained variance of the display components: {explained}.", ""]

        lines += [
            "| Mean | Samples | OOD among them | Distance to true ID mean | PC1 | PC2 |",
            "|---|---|---|---|---|---|",
        ]

        for row in part.itertuples():
            distance = "–" if np.isnan(row.distance_to_true_id_mean) else f"{row.distance_to_true_id_mean:.4f}"
            pc1 = "–" if np.isnan(row.pc1) else f"{row.pc1:.4f}"
            pc2 = "–" if np.isnan(row.pc2) else f"{row.pc2:.4f}"
            lines.append(
                f"| {row.display} | {int(row.n_samples):,} | {int(row.n_ood):,} | {distance} | {pc1} | "
                f"{pc2} |",
            )

        lines.append("")

    return "\n".join(lines) if lines else "no stream analysed\n"


def requested_streams(
    ctx: Context,
) -> list[str]:
    return ctx.option_list("stream", DEFAULT_STREAMS)


def analyze_runs(
    ctx: Context,
    specs: Sequence[RunSpec],
) -> list[WindowReplay]:

    if not ctx.device.startswith("cuda"):
        raise RuntimeError(
            f"motivation needs a CUDA device for the replay (the CLIP network runs on CUDA); "
            f"got --device {ctx.device}.",
        )

    windows: list[WindowReplay] = []
    wanted = requested_streams(ctx)

    for spec in specs:
        run = replay.RecordedRun(spec.dir())
        features = replay.features_of(run, ctx.device)
        found = replay.streams(run, features, leaves=wanted)
        absent = sorted(set(wanted) - {stream.name for stream in found})
        if absent:
            raise KeyError(
                f"{layout.relative(spec.dir())} has no stream named {absent}; its streams are "
                f"{[leaf for _, leaf in features.leaves()]}.",
            )

        for stream in found:
            print(f"[replay] {layout.relative(spec.dir())} stream {stream.name} ({len(stream)} samples)")
            windows.append(replay_window(stream=stream, features=features, cache_dir=ctx.cache))

    return windows


def analyze(
    ctx: Context,
) -> list[Path]:

    from scripts.experiment.motivation import runs as item_runs

    specs = item_runs(ctx)
    recorded = [spec for spec in specs if spec.recorded()]
    missing = [layout.relative(spec.dir()) for spec in specs if not spec.recorded()]
    ctx.notes["missing_runs"] = missing
    ctx.notes["streams"] = requested_streams(ctx)

    if not recorded:
        raise FileNotFoundError("no recorded run for motivation; missing: " + ", ".join(missing))

    for path in missing:
        print(f"[missing] {path}")

    windows = analyze_runs(ctx=ctx, specs=recorded)
    means_parts: list[pd.DataFrame] = []
    pca_parts: list[pd.DataFrame] = []
    written: list[Path] = []
    diffs: dict[str, float] = {}
    batches: dict[str, int] = {}

    for window in windows:
        means, points, pca = summarize_window(window)
        means_parts.append(means)
        pca_parts.append(pca)
        written.append(
            tables.write_csv(
                points,
                points_path(
                    cache=ctx.cache,
                    benchmark=window.benchmark,
                    seed=window.seed,
                    stream=window.stream
                ),
                float_format="%.6f",
            ),
        )

        key = f"{run_key(benchmark=window.benchmark, seed=window.seed)}/{window.stream}"
        diffs[key] = window.score_max_abs_diff
        batches[key] = window.n_batches

    means_frame = pd.concat(means_parts, ignore_index=True)
    pca_frame = pd.concat(pca_parts, ignore_index=True)
    ctx.notes["replay_max_abs_diff"] = diffs
    ctx.notes["replay_max_abs_diff_overall"] = max(diffs.values())
    ctx.notes["replay_batches"] = batches
    ctx.notes["window_size"] = {f"{run_key(benchmark=w.benchmark, seed=w.seed)}/{w.stream}": w.window_size for w in windows}
    written.insert(0, tables.write_csv(means_frame, ctx.tables / "means.csv", float_format="%.6f"))
    written.insert(1, tables.write_markdown(means_markdown(means=means_frame, pca=pca_frame), ctx.tables / "means.md"))
    written.insert(2, tables.write_csv(pca_frame, ctx.tables / "pca.csv", float_format="%.6f"))
    return written


__all__ = [
    "DEFAULT_STREAMS",
    "MEAN_NAMES",
    "MEANS_COLUMNS",
    "PCA_COLUMNS",
    "POINTS_COLUMNS",
    "WindowReplay",
    "analyze",
    "analyze_runs",
    "means_markdown",
    "pca_axes",
    "points_path",
    "replay_window",
    "requested_streams",
    "run_key",
    "summarize_window",
]
