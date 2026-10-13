from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from scripts.experiment.lib import config_io, layout, tables
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.runs import RunSpec

if TYPE_CHECKING:
    import torch

    from openood.postprocessors.camp.config import CAMPPolicy

SPEC_PATH: Path = Path(__file__).with_name("spec.yml")
MARGINS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
DEFAULT_MARGIN = 0.2
METRICS: tuple[str, ...] = (
    "AUROC",
    "FPR@95",
    "share_discarded",
    "ood_ratio_window"
)
METRIC_TITLES: dict[str, str] = {
    "AUROC": "AUROC↑",
    "FPR@95": "FPR@95↓",
    "share_discarded": "Discarded (%)",
    "ood_ratio_window": "OOD ratio of the window (%)"
}
STREAM_COLUMNS: tuple[str, ...] = (
    "benchmark",
    "seed",
    "stream",
    "group",
    "margin",
    *METRICS,
    "n_admitted",
    "n_window",
    "n_stream",
    "camp_max_abs_diff",
    "cos_margin0_replay",
)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "benchmark",
    "margin",
    "metric",
    "mean",
    "std",
    "n_seeds",
    "seeds",
    "missing",
    "streams",
    "n_streams",
)
MARGIN_DECIMALS = 6
SWEEP_VERSION = 2


def load_spec(
    path: Path = SPEC_PATH,
) -> dict[str, Any]:

    spec = config_io.load_yaml(path)
    if not isinstance(spec, Mapping) or "runs" not in spec:
        raise ValueError(f"{path} must hold a mapping with a `runs` list")

    return dict(spec)


def runs(
    ctx: Context,
) -> list[RunSpec]:

    specs: list[RunSpec] = []

    for entry in load_spec()["runs"]:
        for benchmark in ctx.benchmarks:
            for seed in ctx.seeds:
                specs.append(
                    RunSpec(
                        str(entry["method"]),
                        benchmark,
                        int(seed),
                        ctx.backbone,
                        label=entry.get("label"),
                        overrides=dict(entry.get("overrides") or {}),
                        overlays=tuple(entry.get("overlays") or ()),
                    ),
                )

    return specs


def parse_margins(
    text: str | None,
) -> tuple[float, ...]:

    if not text:
        return MARGINS

    values = sorted({round(float(part), MARGIN_DECIMALS) for part in text.split(",") if part.strip()})
    if not values or any(not 0.0 <= value < 1.0 for value in values):
        raise ValueError(f"margins must lie in [0, 1), got {text!r}")

    return tuple(values)


def admitted_mask(
    memory_score: np.ndarray,
    threshold: np.ndarray,
    margin: float,
) -> np.ndarray:

    score = np.asarray(memory_score, dtype=np.float64).reshape(-1)

    if margin <= 0.0:
        return np.ones(score.shape[0], dtype=bool)

    t = np.asarray(threshold, dtype=np.float64).reshape(-1)
    if t.shape != score.shape:
        raise ValueError(f"memory_score {score.shape} and threshold {t.shape} must have one value per sample")

    return (score > t + margin * (1.0 - t)) | (score < t - margin * t)


def window_positions(
    admitted: np.ndarray,
    window_size: int,
) -> np.ndarray:

    positions = np.flatnonzero(np.asarray(admitted, dtype=bool))
    return positions[-int(window_size):] if window_size > 0 else positions


@dataclass(frozen=True)
class WindowSetting:

    window_size: int
    eps: float
    class_num: int
    covariance: str = "full"
    id_reference: str = "classwise"
    weighting: str = "msp"

    @classmethod
    def from_policy(
        cls,
        policy: CAMPPolicy,
        class_num: int,
    ) -> WindowSetting:

        camp = policy.camp
        return cls(
            window_size=int(camp.window_size),
            eps=float(camp.shift_eps),
            class_num=int(class_num),
            covariance=str(camp.covariance),
            id_reference=str(camp.id_reference),
            weighting=str(camp.weighting),
        )


def window_direction(
    features: torch.Tensor,
    msp: torch.Tensor,
    pred: torch.Tensor,
    positions: np.ndarray,
    setting: WindowSetting,
) -> torch.Tensor:

    import torch

    from scripts.experiment.lib import geometry

    index = torch.as_tensor(np.asarray(positions, dtype=np.int64), device=features.device)
    return geometry.camp_geometry(
        features[index],
        msp[index],
        pred[index],
        eps=setting.eps,
        class_num=setting.class_num,
        covariance=setting.covariance,
        id_reference=setting.id_reference,
        weighting=setting.weighting,
    ).u


def stream_margins(
    features: torch.Tensor,
    msp: torch.Tensor,
    pred: torch.Tensor,
    is_ood: np.ndarray,
    memory_score: np.ndarray,
    threshold: np.ndarray,
    margins: Sequence[float],
    setting: WindowSetting,
) -> tuple[pd.DataFrame, dict[float, torch.Tensor]]:

    from scripts.experiment.lib import geometry

    is_ood = np.asarray(is_ood, dtype=bool).reshape(-1)
    n = int(features.shape[0])
    if is_ood.shape[0] != n or np.asarray(memory_score).reshape(-1).shape[0] != n:
        raise ValueError("features, is_ood and the TANL trace must describe the same stream")

    records: list[dict[str, Any]] = []
    directions: dict[float, torch.Tensor] = {}

    for margin in margins:
        margin = round(float(margin), MARGIN_DECIMALS)
        admitted = admitted_mask(memory_score=memory_score, threshold=threshold, margin=margin)
        positions = window_positions(admitted=admitted, window_size=setting.window_size)
        if positions.size == 0:
            records.append(
                {
                    "margin": margin,
                    **{metric: math.nan for metric in METRICS},
                    "n_admitted": 0,
                    "n_window": 0,
                    "n_stream": n,
                },
            )

            continue

        u = window_direction(features=features, msp=msp, pred=pred, positions=positions, setting=setting)
        score = -geometry.project(features, u)
        directions[margin] = u
        records.append(
            {
                "margin": margin,
                "AUROC": 100.0 * geometry.auroc(score, ~is_ood),
                "FPR@95": 100.0 * geometry.fpr95(score, is_ood),
                "share_discarded": 100.0 * float(1.0 - admitted.mean()),
                "ood_ratio_window": 100.0 * float(is_ood[positions].mean()),
                "n_admitted": int(admitted.sum()),
                "n_window": int(positions.size),
                "n_stream": n,
            },
        )

    return pd.DataFrame.from_records(records), directions


def stream_cache_path(
    cache_dir: Path,
    spec: RunSpec,
    leaf: str,
) -> Path:

    run_dir = spec.dir()
    return Path(cache_dir) / f"selection_v{SWEEP_VERSION}__{run_dir.parent.name}__{run_dir.name}__{leaf}.csv"


def compute_run(
    spec: RunSpec,
    leaves: Sequence[str],
    margins: Sequence[float],
    trace_dir: Path,
    device: str,
    sweep_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:

    from scripts.experiment.lib import geometry, replay

    run = replay.RecordedRun(spec.dir())
    features = replay.features_of(run, device)
    setting = WindowSetting.from_policy(run.policy(), features.class_num)
    result: dict[str, pd.DataFrame] = {}

    for stream in replay.streams(run, features, leaves=leaves):
        replay.verify_stream(stream)
        replayed = replay.replay_camp(stream, features)
        difference = replay.check_against_recorded(stream, replayed.score)
        trace = replay.trace_tanl(features, stream, cache_dir=trace_dir)
        rows = stream.features.to(features.device)
        msp, pred, _ = geometry.zero_shot(rows, features.prototypes, features.logit_scale)
        frame, directions = stream_margins(
            features=rows,
            msp=msp,
            pred=pred,
            is_ood=stream.is_ood,
            memory_score=trace.memory_score,
            threshold=trace.threshold,
            margins=margins,
            setting=setting,
        )

        cos = math.nan

        if 0.0 in directions and replayed.final is not None:
            cos = geometry.cosine(directions[0.0], replayed.final.u)

        frame.insert(0, "group", str(stream.group))
        frame.insert(0, "stream", str(stream.name))
        frame.insert(0, "seed", int(spec.seed))
        frame.insert(0, "benchmark", spec.benchmark)
        frame["camp_max_abs_diff"] = float(difference)
        frame["cos_margin0_replay"] = float(cos)
        result[str(stream.name)] = frame[list(STREAM_COLUMNS)]

        if sweep_dir is not None:
            tables.write_csv(
                result[str(stream.name)],
                stream_cache_path(cache_dir=sweep_dir, spec=spec, leaf=str(stream.name)),
                float_format="%.6f",
            )

    return result


def benchmark_leaves(
    benchmark: str,
) -> list[str]:

    names = [
        leaf
        for members in layout.LEAVES[benchmark].values()
        for leaf in members
    ]

    return names


def cached_sweep(
    path: Path,
    margins: Sequence[float],
) -> pd.DataFrame | None:

    if not path.is_file():
        return None

    frame = pd.read_csv(path)
    if frame.empty or not set(STREAM_COLUMNS) <= set(frame.columns):
        return None

    cached = {round(float(m), MARGIN_DECIMALS) for m in frame["margin"]}
    wanted = [round(float(m), MARGIN_DECIMALS) for m in margins]
    if not set(wanted) <= cached:
        return None

    frame = frame.assign(margin=[round(float(m), MARGIN_DECIMALS) for m in frame["margin"]])
    return frame[frame["margin"].isin(wanted)].reset_index(drop=True)


def collect(
    ctx: Context,
    specs: Iterable[RunSpec],
    margins: Sequence[float],
    leaves: Sequence[str] | None,
    trace_dir: Path,
    refresh: bool = False,
) -> tuple[pd.DataFrame, list[RunSpec]]:

    frames: list[pd.DataFrame] = []
    missing: list[RunSpec] = []

    for spec in specs:
        if not spec.recorded():
            missing.append(spec)
            continue

        available = benchmark_leaves(spec.benchmark)
        wanted = [leaf for leaf in leaves if leaf in available] if leaves else available
        pending: list[str] = []

        for leaf in wanted:
            if refresh:
                cached = None
            else:
                cached = cached_sweep(path=stream_cache_path(cache_dir=ctx.cache, spec=spec, leaf=leaf), margins=margins)

            if cached is None:
                pending.append(leaf)
            else:
                frames.append(cached)

        if pending:
            computed = compute_run(
                spec=spec,
                leaves=pending,
                margins=margins,
                trace_dir=trace_dir,
                device=ctx.device,
                sweep_dir=ctx.cache
            )

            absent = [leaf for leaf in pending if leaf not in computed]
            if absent:
                raise KeyError(f"{layout.relative(spec.dir())} has no stream for {absent}")

            frames += [computed[leaf] for leaf in pending]

    streams = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(STREAM_COLUMNS))
    return streams, missing


def aggregate(
    streams: pd.DataFrame,
    requested: Mapping[str, Sequence[int]],
) -> pd.DataFrame:

    all_seeds = sorted(
        {int(s) for seeds in requested.values() for s in seeds}
        | ({int(s) for s in streams["seed"]} if not streams.empty else set()),
    )

    columns = [*SUMMARY_COLUMNS, *(f"seed_{seed}" for seed in all_seeds)]

    if streams.empty:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, Any]] = []

    for benchmark in dict.fromkeys(streams["benchmark"]):
        part = streams[streams["benchmark"] == benchmark]
        per_seed = part.groupby(["seed", "margin"], as_index=False)[list(METRICS)].mean(numeric_only=True)
        stream_names = list(dict.fromkeys(part["stream"]))
        recorded = sorted({int(s) for s in part["seed"]})
        wanted = [int(s) for s in requested.get(benchmark, recorded)]
        absent = sorted(set(wanted) - set(recorded))

        for margin in sorted({round(float(m), MARGIN_DECIMALS) for m in per_seed["margin"]}):
            rows = per_seed[np.isclose(a=per_seed["margin"].astype(float), b=margin)]

            for metric in METRICS:
                values = {
                    int(seed): float(value) for seed, value in zip(rows["seed"], rows[metric])
                    if not math.isnan(float(value))
                }
                series = list(values.values())
                record: dict[str, Any] = {
                    "benchmark": benchmark,
                    "margin": margin,
                    "metric": metric,
                    "mean": sum(series) / len(series) if series else math.nan,
                    "std": tables.sample_std(series),
                    "n_seeds": len(series),
                    "seeds": ",".join(str(s) for s in sorted(values)),
                    "missing": ",".join(str(s) for s in absent),
                    "streams": ",".join(stream_names),
                    "n_streams": len(stream_names),
                }

                for seed in all_seeds:
                    record[f"seed_{seed}"] = values.get(seed, math.nan)

                records.append(record)

    return pd.DataFrame.from_records(records, columns=columns)


def margin_title(
    margin: float,
) -> str:

    text = f"{margin:.2f}"

    if margin == 0.0:
        return f"{text} (CAMP: every sample)"

    if math.isclose(a=margin, b=DEFAULT_MARGIN):
        return f"{text} (TANL default)"

    return text


def markdown(
    summary: pd.DataFrame,
) -> str:

    if summary.empty:
        return "no run recorded\n"

    lines: list[str] = []

    for benchmark in dict.fromkeys(summary["benchmark"]):
        part = summary[summary["benchmark"] == benchmark]
        title = layout.BENCHMARK_NAMES.get(benchmark, benchmark)
        lines += [f"### {title}: CAMP with the window restricted by the TANL rule", "",
            "| Margin | seeds | " + " | ".join(METRIC_TITLES[m] for m in METRICS) + " |",
            "|---|---|" + "---|" * len(METRICS)]

        for margin in sorted({float(m) for m in part["margin"]}):
            rows = part[np.isclose(a=part["margin"].astype(float), b=margin)].set_index("metric")
            first = rows.iloc[0]
            seeds = str(first["seeds"]) + (f" (missing {first['missing']})" if str(first["missing"]) else "")
            cells = [margin_title(margin), seeds]

            for metric in METRICS:
                if metric in rows.index:
                    cells.append(tables.cell(float(rows.loc[metric, "mean"]), float(rows.loc[metric, "std"])))
                else:
                    cells.append(tables.MISSING_CELL)

            lines.append("| " + " | ".join(cells) + " |")

        lines.append("")

    lines.append(
        "Every value is in percent and is the mean over the streams of a seed, then over the seeds (sample "
        "std in parentheses). A margin of 0 admits every sample (CAMP); the window of a wider margin holds "
        "the last N admitted samples and its direction scores the whole stream offline.",
    )

    return "\n".join(lines) + "\n"


def write_tables(
    ctx: Context,
    summary: pd.DataFrame,
    streams: pd.DataFrame,
) -> list[Path]:

    return [
        tables.write_csv(summary, ctx.tables / "selection.csv"),
        tables.write_csv(streams, ctx.tables / "selection_streams.csv", float_format="%.6f"),
        tables.write_markdown(markdown(summary), ctx.tables / "selection.md"),
    ]


def analyze(
    ctx: Context,
) -> list[Path]:

    margins = parse_margins(ctx.option("margins"))
    leaves = ctx.option_list("leaves") or None
    trace_dir = Path(ctx.option("trace_dir") or ctx.cache)
    refresh = ctx.option("refresh") in ("1", "true", "yes")
    specs = runs(ctx)
    streams, missing = collect(
        ctx=ctx,
        specs=specs,
        margins=margins,
        leaves=leaves,
        trace_dir=trace_dir,
        refresh=refresh
    )
    if streams.empty:
        raise FileNotFoundError(
            "no recorded CAMP run to analyse; expected " +
            ", ".join(layout.relative(spec.dir()) for spec in specs),
        )

    requested = {
        benchmark: [spec.seed for spec in specs if spec.benchmark == benchmark]
        for benchmark in ctx.benchmarks
    }
    summary = aggregate(streams=streams, requested=requested)
    per_stream = streams.drop_duplicates(["benchmark", "seed", "stream"])
    ctx.notes["margins"] = list(margins)
    ctx.notes["trace_dir"] = layout.relative(trace_dir)
    ctx.notes["missing_runs"] = [layout.relative(spec.dir()) for spec in missing]
    ctx.notes["camp_max_abs_diff"] = float(per_stream["camp_max_abs_diff"].max())
    ctx.notes["camp_max_abs_diff_per_stream"] = {
        f"{row.benchmark}/s{int(row.seed)}/{row.stream}": float(row.camp_max_abs_diff)
        for row in per_stream.itertuples(index=False)
    }

    ctx.notes["cos_margin0_replay_min"] = float(per_stream["cos_margin0_replay"].min())
    return write_tables(ctx, summary, streams)


__all__ = [
    "DEFAULT_MARGIN",
    "MARGINS",
    "METRICS",
    "METRIC_TITLES",
    "STREAM_COLUMNS",
    "SUMMARY_COLUMNS",
    "WindowSetting",
    "admitted_mask",
    "aggregate",
    "analyze",
    "benchmark_leaves",
    "cached_sweep",
    "collect",
    "compute_run",
    "load_spec",
    "markdown",
    "parse_margins",
    "runs",
    "stream_cache_path",
    "stream_margins",
    "window_direction",
    "window_positions",
    "write_tables",
]
