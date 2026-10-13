from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.experiment.lib import config_io, layout, tables
from scripts.experiment.lib import runs as runs_lib
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.runs import RunSpec

ITEM_DIR: Path = Path(__file__).resolve().parent
SPEC_PATH: Path = ITEM_DIR / "spec.yml"

REFERENCE = "camp"
BASELINE = "mcm"
GAIN_METRICS: dict[str, str] = {
    "FPR@95": "FPR@95 gain",
    "AUROC": "AUROC gain",
}
GROUP_ROWS: dict[str, tuple[str, str]] = {
    "IN-1K": ("imagenet_standard", "farood"),
    "near": ("imagenet_openood", "nearood"),
    "far": ("imagenet_openood", "farood"),
}
GROUP_ORDER: tuple[str, ...] = tuple(GROUP_ROWS)
ARM_FIXED = "fixed"
ARM_NATURAL = "natural"

TABLE_RATIO_METRICS = "ratio_metrics.csv"
TABLE_GAINS = "gains.csv"
TABLE_GAINS_MD = "gains.md"
TABLE_DELTA = "delta_norm.csv"
TABLE_DELTA_SUMMARY = "delta_norm_summary.csv"

DELTA_COLUMNS: tuple[str, ...] = (
    "method",
    "label",
    "arm",
    "ratio",
    "benchmark",
    "leaf",
    "group",
    "seed",
    "stream",
    "stream_size",
    "ood_ratio_stream",
    "window_size",
    "ood_ratio_window",
    "delta_norm",
    "r_norm",
    "k",
    "batches",
    "max_abs_diff",
    "ood_weight_share",
    "max_class_weight",
)


@dataclass(frozen=True)
class Arm:

    label: str | None
    ratio: float | None
    methods: tuple[str, ...]
    overlays: tuple[str, ...] = ()
    overrides: Mapping[str, Any] | None = None

    @property
    def kind(self) -> str:
        return ARM_NATURAL if self.ratio is None else ARM_FIXED


    @property
    def name(self) -> str:
        return ARM_NATURAL if self.ratio is None else f"{self.ratio * 100:g} %"


def load_spec(
    path: Path = SPEC_PATH,
) -> dict[str, Any]:

    spec = config_io.load_yaml(path)
    if not isinstance(spec, Mapping) or "runs" not in spec:
        raise ValueError(f"{path} must be a mapping with a `runs` list")

    return dict(spec)


def arms(
    spec: Mapping[str, Any] | None = None,
) -> list[Arm]:

    spec = load_spec() if spec is None else spec
    result: list[Arm] = []

    for entry in spec["runs"]:
        methods = entry.get("method", REFERENCE)
        methods = (methods,) if isinstance(methods, str) else tuple(methods)
        ratio = entry.get("ratio")
        result.append(
            Arm(
                label=entry.get("label"),
                ratio=None if ratio is None else float(ratio),
                methods=methods,
                overlays=tuple(entry.get("overlays") or ()),
                overrides=dict(entry.get("overrides") or {}),
            ),
        )

    return result


def arm_of(
    label: str | None,
    spec: Mapping[str, Any] | None = None,
) -> Arm:

    for arm in arms(spec):
        if arm.label == label:
            return arm

    raise KeyError(f"no arm labelled {label!r} in {SPEC_PATH}")


def ratio_of(
    label: str | None,
    spec: Mapping[str, Any] | None = None,
) -> float:

    ratio = arm_of(label=label, spec=spec).ratio
    return math.nan if ratio is None else ratio


def expand_runs(
    ctx: Context,
    spec: Mapping[str, Any] | None = None,
) -> list[RunSpec]:

    specs: list[RunSpec] = []

    for arm in arms(spec):
        for method in arm.methods:
            for benchmark in ctx.benchmarks:
                for seed in ctx.seeds:
                    specs.append(
                        RunSpec(
                            method,
                            benchmark,
                            seed,
                            backbone=ctx.backbone,
                            label=arm.label,
                            overrides=dict(arm.overrides or {}),
                            overlays=arm.overlays,
                        ),
                    )

    return specs


def group_of(
    benchmark: str,
    row: str,
) -> str:

    group_row = layout.leaf_group(benchmark, row)

    for name, (bench, grow) in GROUP_ROWS.items():
        if (bench, grow) == (benchmark, group_row):
            return name

    return f"{benchmark}/{group_row}"


def _count_lines(
    path: Path,
) -> int:

    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def natural_ratios(
    benchmark: str,
    root: Path | None = None,
) -> dict[str, float]:

    root = layout.ROOT if root is None else Path(root)
    tree: dict[str, Any] = {}

    for rel in runs_lib.dataset_configs(benchmark):
        tree = config_io.merge_trees(tree, config_io.load_yaml(root / rel))

    id_list = root / str(tree["dataset"]["test"]["imglist_pth"])
    if not id_list.is_file():
        return {}

    n_id = _count_lines(id_list)
    ratios: dict[str, float] = {}
    ood = tree["ood_dataset"]
    for split in ood["split_names"]:
        for leaf in ood[split]["datasets"]:
            path = root / str(ood[split][leaf]["imglist_pth"])
            if path.is_file():
                n_ood = _count_lines(path)
                ratios[str(leaf)] = n_ood / (n_ood + n_id)

    return ratios


def natural_ratio(
    benchmark: str,
    row: str,
    ratios: Mapping[str, float],
) -> float:

    if row in layout.GROUPS:
        values = [ratios[leaf] for leaf in layout.LEAVES[benchmark].get(row, ()) if leaf in ratios]
        return sum(values) / len(values) if values else math.nan

    return ratios.get(row, math.nan)


def _seed_columns(
    frame: pd.DataFrame,
) -> list[str]:

    names = [
        column
        for column in frame.columns
        if column.startswith("seed_")
    ]

    return names


def ratio_metrics(
    specs: Sequence[RunSpec],
    spec: Mapping[str, Any] | None = None,
) -> pd.DataFrame:

    frame = tables.summarize(specs)
    if frame.empty:
        columns = [*tables.KEY_COLUMNS, "arm", "ratio", "group", *tables.VALUE_COLUMNS]
        return pd.DataFrame(columns=columns)

    frame = frame.copy()
    frame["arm"] = [arm_of(label=label, spec=spec).kind for label in frame["label"]]
    frame["ratio"] = [ratio_of(label=label, spec=spec) for label in frame["label"]]
    frame["group"] = [group_of(benchmark=b, row=r) for b, r in zip(frame["benchmark"], frame["row"])]
    front = [*tables.KEY_COLUMNS, "arm", "ratio", "group"]
    return frame[front + [column for column in frame.columns if column not in front]]


def gains(
    metrics: pd.DataFrame,
    requested_seeds: Sequence[int],
    ratios: Mapping[str, Mapping[str, float]],
    spec: Mapping[str, Any] | None = None,
) -> pd.DataFrame:

    seed_columns = _seed_columns(metrics)
    keys = ["label", "benchmark", "backbone", "row", "kind", "metric"]
    camp = metrics[(metrics["method"] == REFERENCE) & metrics["metric"].isin(GAIN_METRICS)]
    mcm = metrics[(metrics["method"] == BASELINE) & metrics["metric"].isin(GAIN_METRICS)]
    records: list[dict[str, Any]] = []

    if not camp.empty and not mcm.empty:
        paired = camp.merge(mcm, on=keys, suffixes=("_camp", "_mcm"))
        for _, pair in paired.iterrows():
            metric = str(pair["metric"])
            values: dict[int, float] = {}

            for column in seed_columns:
                a, b = float(pair[f"{column}_camp"]), float(pair[f"{column}_mcm"])
                if math.isnan(a) or math.isnan(b):
                    continue

                values[int(column[len("seed_"):])] = (b - a) if metric == "FPR@95" else (a - b)

            series = list(values.values())
            label = pair["label"] if isinstance(pair["label"], str) else None
            arm = arm_of(label=label, spec=spec)
            benchmark, row = str(pair["benchmark"]), str(pair["row"])

            if arm.ratio is not None:
                ratio = arm.ratio
            else:
                ratio = natural_ratio(benchmark=benchmark, row=row, ratios=ratios.get(benchmark, {}))

            record: dict[str, Any] = {
                "label": label,
                "arm": arm.kind,
                "ratio": ratio,
                "benchmark": benchmark,
                "backbone": pair["backbone"],
                "row": row,
                "kind": pair["kind"],
                "group": group_of(benchmark=benchmark, row=row),
                "metric": GAIN_METRICS[metric],
                "mean": sum(series) / len(series) if series else math.nan,
                "std": tables.sample_std(series),
                "n_seeds": len(series),
                "seeds": ",".join(str(s) for s in sorted(values)),
                "missing": ",".join(str(s) for s in requested_seeds if s not in values),
            }

            for column in seed_columns:
                record[column] = values.get(int(column[len("seed_"):]), math.nan)

            records.append(record)

    columns = [
        "label",
        "arm",
        "ratio",
        "benchmark",
        "backbone",
        "row",
        "kind",
        "group",
        "metric",
        "mean",
        "std",
        "n_seeds",
        "seeds",
        "missing",
        *seed_columns,
    ]
    frame = pd.DataFrame.from_records(records, columns=columns)
    frame["label"] = frame["label"].astype(object).where(frame["label"].notna(), None)
    return frame.sort_values(["metric", "benchmark", "row", "ratio"], na_position="first", kind="stable").reset_index(
        drop=True,
    )


def gains_markdown(
    frame: pd.DataFrame,
    spec: Mapping[str, Any] | None = None,
) -> str:

    if frame.empty:
        return "### CAMP's gain over MCM (percentage points)\n\nno paired CAMP / MCM run recorded\n"

    part = frame[frame["kind"] == "group"].copy()
    part["method"] = "gain"
    ordered = sorted(arms(spec), key=lambda arm: (arm.ratio is not None, arm.ratio or 0.0))
    present = {value if isinstance(value, str) else None for value in part["label"]}
    rows = [tables.Row("gain", arm.label, name=arm.name) for arm in ordered if arm.label in present]
    columns = [tables.Column(bench, row, metric, header=f"{group} {metric}")
        for metric in GAIN_METRICS.values() for group, (bench, row) in GROUP_ROWS.items()
        if ((part["benchmark"] == bench) & (part["row"] == row)).any()]
    text = tables.markdown_table(
        part,
        columns=columns,
        rows=rows,
        title="CAMP's gain over MCM (percentage points)",
        mark_best=False,
        row_header="OOD ratio",
    )

    return text + ("\nGain = FPR@95(MCM) - FPR@95(CAMP) and AUROC(CAMP) - AUROC(MCM) on the same streams and seeds; "
        "positive = CAMP better. Mean (sample std) over the seeds listed.\n")


def delta_cache_path(
    ctx: Context,
    spec: RunSpec,
    stream: str,
) -> Path:

    run_dir = spec.dir()
    return ctx.cache / f"delta_v2__{run_dir.parent.name}__{run_dir.name}__{stream}.json"


def _read_record(
    path: Path,
) -> dict[str, Any] | None:

    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    return record if all(key in record for key in DELTA_COLUMNS) else None


def replay_allowed(
    ctx: Context,
) -> str | None:

    if ctx.option("replay", "1") in ("0", "false", "no"):
        return "disabled by --opt replay=0"

    if ctx.device == "cpu":
        return "the replay needs a CUDA device (the released CLIP network runs on CUDA); device is cpu"

    return None


def replay_stream(
    spec: RunSpec,
    arm: Arm,
    run: Any,
    features: Any,
    stream: Any,
) -> dict[str, Any]:

    from scripts.experiment.lib import context, replay

    replay.verify_stream(stream)
    result = replay.replay_camp(stream, features)
    diff = replay.check_against_recorded(stream, result.score)
    final = result.final
    window = stream.is_ood[result.final_window]
    ratio_stream = float(stream.is_ood.mean())
    weight_stats = {
        "ood_weight_share": math.nan,
        "max_class_weight": math.nan,
    }

    if result.final_rows is not None:
        _, confidence, pseudo = result.final_rows
        camp = run.policy().camp
        stats = context.weight_context(
            confidence.detach().cpu().numpy(),
            pseudo.detach().cpu().numpy(),
            window,
            id_reference=camp.id_reference,
            weighting=camp.weighting,
        )

        weight_stats = {key: stats[key] for key in weight_stats}

    return {
        "method": spec.method,
        "label": spec.label,
        "arm": arm.kind,
        "ratio": arm.ratio if arm.ratio is not None else ratio_stream,
        "benchmark": spec.benchmark,
        "leaf": str(stream.leaf),
        "group": group_of(benchmark=spec.benchmark, row=str(stream.leaf)),
        "seed": int(run.seed),
        "stream": str(stream.name),
        "stream_size": int(len(stream)),
        "ood_ratio_stream": ratio_stream,
        "window_size": int(window.size),
        "ood_ratio_window": float(window.mean()) if window.size else math.nan,
        "delta_norm": float(final.delta.norm()) if final is not None else math.nan,
        "r_norm": float(final.r.norm()) if final is not None else math.nan,
        "k": int(final.k) if final is not None else 0,
        "batches": int(result.batches),
        "max_abs_diff": float(diff),
        **weight_stats,
    }


def replay_delta_norms(
    ctx: Context,
    specs: Iterable[RunSpec],
    spec: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:

    wanted_seeds = {int(s) for s in ctx.option_list("seeds")} or None
    wanted_leaves = set(ctx.option_list("leaves")) or None
    skip_reason = replay_allowed(ctx)
    notes: dict[str, Any] = {
        "streams": 0,
        "cached": 0,
        "replayed": 0,
        "max_abs_diff": None,
        "skipped_runs": [],
        "mismatches": [],
    }
    records: list[dict[str, Any]] = []

    for run_spec in specs:
        if run_spec.method != REFERENCE or not run_spec.recorded():
            continue

        if wanted_seeds is not None and run_spec.seed not in wanted_seeds:
            continue

        arm = arm_of(label=run_spec.label, spec=spec)
        leaves = [
            leaf for members in layout.LEAVES[run_spec.benchmark].values() for leaf in members
            if wanted_leaves is None or leaf in wanted_leaves
        ]
        cached = {leaf: _read_record(delta_cache_path(ctx=ctx, spec=run_spec, stream=leaf)) for leaf in leaves}
        records += [record for record in cached.values() if record is not None]
        notes["cached"] += sum(record is not None for record in cached.values())
        missing = [leaf for leaf, record in cached.items() if record is None]
        if not missing:
            continue

        if skip_reason is not None:
            notes["skipped_runs"].append(
                {
                    "run": layout.relative(run_spec.dir()),
                    "streams": missing,
                    "reason": skip_reason,
                },
            )

            continue

        records += _replay_run(ctx=ctx, run_spec=run_spec, arm=arm, leaves=missing, notes=notes)

    notes["streams"] = len(records)
    frame = pd.DataFrame.from_records(records, columns=list(DELTA_COLUMNS))
    frame["label"] = frame["label"].astype(object).where(frame["label"].notna(), None)
    if not frame.empty:
        notes["max_abs_diff"] = float(frame["max_abs_diff"].max())

    return frame.sort_values(["benchmark", "ratio", "seed", "leaf"], kind="stable").reset_index(drop=True), notes


def _replay_run(
    ctx: Context,
    run_spec: RunSpec,
    arm: Arm,
    leaves: Sequence[str],
    notes: dict[str, Any],
) -> list[dict[str, Any]]:

    import torch

    from scripts.experiment.lib import replay

    run = replay.RecordedRun(run_spec.dir())
    features = replay.features_of(run, ctx.device)
    records: list[dict[str, Any]] = []

    try:
        for stream in replay.streams(run, features, leaves=leaves):
            if stream.leaf is None:
                raise ValueError(
                    f"{run_spec.dir()}: stream {stream.name!r} evaluates several leaves; this item "
                    "expects the per-leaf composition",
                )

            print(f"[replay] {layout.relative(run_spec.dir())} {stream.name} ({len(stream)} samples)")
            record = replay_stream(spec=run_spec, arm=arm, run=run, features=features, stream=stream)
            if record["max_abs_diff"] > 1e-3:
                notes["mismatches"].append(
                    {
                        "run": layout.relative(run_spec.dir()),
                        "stream": stream.name,
                        "max_abs_diff": record["max_abs_diff"],
                    },
                )

            path = delta_cache_path(ctx=ctx, spec=run_spec, stream=stream.name)
            path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            records.append(record)
            notes["replayed"] += 1

    finally:
        del features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return records


def summarize_delta(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    columns = ["label", "arm", "ratio", "group", "mean", "std", "n_seeds", "seeds", "leaves"]

    if frame.empty:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, Any]] = []
    keyed = frame.assign(_label=frame["label"].map(lambda v: v if isinstance(v, str) else ""))
    for (label, arm, group), part in keyed.groupby(["_label", "arm", "group"], sort=False):
        per_seed = part.groupby("seed")[["delta_norm", "ratio"]].mean(numeric_only=True)
        values = [float(v) for v in per_seed["delta_norm"] if not math.isnan(float(v))]
        records.append({
            "label": label or None,
            "arm": arm,
            "ratio": float(per_seed["ratio"].mean()),
            "group": group,
            "mean": sum(values) / len(values) if values else math.nan,
            "std": tables.sample_std(values),
            "n_seeds": len(values),
            "seeds": ",".join(str(int(s)) for s in sorted(per_seed.index)),
            "leaves": ",".join(sorted(set(part["leaf"]))),
        })

    summary = pd.DataFrame.from_records(records, columns=columns)
    summary["label"] = summary["label"].astype(object).where(summary["label"].notna(), None)
    return summary.sort_values(["group", "ratio"], kind="stable").reset_index(drop=True)


def analyze(
    ctx: Context,
) -> list[Path]:

    spec = load_spec()
    specs = expand_runs(ctx=ctx, spec=spec)
    recorded = [s for s in specs if s.recorded()]
    missing = [s for s in specs if not s.recorded()]

    if not recorded:
        raise FileNotFoundError(
            f"ood_ratio: none of the {len(specs)} runs is recorded "
            f"(run `python -m scripts.experiment ood_ratio plan` for the commands)",
        )

    ratios = {benchmark: natural_ratios(benchmark=benchmark, root=ctx.root) for benchmark in ctx.benchmarks}
    metrics = ratio_metrics(specs=specs, spec=spec)
    gain_frame = gains(metrics=metrics, requested_seeds=ctx.seeds, ratios=ratios, spec=spec)
    delta_frame, replay_notes = replay_delta_norms(ctx=ctx, specs=specs, spec=spec)
    summary = summarize_delta(delta_frame)
    written = [
        tables.write_csv(metrics, ctx.tables / TABLE_RATIO_METRICS),
        tables.write_csv(gain_frame, ctx.tables / TABLE_GAINS),
        tables.write_markdown(gains_markdown(frame=gain_frame, spec=spec), ctx.tables / TABLE_GAINS_MD),
        tables.write_csv(delta_frame, ctx.tables / TABLE_DELTA, float_format="%.6g"),
        tables.write_csv(summary, ctx.tables / TABLE_DELTA_SUMMARY, float_format="%.6g"),
    ]
    paired = {label if isinstance(label, str) else None for label in gain_frame["label"]}
    unpaired = sorted({(s.method, s.label or "") for s in recorded if s.label not in paired})

    ctx.notes.update({
        "recorded_runs": len(recorded),
        "missing_runs": [layout.relative(s.dir()) for s in missing],
        "paired_arms": sorted(label or ARM_NATURAL for label in paired),
        "unpaired_recorded": [f"{method} {label or ARM_NATURAL}" for method, label in unpaired],
        "natural_ratios": ratios,
        "replay": replay_notes,
        "max_abs_diff": replay_notes["max_abs_diff"],
    })

    print(
        f"ood_ratio: {len(recorded)}/{len(specs)} runs recorded, {len(missing)} missing; "
        f"{len(paired)} arm(s) with both CAMP and MCM; replay: {replay_notes['replayed']} stream(s) replayed, "
        f"{replay_notes['cached']} cached, {len(replay_notes['skipped_runs'])} run(s) skipped",
    )

    for entry in replay_notes["skipped_runs"][:3]:
        print(f"  skipped {entry['run']}: {entry['reason']}")

    if missing:
        print(f"  missing e.g. {layout.relative(missing[0].dir())}")

    return written
