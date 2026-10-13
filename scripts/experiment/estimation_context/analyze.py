from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experiment.lib import context, layout, manifest, replay, tables
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.runs import RunSpec

METRICS = (
    "ood_sample_share",
    "ood_weight_share",
    "max_class_weight",
    "delta_norm",
    "cos_to_oracle",
    "projection_gap",
    "gap_id_std",
)


def wanted_leaves(
    ctx: Context,
    benchmark: str,
) -> list[str]:

    leaves = [leaf for group in layout.LEAVES[benchmark].values() for leaf in group]
    selected = ctx.option_list("leaves", leaves)
    unknown = set(selected) - set(leaves)
    if unknown:
        raise ValueError(f"unknown leaves for {benchmark}: {sorted(unknown)}")

    return selected


def replay_run(
    spec: RunSpec,
    ctx: Context,
) -> list[dict[str, Any]]:

    run = replay.RecordedRun(spec.dir())
    features = replay.features_of(run, ctx.device)
    policy = run.policy()
    records = []

    for stream in replay.streams(run, features, wanted_leaves(ctx=ctx, benchmark=spec.benchmark)):
        replay.verify_stream(stream)
        result = replay.replay_camp(stream, features)
        if result.final is None:
            raise RuntimeError(f"{run.dir}/{stream.name}: the stream never filled CAMP's warm-up")

        difference = replay.check_against_recorded(stream, result.score)
        rows = stream.features.to(result.final.u.device)

        if result.final_rows is None:
            raise RuntimeError(f"{run.dir}/{stream.name}: replay returned no final window rows")

        window, msp, pseudo = result.final_rows
        weights = context.weight_context(
            msp.detach().cpu().numpy(),
            pseudo.detach().cpu().numpy(),
            stream.is_ood[result.final_window],
            id_reference=policy.camp.id_reference,
            weighting=policy.camp.weighting,
        )

        separation = context.separation_context(rows, stream.labels, result.final, class_num=features.class_num)
        records.append(
            {
                "benchmark": spec.benchmark,
                "seed": spec.seed,
                "arm": spec.label or "camp",
                "stream": stream.name,
                "leaf": stream.leaf,
                "group": stream.group,
                "window_start": result.final_window.start,
                "window_end": result.final_window.stop,
                "window_rows": int(window.shape[0]),
                "id_reference": policy.camp.id_reference,
                "weighting": policy.camp.weighting,
                **weights,
                **separation,
                "replay_max_abs_diff": float(difference),
            },
        )

    ctx.notes.setdefault("feature_caches", {})[layout.relative(spec.dir())] = list(features.cache_paths)
    return records


def source_signature(
    spec: RunSpec,
    leaves: list[str],
) -> dict[str, str]:

    files = [spec.dir() / name for name in ("config.yml", "ood.csv", "log.txt")]
    files += [spec.dir() / "scores" / f"{leaf}.npz" for leaf in leaves]
    table = {
        str(path): manifest.sha256_of(path) if path.is_file() else "missing"
        for path in files
    }

    return table


def cached_run(
    spec: RunSpec,
    ctx: Context,
) -> list[dict[str, Any]]:

    leaves = wanted_leaves(ctx=ctx, benchmark=spec.benchmark)
    signature = source_signature(spec=spec, leaves=leaves)
    path = ctx.cache / f"context__{spec.dir().parent.name}__{spec.dir().name}.json"
    if path.is_file() and ctx.option("refresh", "0") != "1":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("source_signature") == signature and data.get("leaves") == leaves:
            ctx.notes.setdefault("context_from_cache", []).append(layout.relative(spec.dir()))
            ctx.notes.setdefault("feature_caches", {})[layout.relative(spec.dir())] = data.get("feature_caches", [])
            return data["rows"]

    rows = replay_run(spec=spec, ctx=ctx)
    path.write_text(
        json.dumps(
            {
                "source_signature": signature,
                "leaves": leaves,
                "rows": rows,
                "feature_caches": ctx.notes.get("feature_caches", {}).get(
                    layout.relative(spec.dir()),
                    [],
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return rows


def summarize(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    leaves = frame.copy()
    leaves["scope"] = leaves["leaf"]
    groups = frame.groupby(["benchmark", "arm", "group", "seed"], as_index=False)[list(METRICS)].mean()
    groups["scope"] = groups["group"]
    combined = pd.concat([leaves, groups], ignore_index=True)
    records = []

    for (benchmark, arm, group, scope), part in combined.groupby(["benchmark", "arm", "group", "scope"]):
        record = {
            "benchmark": benchmark,
            "arm": arm,
            "group": group,
            "scope": scope,
            "seeds": ",".join(str(s) for s in sorted(part["seed"].unique())),
            "n_seeds": int(part["seed"].nunique()),
        }

        for metric in METRICS:
            record[metric] = float(part[metric].mean())
            record[f"{metric}_std"] = float(part[metric].std(ddof=1))

        records.append(record)

    return pd.DataFrame(records)


def markdown(
    frame: pd.DataFrame,
) -> str:

    lines = [
        "Estimation context",
        "",
        "Weight shares are percentages. Gap is measured in ID standard deviations; "
        "cosine compares the window direction with the full-ID oracle direction.",
        "",
        "| Benchmark | Arm | Stream / group | Seeds | OOD rows | OOD weight | Max class weight | "
        "Mean difference norm | Oracle cosine | Projection gap |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in frame.to_dict("records"):
        metrics = [f"{row[key]:.4f}" for key in (
                "ood_sample_share",
                "ood_weight_share",
                "max_class_weight",
                "delta_norm",
                "cos_to_oracle",
                "gap_id_std",
            )]
        lines.append(
            f"| {row['benchmark']} | {row['arm']} | {row['scope']} | {row['seeds']} | "
            + " | ".join(metrics) + " |",
        )

    return "\n".join(lines) + "\n"


def analyze(
    ctx: Context,
    specs: list[RunSpec],
) -> list[Path]:

    if ctx.dry_run:
        print("[dry-run] would replay the canonical and weighting-ablation runs for estimation context")
        return []

    records = []

    for spec in specs:
        if spec.recorded():
            records += cached_run(spec=spec, ctx=ctx)
        else:
            ctx.notes.setdefault("missing_runs", []).append(layout.relative(spec.dir()))

    if not records:
        raise FileNotFoundError("no recorded runs for estimation_context; run the item's plan first")

    frame = pd.DataFrame(records)
    summary = summarize(frame)
    difference = float(frame["replay_max_abs_diff"].max())
    ctx.notes.update(
        {
            "replay_max_abs_diff": difference,
            "replay_within_tolerance": bool(np.isfinite(difference) and difference <= 1e-3),
            "window_stage": "end_of_stream_after_update",
        },
    )

    return [
        tables.write_csv(frame, ctx.tables / "context_streams.csv"),
        tables.write_csv(summary, ctx.tables / "context_summary.csv"),
        tables.write_markdown(markdown(summary), ctx.tables / "context_summary.md"),
    ]
