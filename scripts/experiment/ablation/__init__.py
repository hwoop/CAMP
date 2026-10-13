"""Ablation study of CAMP's three components."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.experiment.ablation import analyze as analyze_module
from scripts.experiment.ablation import figure as figure_module
from scripts.experiment.lib import config_io
from scripts.experiment.lib.items import Context, Item
from scripts.experiment.lib.runs import RunSpec

SPEC_PATH: Path = Path(__file__).with_name("spec.yml")


@dataclass(frozen=True)
class Arm:

    method: str
    label: str | None
    name: str
    overrides: Mapping[str, Any] = field(default_factory=dict)
    overlays: tuple[str, ...] = ()


@dataclass(frozen=True)
class Spec:

    benchmarks: tuple[str, ...]
    seeds: tuple[int, ...]
    arms: tuple[Arm, ...]


def load_spec(
    path: Path = SPEC_PATH,
) -> Spec:

    tree = config_io.load_yaml(path)
    if not isinstance(tree, Mapping) or "runs" not in tree:
        raise ValueError(f"{path} must be a mapping with benchmarks, seeds and runs")

    arms: list[Arm] = []

    for index, entry in enumerate(tree["runs"]):
        if not isinstance(entry, Mapping) or "method" not in entry or "name" not in entry:
            raise ValueError(f"{path}: runs[{index}] needs at least method and name")

        label = entry.get("label")
        arms.append(
            Arm(
                method=str(entry["method"]),
                label=None if label is None else str(label),
                name=str(entry["name"]),
                overrides=dict(entry.get("overrides") or {}),
                overlays=tuple(entry.get("overlays") or ()),
            ),
        )

    return Spec(
        benchmarks=tuple(str(b) for b in tree.get("benchmarks", ())),
        seeds=tuple(int(s) for s in tree.get("seeds", ())),
        arms=tuple(arms),
    )


SPEC: Spec = load_spec()
ARMS: tuple[Arm, ...] = SPEC.arms

ITEM = Item(
    name="ablation",
    title="Ablation of CAMP's components: class balancing, covariance adjustment, confidence weighting",
    benchmarks=SPEC.benchmarks,
    seeds=SPEC.seeds,
)


def expand(
    arms: Sequence[Arm],
    benchmarks: Sequence[str],
    seeds: Sequence[int],
    backbone: str,
) -> list[RunSpec]:

    run_specs = [
        RunSpec(
            arm.method,
            benchmark,
            seed,
            backbone=backbone,
            label=arm.label,
            overrides=dict(arm.overrides),
            overlays=arm.overlays,
        )
        for arm in arms
        for benchmark in benchmarks
        for seed in seeds
    ]

    return run_specs


def runs(
    ctx: Context,
) -> list[RunSpec]:
    return expand(ARMS, ctx.benchmarks, ctx.seeds, ctx.backbone)


def analyze(
    ctx: Context,
) -> list[Path]:
    return analyze_module.write_tables(ctx, runs(ctx), ARMS)


def figure(
    ctx: Context,
) -> list[Path]:
    return figure_module.draw(ctx)
