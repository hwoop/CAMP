"""
Motivation figure. 
ID means and the stream mean of the last window of the iNaturalist stream, 
shown in feature space.
"""
from __future__ import annotations

from pathlib import Path

from scripts.experiment.lib.config_io import load_yaml
from scripts.experiment.lib.items import Context, Item
from scripts.experiment.lib.runs import RunSpec

from .analyze import analyze
from .figure import figure

SPEC_PATH = Path(__file__).with_name("spec.yml")

ITEM = Item(
    name="motivation",
    title="ID means and the stream mean of the last iNaturalist window (selection bias of hard selection)",
    benchmarks=("imagenet_openood",),
    seeds=(0,),
)


def spec_seeds(
    ctx: Context,
) -> tuple[int, ...]:

    chosen = ctx.option_list("seed")
    return tuple(int(seed) for seed in chosen) if chosen else tuple(ctx.seeds)


def runs(
    ctx: Context,
) -> list[RunSpec]:

    spec = load_yaml(SPEC_PATH)
    specs: list[RunSpec] = []

    for entry in spec["runs"]:
        for benchmark in ctx.benchmarks:
            for seed in spec_seeds(ctx):
                specs.append(
                    RunSpec(
                        method=str(entry["method"]),
                        benchmark=benchmark,
                        seed=int(seed),
                        backbone=ctx.backbone,
                        label=entry.get("label"),
                        overrides=dict(entry.get("overrides") or {}),
                        overlays=tuple(entry.get("overlays") or ()),
                    ),
                )

    return specs


__all__ = [
    "ITEM",
    "analyze",
    "figure",
    "runs",
    "spec_seeds",
]
