"""
Complementarity table. 
Each compared detector is evaluated alone and with CAMP's standardized score added at equal weight.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.experiment.lib import config_io, layout
from scripts.experiment.lib.items import Context, Item
from scripts.experiment.lib.runs import RunSpec

ITEM_DIR: Path = Path(__file__).resolve().parent
SPEC_PATH: Path = ITEM_DIR / "spec.yml"
PLUGINS: tuple[str, ...] = layout.COMPARED_METHODS
LABEL_PREFIX = "EFT_plugin_"

ITEM = Item(
    name="complementarity",
    title="Compared detectors alone and with CAMP's standardized score added (plug-in use of CAMP)",
    benchmarks=layout.IMAGENET_BENCHMARKS,
)


def plugin_label(
    method: str,
) -> str:

    if method not in PLUGINS:
        raise KeyError(f"{method!r} is not a compared detector; one of {', '.join(PLUGINS)}")

    return f"{LABEL_PREFIX}{method}"


def plugin_of(
    label: str | None,
) -> str | None:

    if label is None or not label.startswith(LABEL_PREFIX):
        return None

    return label[len(LABEL_PREFIX):]


def load_spec(
    path: Path = SPEC_PATH,
) -> dict[str, Any]:

    spec = config_io.load_yaml(path)
    if not isinstance(spec, Mapping) or "runs" not in spec:
        raise ValueError(f"{layout.relative(path)} must be a mapping with a `runs` list")

    return dict(spec)


def expand_spec(
    spec: Mapping[str, Any],
    benchmarks: Sequence[str],
    seeds: Sequence[int],
    backbone: str = "vitb16",
) -> list[RunSpec]:

    specs: list[RunSpec] = []

    for entry in spec["runs"]:
        method = str(entry["method"])
        label = entry.get("label")
        overrides = dict(entry.get("overrides") or {})
        overlays = tuple(str(p) for p in entry.get("overlays") or ())

        for benchmark in benchmarks:
            for seed in seeds:
                specs.append(
                    RunSpec(
                        method,
                        benchmark,
                        int(seed),
                        backbone=backbone,
                        label=None if label is None else str(label),
                        overrides=overrides,
                        overlays=overlays,
                    )
                )

    return specs


def runs(
    ctx: Context,
) -> list[RunSpec]:
    return expand_spec(spec=load_spec(), benchmarks=ctx.benchmarks, seeds=ctx.seeds, backbone=ctx.backbone)


# imported here, under the definitions: the submodules share their names with the stage functions below
from scripts.experiment.complementarity import analyze as _analysis  # noqa: E402
from scripts.experiment.complementarity import figure as _drawing  # noqa: E402


def analyze(
    ctx: Context,
) -> list[Path]:
    return _analysis.analyze(ctx, runs(ctx))


def figure(
    ctx: Context,
) -> list[Path]:
    return _drawing.figure(ctx)


__all__ = [
    "ITEM",
    "LABEL_PREFIX",
    "PLUGINS",
    "analyze",
    "expand_spec",
    "figure",
    "load_spec",
    "plugin_label",
    "plugin_of",
    "runs",
]
