"""
How closely CAMP's window approximates the ID distribution, batch by batch.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.experiment.lib import config_io
from scripts.experiment.lib.items import Context, Item
from scripts.experiment.lib.runs import RunSpec

# imported here, under the definitions: the submodules share their names with the stage functions below
from scripts.experiment.id_approximation import analyze as _analysis  # noqa: E402
from scripts.experiment.id_approximation import figure as _drawing  # noqa: E402

SPEC_PATH: Path = Path(__file__).with_name("spec.yml")


def load_spec(
    path: Path = SPEC_PATH,
) -> dict[str, Any]:

    spec = config_io.load_yaml(path)
    if not isinstance(spec, Mapping) or "runs" not in spec:
        raise ValueError(f"{path} must be a mapping with a `runs` list.")

    return dict(spec)


SPEC: dict[str, Any] = load_spec()

ITEM = Item(
    name="id_approximation",
    title="Approximation of the ID distribution over the stream (CAMP window vs window oracle vs TANL ID memory)",
    benchmarks=tuple(str(b) for b in SPEC["benchmarks"]),
    seeds=tuple(int(s) for s in SPEC["seeds"]),
)


def expand_spec(
    spec: Mapping[str, Any],
    benchmarks: tuple[str, ...],
    seeds: tuple[int, ...],
    backbone: str = "vitb16",
) -> list[RunSpec]:

    result: list[RunSpec] = []

    for entry in spec["runs"]:
        method = str(entry["method"])
        label = entry.get("label")
        overrides = dict(entry.get("overrides") or {})
        overlays = tuple(str(o) for o in entry.get("overlays") or ())

        for benchmark in benchmarks:
            for seed in seeds:
                result.append(
                    RunSpec(
                        method,
                        benchmark,
                        int(seed),
                        backbone=backbone,
                        label=None if label is None else str(label),
                        overrides=overrides,
                        overlays=overlays,
                    ),
                )

    return result


def runs(
    ctx: Context,
) -> list[RunSpec]:
    return expand_spec(spec=SPEC, benchmarks=ctx.benchmarks, seeds=ctx.seeds, backbone=ctx.backbone)


def analyze(
    ctx: Context,
) -> list[Path]:
    return _analysis.analyze(ctx)


def figure(
    ctx: Context,
) -> list[Path]:
    return _drawing.figure(ctx)


__all__ = [
    "ITEM",
    "SPEC",
    "SPEC_PATH",
    "analyze",
    "expand_spec",
    "figure",
    "load_spec",
    "runs",
]
