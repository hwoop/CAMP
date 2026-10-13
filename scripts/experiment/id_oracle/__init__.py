"""
CAMP with oracle ID statistics: the window estimate against the window oracle,
the full-ID oracle and the rank-one ceiling.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.experiment.lib import config_io
from scripts.experiment.lib.items import Context, Item
from scripts.experiment.lib.runs import RunSpec

# imported here, under the definitions: the submodule shares its name with the stage function below
from scripts.experiment.id_oracle import analyze as _analysis  # noqa: E402

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
    name="id_oracle",
    title="CAMP with oracle ID statistics (window estimate vs window oracle vs full-ID oracle vs rank-one ceiling)",
    benchmarks=tuple(str(b) for b in SPEC["benchmarks"]),
    seeds=tuple(int(s) for s in SPEC["seeds"]),
)


def _labelled_overlays(
    label: str,
) -> tuple[str, ...]:

    try:
        from scripts.experiment.ood_ratio import SPEC as ratio_spec
    except ImportError:
        return ()

    for entry in ratio_spec.get("runs", ()):
        if entry.get("label") == label:
            return tuple(str(o) for o in entry.get("overlays") or ())

    return ()


def expand_spec(
    spec: Mapping[str, Any],
    benchmarks: tuple[str, ...],
    seeds: tuple[int, ...],
    backbone: str = "vitb16",
    labels: tuple[str, ...] = (),
) -> list[RunSpec]:

    entries: list[tuple[str, str | None, dict[str, Any], tuple[str, ...]]] = []

    for entry in spec["runs"]:
        entries.append(
            (
                str(entry["method"]),
                None if entry.get("label") is None else str(entry["label"]),
                dict(entry.get("overrides") or {}),
                tuple(str(o) for o in entry.get("overlays") or ()),
            ),
        )

    for label in labels:
        entries.append(("camp", str(label), {}, _labelled_overlays(str(label))))

    result: list[RunSpec] = []

    for method, label, overrides, overlays in entries:
        for benchmark in benchmarks:
            for seed in seeds:
                result.append(
                    RunSpec(
                        method,
                        benchmark,
                        int(seed),
                        backbone=backbone,
                        label=label,
                        overrides=overrides,
                        overlays=overlays,
                    ),
                )

    return result


def runs(
    ctx: Context,
) -> list[RunSpec]:

    labels = tuple(ctx.option_list("labels"))
    return expand_spec(spec=SPEC, benchmarks=ctx.benchmarks, seeds=ctx.seeds, backbone=ctx.backbone, labels=labels)


def analyze(
    ctx: Context,
) -> list[Path]:
    return _analysis.analyze(ctx)


def figure(
    ctx: Context,
) -> list[Path]:
    return []


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
