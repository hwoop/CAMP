from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from scripts.experiment.covariance_adjustment.analyze import DENSITIES_CSV, DIRECTION_NAMES, DIRECTIONS, GAPS_CSV
from scripts.experiment.lib import layout, svgfig, tables
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.svgfig import Canvas

DEFAULT_LEAVES: tuple[str, ...] = ("ssb_hard", "textures")
STEM = "adjustment"
COLUMN_STEM = "adjustment_column"
WIDTH = svgfig.WIDTH
COLUMN_WIDTH = svgfig.COLUMN_WIDTH
# above a pair: the part letter and the leaf name at TOP - 20, each panel's title (the direction) at TOP - 5;
# below it: the ticks at BOTTOM + 14 and the x title at BOTTOM + 30
TOP, BOTTOM = 56, 152
HEIGHT = BOTTOM + 34
PANEL_HEIGHT = BOTTOM - TOP
# the column layout stacks one pair of panels per leaf, the panels at their full-width size, the x title once
COLUMN_TOP = 56
COLUMN_ROW_GAP = 58
COLUMN_BELOW = 34
TICK, LAB, LEG = 11.0, 12.0, 11.5
X_LIMITS: tuple[float, float] = (-4.0, 10.0)
X_TICKS: tuple[int, ...] = (-4, 0, 4, 8)
PAPER_PAIRS: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = (
    ((30, 160), (172, 302)),
    ((330, 460), (472, 596)),
)
PANEL_LETTERS = "abcdefgh"


def read_tables(
    ctx: Context,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    densities_path = ctx.tables / DENSITIES_CSV
    gaps_path = ctx.tables / GAPS_CSV
    for path in (densities_path, gaps_path):
        if not path.is_file():
            raise FileNotFoundError(f"{layout.relative(path)} is missing; run `analyze` first.")

    return pd.read_csv(densities_path), pd.read_csv(gaps_path)


def pair_positions(
    count: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:

    if count == 2:
        return list(PAPER_PAIRS)

    left, right, between, within = 30.0, 596.0, 28.0, 12.0
    width = (right - left - between * (count - 1) - within * count) / (2 * count)
    pairs = []
    x = left

    for _ in range(count):
        pairs.append(((x, x + width), (x + width + within, x + 2 * width + within)))
        x += 2 * width + within + between

    return pairs


def x_limits(
    gaps: pd.DataFrame,
    benchmark: str,
    leaves: Sequence[str],
) -> tuple[float, float]:

    part = gaps[(gaps["bench"] == benchmark) & gaps["leaf"].isin(list(leaves))]["gap_id_std"].dropna()
    upper = X_LIMITS[1]
    if not part.empty and float(part.max()) > upper - 1.0:
        upper = float(math.ceil(part.max()) + 2)

    return X_LIMITS[0], upper


def mean_gap(
    gaps: pd.DataFrame,
    benchmark: str,
    leaf: str,
    direction: str,
) -> float | None:

    part = gaps[(gaps["bench"] == benchmark) & (gaps["leaf"] == leaf) & (gaps["direction"] == direction)]
    values = part["gap_id_std"].dropna()
    return float(values.mean()) if not values.empty else None


def x_ticks(
    limits: tuple[float, float],
) -> list[int]:
    return [v for v in list(X_TICKS) + list(range(12, int(limits[1]) + 1, 4)) if v <= limits[1]]


def curve(
    densities: pd.DataFrame,
    benchmark: str,
    leaf: str,
    direction: str,
    limits: tuple[float, float],
) -> pd.DataFrame:

    part = densities[(densities["bench"] == benchmark) & (densities["leaf"] == leaf)
        & (densities["direction"] == direction)].sort_values("z")
    part = part[part["z"].between(limits[0], limits[1])]
    if part.empty:
        raise ValueError(f"densities.csv holds no {direction!r} curve of {leaf!r} on {benchmark!r} inside {limits}")

    return part


def draw_panel(
    canvas: Canvas,
    part: pd.DataFrame,
    gap: float | None,
    x0: float,
    x1: float,
    top: float,
    bottom: float,
    limits: tuple[float, float],
    title: str,
    first: bool,
    x_title: bool = True,
) -> None:

    ymax = float(max(part["id"].max(), part["ood"].max())) * 1.1 or 1.0
    fx = svgfig.linear(limits[0], limits[1], x0, x1)
    fy = svgfig.linear(0.0, ymax, bottom, top)
    canvas.line(x0, bottom, x1, bottom)
    canvas.line(x0, top, x0, bottom)

    for value in x_ticks(limits):
        canvas.line(fx(value), bottom, fx(value), bottom + 3)
        canvas.text(fx(value), bottom + 14, svgfig.minus(value), size=TICK)

    canvas.area([(fx(z), fy(y)) for z, y in zip(part["z"], part["id"])], bottom, svgfig.ID)
    canvas.area([(fx(z), fy(y)) for z, y in zip(part["z"], part["ood"])], bottom, svgfig.OOD, dash="4,2")
    canvas.line(fx(0.0), top + 4, fx(0.0), bottom, stroke="#999", sw=0.7, dash="1.2,2")
    if gap is not None:
        canvas.line(fx(gap), top + 4, fx(gap), bottom, stroke=svgfig.OOD, sw=0.7, dash="1.2,2")
        canvas.text(x1 - 2, top + 10, f"gap {gap:.2f}", size=10.5, anchor="end", fill="#333")

    if first:
        canvas.text(x0 - 12, (top + bottom) / 2, "Density", size=LAB, rot=-90)

    canvas.text((x0 + x1) / 2, top - 5, title, size=LAB)
    if x_title:
        canvas.text((x0 + x1) / 2, bottom + 30, "Distance from ID mean", size=LAB)


def legend(
    canvas: Canvas,
    width: float,
) -> None:

    lx = width / 2 - 70
    canvas.rect(lx, 6, 16, 9, svgfig.ID, stroke=svgfig.ID, sw=1, fill_opacity=0.25)
    canvas.text(lx + 21, 14.5, "ID", size=LEG, anchor="start")
    canvas.rect(lx + 60, 6, 16, 9, svgfig.OOD, stroke=svgfig.OOD, sw=1, fill_opacity=0.25, dash="4,2")
    canvas.text(lx + 81, 14.5, "OOD", size=LEG, anchor="start")


def draw(
    densities: pd.DataFrame,
    gaps: pd.DataFrame,
    benchmark: str,
    leaves: Sequence[str],
    limits: tuple[float, float] | None = None,
) -> str:

    limits = limits or x_limits(gaps=gaps, benchmark=benchmark, leaves=leaves)
    canvas = Canvas()
    pairs = pair_positions(len(leaves))
    for index, (leaf, pair) in enumerate(zip(leaves, pairs)):
        for (x0, x1), direction in zip(pair, DIRECTIONS):
            part = curve(densities, benchmark, leaf, direction, limits)
            gap = mean_gap(gaps=gaps, benchmark=benchmark, leaf=leaf, direction=direction)
            draw_panel(canvas, part, gap, x0, x1, TOP, BOTTOM, limits, DIRECTION_NAMES[direction],
                direction == DIRECTIONS[0])

        svgfig.panel_letter(canvas, pair[0][0] - 26, TOP - 20, PANEL_LETTERS[index])
        canvas.text((pair[0][0] + pair[1][1]) / 2, TOP - 20, tables.leaf_name(leaf), size=LAB)

    legend(canvas, WIDTH)
    return canvas.svg(WIDTH, HEIGHT)


def column_height(
    count: int,
) -> float:
    return COLUMN_TOP + count * PANEL_HEIGHT + (count - 1) * COLUMN_ROW_GAP + COLUMN_BELOW


def draw_column(
    densities: pd.DataFrame,
    gaps: pd.DataFrame,
    benchmark: str,
    leaves: Sequence[str],
    limits: tuple[float, float] | None = None,
) -> tuple[str, float]:

    limits = limits or x_limits(gaps=gaps, benchmark=benchmark, leaves=leaves)
    canvas = Canvas()
    pair = PAPER_PAIRS[0]
    centre = (pair[0][0] + pair[1][1]) / 2
    for index, leaf in enumerate(leaves):
        top = COLUMN_TOP + index * (PANEL_HEIGHT + COLUMN_ROW_GAP)
        bottom = top + PANEL_HEIGHT
        for (x0, x1), direction in zip(pair, DIRECTIONS):
            part = curve(densities, benchmark, leaf, direction, limits)
            gap = mean_gap(gaps=gaps, benchmark=benchmark, leaf=leaf, direction=direction)
            # the rows share the x axis: its title stands under the last row only
            draw_panel(canvas, part, gap, x0, x1, top, bottom, limits, DIRECTION_NAMES[direction],
                direction == DIRECTIONS[0], x_title=index == len(leaves) - 1)

        svgfig.panel_letter(canvas, pair[0][0] - 26, top - 20, PANEL_LETTERS[index])
        canvas.text(centre, top - 20, tables.leaf_name(leaf), size=LAB)

    legend(canvas, COLUMN_WIDTH)
    height = column_height(len(leaves))
    return canvas.svg(COLUMN_WIDTH, height), height


def figure(
    ctx: Context,
) -> list[Path]:

    densities, gaps = read_tables(ctx)
    benchmark = ctx.benchmarks[0]
    leaves: Sequence[str] = ctx.option_list("leaves", DEFAULT_LEAVES)
    available = list(dict.fromkeys(densities[densities["bench"] == benchmark]["leaf"]))
    unknown = [leaf for leaf in leaves if leaf not in available]
    if unknown:
        raise ValueError(
            f"no densities for {unknown} on {benchmark}; available: {available or 'none'} "
            "(run `analyze` for that benchmark, or pass --opt leaves=...)",
        )

    limits = x_limits(gaps=gaps, benchmark=benchmark, leaves=leaves)
    seeds = sorted({int(s) for s in gaps[gaps["bench"] == benchmark]["seed"]})
    ctx.notes.update({"figure_leaves": list(leaves), "figure_seeds": seeds, "figure_x_limits": list(limits)})
    render = svgfig.Render.from_context(ctx)
    written = svgfig.write_figure(
        draw(densities, gaps, benchmark, leaves, limits),
        ctx.figures / STEM,
        width=WIDTH,
        height=HEIGHT,
        render=render,
    )

    svg, height = draw_column(densities, gaps, benchmark, leaves, limits)
    written += svgfig.write_figure(svg, ctx.figures / COLUMN_STEM, width=COLUMN_WIDTH, height=height, render=render)
    return written


__all__ = [
    "COLUMN_STEM",
    "COLUMN_WIDTH",
    "DEFAULT_LEAVES",
    "HEIGHT",
    "PAPER_PAIRS",
    "STEM",
    "WIDTH",
    "X_LIMITS",
    "X_TICKS",
    "column_height",
    "curve",
    "draw",
    "draw_column",
    "draw_panel",
    "figure",
    "mean_gap",
    "pair_positions",
    "read_tables",
    "x_limits",
]
