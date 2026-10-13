from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.experiment.direction_warmup.analyze import DEFAULT_WARMUP, TABLE_SUMMARY
from scripts.experiment.lib import layout, svgfig
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.svgfig import Canvas

STEM = "direction_warmup"
COLUMN_STEM = "direction_warmup_column"
WIDTH, HEIGHT = svgfig.WIDTH, 216
TOP, BOTTOM = 50, 180
TICK, LAB, LEG = 11.0, 12.0, 11.5
PANEL_X: tuple[tuple[int, int], ...] = ((56, 296), (352, 592))
# the whole-stream qualifier of the two metrics goes to the caption: the titles must fit the panel height
PANELS: tuple[tuple[str, str, str], ...] = (
    ("FPR@95", "FPR@95 (%)", "a"),
    ("AUROC", "AUROC (%)", "b"),
)
X_LABEL = "Samples in the window"
MCM_LABEL = "Warm-up score"
MCM_DASH = "1.2,2"
LEGEND_GAP, LEGEND_CHAR = 30.0, 5.6


@dataclass(frozen=True)
class Layout:
    stem: str
    width: float
    height: float
    panels: tuple[tuple[float, float, float, float], ...]
    ylabels: tuple[str, ...]
    stacked: bool


# side by side at the text width (the paper figure), and stacked at the column width with flat panels: the x title
# under the last panel only, the legend on two rows with short group names; the part labels at the top left in both
FULL = Layout(STEM, WIDTH, HEIGHT, tuple((x0, x1, TOP, BOTTOM) for x0, x1 in PANEL_X), tuple(p[1] for p in PANELS), False)
COLUMN = Layout(
    COLUMN_STEM,
    svgfig.COLUMN_WIDTH,
    334,
    ((50, 302, 56, 152), (50, 302, 198, 294)),
    tuple(p[1] for p in PANELS),
    True,
)


def read_summary(
    ctx: Context,
) -> pd.DataFrame:

    path = ctx.tables / TABLE_SUMMARY
    if not path.is_file():
        raise FileNotFoundError(f"{layout.relative(path)} is missing; run `python -m scripts.experiment direction_warmup analyze` first")

    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"{layout.relative(path)} holds no row; nothing to draw")

    return frame


def nice_range(
    values: list[float],
) -> tuple[float, float, list[float]]:

    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return 0.0, 100.0, [0.0, 50.0, 100.0]

    lo, hi = min(finite), max(finite)
    span = (hi - lo) or 1.0
    step = 10 ** math.floor(math.log10(span))
    for candidate in (step * 0.1, step * 0.2, step * 0.5, step, step * 2, step * 5, step * 10):
        if span / candidate <= 6:
            step = candidate
            break

    low = math.floor(lo / step) * step
    high = math.ceil(hi / step) * step
    if high == low:
        high = low + step

    ticks = []
    value = low
    while value <= high + 1e-9:
        ticks.append(round(value, 10))
        value += step

    return low, high, ticks


def is_power_of_two(
    t: int,
) -> bool:
    return t >= 1 and (t & (t - 1)) == 0


def draw(
    summary: pd.DataFrame,
    layout_: Layout = FULL,
) -> str:

    ts = sorted(set(int(v) for v in summary["t"]))
    if len(ts) < 2:
        raise RuntimeError("direction_warmup.csv holds fewer than two window fills; nothing to draw against")

    canvas = Canvas()
    entries = [(label, color, dash, 1.6) for _, label, color, dash in svgfig.GROUP_LINES]
    entries.append((MCM_LABEL, "#555", MCM_DASH, 0.9))
    if layout_.stacked:
        entries = [(label.replace(" (OpenOOD)", ""), color, dash, sw) for label, color, dash, sw in entries]
        for row, start in enumerate(range(0, len(entries), 2)):
            svgfig.legend_row(canvas, entries[start:start + 2], width=layout_.width, y=11.0 + 14.0 * row,
                gap=LEGEND_GAP, char_width=LEGEND_CHAR, size=LEG)
    else:
        svgfig.legend_row(canvas, entries, width=layout_.width, gap=LEGEND_GAP, char_width=LEGEND_CHAR, size=LEG)

    last = len(layout_.panels) - 1
    for index, ((x0, x1, top, bottom), (metric, _, tag), ylab) in enumerate(zip(layout_.panels, PANELS, layout_.ylabels)):
        part = summary[summary["metric"] == metric]
        fx = svgfig.log_scale(ts[0], ts[-1], x0 + 6, x1 - 6)
        values = part["camp_mean"].dropna().tolist() + part["mcm_mean"].dropna().tolist()
        values += (part["camp_mean"] + part["camp_std"].fillna(0.0)).dropna().tolist()
        values += (part["camp_mean"] - part["camp_std"].fillna(0.0)).dropna().tolist()
        lo, hi, yticks = nice_range(values)
        lo, hi = max(lo, 0.0), min(hi, 100.0)
        yticks = [v for v in yticks if lo <= v <= hi]
        fy = svgfig.linear(lo, hi, bottom, top)
        canvas.line(x0, bottom, x1, bottom)
        canvas.line(x0, top, x0, bottom)

        for t in ts:
            if not is_power_of_two(t):
                continue

            canvas.line(fx(t), bottom, fx(t), bottom + 3)

            if int(math.log2(t)) % 2 == 1:
                canvas.text(fx(t), bottom + 14, svgfig.thousands(t), size=TICK)

        for value in yticks:
            canvas.line(x0 - 3, fy(value), x0, fy(value))
            canvas.text(x0 - 5, fy(value) + 3.8, f"{value:g}", size=TICK, anchor="end")

        canvas.text(x0 - 34, (top + bottom) / 2, ylab, size=LAB, rot=-90)

        svgfig.panel_letter(canvas, x0 - 42, top - 6, tag)
        if not layout_.stacked or index == last:
            canvas.text((x0 + x1) / 2, bottom + 30, X_LABEL, size=LAB)

        if ts[0] <= DEFAULT_WARMUP <= ts[-1]:
            canvas.line(fx(DEFAULT_WARMUP), top, fx(DEFAULT_WARMUP), bottom, stroke="#999", sw=0.8, dash="1.2,2")

        drawn = False
        canvas.begin_clip(f"clip_{tag}", x0, top, x1 - x0, bottom - top)

        for group, _, color, dash in svgfig.GROUP_LINES:
            rows = part[part["group"] == group].dropna(subset=["camp_mean"]).sort_values("t")
            if rows.empty:
                continue

            drawn = True
            mcm = float(rows["mcm_mean"].iloc[0])
            if math.isfinite(mcm):
                canvas.line(x0, fy(mcm), x1, fy(mcm), stroke=color, sw=0.9, dash=MCM_DASH)

            points = [(fx(int(t)), fy(float(v))) for t, v in zip(rows["t"], rows["camp_mean"])]
            spread = [(x, float(v), float(sd) if int(n) > 1 else None)
                for (x, _y), v, sd, n in zip(points, rows["camp_mean"], rows["camp_std"], rows["n_seeds"])]
            svgfig.std_band(canvas, spread, fy, color)

            if len(points) >= 2:
                canvas.poly(points, color, dash=dash)

            for (x, y), t in zip(points, rows["t"]):
                if is_power_of_two(int(t)):
                    canvas.circle(x, y, 2.4, color)

        canvas.end_clip()

        if not drawn:
            canvas.text((x0 + x1) / 2, (top + bottom) / 2, "no recorded stream", size=9, fill=svgfig.NEUTRAL)

    return canvas.svg(layout_.width, layout_.height)


def figure(
    ctx: Context,
) -> list[Path]:

    summary = read_summary(ctx)
    ctx.notes["figure_groups"] = sorted(set(summary["group"]))
    render = svgfig.Render.from_context(ctx)
    written: list[Path] = []

    for layout_ in (FULL, COLUMN):
        written += svgfig.write_figure(
            draw(summary, layout_),
            ctx.figures / layout_.stem,
            width=layout_.width,
            height=layout_.height,
            render=render,
        )

    return written


__all__ = [
    "COLUMN",
    "COLUMN_STEM",
    "FULL",
    "HEIGHT",
    "Layout",
    "MCM_LABEL",
    "PANELS",
    "STEM",
    "WIDTH",
    "draw",
    "figure",
    "is_power_of_two",
    "nice_range",
    "read_summary",
]
