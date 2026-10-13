from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.experiment.lib import svgfig
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.svgfig import Canvas
from scripts.experiment.sensitivity import FIGURE_STEM, TABLE_LONG, Axis, Spec, spec_for

WIDTH = svgfig.WIDTH
COLUMN_STEM = FIGURE_STEM + "_column"
TICK, LAB, LEG = 11.0, 12.0, 11.5
ROWS: tuple[tuple[str, tuple[float, float], tuple[int, ...], str], ...] = (
    ("FPR@95", (0.0, 100.0), (0, 20, 40, 60, 80, 100), "FPR@95 (%)"),
    ("AUROC", (60.0, 100.0), (60, 70, 80, 90, 100), "AUROC (%)"),
)
FIGURE_AXES: tuple[str, ...] = ("window_size", "shift_eps")
PAPER_COLUMNS: dict[str, tuple[float, float]] = {
    "window_size": (60, 218),
    "min_samples": (262, 420),
    "shift_eps": (464, 582),
}
TWO_COLUMNS: tuple[tuple[float, float], ...] = ((60, 300), (344, 582))
LEGEND_GAP, LEGEND_CHAR = 30.0, 5.6


@dataclass(frozen=True)
class Layout:
    stem: str
    width: float
    rows: tuple[tuple[float, float], ...]
    left: float
    right: float
    gap: float
    marker: float
    legend_gap: float
    short_legend: bool
    sparse_ticks: bool


# the rows are the metrics (FPR@95 above AUROC), the columns the swept settings; the tick labels, the x title and
# the part letter stand under the last row only, the y titles beside the first column only
FULL = Layout(FIGURE_STEM, WIDTH, ((30, 140), (160, 270)), 60.0, 582.0, 44.0, 2.4, LEGEND_GAP, False, False)
COLUMN = Layout(COLUMN_STEM, svgfig.COLUMN_WIDTH, ((40, 150), (172, 282)), 46.0, 302.0, 32.0, 2.0, 20.0, True, True)


def read_table(
    ctx: Context,
) -> pd.DataFrame:

    path = ctx.tables / TABLE_LONG
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run `python -m scripts.experiment sensitivity analyze` first")

    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"{path} holds no row; rerun analyze")

    return frame


def column_positions(
    axes: list[str],
    layout_: Layout = FULL,
) -> list[tuple[float, float]]:

    if layout_ is FULL and axes == list(PAPER_COLUMNS):
        return list(PAPER_COLUMNS.values())

    if layout_ is FULL and len(axes) == 2:
        return list(TWO_COLUMNS)

    width = (layout_.right - layout_.left - layout_.gap * (len(axes) - 1)) / len(axes)
    rows = [
        (layout_.left + i * (width + layout_.gap), layout_.left + i * (width + layout_.gap) + width)
        for i in range(len(axes))
    ]

    return rows


def tick_labels(
    axis: Axis,
    values: list[float],
    sparse: bool = False,
) -> list[str]:

    if axis.scale == "log2":
        every = 3 if sparse else 2
        names_ = [
            svgfig.thousands(v) if i % every == 0 else ""
            for i, v in enumerate(values)
        ]

        return names_

    names = [
        svgfig.power_of_ten(v)
        for v in values
    ]

    return names


def draw(
    frame: pd.DataFrame,
    spec: Spec,
    axes: list[str],
    layout_: Layout = FULL,
) -> tuple[str, float]:

    metrics = [(row, span) for row, span in zip(ROWS, layout_.rows) if row[0] in set(frame["metric"])]
    if not metrics:
        raise RuntimeError("sensitivity.csv has none of the metrics FPR@95 / AUROC")

    canvas = Canvas()
    entries = [(label, color, dash, 1.6) for _, label, color, dash in svgfig.GROUP_LINES]
    if layout_.short_legend:
        entries = [(label.replace(" (OpenOOD)", ""), color, dash, sw) for label, color, dash, sw in entries]

    svgfig.legend_row(canvas, entries, width=layout_.width, gap=layout_.legend_gap, char_width=LEGEND_CHAR, size=LEG)
    drawn = 0

    for r, ((metric, ylim, yticks, ylab), (y0, y1)) in enumerate(metrics):
        for c, (axis_name, (x0, x1)) in enumerate(zip(axes, column_positions(axes, layout_))):
            axis = spec.axes[axis_name]
            part = frame[(frame["axis"] == axis_name) & (frame["metric"] == metric)]
            values = sorted(set(part["value"].astype(float)))
            if len(values) < 2:
                raise RuntimeError(f"sensitivity.csv holds fewer than two values of {axis_name}")

            fx = svgfig.log_scale(values[0], values[-1], x0, x1)
            fy = svgfig.linear(ylim[0], ylim[1], y1, y0)
            canvas.line(x0, y1, x1, y1)
            canvas.line(x0, y0, x0, y1)
            bottom = r == len(metrics) - 1

            for value, label in zip(values, tick_labels(axis=axis, values=values, sparse=layout_.sparse_ticks)):
                canvas.line(fx(value), y1, fx(value), y1 + 3)

                if label and bottom:
                    canvas.text(fx(value), y1 + 14, label, size=TICK)

            for value in yticks:
                canvas.line(x0 - 3, fy(value), x0, fy(value))
                canvas.text(x0 - 5, fy(value) + 3.8, str(value), size=TICK, anchor="end")

            if c == 0:
                canvas.text(x0 - 30, (y0 + y1) / 2, ylab, size=LAB, rot=-90)

            if bottom:
                canvas.text((x0 + x1) / 2, y1 + 30, axis.title, size=LAB)
                canvas.text((x0 + x1) / 2, y1 + 48, f"({chr(ord('a') + c)})", size=LAB)

            if values[0] <= axis.default <= values[-1]:
                canvas.line(fx(axis.default), y0, fx(axis.default), y1, stroke="#999", sw=0.8, dash="1.2,2")

            for group, _, color, dash in svgfig.GROUP_LINES:
                rows = part[part["group"] == group].dropna(subset=["mean"]).sort_values("value")
                if rows.empty:
                    continue

                drawn += 1
                points = [(fx(float(v)), fy(float(m))) for v, m in zip(rows["value"], rows["mean"])]
                spread = [(x, float(m), float(sd) if int(n) > 1 and pd.notna(sd) else None)
                    for (x, _y), m, sd, n in zip(points, rows["mean"], rows["std"], rows["n_seeds"])]
                svgfig.std_band(canvas, spread, fy, color)

                if len(points) >= 2:
                    canvas.poly(points, color, dash=dash)

                for x, y in points:
                    canvas.circle(x, y, layout_.marker, color)

    if drawn == 0:
        raise RuntimeError("sensitivity.csv has no recorded value to draw")

    height = metrics[-1][1][1] + 54
    return canvas.svg(layout_.width, height), height


def figure(
    ctx: Context,
) -> list[Path]:

    spec = spec_for(ctx)
    frame = read_table(ctx)
    requested = ctx.option_list("axes", FIGURE_AXES)
    unknown = [axis for axis in requested if axis not in spec.axes]
    if unknown:
        raise ValueError(f"unknown axes {unknown}; one of {', '.join(spec.axes)}")

    axes = [axis for axis in requested if axis in set(frame["axis"])]
    if not axes:
        raise RuntimeError("sensitivity.csv holds none of the requested axes; rerun analyze with --opt axes= covering them")

    render = svgfig.Render.from_context(ctx)
    written: list[Path] = []

    for layout_ in (FULL, COLUMN):
        svg, height = draw(frame, spec, axes, layout_)
        written += svgfig.write_figure(svg, ctx.figures / layout_.stem, width=layout_.width, height=height, render=render)

    return written


__all__ = [
    "COLUMN",
    "COLUMN_STEM",
    "FIGURE_AXES",
    "FULL",
    "Layout",
    "PAPER_COLUMNS",
    "ROWS",
    "TWO_COLUMNS",
    "WIDTH",
    "column_positions",
    "draw",
    "figure",
    "read_table",
    "tick_labels",
]
