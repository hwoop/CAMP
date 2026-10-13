from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment.cost_methods.analyze import TABLE_SUMMARY
from scripts.experiment.lib import layout, svgfig
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.svgfig import Canvas

STEM = "cost_methods"
WIDTH = svgfig.WIDTH
TOP = 28
ROW = 17.0
BAR = 10.0
TICK, LAB, NOTE = 11.0, 12.0, 10.0
TITLE_LINE = 14.0
SHARED = "#c8c8c8"
POSTPROCESSOR = "#6e6e6e"
VALUE = "#333333"
ERROR_CAP = 2.5
ERROR_MIN = 0.5
LABEL_X = 92
GROUP_X = 4
PANEL_X: tuple[tuple[float, float], ...] = ((100, 240), (266, 406), (432, 572))
LOG_TIME = (0.3, 3000.0)
LOG_MEMORY = (3.0, 60000.0)
KIND_TITLES: dict[str, str] = {
    layout.ZERO_SHOT: "Zero-shot",
    layout.TEST_TIME: "Test-time",
}


def read_table(
    ctx: Context,
) -> pd.DataFrame:

    path = ctx.tables / TABLE_SUMMARY
    if not path.is_file():
        raise FileNotFoundError(
            f"{layout.relative(path)} is missing; "
            "run `python -m scripts.experiment cost_methods analyze` first",
        )

    table = pd.read_csv(path)

    if table.empty:
        raise ValueError(f"{layout.relative(path)} holds no row; nothing to draw")

    return table


def axis_rows(
    table: pd.DataFrame,
) -> list[tuple[str, pd.Series | None]]:

    rows: list[tuple[str, pd.Series | None]] = []

    for kind, title in KIND_TITLES.items():
        part = table[table["kind"] == kind]
        if part.empty:
            continue

        rows.append((title, None))
        rows += [(str(row["name"]), row) for _, row in part.iterrows()]

    return rows


def nice_limit(
    value: float,
    steps: tuple[float, ...] = (1, 2, 2.5, 5, 10),
) -> tuple[float, float]:

    magnitude = 10 ** math.floor(math.log10(value / 4))
    step = next(magnitude * s for s in steps if value / (magnitude * s) <= 4)
    return step * math.ceil(value / step), step


def x_axis(
    canvas: Canvas,
    x0: float,
    x1: float,
    bottom: float,
    ticks: list[tuple[float, str]],
    fx: Callable[[float], float],
    title: tuple[str, ...],
    letter: str,
) -> None:

    canvas.line(x0, TOP, x0, bottom)
    canvas.line(x0, bottom, x1, bottom)

    for value, text in ticks:
        canvas.line(fx(value), bottom, fx(value), bottom + 3.5)
        canvas.text(fx(value), bottom + 15, text, size=TICK)

    for line, text in enumerate(title):
        canvas.text((x0 + x1) / 2, bottom + 31 + TITLE_LINE * line, text, size=LAB)

    canvas.text((x0 + x1) / 2, bottom + 33 + TITLE_LINE * len(title) + 4, letter, size=LAB)


def bar(
    canvas: Canvas,
    x_from: float,
    x_to: float,
    y: float,
    color: str,
) -> None:

    if x_to > x_from:
        canvas.rect(x_from, y - BAR / 2, x_to - x_from, BAR, color)


def spread(
    mean: float,
    std: float,
) -> tuple[float, float]:

    if not math.isfinite(std):
        return mean, mean

    return mean - std, mean + std


def error_bar(
    canvas: Canvas,
    x_low: float,
    x_high: float,
    y: float,
) -> None:

    if x_high - x_low < ERROR_MIN:
        return

    canvas.line(x_low, y, x_high, y, stroke=VALUE, sw=0.7)
    canvas.line(x_low, y - ERROR_CAP, x_low, y + ERROR_CAP, stroke=VALUE, sw=0.7)
    canvas.line(x_high, y - ERROR_CAP, x_high, y + ERROR_CAP, stroke=VALUE, sw=0.7)


def std_of(
    row: pd.Series,
    column: str,
) -> float:
    return float(row[column]) if column in row.index and pd.notna(row[column]) else float("nan")


def value_label(
    canvas: Canvas,
    x: float,
    y: float,
    value: float,
    digits: int,
) -> None:

    if math.isfinite(value):
        canvas.text(x + 3, y + 3.6, f"{value:,.{digits}f}", size=NOTE, anchor="start", fill=VALUE)


def draw_svg(
    table: pd.DataFrame,
) -> tuple[str, float]:

    rows = axis_rows(table)
    bottom = TOP + ROW * len(rows)
    height = bottom + 82
    canvas = Canvas()
    svgfig.legend_row(
        canvas,
        [
            ("Copy + image encoder (every method)", SHARED, None, 7.0),
            ("Postprocessor (detection + adaptation)", POSTPROCESSOR, None, 7.0),
        ],
        y=10.0,
    )

    for index, (label, row) in enumerate(rows):
        y = TOP + ROW * (index + 0.5)

        if row is None:
            canvas.add(f'<text x="{GROUP_X}" y="{y + 4:.1f}" font-family="{svgfig.FONT}" font-size="{TICK}" '
                f'font-style="italic" fill="#000">{label}</text>')
        else:
            canvas.text(LABEL_X, y + 4, label, size=TICK, anchor="end")

    methods = [(TOP + ROW * (index + 0.5), row) for index, (_, row) in enumerate(rows) if row is not None]

    (x0, x1) = PANEL_X[0]
    longest = np.nanmax([spread(float(row["total_ms"]), std_of(row, "total_std"))[1] for _, row in methods])
    end, step = nice_limit(float(longest) * 1.1)
    fx = svgfig.linear(0.0, end, x0, x1)

    for y, row in methods:
        total, shared = float(row["total_ms"]), float(row["shared_ms"])
        bar(canvas=canvas, x_from=x0, x_to=fx(min(shared, total)), y=y, color=SHARED)
        bar(canvas=canvas, x_from=fx(min(shared, total)) + 0.8, x_to=fx(total), y=y, color=POSTPROCESSOR)
        low, high = spread(total, std_of(row, "total_std"))
        error_bar(canvas=canvas, x_low=fx(max(low, 0.0)), x_high=fx(high), y=y)
        value_label(canvas, fx(high), y, total, 0)

    x_axis(
        canvas=canvas,
        x0=x0,
        x1=x1,
        bottom=bottom,
        ticks=[(value, f"{value:g}") for value in np.arange(0.0, end + step / 2, step)],
        fx=fx,
        title=("End-to-end time", "per batch (ms)"),
        letter="(a)",
    )

    (x0, x1) = PANEL_X[1]
    fx = svgfig.log_scale(LOG_TIME[0], LOG_TIME[1], x0, x1)

    for y, row in methods:
        detect = float(row["detect_ms"])
        bar(canvas=canvas, x_from=x0, x_to=fx(detect), y=y, color=POSTPROCESSOR)
        low, high = spread(detect, std_of(row, "detect_std"))
        error_bar(canvas=canvas, x_low=fx(max(low, LOG_TIME[0])), x_high=fx(high), y=y)
        value_label(canvas, fx(high), y, detect, 1)

    x_axis(
        canvas=canvas,
        x0=x0,
        x1=x1,
        bottom=bottom,
        ticks=[(10.0 ** k, svgfig.power_of_ten(10.0 ** k)) for k in range(4)],
        fx=fx,
        title=("Postprocessor time", "per batch (ms)"),
        letter="(b)",
    )

    (x0, x1) = PANEL_X[2]
    fx = svgfig.log_scale(LOG_MEMORY[0], LOG_MEMORY[1], x0, x1)

    for y, row in methods:
        memory = float(row["post_gpu_mb"])
        bar(canvas=canvas, x_from=x0, x_to=fx(memory), y=y, color=POSTPROCESSOR)
        value_label(canvas, fx(memory), y, memory, 0)

    x_axis(
        canvas=canvas,
        x0=x0,
        x1=x1,
        bottom=bottom,
        ticks=[(10.0 ** k, svgfig.power_of_ten(10.0 ** k)) for k in range(1, 5)],
        fx=fx,
        title=("Postprocessor peak", "GPU memory (MB)"),
        letter="(c)",
    )

    return canvas.svg(WIDTH, height), height


def draw(
    table: pd.DataFrame,
    stem: Path,
    render: svgfig.Render | None = None,
) -> list[Path]:

    svg, height = draw_svg(table)
    return svgfig.write_figure(svg, stem, width=WIDTH, height=height, render=render)


def figure(
    ctx: Context,
) -> list[Path]:
    return draw(read_table(ctx), ctx.figures / STEM, svgfig.Render.from_context(ctx))


__all__ = [
    "KIND_TITLES",
    "PANEL_X",
    "STEM",
    "axis_rows",
    "draw",
    "draw_svg",
    "error_bar",
    "figure",
    "nice_limit",
    "read_table",
    "spread",
]
