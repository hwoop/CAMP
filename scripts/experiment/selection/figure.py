from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.experiment.lib import layout, svgfig
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.svgfig import Canvas
from scripts.experiment.selection.analyze import DEFAULT_MARGIN

STEM = "selection"
MAIN_BENCHMARK = "imagenet_openood"
WIDTH, HEIGHT = svgfig.WIDTH, 226
TOP, BOTTOM = 22, 160
TICK, LAB, NOTE = 11.5, 12.5, 10.5
GRAY = "#8a8a8a"
BAND, BAND_OPACITY = svgfig.TANL, 0.45
X_TICKS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)
PANELS: tuple[tuple[str, tuple[float, float], tuple[float, ...], str, str, float], ...] = (
    ("FPR@95", (0.0, 80.0), (0, 20, 40, 60, 80), "FPR@95 (%)", "(a)", -8.0),
    ("AUROC", (60.0, 100.0), (60, 70, 80, 90, 100), "AUROC (%)", "(b)", 15.0),
    ("share_discarded", (0.0, 80.0), (0, 20, 40, 60, 80), "Share discarded (%)", "(c)", -22.0),
)
PANEL_X: tuple[tuple[int, int], ...] = ((48, 190), (248, 390), (448, 590))


@dataclass(frozen=True)
class Layout:
    suffix: str
    width: float
    height: float
    panels: tuple[tuple[float, float, float, float], ...]
    yticks: tuple[tuple[float, ...], ...]
    ylabels: tuple[str, ...]
    stacked: bool


# side by side at the text width (the paper figure), and stacked at the column width with flat panels sharing the
# x axis: its ticks and title under the last panel only, the part letters above the top left, shorter y titles
FULL = Layout("", WIDTH, HEIGHT, tuple((x0, x1, TOP, BOTTOM) for x0, x1 in PANEL_X), tuple(p[2] for p in PANELS),
    tuple(p[3] for p in PANELS), False)
COLUMN = Layout(
    "_column",
    svgfig.COLUMN_WIDTH,
    354,
    ((60, 305, 20, 100), (60, 305, 128, 208), (60, 305, 236, 316)),
    ((0, 40, 80), (60, 80, 100), (0, 40, 80)),
    ("FPR@95 (%)", "AUROC (%)", "Discarded (%)"),
    True,
)
LETTER_X = 1.0   # stacked layout: the part letter at the canvas edge, left of the y title column


def read_summary(
    path: Path,
) -> pd.DataFrame:

    if not path.is_file():
        raise FileNotFoundError(f"{layout.relative(path)} is missing; run `selection analyze` first")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{layout.relative(path)} holds no row; nothing to draw")

    frame["margin"] = frame["margin"].astype(float)
    return frame


def figure_stem(
    benchmark: str,
    layout_: Layout = FULL,
) -> str:
    return (STEM if benchmark == MAIN_BENCHMARK else f"{STEM}_{benchmark}") + layout_.suffix


def margin_scale(
    x0: float,
    x1: float,
):
    return lambda margin: x0 + (margin + 0.03) / 1.01 * (x1 - x0)


def draw_benchmark(
    summary: pd.DataFrame,
    layout_: Layout = FULL,
) -> str:

    canvas = Canvas()
    drawn = 0
    last = len(layout_.panels) - 1

    for index, ((x0, x1, top, bottom), (metric, ylim, _, _, tag, camp_dy), yticks, ylab) in enumerate(
            zip(layout_.panels, PANELS, layout_.yticks, layout_.ylabels)):
        fx = margin_scale(x0=x0, x1=x1)
        fy = svgfig.linear(ylim[0], ylim[1], bottom, top)
        canvas.line(x0, bottom, x1, bottom)
        canvas.line(x0, top, x0, bottom)
        axis_row = not layout_.stacked or index == last

        for value in X_TICKS:
            canvas.line(fx(value), bottom, fx(value), bottom + 3.5)

            if axis_row:
                canvas.text(fx(value), bottom + 15, f"{value:g}", size=TICK)

        for value in yticks:
            canvas.line(x0 - 3.5, fy(value), x0, fy(value))
            canvas.text(x0 - 6, fy(value) + 4, str(value), size=TICK, anchor="end")

        canvas.text(x0 - 29, (top + bottom) / 2, ylab, size=LAB, rot=-90)

        if axis_row:
            canvas.text((x0 + x1) / 2, bottom + 32, "Margin of the TANL rule", size=LAB)

        if layout_.stacked:
            svgfig.panel_letter(canvas, LETTER_X, top - 6, tag[1])
        else:
            canvas.text((x0 + x1) / 2, bottom + 52, tag, size=LAB)

        canvas.line(fx(DEFAULT_MARGIN), top, fx(DEFAULT_MARGIN), bottom, stroke="#bbbbbb", sw=0.8, dash="3,2.5")
        canvas.text(fx(DEFAULT_MARGIN) + 3, top + 9, "Default", size=NOTE - 0.5, anchor="start", fill="#666")
        part = summary[summary["metric"] == metric].dropna(subset=["mean"]).sort_values("margin")
        if part.empty:
            continue

        drawn += 1
        points = [(fx(m), fy(v)) for m, v in zip(part["margin"], part["mean"])]
        spread = [(fx(m), float(v), float(sd) if int(n) > 1 else None)
            for m, v, sd, n in zip(part["margin"], part["mean"], part["std"], part["n_seeds"])]
        svgfig.std_band(canvas, spread, fy, BAND, fill_opacity=BAND_OPACITY)

        if len(points) >= 2:
            canvas.poly(points, GRAY, sw=1.3)

        for margin, mean in zip(part["margin"], part["mean"]):
            if margin > 0:
                canvas.circle(fx(margin), fy(mean), 2.6, "#fff", stroke=GRAY, sw=1.2)

        camp = part[part["margin"] == 0.0]
        if not camp.empty:
            mean = float(camp["mean"].iloc[0])
            canvas.circle(fx(0.0), fy(mean), 4.2, svgfig.CAMP)
            canvas.text(fx(0.0) + 4, fy(mean) + camp_dy, "CAMP", size=NOTE, anchor="start")

    if drawn == 0:
        raise ValueError("selection.csv holds no value of AUROC, FPR@95 or share_discarded to draw")

    return canvas.svg(layout_.width, layout_.height)


def draw(
    summary: pd.DataFrame,
    stem: Path,
    render: svgfig.Render | None = None,
) -> list[Path]:

    stem = Path(stem)
    benchmarks = list(dict.fromkeys(summary["benchmark"]))
    written: list[Path] = []

    for benchmark in benchmarks:
        for layout_ in (FULL, COLUMN):
            svg = draw_benchmark(summary[summary["benchmark"] == benchmark], layout_)

            if len(benchmarks) == 1 or benchmark == MAIN_BENCHMARK:
                target = stem.with_name(stem.name + layout_.suffix)
            else:
                target = stem.with_name(figure_stem(benchmark, layout_))

            written += svgfig.write_figure(svg, target, width=layout_.width, height=layout_.height, render=render)

    return written


def figure(
    ctx: Context,
) -> list[Path]:
    return draw(read_summary(ctx.tables / "selection.csv"), ctx.figures / STEM, svgfig.Render.from_context(ctx))


__all__ = [
    "COLUMN",
    "FULL",
    "GRAY",
    "HEIGHT",
    "Layout",
    "MAIN_BENCHMARK",
    "PANELS",
    "STEM",
    "WIDTH",
    "X_TICKS",
    "draw",
    "draw_benchmark",
    "figure",
    "figure_stem",
    "margin_scale",
    "read_summary",
]
