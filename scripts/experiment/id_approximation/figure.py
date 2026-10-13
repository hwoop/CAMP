from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.experiment.id_approximation.analyze import ESTIMATOR_NAMES, ESTIMATORS, METRICS
from scripts.experiment.lib import layout, svgfig
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.svgfig import Canvas

STEM = "id_approximation"
MAIN_BENCHMARK = "imagenet_openood"
DEFAULT_XMIN = 512
DEFAULT_XMAX = 50_000
DEFAULT_WINDOW = 8192
WIDTH, HEIGHT = svgfig.WIDTH, 222
TOP, BOTTOM = 30, 150
TICK, LAB, LEG = 11.5, 12.5, 11.5
LINE_STYLES: dict[str, tuple[str, str | None, float]] = {
    "camp": (svgfig.CAMP, None, 1.6),
    "lower_bound": (svgfig.LOWER_BOUND, "1.6,2.2", 1.2),
    "tanl_memory": (svgfig.TANL, "5,3", 1.4),
}
PANELS: tuple[tuple[str, tuple[float, float], tuple[float, ...], str, str], ...] = (
    ("mu_mahalanobis", (0.0, 1.0), (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), "ID mean error", "(a)"),
    ("sigma_whitened_rel_error", (0.0, 1.4), (0.0, 0.4, 0.8, 1.2), "ID covariance error", "(b)"),
    ("cos_to_oracle", (0.0, 1.0), (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), "Cosine similarity", "(c)"),
)
PANEL_X: tuple[tuple[int, int], ...] = ((52, 196), (254, 398), (456, 596))
SHORT_NAMES: dict[str, str] = {"CAMP (Proposed)": "CAMP"}
WINDOW_LABEL = "Window full"
WINDOW_LABEL_SIZE = 10.5
WINDOW_LABEL_DX = 3.0
WINDOW_LABEL_DY = 10.0
WINDOW_LABEL_BOTTOM_DY = 5.0
WINDOW_LABEL_CLEAR = 4.0
WINDOW_LABEL_CHAR = 0.452


@dataclass(frozen=True)
class Layout:
    suffix: str
    width: float
    height: float
    panels: tuple[tuple[float, float, float, float], ...]
    yticks: tuple[tuple[float, ...], ...]
    ylabels: tuple[str, ...]
    stacked: bool
    legend_gap: float


# side by side at the text width (the paper figure), and stacked at the column width with flat panels sharing the
# x axis: its ticks and title under the last panel only, the part letters above the top left, shorter y titles
FULL = Layout("", WIDTH, HEIGHT, tuple((x0, x1, TOP, BOTTOM) for x0, x1 in PANEL_X), tuple(p[2] for p in PANELS),
    tuple(p[3] for p in PANELS), False, 26.0)
COLUMN = Layout(
    "_column",
    svgfig.COLUMN_WIDTH,
    368,
    ((60, 305, 34, 114), (60, 305, 142, 222), (60, 305, 250, 330)),
    ((0.0, 0.5, 1.0), (0.0, 0.5, 1.0), (0.0, 0.5, 1.0)),
    ("Mean error", "Covariance error", "Cosine similarity"),
    True,
    14.0,
)
LETTER_X = 1.0   # stacked layouts: the part letter at the canvas edge, left of the y title column


def read_curves(
    path: Path,
) -> pd.DataFrame:

    frame = pd.read_csv(path)
    needed = ["benchmark", "estimator", "order_idx", "n_seeds", *(f"{m}_{s}" for m in METRICS for s in ("mean", "std"))]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is not the curves table: it lacks {missing}.")

    return frame


def figure_stem(
    benchmark: str,
    layout_: Layout = FULL,
) -> str:
    return (STEM if benchmark == MAIN_BENCHMARK else f"{STEM}_{benchmark}") + layout_.suffix


def x_ticks(
    xmin: float,
    xmax: float,
) -> list[tuple[float, str]]:

    unit = 1000.0 if xmax >= 10_000 else 100.0 if xmax >= 1_000 else 1.0
    middle = round(xmax / 2 / unit) * unit
    if not xmin < middle < xmax:
        middle = (xmin + xmax) / 2

    return [(xmin, svgfig.thousands(xmin)), (middle, svgfig.thousands(middle)), (xmax, svgfig.thousands(xmax))]


def window_label_y(
    part: pd.DataFrame,
    metric: str,
    estimators: list[str],
    fx,
    fy,
    *,
    x_from: float,
    x_to: float,
    side: str = "auto",
    top: float = TOP,
    bottom: float = BOTTOM,
) -> float:

    top = top + WINDOW_LABEL_DY
    bottom = bottom - WINDOW_LABEL_BOTTOM_DY

    if side == "top":
        return top

    if side == "bottom":
        return bottom

    highest = lowest = None

    for estimator in estimators:
        line = part[part["estimator"] == estimator].dropna(subset=[f"{metric}_mean"])
        for value, y, std in zip(line["order_idx"], line[f"{metric}_mean"], line[f"{metric}_std"]):
            if x_from <= fx(value) <= x_to:
                spread = float(std) if pd.notna(std) and float(std) > 0 else 0.0
                top_py, bottom_py = fy(y + spread), fy(y - spread)
                highest = top_py if highest is None else min(highest, top_py)
                lowest = bottom_py if lowest is None else max(lowest, bottom_py)

    if highest is None:
        return top

    if highest - top >= WINDOW_LABEL_CLEAR:
        return top

    if bottom - lowest >= WINDOW_LABEL_CLEAR:
        return bottom

    return top


def draw(
    curves: pd.DataFrame,
    benchmark: str,
    *,
    xmax: float | None = DEFAULT_XMAX,
    window_size: int = DEFAULT_WINDOW,
    xmin: float = DEFAULT_XMIN,
    window_label: str = "auto",
    layout_: Layout = FULL,
) -> str:

    part = curves[curves["benchmark"] == benchmark]
    if part.empty:
        raise ValueError(f"curves.csv holds no rows for {benchmark!r}.")

    if xmax is None:
        limit = float(part["order_idx"].max())
    else:
        limit = min(float(xmax), float(part["order_idx"].max()))

    if limit <= xmin:
        raise ValueError(f"curves.csv for {benchmark!r} ends at {limit:g} samples, before the first tick at {xmin:g}.")

    part = part[(part["order_idx"] >= xmin) & (part["order_idx"] <= limit)]
    estimators = [e for e in ESTIMATORS if e in set(part["estimator"])]
    canvas = Canvas()
    last = len(layout_.panels) - 1

    for index, ((x0, x1, top, bottom), (metric, ylim, _, _, tag), yticks, ylab) in enumerate(
            zip(layout_.panels, PANELS, layout_.yticks, layout_.ylabels)):
        fx = svgfig.linear(xmin, limit, x0, x1)
        fy = svgfig.linear(ylim[0], ylim[1], bottom, top)
        canvas.line(x0, bottom, x1, bottom)
        canvas.line(x0, top, x0, bottom)
        axis_row = not layout_.stacked or index == last

        for i, (value, label) in enumerate(x_ticks(xmin=xmin, xmax=limit)):
            canvas.line(fx(value), bottom, fx(value), bottom + 3.5)

            if axis_row:
                canvas.text(fx(value), bottom + 15, label, size=TICK, anchor="start" if i == 0 else ("end" if i == 2 else "middle"))

        for value in yticks:
            canvas.line(x0 - 3.5, fy(value), x0, fy(value))
            canvas.text(x0 - 6, fy(value) + 4, f"{value:.1f}", size=TICK, anchor="end")

        canvas.text(x0 - 33, (top + bottom) / 2, ylab, size=LAB, rot=-90)

        if axis_row:
            canvas.text((x0 + x1) / 2, bottom + 32, "Samples seen", size=LAB)

        if layout_.stacked:
            svgfig.panel_letter(canvas, LETTER_X, top - 6, tag[1])
        else:
            canvas.text((x0 + x1) / 2, bottom + 52, tag, size=LAB)

        if xmin <= window_size <= limit:
            canvas.line(fx(window_size), top, fx(window_size), bottom, stroke="#999", sw=0.8, dash="1.2,2")
            label_x = fx(window_size) + WINDOW_LABEL_DX
            label_w = len(WINDOW_LABEL) * WINDOW_LABEL_SIZE * WINDOW_LABEL_CHAR
            label_y = window_label_y(
                part=part,
                metric=metric,
                estimators=estimators,
                fx=fx,
                fy=fy,
                x_from=label_x,
                x_to=min(label_x + label_w, x1),
                side=window_label,
                top=top,
                bottom=bottom,
            )

            canvas.text(label_x, label_y, WINDOW_LABEL, size=WINDOW_LABEL_SIZE, anchor="start", fill="#555")

        canvas.begin_clip(f"clip_{metric}", x0, top, x1 - x0, bottom - top)

        for estimator in estimators:
            color, dash, sw = LINE_STYLES.get(estimator, (svgfig.NEUTRAL, None, 1.3))
            line = part[part["estimator"] == estimator].dropna(subset=[f"{metric}_mean"]).sort_values("order_idx")
            points = [(fx(x), fy(y)) for x, y in zip(line["order_idx"], line[f"{metric}_mean"])]
            spread = [(fx(x), float(y), float(sd) if int(n) > 1 else None)
                for x, y, sd, n in zip(
                    line["order_idx"],
                    line[f"{metric}_mean"],
                    line[f"{metric}_std"],
                    line["n_seeds"],
                )]

            svgfig.std_band(canvas, spread, fy, color)

            if len(points) >= 2:
                canvas.poly(points, color, sw=sw, dash=dash)
            elif points:
                canvas.circle(points[0][0], points[0][1], 2.0, color)

        canvas.end_clip()

    entries = [(
            ESTIMATOR_NAMES.get(e, e),
            *LINE_STYLES.get(e, (svgfig.NEUTRAL, None, 1.3))[:2],
            LINE_STYLES.get(e, (svgfig.NEUTRAL, None, 1.3))[2] + 0.1,
        ) for e in estimators]
    if layout_.stacked:
        entries = [(SHORT_NAMES.get(label, label), color, dash, sw) for label, color, dash, sw in entries]

    svgfig.legend_row(canvas, entries, width=layout_.width, gap=layout_.legend_gap, char_width=5.2, size=LEG)
    return canvas.svg(layout_.width, layout_.height)


def figure(
    ctx: Context,
) -> list[Path]:

    path = ctx.tables / "curves.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{layout.relative(path)} is missing; run `analyze` first.")

    curves = read_curves(path)
    xmax_option = ctx.option("xmax")
    if xmax_option is None:
        xmax = DEFAULT_XMAX
    else:
        xmax = None if xmax_option in ("", "none", "all") else float(xmax_option)

    window_size = int(ctx.option("window", str(DEFAULT_WINDOW)) or DEFAULT_WINDOW)
    window_label = (ctx.option("window_label", "auto") or "auto").lower()
    if window_label not in ("auto", "top", "bottom"):
        raise ValueError(f"--opt window_label must be auto, top or bottom, not {window_label!r}")

    render = svgfig.Render.from_context(ctx)
    written: list[Path] = []
    benchmarks = [b for b in dict.fromkeys(curves["benchmark"]) if b in ctx.benchmarks] or list(
        dict.fromkeys(curves["benchmark"]),
    )
    for benchmark in benchmarks:
        for layout_ in (FULL, COLUMN):
            svg = draw(curves, benchmark, xmax=xmax, window_size=window_size, window_label=window_label, layout_=layout_)
            written += svgfig.write_figure(
                svg,
                ctx.figures / figure_stem(benchmark, layout_),
                width=layout_.width,
                height=layout_.height,
                render=render,
            )

    return written


__all__ = [
    "COLUMN",
    "DEFAULT_WINDOW",
    "DEFAULT_XMAX",
    "DEFAULT_XMIN",
    "FULL",
    "HEIGHT",
    "LINE_STYLES",
    "Layout",
    "MAIN_BENCHMARK",
    "PANELS",
    "STEM",
    "WIDTH",
    "WINDOW_LABEL",
    "draw",
    "figure",
    "figure_stem",
    "read_curves",
    "window_label_y",
    "x_ticks",
]
