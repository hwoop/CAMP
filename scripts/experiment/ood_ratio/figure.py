from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.experiment.lib import layout, svgfig
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.svgfig import Canvas
from scripts.experiment.ood_ratio.analyze import ARM_FIXED, ARM_NATURAL, GROUP_ORDER, TABLE_DELTA_SUMMARY, TABLE_GAINS

STEM = "ood_ratio"
WIDTH, HEIGHT = svgfig.WIDTH, 222
TOP, BOTTOM = 34, 164
TICK, LAB, LEG = 11.0, 12.0, 11.5
DEFAULT_RATIOS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99)
X_LABEL = 'OOD ratio <tspan font-style="italic">π</tspan> (%)'
PANELS: tuple[tuple[str, tuple[float, float], tuple[float, ...], str, str, bool], ...] = (
    ("FPR@95 gain", (-30.0, 50.0), (-20, 0, 20, 40), "Gain in FPR@95", "(a)", True),
    ("AUROC gain", (-15.0, 25.0), (-10, 0, 10, 20), "Gain in AUROC", "(b)", True),
    ("delta_norm", (0.0, 0.25), (0.0, 0.1, 0.2), 'Mean-difference norm ‖<tspan font-style="italic">Δμ</tspan>‖',
        "(c)", False),
)
PANEL_X: tuple[tuple[int, int], ...] = ((48, 202), (246, 400), (446, 596))


@dataclass(frozen=True)
class Layout:
    stem: str
    width: float
    height: float
    panels: tuple[tuple[float, float, float, float], ...]
    yticks: tuple[tuple[float, ...], ...]
    ylabels: tuple[str, ...]
    stacked: bool
    legend_gap: float


# side by side at the text width (the paper figure), and stacked at the column width with flat panels sharing the
# x axis: its ticks and title under the last panel only, the part letters above the top left, shorter y titles
FULL = Layout(STEM, WIDTH, HEIGHT, tuple((x0, x1, TOP, BOTTOM) for x0, x1 in PANEL_X), tuple(p[2] for p in PANELS),
    tuple(p[3] for p in PANELS), False, 30.0)
COLUMN = Layout(
    STEM + "_column",
    svgfig.COLUMN_WIDTH,
    366,
    ((60, 305, 34, 114), (60, 305, 142, 222), (60, 305, 250, 330)),
    ((-20, 0, 20, 40), (-10, 0, 10, 20), (0.0, 0.1, 0.2)),
    ("Gain in FPR@95", "Gain in AUROC", '‖<tspan font-style="italic">Δμ</tspan>‖'),
    True,
    20.0,
)
LETTER_X = 1.0   # stacked layout: the part letter at the canvas edge, left of the y title column
EMPTY_NOTES: dict[str, str] = {
    "delta_norm": "no fixed-ratio CAMP stream replayed",
}
EMPTY_NOTE_GAIN = "no paired CAMP / MCM runs recorded"


def _read(
    path: Path,
) -> pd.DataFrame:

    if not path.is_file():
        return pd.DataFrame()

    frame = pd.read_csv(path)
    if "label" in frame.columns:
        frame["label"] = frame["label"].astype(object).where(frame["label"].notna(), None)

    return frame


def gain_series(
    gains: pd.DataFrame,
    group: str,
    metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if gains.empty:
        return pd.DataFrame(), pd.DataFrame()

    part = gains[(gains["group"] == group) & (gains["metric"] == metric) & (gains["kind"] == "group")]
    fixed = part[part["arm"] == ARM_FIXED].sort_values("ratio")
    natural = part[part["arm"] == ARM_NATURAL]
    return fixed, natural


def delta_series(
    summary: pd.DataFrame,
    group: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    part = summary[summary["group"] == group]
    fixed = part[part["arm"] == ARM_FIXED].sort_values("ratio")
    natural = part[part["arm"] == ARM_NATURAL]
    return fixed, natural


def fixed_series(
    gains: pd.DataFrame,
    summary: pd.DataFrame,
    group: str,
    metric: str,
) -> pd.DataFrame:

    if metric == "delta_norm":
        fixed, _ = delta_series(summary=summary, group=group)
    else:
        fixed, _ = gain_series(gains=gains, group=group, metric=metric)

    if fixed.empty:
        return fixed

    part = fixed.dropna(subset=["ratio", "mean"]).sort_values("ratio")
    return part[["ratio", "mean", "std", "n_seeds"]]


def ratios_present(
    gains: pd.DataFrame,
    summary: pd.DataFrame,
) -> list[float]:

    found: set[float] = set()

    for frame in (gains, summary):
        if not frame.empty and "arm" in frame.columns:
            found |= {float(r) for r in frame[frame["arm"] == ARM_FIXED]["ratio"].dropna()}

    return sorted(found) if found else list(DEFAULT_RATIOS)


def ratio_label(
    ratio: float,
) -> str:
    return f"{ratio * 100:g}"


def draw(
    gains: pd.DataFrame,
    summary: pd.DataFrame,
    stem: Path | str,
    render: svgfig.Render | None = None,
) -> list[Path]:

    stem = Path(stem)
    written: list[Path] = []

    for layout_ in (FULL, COLUMN):
        svg = build(gains=gains, summary=summary, layout_=layout_)
        written += svgfig.write_figure(svg, stem.with_name(layout_.stem), width=layout_.width, height=layout_.height,
            render=render)

    return written


def build(
    gains: pd.DataFrame,
    summary: pd.DataFrame,
    layout_: Layout = FULL,
) -> str:

    ratios = ratios_present(gains=gains, summary=summary)
    index = {ratio: i for i, ratio in enumerate(ratios)}
    mcm_slot = min(range(len(ratios)), key=lambda i: abs(ratios[i] - 0.5))
    canvas = Canvas()
    last = len(layout_.panels) - 1

    for panel, ((x0, x1, top, bottom), (metric, ylim, _, _, tag, zero), yticks, ylab) in enumerate(
            zip(layout_.panels, PANELS, layout_.yticks, layout_.ylabels)):
        axis_row = not layout_.stacked or panel == last
        step = (x1 - x0 - 12) / max(len(ratios) - 1, 1)

        def fx(
            i: int,
            x0: float = x0,
            step: float = step,
        ) -> float:
            return x0 + 6 + i * step

        fy = svgfig.linear(ylim[0], ylim[1], bottom, top)
        canvas.line(x0, bottom, x1, bottom)
        canvas.line(x0, top, x0, bottom)

        for i, ratio in enumerate(ratios):
            canvas.line(fx(i), bottom, fx(i), bottom + 3)

            if axis_row:
                canvas.text(fx(i), bottom + 14, ratio_label(ratio), size=TICK)

        for value in yticks:
            canvas.line(x0 - 3, fy(value), x0, fy(value))

            if metric == "delta_norm":
                label = "0" if value == 0 else f"{value:.1f}"
            else:
                label = svgfig.minus(str(value))

            canvas.text(x0 - 5, fy(value) + 3.8, label, size=TICK, anchor="end")

        canvas.text(x0 - 31, (top + bottom) / 2, ylab, size=LAB, rot=-90)

        if axis_row:
            canvas.text((x0 + x1) / 2, bottom + 30, X_LABEL, size=LAB)

        if layout_.stacked:
            svgfig.panel_letter(canvas, LETTER_X, top - 6, tag[1])
        else:
            canvas.text((x0 + x1) / 2, bottom + 48, tag, size=LAB)

        if zero:
            canvas.line(x0, fy(0.0), x1, fy(0.0), stroke="#888", sw=0.8, dash="3,2")
            canvas.text(fx(mcm_slot), fy(0.0) + 12, "MCM", size=10, fill="#666")

        drawn = False
        canvas.begin_clip(f"clip_{tag[1]}", x0, top, x1 - x0, bottom - top)

        for group, _, color, dash in svgfig.GROUP_LINES:
            series = fixed_series(gains=gains, summary=summary, group=group, metric=metric)
            if series.empty:
                continue

            drawn = True
            points = [(fx(index[float(r)]), fy(float(m))) for r, m in zip(series["ratio"], series["mean"])]
            spread = [(x, float(m), float(sd) if int(n) > 1 else None)
                for (x, _y), m, sd, n in zip(points, series["mean"], series["std"], series["n_seeds"])]
            svgfig.std_band(canvas, spread, fy, color)

            if len(points) >= 2:
                canvas.poly(points, color, dash=dash)

            for x, y in points:
                canvas.circle(x, y, 2.4, color)

        canvas.end_clip()

        if not drawn:
            canvas.text(
                (x0 + x1) / 2,
                (top + bottom) / 2,
                EMPTY_NOTES.get(metric, EMPTY_NOTE_GAIN),
                size=9,
                fill=svgfig.NEUTRAL,
            )

    entries = [(label, color, dash, 1.6) for _, label, color, dash in svgfig.GROUP_LINES]
    if layout_.stacked:
        entries = [(label.replace(" (OpenOOD)", ""), color, dash, sw) for label, color, dash, sw in entries]

    svgfig.legend_row(
        canvas,
        entries,
        width=layout_.width,
        gap=layout_.legend_gap,
        char_width=5.6,
        size=LEG,
        dots=True,
    )

    return canvas.svg(layout_.width, layout_.height)


def figure(
    ctx: Context,
) -> list[Path]:

    gains_path = ctx.tables / TABLE_GAINS
    summary_path = ctx.tables / TABLE_DELTA_SUMMARY
    if not gains_path.is_file() and not summary_path.is_file():
        raise FileNotFoundError(f"ood_ratio: no tables under {ctx.tables}; run `analyze` first")

    gains = _read(gains_path)
    summary = _read(summary_path)
    groups = set(gains["group"]) if not gains.empty else set()
    groups |= set(summary["group"]) if not summary.empty else set()
    ctx.notes["figure_groups"] = sorted(groups)
    ctx.notes["figure_tables"] = [layout.relative(p) for p in (gains_path, summary_path) if p.is_file()]
    ctx.notes["figure_ratios"] = ratios_present(gains=gains, summary=summary)
    return draw(gains, summary, ctx.figures / STEM, svgfig.Render.from_context(ctx))


__all__ = [
    "COLUMN",
    "DEFAULT_RATIOS",
    "FULL",
    "GROUP_ORDER",
    "HEIGHT",
    "Layout",
    "PANELS",
    "STEM",
    "WIDTH",
    "build",
    "delta_series",
    "draw",
    "figure",
    "fixed_series",
    "gain_series",
    "ratio_label",
    "ratios_present",
]
