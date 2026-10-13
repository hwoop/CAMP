from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from scripts.experiment.lib import layout, svgfig
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.svgfig import Canvas
from scripts.experiment.warmup_batch1.analyze import DEFAULT_WARMUP, TABLE_SUMMARY

STEM = "warmup_batch1"
WIDTH, HEIGHT = svgfig.WIDTH, 226
TOP, BOTTOM = 34, 164
TICK, LAB, LEG = 11.0, 12.0, 11.5
PANEL_X: tuple[tuple[int, int], ...] = ((56, 300), (356, 596))
PANELS: tuple[tuple[str, str, str], ...] = (
    ("FPR@95", "Gain in FPR@95 over MCM", "(a)"),
    ("AUROC", "Gain in AUROC over MCM", "(b)"),
)
X_LABEL = 'Warm-up length <tspan font-style="italic">N</tspan><tspan font-size="8.5" dy="3">min</tspan>'
DEFAULT_LABEL = "Batch 256 (default)"


def read_summary(
    ctx: Context,
) -> pd.DataFrame:

    path = ctx.tables / TABLE_SUMMARY
    if not path.is_file():
        raise FileNotFoundError(f"{layout.relative(path)} is missing; run `python -m scripts.experiment warmup_batch1 analyze` first")

    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"{layout.relative(path)} holds no row; nothing to draw")

    return frame


def nice_range(
    values: list[float],
    *,
    include_zero: bool = True,
) -> tuple[float, float, list[float]]:

    finite = [v for v in values if math.isfinite(v)]

    if include_zero:
        finite.append(0.0)

    if not finite:
        return -1.0, 1.0, [-1.0, 0.0, 1.0]

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


def tick_text(
    value: float,
) -> str:

    text = f"{value:g}"
    return svgfig.minus(text)


def draw(
    summary: pd.DataFrame,
) -> str:

    n_mins = sorted(set(int(v) for v in summary["n_min"]))
    if len(n_mins) < 2:
        raise RuntimeError("warmup.csv holds fewer than two warm-up lengths; nothing to draw against")

    canvas = Canvas()
    has_default = summary["default_mean"].notna().any() and DEFAULT_WARMUP in n_mins
    entries = [(label, color, dash, 1.6) for _, label, color, dash in svgfig.GROUP_LINES]
    extra = (27.0 + len(DEFAULT_LABEL) * 5.6 + 30.0) if has_default else 0.0
    end = svgfig.legend_row(canvas, entries, width=WIDTH - extra, gap=30.0, char_width=5.6, size=LEG, dots=True)

    for (x0, x1), (metric, ylab, tag) in zip(PANEL_X, PANELS):
        part = summary[summary["metric"] == metric]
        fx = svgfig.log_scale(n_mins[0], n_mins[-1], x0 + 6, x1 - 6)
        gains = part["gain_mean"].dropna().tolist()
        spread = (part["gain_mean"] + part["gain_std"].fillna(0.0)).dropna().tolist() + \
                 (part["gain_mean"] - part["gain_std"].fillna(0.0)).dropna().tolist()
        default_gains: dict[str, float] = {}

        if has_default:
            at_default = part[part["n_min"] == DEFAULT_WARMUP]
            for _, row in at_default.iterrows():
                if pd.notna(row["default_mean"]) and pd.notna(row["mcm_mean"]):
                    default_gains[str(row["group"])] = (float(row["mcm_mean"]) - float(row["default_mean"])
                        if metric == "FPR@95" else
                        float(row["default_mean"]) - float(row["mcm_mean"]))

        lo, hi, yticks = nice_range(gains + spread + list(default_gains.values()))
        fy = svgfig.linear(lo, hi, BOTTOM, TOP)
        canvas.line(x0, BOTTOM, x1, BOTTOM)
        canvas.line(x0, TOP, x0, BOTTOM)

        for i, n_min in enumerate(n_mins):
            canvas.line(fx(n_min), BOTTOM, fx(n_min), BOTTOM + 3)

            if i % 2 == 0 or n_min == n_mins[-1]:
                canvas.text(fx(n_min), BOTTOM + 14, svgfig.thousands(n_min), size=TICK)

        for value in yticks:
            canvas.line(x0 - 3, fy(value), x0, fy(value))
            canvas.text(x0 - 5, fy(value) + 3.8, tick_text(value), size=TICK, anchor="end")

        canvas.text(x0 - 34, (TOP + BOTTOM) / 2, ylab, size=LAB, rot=-90)
        canvas.text((x0 + x1) / 2, BOTTOM + 30, X_LABEL, size=LAB)
        canvas.text((x0 + x1) / 2, BOTTOM + 48, tag, size=LAB)

        if n_mins[0] <= DEFAULT_WARMUP <= n_mins[-1]:
            canvas.line(fx(DEFAULT_WARMUP), TOP, fx(DEFAULT_WARMUP), BOTTOM, stroke="#999", sw=0.8, dash="1.2,2")

        if lo <= 0 <= hi:
            zero = fy(0.0)
            canvas.line(x0, zero, x1, zero, stroke="#888", sw=0.8, dash="3,2")
            label_y = zero + 12 if BOTTOM - zero > 16 else zero - 4
            canvas.text(x1 - 4, label_y, "MCM", size=10, fill="#666", anchor="end")

        drawn = False
        canvas.begin_clip(f"clip_{tag[1]}", x0, TOP, x1 - x0, BOTTOM - TOP)

        for group, _, color, dash in svgfig.GROUP_LINES:
            rows = part[part["group"] == group].dropna(subset=["gain_mean"]).sort_values("n_min")
            if rows.empty:
                continue

            drawn = True
            points = [(fx(int(n)), fy(float(g))) for n, g in zip(rows["n_min"], rows["gain_mean"])]
            spread = [(x, float(g), float(sd) if int(n) > 1 else None)
                for (x, _y), g, sd, n in zip(points, rows["gain_mean"], rows["gain_std"], rows["n_seeds"])]
            svgfig.std_band(canvas, spread, fy, color)

            if len(points) >= 2:
                canvas.poly(points, color, dash=dash)

            for x, y in points:
                canvas.circle(x, y, 2.4, color)

            if group in default_gains:
                x, y = fx(DEFAULT_WARMUP), fy(default_gains[group])
                canvas.polygon([(x, y - 4.2), (x + 4.2, y), (x, y + 4.2), (x - 4.2, y)], "#fff", stroke=color, sw=1.2)

        canvas.end_clip()

        if not drawn:
            canvas.text((x0 + x1) / 2, (TOP + BOTTOM) / 2, "no MCM reference run recorded", size=9, fill=svgfig.NEUTRAL)

    if has_default:
        x = end + 30
        canvas.polygon([(x + 11, 11 - 4.2), (x + 15.2, 11), (x + 11, 11 + 4.2), (x + 6.8, 11)], "#fff",
            stroke="#555", sw=1.2)

        canvas.text(x + 27, 15, DEFAULT_LABEL, size=LEG, anchor="start")

    return canvas.svg(WIDTH, HEIGHT)


def figure(
    ctx: Context,
) -> list[Path]:

    summary = read_summary(ctx)
    ctx.notes["figure_groups"] = sorted(set(summary["group"]))
    ctx.notes["figure_n_min"] = sorted(set(int(v) for v in summary["n_min"]))
    return svgfig.write_figure(
        draw(summary),
        ctx.figures / STEM,
        width=WIDTH,
        height=HEIGHT,
        render=svgfig.Render.from_context(ctx),
    )


__all__ = [
    "DEFAULT_LABEL",
    "HEIGHT",
    "PANELS",
    "STEM",
    "WIDTH",
    "draw",
    "figure",
    "nice_range",
    "read_summary",
]
