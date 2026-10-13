from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment.lib import layout, svgfig
from scripts.experiment.lib.items import Context
from scripts.experiment.lib.svgfig import Canvas

from .analyze import points_path, requested_streams

PAPER_SELECTION: tuple[str, int, str] = ("imagenet_openood", 0, "inaturalist")
STEM = "motivation"
WIDTH = svgfig.WIDTH
TOP, PANEL_WIDTH = 60.0, 240.0
PANEL_X: tuple[float, float] = (50.0, 350.0)
RATE = 1.0
DOT_RADIUS = 0.8
SAMPLE_SEED = 0
TICK, LAB, LEG = 11.0, 12.0, 11.5
ID_FILL, OOD_C = svgfig.ID, svgfig.OOD
CAMP_C, HARD_C, ARROW = "#1f4e9c", "#c00000", "#86213f"
AXIS_LABELS: tuple[str, str] = ("Principal component 1", "Principal component 2")
LEGEND_ROWS: tuple[tuple[tuple[str, str], ...], ...] = (
    (("id", "ID sample"), ("ood", "OOD sample"), ("true", "True ID mean")),
    (("hard", "Estimated ID mean of hard selection"), ("camp", "Estimated ID mean of CAMP (Proposed)")),
    (("stream", "Stream mean"), ("ood_mean", "OOD mean"), ("arrow", "Direction of the mean difference")),
)
LEGEND_MARGIN = 30.0


def figure_stem(
    benchmark: str,
    seed: int,
    stream: str,
) -> str:

    if (benchmark, int(seed), stream) == PAPER_SELECTION:
        return STEM

    return f"{STEM}_{benchmark}_{stream}_s{int(seed)}"


def selections(
    means: pd.DataFrame,
    ctx: Context,
) -> list[tuple[str, int, str]]:

    from scripts.experiment.motivation import spec_seeds

    seeds = set(spec_seeds(ctx))
    streams = set(requested_streams(ctx))
    found = list(dict.fromkeys(zip(means["benchmark"], means["seed"].astype(int), means["stream"])))
    rows = [
        (b, s, l)
        for b, s, l in found
        if b in ctx.benchmarks and s in seeds and l in streams
    ]

    return rows


def _mean_row(
    means: pd.DataFrame,
    name: str,
) -> pd.Series:

    hit = means[means["name"] == name]
    if hit.empty:
        raise KeyError(f"means.csv lacks the row {name!r}")

    return hit.iloc[0]


def coordinates(
    means: pd.DataFrame,
    points: pd.DataFrame,
    rate: float = RATE,
) -> dict[str, np.ndarray]:

    true = _mean_row(means=means, name="true_id_mean")
    origin = np.array([true["pc1"], true["pc2"]], dtype=float)
    is_ood = points["is_ood"].astype(int).to_numpy() == 1
    xy = points[["pc1", "pc2"]].to_numpy(dtype=float) - origin

    if not is_ood.any():
        raise ValueError("the window holds no OOD sample; the motivation figure needs the OOD mean")

    ood_mean = xy[is_ood].mean(axis=0)
    sign = np.where(ood_mean >= 0, 1.0, -1.0)
    xy = xy * sign
    rng = np.random.default_rng(SAMPLE_SEED)
    id_index = np.flatnonzero(~is_ood)
    ood_index = np.flatnonzero(is_ood)
    rate = float(np.clip(a=rate, a_min=0.0, a_max=1.0))

    if id_index.size:
        id_draw = rng.choice(id_index, int(round(rate * id_index.size)), replace=False)
    else:
        id_draw = id_index

    ood_draw = rng.choice(ood_index, int(round(rate * ood_index.size)), replace=False)
    out = {
        "id": xy[id_draw],
        "ood": xy[ood_draw],
        "true": np.zeros(2),
        "ood_mean": ood_mean * sign,
    }

    for key, name in (("hard", "hard_selection_mean"), ("camp", "camp_id_mean"), ("stream", "stream_mean")):
        row = _mean_row(means=means, name=name)
        out[key] = (np.array([row["pc1"], row["pc2"]], dtype=float) - origin) * sign

    return out


def extent(
    coords: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:

    drawn = np.vstack([coords["id"], coords["ood"]]) if coords["id"].size else coords["ood"]
    return np.percentile(a=drawn, q=0.5, axis=0) - 0.02, np.percentile(a=drawn, q=99.5, axis=0) + 0.02


def box(
    coords: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:

    lo, hi = extent(coords)
    centre, span = (lo + hi) / 2, hi - lo
    return centre - span / 2, centre + span / 2


def fmt(
    value: float,
) -> str:
    return "0.0" if abs(value) < 1e-9 else svgfig.minus(f"{value:.1f}")


def star(
    canvas: Canvas,
    x: float,
    y: float,
    r: float,
    fill: str,
    stroke: str,
    sw: float,
) -> None:

    points = []

    for k in range(10):
        angle = -math.pi / 2 + k * math.pi / 5
        radius = r if k % 2 == 0 else r * 0.45
        points.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))

    canvas.polygon(points, fill, stroke=stroke, sw=sw, linejoin="round")


def plus(
    canvas: Canvas,
    x: float,
    y: float,
    arm: float,
    sw: float,
) -> None:

    for stroke, width in (("#fff", sw + 1.6), ("#000", sw)):
        canvas.line(x - arm, y, x + arm, y, stroke, width)
        canvas.line(x, y - arm, x, y + arm, stroke, width)


def arrow(
    canvas: Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    sw: float = 1.8,
    head: float = 6.0,
) -> None:

    angle = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(angle), y2 - head * math.sin(angle)
    canvas.line(x1, y1, bx, by, color, sw)
    h1 = (bx + head * 0.42 * math.sin(angle), by - head * 0.42 * math.cos(angle))
    h2 = (bx - head * 0.42 * math.sin(angle), by + head * 0.42 * math.cos(angle))
    canvas.polygon([(x2, y2), h1, h2], color)


def diamond(
    canvas: Canvas,
    x: float,
    y: float,
    r: float,
) -> None:
    canvas.polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)], OOD_C, stroke="#262626", sw=1.0)


def value_label(
    canvas: Canvas,
    x: float,
    y: float,
    text: str,
    color: str,
) -> None:

    canvas.rect(x - 16, y - 11, 32, 15, "#fff", fill_opacity=0.9, rx=2)
    canvas.text(x, y, text, size=LAB, fill=color)


def panel(
    canvas: Canvas,
    coords: dict[str, np.ndarray],
    x0: float,
    y0: float,
    width: float,
    letter: str,
    lo: np.ndarray,
    hi: np.ndarray,
    dot: float = DOT_RADIUS,
) -> tuple[Callable[[np.ndarray], tuple[float, float]], float]:

    scale = width / (hi[0] - lo[0])
    height = (hi[1] - lo[1]) * scale

    def to(
        q: np.ndarray,
    ) -> tuple[float, float]:
        return x0 + (q[0] - lo[0]) * scale, y0 + (hi[1] - q[1]) * scale

    canvas.rect(x0, y0, width, height, "none", stroke="#000", sw=0.8)

    for value in np.arange(math.ceil(lo[0] / 0.2 - 1e-9) * 0.2, hi[0], 0.2):
        x = to(np.array([value, 0.0]))[0]
        canvas.line(x, y0 + height, x, y0 + height + 3.5)
        canvas.text(x, y0 + height + 15, fmt(value), size=TICK)

    for value in np.arange(math.ceil(lo[1] / 0.1 - 1e-9) * 0.1, hi[1], 0.1):
        y = to(np.array([0.0, value]))[1]
        canvas.line(x0 - 3.5, y, x0, y)
        canvas.text(x0 - 6, y + 3.8, fmt(value), size=TICK, anchor="end")

    canvas.text(x0 + width / 2, y0 + height + 31, AXIS_LABELS[0], size=LAB)
    canvas.text(x0 - 30, y0 + height / 2, AXIS_LABELS[1], size=LAB, rot=-90)
    canvas.begin_clip(f"clip_{letter}", x0, y0, width, height)

    for q in coords["id"]:
        canvas.circle(*to(q), dot, ID_FILL, opacity=0.45)

    for q in coords["ood"]:
        canvas.circle(*to(q), dot, OOD_C, opacity=0.55)

    return to, height


def draw(
    means: pd.DataFrame,
    points: pd.DataFrame,
    rate: float = RATE,
    dot: float = DOT_RADIUS,
) -> tuple[str, float]:

    coords = coordinates(means, points, rate)
    distances = {key: float(_mean_row(means=means, name=name)["distance_to_true_id_mean"])
        for key, name in (("hard", "hard_selection_mean"), ("camp", "camp_id_mean"))}
    lo, hi = box(coords)
    canvas = Canvas()

    to, height = panel(
        canvas=canvas,
        coords=coords,
        x0=PANEL_X[0],
        y0=TOP,
        width=PANEL_WIDTH,
        letter="a",
        lo=lo,
        hi=hi,
        dot=dot
    )

    tx, ty = to(coords["true"])
    marks = [(key, color) for key, color in (("hard", HARD_C), ("camp", CAMP_C)) if np.isfinite(coords[key]).all()]
    for key, color in marks:
        canvas.line(tx, ty, *to(coords[key]), color, 1.3, "3,2")

    if "hard" in dict(marks):
        star(canvas, *to(coords["hard"]), 8.5, "#fff", HARD_C, 1.7)

    if "camp" in dict(marks):
        star(canvas, *to(coords["camp"]), 9.5, CAMP_C, "#fff", 0.7)

    plus(canvas=canvas, x=tx, y=ty, arm=3.6, sw=1.3)

    for key, color in marks:
        x, y = to(coords[key])
        dx, dy = x - tx, y - ty
        norm = math.hypot(dx, dy) or 1.0

        if math.isfinite(distances[key]):
            value_label(canvas, x + dx / norm * 22, y + dy / norm * 22 + 4.5, f"{distances[key]:.3f}", color)

    canvas.end_clip()

    to, height_b = panel(
        canvas=canvas,
        coords=coords,
        x0=PANEL_X[1],
        y0=TOP,
        width=PANEL_WIDTH,
        letter="b",
        lo=lo,
        hi=hi,
        dot=dot
    )

    tx, ty = to(coords["true"])
    direction = coords["stream"] - coords["true"]
    norm = float(np.linalg.norm(direction))
    if norm > 0:
        unit = direction / norm
        foot = coords["true"] + unit * float((coords["ood_mean"] - coords["true"]) @ unit)
        ex, ey = to(foot)
        ox, oy = to(coords["ood_mean"])
        canvas.line(ex, ey, ox, oy, "#595959", 0.8, "2,1.6")
        arrow(canvas=canvas, x1=tx, y1=ty, x2=ex, y2=ey, color=ARROW, sw=1.8, head=6.5)

    canvas.circle(*to(coords["stream"]), 4.0, "#fff", stroke="#262626", sw=1.4)
    plus(canvas=canvas, x=tx, y=ty, arm=3.6, sw=1.3)
    diamond(canvas, *to(coords["ood_mean"]), 5.0)
    canvas.end_clip()
    ly = TOP + max(height, height_b) + 50
    canvas.text(PANEL_X[0] + PANEL_WIDTH / 2, ly, "(a)", size=LAB)
    canvas.text(PANEL_X[1] + PANEL_WIDTH / 2, ly, "(b)", size=LAB)

    for r, row in enumerate(LEGEND_ROWS):
        step = (WIDTH - 2 * LEGEND_MARGIN) / len(row)

        for c, (mark, label) in enumerate(row):
            x, y = LEGEND_MARGIN + c * step + 8, 12 + r * 17

            if mark == "id":
                canvas.circle(x, y, 2.8, ID_FILL, opacity=0.6)
            elif mark == "ood":
                canvas.circle(x, y, 2.8, OOD_C, opacity=0.65)
            elif mark == "true":
                plus(canvas=canvas, x=x, y=y, arm=4.5, sw=1.8)
            elif mark == "hard":
                star(canvas=canvas, x=x, y=y, r=7, fill="#fff", stroke=HARD_C, sw=1.5)
            elif mark == "camp":
                star(canvas=canvas, x=x, y=y, r=7, fill=CAMP_C, stroke="#fff", sw=0.6)
            elif mark == "stream":
                canvas.circle(x, y, 3.8, "#fff", stroke="#262626", sw=1.4)
            elif mark == "ood_mean":
                diamond(canvas=canvas, x=x, y=y, r=4.6)
            else:
                arrow(canvas=canvas, x1=x - 9, y1=y, x2=x + 9, y2=y, color=ARROW, sw=1.8, head=5.5)

            canvas.text(x + 14, y + 4, label, size=LEG, anchor="start")

    total = TOP + max(height, height_b) + 58
    return canvas.svg(WIDTH, total), total


def figure(
    ctx: Context,
) -> list[Path]:

    means_path = ctx.tables / "means.csv"
    if not means_path.is_file():
        raise FileNotFoundError(f"{layout.relative(means_path)} is missing; run `motivation analyze` first.")

    means = pd.read_csv(means_path)
    chosen = selections(means=means, ctx=ctx)
    if not chosen:
        raise KeyError(
            f"{layout.relative(means_path)} holds no stream for benchmarks {list(ctx.benchmarks)}, seeds "
            f"{list(ctx.seeds)} and streams {requested_streams(ctx)}; rerun analyze for this scope.",
        )

    rate = float(ctx.option("rate", str(RATE)) or RATE)
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"--opt rate must be in (0, 1], not {rate}")

    dot = float(ctx.option("dot", str(DOT_RADIUS)) or DOT_RADIUS)
    if not 0.0 < dot <= 6.0:
        raise ValueError(f"--opt dot must be in (0, 6] px, not {dot}")

    render = svgfig.Render.from_context(ctx)
    written: list[Path] = []

    for benchmark, seed, stream in chosen:
        cache = points_path(
            cache=ctx.cache,
            benchmark=benchmark,
            seed=seed,
            stream=stream
        )
        if not cache.is_file():
            raise FileNotFoundError(f"{layout.relative(cache)} is missing; rerun `motivation analyze`.")

        part = means[
            (means["benchmark"] == benchmark) &
            (means["seed"].astype(int) == seed) &
            (means["stream"] == stream)
        ]

        svg, height = draw(part, pd.read_csv(cache), rate, dot)

        written += svgfig.write_figure(
            svg,
            ctx.figures / figure_stem(benchmark, seed, stream),
            width=WIDTH,
            height=height,
            render=render,
        )

    return written


__all__ = [
    "AXIS_LABELS",
    "DOT_RADIUS",
    "LEGEND_ROWS",
    "PAPER_SELECTION",
    "RATE",
    "STEM",
    "WIDTH",
    "coordinates",
    "draw",
    "figure",
    "figure_stem",
    "selections",
]
