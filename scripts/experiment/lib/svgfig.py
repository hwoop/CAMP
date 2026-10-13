"""
Helpers for writing SVG figures in the shared style and rendering their PNG copies.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CAMP = "#1a9e77"
LOWER_BOUND = "#222222"
TANL = "#7a3db8"
IN1K = CAMP
NEAR = "#1f5fbf"
FAR = "#d9601a"
ID = "#6fa0d6"
OOD = "#d97545"
NEUTRAL = "#555555"
GRID = "#e6e5e0"

FONT = "Times New Roman"
WIDTH = 602            # the text width of a two-column page (6.27 in at 96 dpi)
COLUMN_WIDTH = 317     # one column of it (3.3 in)
TICK, LAB, LEG = 11.0, 12.0, 11.5
MINUS = "\u2212"
_SUPERSCRIPTS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

GROUP_LINES: tuple[tuple[str, str, str, str | None], ...] = (
    ("IN-1K", "ImageNet-1K", IN1K, None),
    ("near", "Near-OOD (OpenOOD)", NEAR, "5,3"),
    ("far", "Far-OOD (OpenOOD)", FAR, "1.6,2.2"),
)

ENV_BROWSER = "CAMP_FIGURE_BROWSER"
BROWSER_PATHS: tuple[str, ...] = (
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)
BROWSER_NAMES: tuple[str, ...] = (
    "msedge",
    "microsoft-edge",
    "google-chrome",
    "google-chrome-stable",
    "chrome",
    "chromium",
    "chromium-browser",
)
RENDER_TIMEOUT = 120.0

# the canvas is trimmed under its lowest element: a text's ink reaches TEXT_DESCENT x font-size under its baseline
# (Times New Roman: 0.22), its glyphs are CHAR_WIDTH x font-size wide on average, and BOTTOM_PAD px are kept below
TEXT_ASCENT = 0.72
TEXT_DESCENT = 0.25
CHAR_WIDTH = 0.5
BOTTOM_PAD = 3.0
_TAG = re.compile(
    r"<(?P<name>text|line|polyline|polygon|circle|rect|clipPath|g|/g|/clipPath)\b(?P<attrs>[^>]*)>"
    r"(?:(?P<body>[^<]*)</text>)?",
)
_ROTATE = re.compile(r"rotate\(\s*(?P<angle>[-+0-9.eE]+)\s+(?P<x>[-+0-9.eE]+)\s+(?P<y>[-+0-9.eE]+)\s*\)")
_ANCHOR_SPAN: dict[str, tuple[float, float]] = {"start": (0.0, 1.0), "middle": (0.5, 0.5), "end": (1.0, 0.0)}


def minus(
    value: Any,
) -> str:
    return str(value).replace("-", MINUS)


def thousands(
    value: float,
) -> str:
    return f"{int(round(value)):,}"


def power_of_ten(
    value: float,
) -> str:

    exponent = int(round(math.log10(value)))
    return "10" + str(exponent).translate(_SUPERSCRIPTS)


def linear(
    lo: float,
    hi: float,
    p0: float,
    p1: float,
) -> Callable[[float], float]:

    span = (hi - lo) or 1.0
    return lambda value: p0 + (value - lo) / span * (p1 - p0)


def log_scale(
    lo: float,
    hi: float,
    p0: float,
    p1: float,
) -> Callable[[float], float]:

    a, b = math.log10(lo), math.log10(hi)
    span = (b - a) or 1.0
    return lambda value: p0 + (math.log10(value) - a) / span * (p1 - p0)


def _dash(
    dash: str | None,
) -> str:
    return f' stroke-dasharray="{dash}"' if dash else ""


def _opt(
    name: str,
    value: Any,
) -> str:
    return "" if value is None else f' {name}="{value}"'


class Canvas:

    def __init__(self) -> None:
        self.elements: list[str] = []


    def add(
        self,
        element: str,
    ) -> None:
        self.elements.append(element)


    def text(self, x: float, y: float, s: str, size: float = TICK, anchor: str = "middle", fill: str = "#000",
        rot: float | None = None, family: str = FONT) -> None:

        transform = f' transform="rotate({rot} {x:.1f} {y:.1f})"' if rot else ""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{fill}"{transform}>{s}</text>',
        )


    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = "#000", sw: float = 0.8,
        dash: str | None = None) -> None:

        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" '
            f'stroke-width="{sw}"{_dash(dash)}/>',
        )


    def poly(
        self,
        xy: Iterable[tuple[float, float]],
        stroke: str,
        sw: float = 1.3,
        dash: str | None = None,
    ) -> None:

        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in xy)
        self.add(
            f'<polyline points="{points}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linejoin="round"{_dash(dash)}/>',
        )


    def polygon(
        self,
        xy: Iterable[tuple[float, float]],
        fill: str,
        stroke: str = "none",
        sw: float = 0.0,
        dash: str | None = None,
        fill_opacity: float | None = None,
        linejoin: str | None = None,
    ) -> None:

        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in xy)
        self.add(
            f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{_dash(dash)}'
            f'{_opt(name="fill-opacity", value=fill_opacity)}{_opt(name="stroke-linejoin", value=linejoin)}/>',
        )


    def circle(
        self,
        x: float,
        y: float,
        r: float,
        fill: str,
        stroke: str = "none",
        sw: float = 0.0,
        opacity: float | None = None,
        stroke_opacity: float | None = None,
    ) -> None:

        self.add(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
            f'{_opt(name="fill-opacity", value=opacity)}{_opt(name="stroke-opacity", value=stroke_opacity)}/>',
        )


    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        stroke: str = "none",
        sw: float = 0.0,
        fill_opacity: float | None = None,
        dash: str | None = None,
        rx: float | None = None,
    ) -> None:

        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"{_opt(name="fill-opacity", value=fill_opacity)}{_dash(dash)}{_opt(name="rx", value=rx)}/>',
        )


    def area(
        self,
        pts: Sequence[tuple[float, float]],
        base_y: float,
        color: str,
        dash: str | None = None,
        fill_opacity: float = 0.18,
        sw: float = 1.3,
    ) -> None:

        if not pts:
            return

        self.polygon([(pts[0][0], base_y), *pts, (pts[-1][0], base_y)], color, fill_opacity=fill_opacity)
        self.poly(pts, color, sw=sw, dash=dash)


    def begin_clip(
        self,
        name: str,
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> None:

        self.add(f'<clipPath id="{name}"><rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}"/></clipPath>')
        self.add(f'<g clip-path="url(#{name})">')


    def end_clip(self) -> None:
        self.add("</g>")


    def svg(
        self,
        width: float,
        height: float,
    ) -> str:

        w, h = float(width), float(height)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w / 96:.3f}in" height="{h / 96:.3f}in" '
            f'viewBox="0 0 {w:g} {h:g}"><rect width="{w:g}" height="{h:g}" fill="#fff"/>'
            + "".join(self.elements) + "</svg>")


def _attr(
    attrs: str,
    name: str,
) -> str | None:

    match = re.search(rf'(?<![\w-]){re.escape(name)}="([^"]*)"', attrs)
    return None if match is None else match.group(1)


def _points(
    attrs: str,
) -> list[tuple[float, float]]:

    text = _attr(attrs, "points") or ""
    pairs = [pair.split(",") for pair in text.split()]
    return [(float(x), float(y)) for x, y in pairs if x and y]


def element_bottom(
    name: str,
    attrs: str,
    body: str | None = None,
) -> float | None:

    if name == "text":
        x, y = float(_attr(attrs, "x") or 0.0), float(_attr(attrs, "y") or 0.0)
        size = float(_attr(attrs, "font-size") or TICK)
        rotate = _ROTATE.search(_attr(attrs, "transform") or "")
        if rotate is None:
            return y + TEXT_DESCENT * size

        before, after = _ANCHOR_SPAN.get(_attr(attrs, "text-anchor") or "start", (0.0, 1.0))
        length = CHAR_WIDTH * size * len(body or "")
        angle = math.radians(float(rotate.group("angle")))
        px, py = float(rotate.group("x")), float(rotate.group("y"))
        corners = [(x - before * length, y - TEXT_ASCENT * size), (x + after * length, y - TEXT_ASCENT * size),
            (x - before * length, y + TEXT_DESCENT * size), (x + after * length, y + TEXT_DESCENT * size)]
        return max(py + (cx - px) * math.sin(angle) + (cy - py) * math.cos(angle) for cx, cy in corners)

    if name == "line":
        return max(float(_attr(attrs, "y1") or 0.0), float(_attr(attrs, "y2") or 0.0)) + float(_attr(attrs, "stroke-width") or 0.0) / 2

    if name in ("polyline", "polygon"):
        points = _points(attrs)
        return max(y for _x, y in points) + float(_attr(attrs, "stroke-width") or 0.0) / 2 if points else None

    if name == "circle":
        return float(_attr(attrs, "cy") or 0.0) + float(_attr(attrs, "r") or 0.0) + float(_attr(attrs, "stroke-width") or 0.0) / 2

    if name == "rect":
        y = _attr(attrs, "y")
        if y is None:
            return None  # the canvas background of ``Canvas.svg`` has no position

        return float(y) + float(_attr(attrs, "height") or 0.0) + float(_attr(attrs, "stroke-width") or 0.0) / 2

    return None


def content_bottom(
    svg: str,
) -> float | None:

    clips: dict[str, float] = {}
    stack: list[float | None] = []
    defining: str | None = None
    bottom: float | None = None

    for match in _TAG.finditer(svg):
        name, attrs = match.group("name"), match.group("attrs")

        if name == "clipPath":
            defining = _attr(attrs, "id")
            continue

        if name == "/clipPath":
            defining = None
            continue

        if defining is not None:
            if name == "rect":
                clips[defining] = float(_attr(attrs, "y") or 0.0) + float(_attr(attrs, "height") or 0.0)

            continue

        if name == "g":
            clip = re.search(r"url\(#([^)]+)\)", _attr(attrs, "clip-path") or "")
            stack.append(None if clip is None else clips.get(clip.group(1)))
            continue

        if name == "/g":
            if stack:
                stack.pop()

            continue

        reach = element_bottom(name=name, attrs=attrs, body=match.group("body"))
        if reach is None:
            continue

        limits = [limit for limit in stack if limit is not None]
        if limits:
            reach = min(reach, min(limits))

        bottom = reach if bottom is None else max(bottom, reach)

    return bottom


def trim_bottom(
    svg: str,
    *,
    width: float,
    height: float,
    pad: float = BOTTOM_PAD,
) -> tuple[str, float]:

    bottom = content_bottom(svg)
    if bottom is None:
        return svg, float(height)

    trimmed = float(math.ceil(bottom + pad))
    if trimmed >= float(height):
        return svg, float(height)

    w, h = float(width), float(height)
    before = (f'width="{w / 96:.3f}in" height="{h / 96:.3f}in" viewBox="0 0 {w:g} {h:g}">'
        f'<rect width="{w:g}" height="{h:g}" fill="#fff"/>')
    after = (f'width="{w / 96:.3f}in" height="{trimmed / 96:.3f}in" viewBox="0 0 {w:g} {trimmed:g}">'
        f'<rect width="{w:g}" height="{trimmed:g}" fill="#fff"/>')
    if before not in svg:
        return svg, float(height)

    return svg.replace(before, after, 1), trimmed


BAND_OPACITY = 0.18


def std_band(
    canvas: Canvas,
    samples: Sequence[tuple[float, float, float | None]],
    fy: Callable[[float], float],
    color: str,
    *,
    fill_opacity: float = BAND_OPACITY,
) -> bool:

    runs: list[list[tuple[float, float, float]]] = []
    current: list[tuple[float, float, float]] = []

    for x, mean, std in samples:
        if mean is None or not math.isfinite(float(mean)):
            if current:
                runs.append(current)
                current = []

            continue

        spread = float(std) if std is not None and math.isfinite(float(std)) and float(std) > 0 else 0.0
        current.append((float(x), float(mean), spread))

    if current:
        runs.append(current)

    drawn = False

    for run in runs:
        if len(run) < 2 or not any(spread > 0 for _x, _mean, spread in run):
            continue

        upper = [(x, fy(mean + spread)) for x, mean, spread in run]
        lower = [(x, fy(mean - spread)) for x, mean, spread in reversed(run)]
        canvas.polygon(upper + lower, color, fill_opacity=fill_opacity)
        drawn = True

    return drawn


def panel_letter(
    canvas: Canvas,
    x: float,
    y: float,
    letter: str,
    size: float = LAB,
) -> None:
    # the label of a figure part, placed as Springer sets them: above the top left corner of the part
    canvas.text(x, y, f"({letter})", size=size, anchor="start")


def legend_row(
    canvas: Canvas,
    entries: Sequence[tuple[str, str, str | None, float]],
    *,
    width: float = WIDTH,
    y: float = 11.0,
    gap: float = 26.0,
    char_width: float = 5.2,
    size: float = LEG,
    dots: bool = False,
) -> float:

    widths = [27.0 + len(label) * char_width for label, *_ in entries]
    x = (width - (sum(widths) + gap * (len(entries) - 1))) / 2

    for (label, color, dash, sw), w in zip(entries, widths):
        canvas.line(x, y, x + 22, y, stroke=color, sw=sw, dash=dash)

        if dots:
            canvas.circle(x + 11, y, 2.4, color)

        canvas.text(x + 27, y + 4, label, size=size, anchor="start")
        x += w + gap

    return x - gap


def html_page(
    svg: str,
    width: float,
    height: float,
) -> str:

    return ('<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;padding:0;background:#fff}'
        f'svg{{display:block;width:{width:g}px;height:{height:g}px}}</style></head><body>{svg}</body></html>')


def find_browser(
    explicit: str | os.PathLike[str] | None = None,
) -> Path | None:

    candidates: list[str] = []

    if explicit:
        candidates.append(str(explicit))

    if os.environ.get(ENV_BROWSER):
        candidates.append(os.environ[ENV_BROWSER])

    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path

        found = shutil.which(candidate)
        if found:
            return Path(found)

    for candidate in BROWSER_PATHS:
        if Path(candidate).is_file():
            return Path(candidate)

    for name in BROWSER_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)

    return None


def _render_with_browser(
    browser: Path,
    svg: str,
    png_path: Path,
    width: float,
    height: float,
    scale: int,
) -> bool:

    with tempfile.TemporaryDirectory(prefix="camp_figure_") as tmp:
        folder = Path(tmp)
        page = folder / "figure.html"
        page.write_text(html_page(svg=svg, width=width, height=height), encoding="utf-8")
        shot = folder / "figure.png"
        command = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-sandbox",
            f"--user-data-dir={folder / 'profile'}",
            f"--force-device-scale-factor={int(scale)}",
            f"--window-size={int(round(width))},{int(round(height))}",
            f"--screenshot={shot}",
            page.as_uri(),
        ]

        try:
            subprocess.run(command, capture_output=True, timeout=RENDER_TIMEOUT, cwd=folder, check=False)
        except (subprocess.TimeoutExpired, OSError):
            pass

        if not shot.is_file() or shot.stat().st_size == 0:
            return False

        png_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src=str(shot), dst=str(png_path))

    return True


def _render_with_rsvg(
    svg: str,
    png_path: Path,
    width: float,
    height: float,
    scale: int,
) -> bool:

    tool = shutil.which("rsvg-convert")
    if tool is None:
        return False

    png_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        tool,
        "-w",
        str(int(round(width * scale))),
        "-h",
        str(int(round(height * scale))),
        "-b",
        "white",
        "-o",
        str(png_path),
    ]

    try:
        done = subprocess.run(command, input=svg.encode("utf-8"), capture_output=True, timeout=RENDER_TIMEOUT)
    except (subprocess.TimeoutExpired, OSError):
        return False

    if done.returncode != 0:
        return False

    return png_path.is_file() and png_path.stat().st_size > 0


def _render_with_cairosvg(
    svg: str,
    png_path: Path,
    width: float,
    height: float,
    scale: int,
) -> bool:

    try:
        import cairosvg  # type: ignore[import-not-found]
    except ImportError:
        return False

    png_path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=str(png_path),
        output_width=int(round(width * scale)),
        output_height=int(round(height * scale)),
        background_color="white",
    )

    return png_path.is_file() and png_path.stat().st_size > 0


def render_png(
    svg: str,
    png_path: Path | str,
    *,
    width: float,
    height: float,
    scale: int = 6,
    browser: str | os.PathLike[str] | None = None,
) -> str | None:

    png_path = Path(png_path)
    found = find_browser(browser)
    if found is not None and _render_with_browser(browser=found, svg=svg, png_path=png_path, width=width, height=height, scale=scale):
        return f"browser:{found.name}"

    if _render_with_rsvg(svg=svg, png_path=png_path, width=width, height=height, scale=scale):
        return "rsvg"

    if _render_with_cairosvg(svg=svg, png_path=png_path, width=width, height=height, scale=scale):
        return "cairosvg"

    return None


@dataclass(frozen=True)
class Render:

    png: bool = True
    scale: int = 6
    outline: bool = False
    browser: str | None = None
    trim: bool = True

    @classmethod
    def from_context(
        cls,
        ctx: Any,
    ) -> "Render":

        option = getattr(ctx, "option", None)
        if option is None:
            return cls()

        png = str(option("png", "1")).lower() not in ("0", "false", "no")
        scale = int(float(option("png_scale", "6") or 6))
        if scale < 1:
            raise ValueError(f"--opt png_scale must be a positive integer, got {scale}")

        outline = str(option("outline", "0")).lower() in ("1", "true", "yes")
        trim = str(option("trim", "1")).lower() not in ("0", "false", "no")
        return cls(png=png, scale=scale, outline=outline, browser=option("browser") or None, trim=trim)


def write_figure(
    svg: str,
    stem: Path | str,
    *,
    width: float,
    height: float,
    render: Render | None = None,
) -> list[Path]:

    render = render or Render()
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    if render.trim:
        svg, height = trim_bottom(svg, width=width, height=height)

    if render.outline:
        from scripts.experiment.lib import svgtext

        svg = svgtext.outline_svg(svg)

    svg_path = stem.with_suffix(".svg")
    svg_path.write_text(svg, encoding="utf-8")
    written = [svg_path]

    if render.png:
        png_path = stem.with_suffix(".png")
        used = render_png(svg=svg, png_path=png_path, width=width, height=height, scale=render.scale, browser=render.browser)
        if used is None:
            print(
                f"[figure] {png_path.name} not written: no PNG renderer found (install Microsoft Edge or Google "
                f"Chrome, point ${ENV_BROWSER} or --opt browser= at one, or pip install cairosvg)",
            )
        else:
            written.append(png_path)

    return written


__all__ = [
    "BAND_OPACITY",
    "BOTTOM_PAD",
    "BROWSER_NAMES",
    "BROWSER_PATHS",
    "CAMP",
    "COLUMN_WIDTH",
    "Canvas",
    "ENV_BROWSER",
    "FAR",
    "FONT",
    "GRID",
    "GROUP_LINES",
    "ID",
    "IN1K",
    "LAB",
    "LEG",
    "LOWER_BOUND",
    "MINUS",
    "NEAR",
    "NEUTRAL",
    "OOD",
    "Render",
    "TANL",
    "TICK",
    "WIDTH",
    "content_bottom",
    "element_bottom",
    "find_browser",
    "html_page",
    "legend_row",
    "linear",
    "log_scale",
    "minus",
    "panel_letter",
    "power_of_ten",
    "render_png",
    "std_band",
    "thousands",
    "trim_bottom",
    "write_figure",
]
