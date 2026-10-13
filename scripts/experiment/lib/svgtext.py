"""
Replace the ``<text>`` elements of a generated SVG with glyph outlines of their strings (Times New Roman).
"""
from __future__ import annotations

import html
import os
import re
from pathlib import Path

FONT = "Times New Roman"
ENV_FONT_DIR = "CAMP_FIGURE_FONT_DIR"
FONT_FILE_NAMES: dict[str, tuple[str, ...]] = {
    "normal": ("times.ttf", "Times New Roman.ttf", "Times_New_Roman.ttf"),
    "italic": ("timesi.ttf", "Times New Roman Italic.ttf", "Times_New_Roman_Italic.ttf"),
}
FONT_DIRS: tuple[str, ...] = (
    "C:/Windows/Fonts",
    "/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
    "/usr/share/fonts/truetype/msttcorefonts",
    "/usr/share/fonts/TTF",
    "/usr/local/share/fonts",
)

TEXT = re.compile(r"<text ([^>]*)>([^<]*)</text>")
TEXT_TSPAN = re.compile(
    r"<text ([^>]*)>((?:[^<]|<tspan [^>]*>[^<]*</tspan>)*<tspan [^>]*>[^<]*</tspan>"
    r"(?:[^<]|<tspan [^>]*>[^<]*</tspan>)*)</text>",
)

SEGMENT = re.compile(r"<tspan ([^>]*)>([^<]*)</tspan>|([^<]+)")
ATTR = re.compile(r'([\w:-]+)="([^"]*)"')


def font_file(
    style: str = "normal",
) -> Path | None:

    folders = [os.environ[ENV_FONT_DIR]] if os.environ.get(ENV_FONT_DIR) else []
    folders += list(FONT_DIRS)
    for folder in folders:
        for name in FONT_FILE_NAMES[style]:
            path = Path(folder) / name
            if path.is_file():
                return path

    return None


def available() -> bool:

    try:
        import fontTools  # noqa: F401
    except ImportError:
        return False

    return font_file("normal") is not None and font_file("italic") is not None


class Outliner:

    def __init__(
        self,
        path: Path | str,
    ) -> None:

        from fontTools.ttLib import TTFont

        self.font = TTFont(str(path))
        self.cmap = self.font.getBestCmap()
        self.glyphs = self.font.getGlyphSet()
        self.upem = int(self.font["head"].unitsPerEm)
        self.advance = {name: self.font["hmtx"][name][0] for name in self.font.getGlyphOrder()}
        self.pairs = self._kerning_subtables()


    def _kerning_subtables(self) -> list[list]:

        if "GPOS" not in self.font:
            return []

        gpos = self.font["GPOS"].table
        indices: list[int] = []

        for record in gpos.ScriptList.ScriptRecord:
            if record.ScriptTag in ("latn", "DFLT") and record.Script.DefaultLangSys is not None:
                for index in record.Script.DefaultLangSys.FeatureIndex:
                    feature = gpos.FeatureList.FeatureRecord[index]
                    if feature.FeatureTag == "kern":
                        indices.extend(feature.Feature.LookupListIndex)

                if indices:
                    break

        lookups = []

        for index in dict.fromkeys(indices):
            lookup = gpos.LookupList.Lookup[index]
            subtables = [st.ExtSubTable if lookup.LookupType == 9 else st for st in lookup.SubTable]
            lookups.append([st for st in subtables if getattr(st, "LookupType", 2) == 2])

        return lookups


    @staticmethod
    def _x_advance(
        value,
    ) -> int:
        return int(getattr(value, "XAdvance", 0) or 0) if value is not None else 0


    def kern(
        self,
        left: str,
        right: str,
    ) -> int:

        total = 0

        for subtables in self.pairs:
            for st in subtables:
                glyphs = st.Coverage.glyphs
                if left not in glyphs:
                    continue

                if st.Format == 1:
                    records = st.PairSet[glyphs.index(left)].PairValueRecord
                    hit = next((r for r in records if r.SecondGlyph == right), None)
                    if hit is None:
                        continue

                    total += self._x_advance(hit.Value1)
                    break

                c1 = st.ClassDef1.classDefs.get(left, 0)
                c2 = st.ClassDef2.classDefs.get(right, 0)
                total += self._x_advance(st.Class1Record[c1].Class2Record[c2].Value1)
                break

        return total


    def path(
        self,
        string: str,
        start: int = 0,
    ) -> tuple[str, int]:

        from fontTools.pens.svgPathPen import SVGPathPen
        from fontTools.pens.transformPen import TransformPen

        names = []

        for char in string:
            name = self.cmap.get(ord(char))
            if name is None:
                raise ValueError(f"the font has no glyph for {char!r} (U+{ord(char):04X})")

            names.append(name)

        pen = SVGPathPen(self.glyphs)
        cursor = start

        for i, name in enumerate(names):
            self.glyphs[name].draw(TransformPen(pen, (1, 0, 0, 1, cursor, 0)))
            cursor += self.advance[name]

            if i + 1 < len(names):
                cursor += self.kern(name, names[i + 1])

        return pen.getCommands(), cursor - start


_OUTLINERS: dict[str, Outliner] = {}


def _outliner(
    style: str,
) -> Outliner:

    if style not in _OUTLINERS:
        path = font_file(style)
        if path is None:
            raise RuntimeError(
                f"Times New Roman ({style}) is not installed; outlining needs times.ttf and timesi.ttf "
                f"(set ${ENV_FONT_DIR} to a folder that holds them)",
            )

        _OUTLINERS[style] = Outliner(path)

    return _OUTLINERS[style]


def _place(
    attrs: dict[str, str],
    width: int,
    upem: int,
) -> str:

    size = float(attrs["font-size"])
    scale = size / upem
    shift = {"start": 0.0, "middle": -width / 2, "end": -width}[attrs.get("text-anchor", "start")]
    x, y = float(attrs.get("x", 0)), float(attrs.get("y", 0))
    placement = f"translate({x:.3f} {y:.3f}) scale({scale:.6f} {-scale:.6f}) translate({shift:.1f} 0)"
    return f'{attrs["transform"]} {placement}' if "transform" in attrs else placement


def _outline_mixed(
    svg: str,
) -> str:

    def replace(
        match: re.Match[str],
    ) -> str:

        attrs = dict(ATTR.findall(match.group(1)))
        base = attrs.get("font-style", "normal")
        cursor, parts, label, upem = 0, [], "", None

        for segment in SEGMENT.finditer(match.group(2)):
            if segment.group(3) is not None:
                style, string = base, html.unescape(segment.group(3))
            else:
                style = dict(ATTR.findall(segment.group(1))).get("font-style", base)
                string = html.unescape(segment.group(2))

            outliner = _outliner(style)

            if upem is None:
                upem = outliner.upem

            if outliner.upem != upem:
                raise RuntimeError("the regular and italic fonts have different units per em")

            d, width = outliner.path(string, start=cursor)
            parts.append(d)
            cursor += width
            label += string

        fill = attrs.get("fill", "#000")
        return (f'<path aria-label="{html.escape(label)}" transform="{_place(attrs=attrs, width=cursor, upem=upem or 1000)}" '
            f'fill="{fill}" d="{"".join(parts)}"/>')

    return TEXT_TSPAN.sub(replace, svg)


def outline_svg(
    svg: str,
) -> str:

    if not available():
        raise RuntimeError("outlining needs fontTools and the Times New Roman font files; see lib/svgtext.py")

    svg = _outline_mixed(svg)

    def replace(
        match: re.Match[str],
    ) -> str:

        attrs = dict(ATTR.findall(match.group(1)))
        string = html.unescape(match.group(2))
        outliner = _outliner(attrs.get("font-style", "normal"))
        d, width = outliner.path(string)
        fill = attrs.get("fill", "#000")
        return (f'<path aria-label="{html.escape(string)}" transform="{_place(attrs=attrs, width=width, upem=outliner.upem)}" '
            f'fill="{fill}" d="{d}"/>')

    return TEXT.sub(replace, svg)


__all__ = [
    "ENV_FONT_DIR",
    "FONT",
    "FONT_DIRS",
    "FONT_FILE_NAMES",
    "Outliner",
    "available",
    "font_file",
    "outline_svg",
]
