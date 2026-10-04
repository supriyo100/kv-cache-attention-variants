"""MkDocs hook: turn three Markdown markers into embedded figures.

Excalidraw drawing (a pan/zoom poster with section jumps, plus a live viewer on demand)::

    ![GQA: g query heads share each K/V head](../assets/excalidraw/gqa.svg){ .excalidraw }

Story plate (an SVG from tools/story_plates.py, inlined so it follows the site's colors)::

    ![Count the cells](../assets/plates/cells.svg){ .plate }

Interactive HTML diagram (iframe)::

    [D06: GQA group-sharing spectrum](../assets/diagrams/D06_gqa_spectrum.html){ .diagram }

All stay readable on GitHub (an image or a link). Paths are relative to the Markdown
file; that matches the built site because mkdocs.yml sets use_directory_urls: false.
The Excalidraw SVGs come from tools/excalidraw_export (re-run it after editing a drawing).
"""

from __future__ import annotations

import html
import json
import posixpath
import re
from functools import lru_cache
from pathlib import Path

EXCALIDRAW = re.compile(r"!\[([^\]]*)\]\(([^)\s]+?/excalidraw/([\w\-]+)\.svg)\)\{\s*\.excalidraw\s*\}")
PLATE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+?/plates/([\w\-]+)\.svg)\)\{\s*\.plate\s*\}")
DIAGRAM = re.compile(r"\[([^\]]+)\]\(([^)\s]+?\.html)\)\{\s*\.diagram\s*\}")
SECTION_FONT = 26  # drawing text at this size or larger, below the title, starts a section
EXPORT_PAD = 10  # tools/excalidraw_export pads the scene by this many px on every side
READ_ZOOM = 0.7  # default poster zoom: 15 px drawing text shows at ~10.5 px


@lru_cache(maxsize=64)
def _sheet(path: str) -> tuple[float, float, tuple[tuple[str, float, float], ...]]:
    """Width, height and section anchors (label, x, y as fractions) of an .excalidraw scene."""
    scene = json.loads(Path(path).read_text(encoding="utf-8"))
    els = [e for e in scene["elements"] if not e.get("isDeleted")]
    xs: list[float] = []
    ys: list[float] = []
    for e in els:
        xs += [e["x"], e["x"] + e["width"]]
        ys += [e["y"], e["y"] + e["height"]]
        for px, py in e.get("points", ()):
            xs.append(e["x"] + px)
            ys.append(e["y"] + py)
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 2 * EXPORT_PAD, max(ys) - y0 + 2 * EXPORT_PAD
    texts = [e for e in els if e["type"] == "text" and (e.get("fontSize") or 0) >= SECTION_FONT]
    title_size = max((e["fontSize"] for e in texts), default=0)
    sections = []
    for e in sorted(texts, key=lambda e: (round(e["y"] / 40), e["x"])):
        label = " ".join(e["text"].split())
        if e["fontSize"] == title_size or len(label) < 6 or label.startswith(("M_KV", "OBJECTIVE")):
            continue
        label = re.split(r"[:(—]| - |\|", label)[0].strip()
        if len(label) > 44:
            label = label[:42].rsplit(" ", 1)[0] + " …"
        sections.append((label, (e["x"] - x0 + EXPORT_PAD) / w, (e["y"] - y0 + EXPORT_PAD) / h))
    return w, h, tuple(sections)


def _excalidraw(m: re.Match[str], src_dir: Path) -> str:
    caption, svg, name = html.escape(m[1]), m[2], m[3]
    folder = posixpath.dirname(svg)
    viewer = f"{posixpath.dirname(folder)}/viewer/excalidraw.html?file={name}"
    source = f"{folder}/{name}.excalidraw"
    w = h = 0.0
    sections: tuple[tuple[str, float, float], ...] = ()
    scene = src_dir / source
    if scene.is_file():
        w, h, sections = _sheet(str(scene.resolve()))
    size = f' width="{w:.0f}" height="{h:.0f}"' if w else ""
    chips = "".join(
        f'<button type="button" data-x="{x:.4f}" data-y="{y:.4f}">{html.escape(label)}</button>'
        for label, x, y in sections
    )
    nav = (f'<nav class="xd-sections" aria-label="Jump to a section of the drawing">'
           f'<span class="xd-sections__label">Sections</span>{chips}</nav>\n' if chips else "")
    return (
        f'\n<figure class="xd-embed xd-poster" data-zoom="{READ_ZOOM}">\n'
        f'<div class="xd-bar"><strong class="xd-title">{caption}</strong>'
        '<span class="xd-tools" role="toolbar" aria-label="Zoom">'
        '<button type="button" data-act="out" aria-label="Zoom out" title="Zoom out (−)">−</button>'
        '<output class="xd-level" aria-live="polite">Fit</output>'
        '<button type="button" data-act="in" aria-label="Zoom in" title="Zoom in (+)">+</button>'
        '<button type="button" data-act="fit" title="Fit the whole drawing (0)">Fit</button>'
        '<button type="button" data-act="full" title="Full screen (F)">Full screen</button></span></div>\n'
        f"{nav}"
        f'<div class="xd-stage" tabindex="0" aria-label="{caption}. Click to zoom in; drag or scroll to pan.">'
        f'<img class="xd-svg" src="{svg}" alt="{caption}" loading="lazy" draggable="false"{size}></div>\n'
        '<figcaption><span class="xd-hint">Click to zoom · drag to pan · Ctrl + wheel to zoom at the cursor</span>'
        f'<span class="xd-actions"><a href="{svg}" target="_blank" rel="noopener">Open image ↗</a>'
        f' · <a href="{viewer}" target="_blank" rel="noopener">Excalidraw viewer ↗</a>'
        f' · <a href="{source}" download>Download .excalidraw</a></span></figcaption>\n'
        '<details class="xd-live"><summary>Explore here: live Excalidraw pan &amp; zoom</summary>'
        f'<iframe data-src="{viewer}" title="{caption} (interactive)"></iframe></details>\n'
        "</figure>\n"
    )


def _plate(m: re.Match[str], src_dir: Path, counter: list[int]) -> str:
    title, svg, name = html.escape(m[1]), m[2], m[3]
    path = src_dir / svg
    if not path.is_file():
        return m[0]
    counter[0] += 1
    markup = path.read_text(encoding="utf-8").strip()
    return (
        f'\n<figure class="vs-plate" id="plate-{name}">\n'
        f'<div class="vs-plate__bar"><span class="vs-plate__no">Plate {counter[0]:02d}</span>'
        f'<span class="vs-plate__title">{title}</span>'
        f'<a class="vs-plate__open" href="{svg}" target="_blank" rel="noopener" title="Open the plate on its own">↗</a></div>\n'
        f'<div class="vs-plate__canvas">{markup}</div>\n'
        "</figure>\n"
    )


def _diagram(m: re.Match[str]) -> str:
    title, src = html.escape(m[1]), m[2]
    return (
        '\n<figure class="dg-embed">\n'
        f'<iframe src="{src}" title="{title}" loading="lazy"></iframe>\n'
        f'<figcaption><strong>{title}</strong><span class="xd-actions">'
        f'<a href="{src}" target="_blank" rel="noopener">Full screen ↗</a></span></figcaption>\n'
        "</figure>\n"
    )


STEPS = 12


def _step_header(meta: dict) -> str:
    """Block-strip progress marker for pages of the core path (front matter ``step``/``layer``)."""
    n = int(meta["step"])
    cells = "".join(f'<i class="{"on" if i < n else ""}"></i>' for i in range(STEPS))
    layer = html.escape(str(meta.get("layer", "")))
    label = f"Step {n} of {STEPS}" + (f", {layer}" if layer else "")
    return f'<div class="ie-step"><span class="ie-strip" aria-hidden="true">{cells}</span>{label}</div>\n\n'


def on_page_markdown(markdown: str, page=None, **kwargs) -> str:
    src_dir = Path(page.file.abs_src_path).parent if page is not None and page.file.abs_src_path else Path(".")
    counter = [0]
    markdown = EXCALIDRAW.sub(lambda m: _excalidraw(m, src_dir), markdown)
    markdown = PLATE.sub(lambda m: _plate(m, src_dir, counter), markdown)
    markdown = DIAGRAM.sub(_diagram, markdown)
    if page is not None and "step" in page.meta:
        markdown = _step_header(page.meta) + markdown
    return markdown
