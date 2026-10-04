"""MkDocs hook: turn two Markdown markers into embedded diagrams.

Excalidraw drawing (static SVG, plus a live pan/zoom viewer on demand)::

    ![GQA: g query heads share each K/V head](../assets/excalidraw/gqa.svg){ .excalidraw }

Interactive HTML diagram (iframe)::

    [D06: GQA group-sharing spectrum](../assets/diagrams/D06_gqa_spectrum.html){ .diagram }

Both stay readable on GitHub (an image and a link). Paths are relative to the Markdown
file; that matches the built site because mkdocs.yml sets use_directory_urls: false.
The SVGs come from tools/excalidraw_export (re-run it after editing a drawing).
"""

from __future__ import annotations

import html
import posixpath
import re

EXCALIDRAW = re.compile(r"!\[([^\]]*)\]\(([^)\s]+?/excalidraw/([\w\-]+)\.svg)\)\{\s*\.excalidraw\s*\}")
DIAGRAM = re.compile(r"\[([^\]]+)\]\(([^)\s]+?\.html)\)\{\s*\.diagram\s*\}")


def _excalidraw(m: re.Match[str]) -> str:
    caption, svg, name = html.escape(m[1]), m[2], m[3]
    folder = posixpath.dirname(svg)
    viewer = f"{posixpath.dirname(folder)}/viewer/excalidraw.html?file={name}"
    source = f"{folder}/{name}.excalidraw"
    return (
        '\n<figure class="xd-embed">\n'
        f'<a href="{viewer}" target="_blank" rel="noopener" title="Open full screen">'
        f'<img class="xd-svg" src="{svg}" alt="{caption}" loading="lazy"></a>\n'
        f"<figcaption><strong>{caption}</strong>"
        f'<span class="xd-actions"><a href="{viewer}" target="_blank" rel="noopener">Full screen ↗</a>'
        f' · <a href="{source}" download>Download .excalidraw</a></span></figcaption>\n'
        '<details class="xd-live"><summary>Explore here: pan &amp; zoom</summary>'
        f'<iframe data-src="{viewer}" title="{caption} (interactive)"></iframe></details>\n'
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


def on_page_markdown(markdown: str, **kwargs) -> str:
    return DIAGRAM.sub(_diagram, EXCALIDRAW.sub(_excalidraw, markdown))
