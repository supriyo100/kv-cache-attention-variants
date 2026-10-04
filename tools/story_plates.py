"""Render the story plates to docs/assets/plates/*.svg.

    uv run python tools/story_plates.py

Each plate is a plain SVG drawn with semantic classes. hooks/embeds.py inlines it into the page
(`![title](../assets/plates/x.svg){ .plate }`), where the classes pick up the site's color tokens and
follow light/dark mode. The <style> inside each file carries fallback colors, so the files also
render on their own (GitHub, a browser tab). Every number is computed here, not typed in.
The content is distilled from the hand-drawn sheets in docs/assets/excalidraw/: the first eleven
plates (Attention variants compared) from comparison and architecture_evolution, the rest (The
visual story) from the other sheets.
"""

from __future__ import annotations

import itertools
import math
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "plates"
W = 1000
KINDS = "icmsx"  # ink, compute, memory, shared, waste

STYLE = """
.vsp{font-family:var(--ie-font-mono,"IBM Plex Mono",ui-monospace,Consolas,monospace)}
.vsp text{font-size:13px;fill:var(--ie-ink,#1c232b)}
.vsp .sm{font-size:11.5px;fill:var(--ie-ink-2,#4a5561)}
.vsp .xs{font-size:10.5px;fill:var(--ie-ink-2,#4a5561)}
.vsp .b{font-weight:600}
.vsp .hd{font-family:var(--ie-font-ui,Archivo,"Helvetica Neue",Arial,sans-serif);font-size:15px;font-weight:700;fill:var(--ie-ink,#1c232b)}
.vsp .lg{font-size:20px;font-weight:700}
.vsp .n{fill:var(--ie-free,#fff);stroke:var(--ie-ink-2,#4a5561);stroke-width:1.2}
.vsp .n-c{fill:var(--ie-compute-bg,#d9efec);stroke:var(--ie-compute,#0e7c72);stroke-width:1.3}
.vsp .n-m{fill:var(--ie-memory-bg,#f6e8c8);stroke:var(--ie-memory,#b5790a);stroke-width:1.3}
.vsp .n-s{fill:var(--ie-shared-bg,#e5e1f8);stroke:var(--ie-shared,#5b47c9);stroke-width:1.3}
.vsp .n-x{fill:var(--ie-waste-bg,#f6dade);stroke:var(--ie-waste,#b42b3e);stroke-width:1.3}
.vsp .n-p{fill:var(--ie-paper-2,#eceee9);stroke:var(--ie-rule,#d5d9d3);stroke-width:1}
.vsp .n-mm{fill:var(--ie-memory,#b5790a);fill-opacity:.5;stroke:var(--ie-memory,#b5790a);stroke-width:1.3}
.vsp .grp{fill:none;stroke:var(--ie-ink-2,#4a5561);stroke-width:1;stroke-dasharray:5 4;opacity:.75}
.vsp .grp-m{fill:var(--ie-memory,#b5790a);fill-opacity:.06;stroke:var(--ie-memory,#b5790a);stroke-width:1.2;stroke-dasharray:6 4}
.vsp .w-i{fill:none;stroke:var(--ie-ink-2,#4a5561);stroke-width:1.3}
.vsp .w-c{fill:none;stroke:var(--ie-compute,#0e7c72);stroke-width:1.4}
.vsp .w-m{fill:none;stroke:var(--ie-memory,#b5790a);stroke-width:1.4}
.vsp .w-s{fill:none;stroke:var(--ie-shared,#5b47c9);stroke-width:1.4}
.vsp .w-x{fill:none;stroke:var(--ie-waste,#b42b3e);stroke-width:1.4}
.vsp .dash{stroke-dasharray:4 3}
.vsp .rule{fill:none;stroke:var(--ie-rule,#d5d9d3);stroke-width:1}
.vsp .f-i{fill:var(--ie-ink-2,#4a5561)}
.vsp .f-c{fill:var(--ie-compute,#0e7c72)}
.vsp .f-m{fill:var(--ie-memory,#b5790a)}
.vsp .f-s{fill:var(--ie-shared,#5b47c9)}
.vsp .f-x{fill:var(--ie-waste,#b42b3e)}
"""

# ---- shared numbers (Llama-3-8B shape: H = 32, d_h = 128, L = 32, BF16) ----------------------
H, DH, LAYERS, BYTES, CTX, CHATS = 32, 128, 32, 2, 8192, 4
D_C, D_R = 512, 64
VARIANTS = [  # name, elements per token per layer, formula
    ("MHA", 2 * H * DH, "2·H·d_h"),
    ("GQA-8", 2 * 8 * DH, "2·G·d_h"),
    ("MQA", 2 * 1 * DH, "2·d_h"),
    ("MLA", D_C + D_R, "d_c + d_R"),
]
HBM_BW = 1.555e12  # A100-40GB, bytes/s
WEIGHTS = 14e9  # 7B parameters in BF16


def binary(n: float) -> str:
    for unit, size in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= size:
            v = n / size
            return f"{v:.3f}".rstrip("0").rstrip(".") + " " + unit
    return f"{n:,.0f} B"


class Plate:
    def __init__(self, name: str, height: int, label: str):
        self.name, self.h, self.label, self.parts = name, height, label, []

    # primitives
    def add(self, s: str) -> None:
        self.parts.append(s)

    def rect(self, x, y, w, h, cls, rx=7, extra=""):
        self.add(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx}" class="{cls}"{extra}/>')

    def text(self, x, y, s, cls="", anchor="start"):
        a = "" if anchor == "start" else f' text-anchor="{anchor}"'
        c = f' class="{cls}"' if cls else ""
        self.add(f'<text x="{x:g}" y="{y:g}"{a}{c}>{escape(s)}</text>')

    def lines(self, x, y, rows, cls="sm", anchor="start", step=15):
        for i, s in enumerate(rows):
            self.text(x, y + i * step, s, cls, anchor)

    def wire(self, pts, k="i", arrow=True, dash=False, both=False):
        d = "M" + " L".join(f"{x:g},{y:g}" for x, y in pts)
        cls = f"w-{k}" + (" dash" if dash else "")
        m = f' marker-end="url(#{self.name}-a{k})"' if arrow else ""
        if both:
            m += f' marker-start="url(#{self.name}-a{k})"'
        self.add(f'<path d="{d}" class="{cls}"{m}/>')

    def node(self, x, y, w, h, kind, title, sub=None, dot=True, anchor="middle", tcls="b"):
        self.rect(x, y, w, h, kind)
        if dot:
            self.add(f'<circle cx="{x + 8:g}" cy="{y + 8:g}" r="2.6" class="f-{kind[-1] if kind != "n" else "i"}"/>')
        cx = x + w / 2 if anchor == "middle" else x + 14
        if sub is None:
            self.text(cx, y + h / 2 + 4.5, title, tcls, anchor)
        else:
            self.text(cx, y + h / 2 - 2, title, tcls, anchor)
            self.text(cx, y + h / 2 + 13, sub, "xs", anchor)

    def group(self, x, y, w, h, label, cls="grp", lcls="xs"):
        self.rect(x, y, w, h, cls, rx=10)
        if label:
            self.text(x + 10, y + 15, label, lcls)

    def diamond(self, cx, cy, hw, hh, label, cls="sm"):
        pts = f"{cx - hw:g},{cy:g} {cx:g},{cy - hh:g} {cx + hw:g},{cy:g} {cx:g},{cy + hh:g}"
        self.add(f'<polygon points="{pts}" class="n"/>')
        self.text(cx, cy + 4.5, label, cls, "middle")

    def svg(self) -> str:
        defs = "".join(
            f'<marker id="{self.name}-a{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="f-{k}"/></marker>'
            for k in KINDS
        )
        style = " ".join(line.strip() for line in STYLE.strip().splitlines())
        body = "\n".join(self.parts)
        return (
            f'<svg class="vsp" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {self.h}" '
            f'role="img" aria-label="{escape(self.label)}">\n<style>{style}</style>\n'
            f"<defs>{defs}</defs>\n{body}\n</svg>\n"
        )


# ---- plates ------------------------------------------------------------------------------------

def plate_terms() -> Plate:
    p = Plate("terms", 400, "The seven terms of the KV cache formula and the techniques that attack each one")
    terms = [("2", "K and V"), ("L", "layers"), ("B", "sequences"), ("T", "tokens"),
             ("H_kv", "KV heads"), ("d_h", "head dim"), ("S", "bytes/elem")]
    chips = {
        0: [("n-s", "MLA: K and V", "from one latent")],
        1: [("n-s", "CLA: adjacent", "layers share, 2×"), ("n-s", "YOCO: global KV", "cached once"),
            ("n-c", "Hybrid SSM: KV", "in 1/8–1/4 layers")],
        2: [("n-p", "demand, not", "a target"), ("n-c", "Paged KV: <4%", "waste, more fit")],
        3: [("n-c", "Sliding window", "128K→8K = 16×"), ("n-c", "Sinks, H2O,", "SnapKV eviction"),
            ("n-c", "SSM: fixed", "state, no T")],
        4: [("n-s", "GQA-8: 32 → 8", "4× smaller"), ("n-s", "MQA: 32 → 1", "32× smaller")],
        5: [("n-p", "fixed by the", "model width")],
        6: [("n-m", "FP8 KV: 2 → 1 B", "2× smaller"), ("n-m", "INT4 KV: 0.5 B", "4× smaller")],
    }
    tw, pitch = 118, 140
    for i, (sym, meaning) in enumerate(terms):
        x = 20 + i * pitch
        p.rect(x, 16, tw, 70, "n")
        p.text(x + tw / 2, 48, sym, "lg", "middle")
        p.text(x + tw / 2, 72, meaning, "xs", "middle")
        if i < len(terms) - 1:
            p.text(x + tw + 11, 56, "×", "lg f-i", "middle")
        for j, (cls, a, b) in enumerate(chips[i]):
            y = 104 + j * 50
            p.rect(x, y, tw, 42, cls, rx=5)
            p.text(x + tw / 2, y + 18, a, "xs b", "middle")
            p.text(x + tw / 2, y + 33, b, "xs", "middle")
            p.wire([(x + tw / 2, y - 18 if j == 0 else y - 8), (x + tw / 2, y)], "i", arrow=False, dash=True)
    # MLA bracket over H_kv and d_h
    x0, x1, y = 20 + 4 * pitch, 20 + 5 * pitch + tw, 268
    p.wire([(x0, y - 8), (x0, y), (x1, y), (x1, y - 8)], "s", arrow=False)
    cx = (x0 + x1) / 2
    p.text(cx, y + 20, "MLA attacks H_kv·d_h jointly", "sm b f-s", "middle")
    p.text(cx, y + 36, "2·128·128 = 32,768 → 512 + 64 = 576 (~57×)", "xs", "middle")
    # outside the formula
    p.text(20, 337, "Not in the formula:", "sm b")
    for k, (a, b) in enumerate([("fragmentation", "→ PagedAttention"), ("repeated prefill", "→ prefix caching"),
                                ("HBM traffic per step", "→ FlashAttention")]):
        x = 180 + k * 268
        p.rect(x, 318, 250, 32, "n-x", rx=5)
        p.text(x + 125, 338, f"{a} {b}", "xs b", "middle")
    for k, (cls, lab) in enumerate([("n-s", "architecture (retrain)"), ("n-c", "serving / runtime"),
                                    ("n-m", "number format"), ("n-x", "outside the formula")]):
        x = 20 + k * 240
        p.rect(x, 372, 14, 14, cls, rx=3)
        p.text(x + 22, 383, lab, "xs")
    return p


def plate_heads() -> Plate:
    p = Plate("heads", 430, "Head wiring of MHA, GQA-8 and MQA: which cached K/V each query head reads")
    panels = [("MHA", "H_kv = H_q = 32", 32), ("GQA-8", "H_kv = 8, groups of 4", 8), ("MQA", "H_kv = 1", 1)]
    full = 2 * H * DH
    for c, (name, sub, hkv) in enumerate(panels):
        x0 = 20 + c * 330
        p.text(x0, 22, name, "hd")
        p.text(x0 + 300, 22, sub, "xs", "end")
        p.group(x0, 34, 300, 296, "")
        ys = [52 + i * 32 for i in range(8)]
        for i, y in enumerate(ys):
            p.node(x0 + 20, y, 70, 24, "n-c", f"Q{i + 1}", dot=False, tcls="sm b")
        if hkv == 32:
            kvs = [(y, 24, f"K{i + 1} V{i + 1}", None) for i, y in enumerate(ys)]
        elif hkv == 8:
            kvs = [(ys[0], 120, "K1 V1", "serves Q1–Q4"), (ys[4], 120, "K2 V2", "serves Q5–Q8")]
        else:
            kvs = [(ys[0], 248, "K V", "shared by all")]
        for y, h, lab, s in kvs:
            p.node(x0 + 190, y, 92, h, "n-m", lab, s, dot=h > 24, tcls="sm b")
        for y in ys:
            p.wire([(x0 + 90, y + 12), (x0 + 188, y + 12)], "c")
        p.text(x0 + 150, 322, "8 of 32 query heads drawn", "xs", "middle")
        elems = 2 * hkv * DH
        p.text(x0, 356, f"2 · {hkv} · {DH} = {elems:,}", "b")
        ratio = full // elems
        p.text(x0 + 300, 356, "baseline" if ratio == 1 else f"{ratio}× smaller", "sm b f-s" if ratio > 1 else "sm",
               "end")
        p.text(x0, 374, "numbers cached per token per layer", "xs")
        p.rect(x0, 386, 300, 12, "n-p", rx=2)
        p.rect(x0, 386, max(300 * elems / full, 2), 12, "n-mm", rx=2)
        p.text(x0, 416, ["every Q head owns its K, V", "4 Q heads share each K, V", "all 32 Q heads read one K, V"][c],
               "xs")
    return p


def plate_cells() -> Plate:
    p = Plate("cells", 360, "Count the cells: KV numbers per token per layer for MHA, GQA-8, MQA and MLA")
    p.text(20, 16, f"one cell = {DH} numbers = one head's K or V vector, for one token in one layer"
                   f"  (H = {H}, d_h = {DH}, L = {LAYERS}, BF16)", "xs")
    x0, cw, pitch, ch = 200, 10.6, 12.2, 24
    full = VARIANTS[0][1]
    for r, (name, elems, formula) in enumerate(VARIANTS):
        y = 34 + r * 48
        p.text(20, y + 11, name, "hd")
        p.text(20, y + 27, f"{formula} = {elems:,}", "xs")
        x = x0
        if name == "MLA":
            for _ in range(D_C // DH):
                p.rect(x, y, cw, ch, "n-s", rx=2)
                x += pitch
            p.rect(x, y, cw * D_R / DH, ch, "n-c", rx=2)
            x += pitch
            p.text(x0, y + ch + 12, f"c^KV = {D_C // DH} cells ({D_C})  +  k^R = half a cell ({D_R}, RoPE)", "xs")
        else:
            kv = elems // (2 * DH)
            for half, cls in ((0, "n-m"), (1, "n-mm")):
                for _ in range(kv):
                    p.rect(x, y, cw, ch, cls, rx=2)
                    x += pitch
            if name == "MHA":
                p.text(x0, y + ch + 12, f"K_1 … K_{kv}", "xs")
                p.text(x0 + kv * pitch, y + ch + 12, f"V_1 … V_{kv}", "xs")
        if name != "MHA":
            p.text(x + 10, y + 16, f"= {elems:,}  ·  {full / elems:.3g}× smaller", "sm b")
    # table
    cols = [20, 150, 290, 420, 560, 700, 860]
    heads = ["method", "per layer", "× 2 B", f"× {LAYERS} layers", "one 8K chat", "4 chats × 8K", "vs MHA"]
    ty = 236
    for x, hcell in zip(cols, heads, strict=True):
        p.text(x, ty, hcell, "xs b")
    p.add(f'<path d="M20,{ty + 7} L980,{ty + 7}" class="rule"/>')
    for r, (name, elems, _) in enumerate(VARIANTS):
        y = ty + 26 + r * 22
        per_layer = elems * BYTES
        per_tok = per_layer * LAYERS
        row = [name, f"{elems:,}", binary(per_layer), binary(per_tok), binary(per_tok * CTX),
               binary(per_tok * CTX * CHATS), f"{full / elems:.3g}×"]
        for i, (x, v) in enumerate(zip(cols, row, strict=True)):
            p.text(x, y, v, "sm b" if i in (0, 6) else "sm")
        p.add(f'<path d="M20,{y + 7} L980,{y + 7}" class="rule"/>')
    return p


def plate_mla() -> Plate:
    p = Plate("mla", 400, "MLA dataflow: contract the hidden state to a cached latent, expand only inside the math")
    p.text(190, 34, f"contract  {2 * H * DH:,} → {D_C + D_R}", "sm b f-s")
    p.text(610, 34, "expand only when needed", "sm b f-c")
    p.node(20, 170, 110, 50, "n", "h_t", f"[{H * DH:,}]")
    p.node(190, 90, 140, 50, "n-c", "W^DKV  ↓", f"{H * DH:,} × {D_C}")
    p.node(190, 250, 140, 50, "n-c", "W^KR + RoPE", f"{H * DH:,} × {D_R}")
    p.group(370, 50, 190, 300, "cached in HBM, per token", "grp-m", "xs b")
    p.node(390, 90, 150, 50, "n-s", "c^KV", f"[{D_C}] latent")
    p.node(390, 250, 150, 50, "n-c", "k^R", f"[{D_R}] · all heads")
    p.text(465, 186, f"{D_C + D_R} numbers", "b", "middle")
    p.text(465, 203, f"vs {2 * H * DH:,} for MHA", "xs", "middle")
    p.text(465, 219, f"{2 * H * DH / (D_C + D_R):.1f}× smaller", "xs b f-s", "middle")
    p.node(610, 60, 140, 50, "n-c", "W^UK", f"{D_C} → {H}×{DH}")
    p.node(610, 140, 140, 50, "n-c", "W^UV", f"{D_C} → {H}×{DH}")
    p.node(800, 60, 180, 50, "n", "k^C_1 … k^C_32", f"{H} heads × {DH}")
    p.node(800, 140, 180, 50, "n", "v_1 … v_32", f"{H} heads × {DH}")
    p.node(800, 250, 180, 50, "n", "key of head i", f"[k^C_i ; k^R] = {DH + D_R}")
    p.wire([(130, 195), (160, 195), (160, 115), (188, 115)], "c")
    p.wire([(160, 195), (160, 275), (188, 275)], "c")
    p.wire([(330, 115), (388, 115)], "s")
    p.wire([(330, 275), (388, 275)], "c")
    p.wire([(540, 115), (575, 115), (575, 85), (608, 85)], "s")
    p.wire([(575, 115), (575, 165), (608, 165)], "s")
    p.wire([(750, 85), (798, 85)], "c")
    p.wire([(750, 165), (798, 165)], "c")
    p.wire([(980, 85), (994, 85), (994, 275), (982, 275)], "i")
    p.wire([(540, 275), (798, 275)], "c")
    p.rect(590, 318, 390, 66, "n-p", rx=6)
    p.text(604, 338, "decode trick (absorption)", "sm b")
    p.text(604, 355, "W^UK folds into the query, W^UV into W^O:", "xs")
    p.text(604, 370, "per-head K and V are never materialised.", "xs")
    p.text(20, 372, "K and V are both decoded from the same c^KV, so the factor 2 disappears.", "xs")
    return p


def plate_quality() -> Plate:
    p = Plate("quality", 380, "Schematic: quality against KV compression; MLA sits off the head-count curve")
    x0, x1, y0, y1 = 80, 600, 34, 300  # plot box

    def px(c):
        return x0 + math.log2(c) / 6 * (x1 - x0)

    def py(q):
        return y1 - (q - .45) / .5 * (y1 - y0)

    p.add(f'<path d="M{x0},{y0} L{x0},{y1} L{x1},{y1}" class="w-i"/>')
    for c in (1, 2, 4, 8, 16, 32, 64):
        p.add(f'<path d="M{px(c):g},{y1} L{px(c):g},{y1 + 5}" class="w-i"/>')
        p.text(px(c), y1 + 18, f"{c}×", "xs", "middle")
    p.text((x0 + x1) / 2, y1 + 36, "KV compression vs MHA (log scale)", "xs", "middle")
    p.add(f'<text x="{x0 - 14}" y="{(y0 + y1) / 2}" class="xs" text-anchor="middle" '
          f'transform="rotate(-90 {x0 - 14} {(y0 + y1) / 2})">quality / K-V diversity</text>')
    p.text(x0 + 10, y0 + 6, "schematic, not measured", "xs b f-x")
    curve = [(1, .9), (2, .88), (4, .85), (8, .78), (16, .69), (32, .58)]
    d = "M" + " L".join(f"{px(c):g},{py(q):g}" for c, q in curve)
    p.add(f'<path d="{d}" class="w-s" stroke-linejoin="round"/>')
    for name, c, q in (("MHA", 1, .9), ("GQA-16", 2, .88), ("GQA-8", 4, .85), ("GQA-4", 8, .78), ("MQA", 32, .58)):
        p.add(f'<circle cx="{px(c):g}" cy="{py(q):g}" r="5" class="n-s"/>')
        p.text(px(c) + 8, py(q) + 18, name, "xs b")
    mla_c = 2 * H * DH / (D_C + D_R)
    on_curve = .69 + (.78 - .69) * (math.log2(16) - math.log2(mla_c))  # interpolate between 8× and 16×
    p.wire([(px(mla_c), py(on_curve) - 6), (px(mla_c), py(.9) + 9)], "m", dash=True)
    p.add(f'<rect x="{px(mla_c) - 6:g}" y="{py(.9) - 6:g}" width="12" height="12" rx="2" class="n-m"/>')
    p.text(px(mla_c) + 10, py(.9) - 10, f"MLA ({mla_c:.1f}× here)", "sm b f-m")
    p.text(px(mla_c) + 10, py(.9) + 5, "changes WHAT is cached", "xs")
    p.add(f'<circle cx="{px(4):g}" cy="{py(.62):g}" r="5" class="n-c"/>')
    p.text(px(4) - 9, py(.62) + 4, "hybrid (KV in 25% of layers)", "xs b", "end")
    p.text(px(4) - 9, py(.62) + 18, "weaker exact recall", "xs", "end")
    cards = [
        ("n-s", "GQA / MQA: one curve", ["fewer KV heads → fewer bytes,", "less K/V diversity; MQA pays", "the most quality."]),
        ("n-m", "MLA: off the curve", ["32 distinct per-head K/V views", "from one 576-number latent.", "Pays in FLOPs, a RoPE split", "and retraining, not quality."]),
        ("n-c", "Hybrid: a different trade", ["swaps exact token retrieval", "for a fixed-size state."]),
    ]
    y = 30
    for cls, title, rows in cards:
        h = 30 + 15 * len(rows)
        p.rect(640, y, 340, h, cls, rx=6)
        p.text(654, y + 20, title, "sm b")
        p.lines(654, y + 38, rows, "xs")
        y += h + 14
    return p


def plate_bandwidth() -> Plate:
    p = Plate("bandwidth", 310, "Per decode step, weights plus the whole KV cache are read from HBM")
    p.text(20, 16, f"B = {CHATS} chats × 8K tokens  ·  A100-40GB, {HBM_BW / 1e12:.3f} TB/s  ·  "
                   "step time ≥ bytes ÷ bandwidth  (theoretical lower bound)", "xs")
    scale = 560 / 21  # px per ms
    bx = 210
    w_ms = WEIGHTS / HBM_BW * 1e3
    rows = sorted(VARIANTS, key=lambda v: -v[1])
    for r, (name, elems, _) in enumerate(rows):
        y = 50 + r * 50
        kv_bytes = elems * BYTES * LAYERS * CTX * CHATS
        kv_ms = kv_bytes / HBM_BW * 1e3
        total = w_ms + kv_ms
        p.text(20, y + 14, name, "hd")
        p.text(20, y + 30, f"KV {kv_bytes / 1e9:.2f} GB", "xs")
        p.rect(bx, y, w_ms * scale, 30, "n-p", rx=3)
        p.text(bx + 8, y + 19, f"weights {w_ms:.1f} ms", "xs")
        p.rect(bx + w_ms * scale, y, max(kv_ms * scale, 3), 30, "n-mm" if name == "MHA" else "n-m", rx=3)
        if kv_ms * scale > 70:
            p.text(bx + w_ms * scale + 8, y + 19, f"KV {kv_ms:.2f} ms", "xs b")
        tx = bx + total * scale + 12
        p.text(tx, y + 14, f"≥ {total:.1f} ms", "sm b")
        p.text(tx, y + 29, f"≈ {1000 / total:.0f} tokens/s per chat" + ("" if kv_ms * scale > 70 else f"  (KV {kv_ms:.2f} ms)"),
               "xs")
    gx = bx + w_ms * scale
    p.wire([(gx, 40), (gx, 246)], "i", arrow=False, dash=True)
    p.text(gx, 36, "weights alone", "xs", "middle")
    p.text(20, 272, f"With MHA the KV read ({rows[0][1] * BYTES * LAYERS * CTX * CHATS / HBM_BW * 1e3:.1f} ms) exceeds "
                    f"the weight read ({w_ms:.1f} ms). After GQA-8 or MLA, the weights dominate again.", "xs")
    p.text(20, 290, "Bytes per step scale with the cache: shrinking KV 4× or 14× shrinks this read by the same factor.",
           "xs")
    return p


def plate_layers() -> Plate:
    p = Plate("layers", 410, "Per-layer caches: MHA/GQA own one per layer, MLA thins it, CLA shares pairs, YOCO caches once")
    cols = [("A · MHA / GQA / MQA", "every layer owns KV"), ("B · MLA", "every layer owns a thin latent"),
            ("C · CLA", "adjacent layers share one KV"), ("D · YOCO", "cache one global KV")]
    ys = [56 + i * 36 for i in range(8)]
    for c, (title, sub) in enumerate(cols):
        x0 = 20 + c * 245
        p.text(x0, 22, title, "hd")
        p.text(x0, 40, sub, "xs")
        for i, y in enumerate(ys):
            L = i + 1
            if c == 0:
                p.node(x0, y, 120, 28, "n-c", f"L{L} attention", dot=False, tcls="sm")
                p.node(x0 + 132, y, 90, 28, "n-m", "KV", dot=False, tcls="sm b")
            elif c == 1:
                p.node(x0, y, 120, 28, "n-c", f"L{L} attn (MLA)", dot=False, tcls="sm")
                p.rect(x0 + 132, y + 6, 18, 16, "n-s", rx=2)
                p.text(x0 + 158, y + 18, f"{D_C + D_R}", "xs")
            elif c == 2:
                if i % 2 == 0:
                    p.node(x0, y, 120, 28, "n-c", f"L{L} attention", dot=False, tcls="sm")
                    p.node(x0 + 132, y, 90, 28, "n-m", f"KV{i // 2 + 1}", dot=False, tcls="sm b")
                else:
                    p.node(x0, y, 120, 28, "n-c", f"L{L} reuses KV{i // 2 + 1}", dot=False, tcls="sm")
                    p.wire([(x0 + 120, y + 14), (x0 + 177, y + 14), (x0 + 177, y - 9)], "s", dash=True)
            else:
                if i < 4:
                    p.node(x0, y, 120, 28, "n-c", f"L{L} self-dec", dot=False, tcls="sm")
                    p.rect(x0 + 132, y + 8, 16, 12, "n-c", rx=2)
                else:
                    p.node(x0, y, 120, 28, "n-c", f"L{L} cross-dec", dot=False, tcls="sm")
                    p.wire([(x0 + 120, y + 14), (x0 + 148, y + 14)], "s")
        if c == 3:
            gy = ys[4]
            p.node(x0 + 150, gy, 72, 4 * 36 - 8, "n-m", "global", "KV", dot=True)
            p.wire([(x0 + 186, ys[3] + 28), (x0 + 186, gy - 2)], "m")
            p.text(x0 + 156, ys[0] + 18, "small", "xs")
            p.text(x0 + 156, ys[0] + 31, "fixed cache", "xs")
    foot = [("8 layers → 8 KV copies", "GQA/MQA thin each copy,", "never make fewer"),
            ("8 copies, ~14× thinner", "shrinks the width,", "not the layer count"),
            ("8 layers → 4 copies (2×)", "arXiv 2405.12981:", "~2× vs MQA, near-equal acc."),
            ("1 global KV + tiny caches", "arXiv 2405.05254:", "You Only Cache Once")]
    for c, (a, b, d) in enumerate(foot):
        x0 = 20 + c * 245
        p.add(f'<path d="M{x0},{350} L{x0 + 225},{350}" class="rule"/>')
        p.text(x0, 368, a, "sm b")
        p.text(x0, 384, b, "xs")
        p.text(x0, 398, d, "xs")
    return p


def plate_state() -> Plate:
    p = Plate("state", 440, "Full attention reads every cached token; a recurrent layer folds tokens into one state")
    p.text(20, 22, "Full attention: Q reads every cached K, V", "hd")
    p.text(520, 22, "Recurrent / SSM: s_t = f(s_t-1, x_t)", "hd")
    p.add('<path d="M500,8 L500,280" class="rule"/>')
    for i in range(6):
        x = 20 + i * 76
        p.node(x, 44, 64, 24, "n-p", f"tok {i + 1}", dot=False, tcls="xs")
        p.node(x, 96, 64, 26, "n-m", "K, V", dot=False, tcls="xs b")
        p.wire([(x + 32, 68), (x + 32, 94)], "i")
        p.wire([(240, 196), (x + 32, 124)], "c")
        x2 = 520 + i * 76
        p.node(x2, 44, 64, 24, "n-p", f"tok {i + 1}", dot=False, tcls="xs")
        p.node(x2, 96, 64, 26, "n-s", f"s_{i + 1}", dot=False, tcls="xs b")
        p.wire([(x2 + 32, 68), (x2 + 32, 94)], "i")
        if i:
            p.wire([(x2 - 12, 109), (x2 - 2, 109)], "s")
    p.node(180, 196, 120, 32, "n-c", "query Q_t", dot=False, tcls="sm b")
    p.node(860, 196, 120, 32, "n-c", "query Q_t", dot=False, tcls="sm b")
    p.wire([(932, 124), (932, 194)], "s")
    p.text(924, 166, "reads s_6 only", "xs", "end")
    p.text(20, 252, "exact retrieval of any earlier token", "sm b f-c")
    p.text(20, 268, "bytes per step ∝ T · H_kv · d_h  (grows with T)", "xs f-x")
    p.text(520, 252, "fixed-size state: memory independent of T", "sm b f-c")
    p.text(520, 268, "lossy summary: weak at recalling one exact token", "xs f-x")
    p.text(20, 306, "KV budget across 8 layers", "hd")
    rows = [("Full attention", "AAAAAAAA", "KV in 100% of layers"),
            ("Qwen3-Next (3:1)", "SSSASSSA", "25% + fixed state"),
            ("Jamba (1:7)", "SSSASSSS", "12.5% + fixed state"),
            ("Pure SSM / linear", "SSSSSSSS", "0%, state only")]
    for r, (name, pattern, note) in enumerate(rows):
        y = 320 + r * 28
        p.text(20, y + 15, name, "sm b")
        for i, ch in enumerate(pattern):
            x = 200 + i * 64
            attn = ch == "A"
            p.rect(x, y, 58, 22, "n-m" if attn else "n-s", rx=3)
            p.text(x + 29, y + 15, "attn" if attn else "state", "xs", "middle")
        p.text(726, y + 15, note, "sm b" if r < 3 else "sm")
    return p


def plate_timeline() -> Plate:
    p = Plate("timeline", 470, "Architecture evolution from MHA to hybrids, and which term of the formula each attacks")
    cards = [
        ("MHA", "2017", "n-p", "baseline", ["every Q head owns", "its own K and V"], ["largest cache; every", "head re-read per step"]),
        ("MQA", "2019", "n-s", "H_kv: 32 → 1", ["one K/V head shared", "by all Q heads"], ["K/V diversity collapses;", "quality can drop"]),
        ("GQA", "2023", "n-s", "H_kv: 32 → 8", ["groups of 4 Q heads", "share one K/V head"], ["uptrain from MHA", "with ~5% compute"]),
        ("MLA", "2024", "n-m", "H_kv·d_h → d_c + d_R", ["cache a 512 latent", "+ a 64-dim RoPE key"], ["decoupled RoPE; retrain", "or convert (SVD)"]),
        ("CLA", "2024", "n-s", "L → L/2", ["adjacent layers", "share one KV"], ["layers lose private KV;", "retrain"]),
        ("YOCO", "2024", "n-s", "L → cache once", ["self-decoder builds ONE", "global KV for the rest"], ["new two-part decoder;", "retrain"]),
        ("SSM / linear", "2023", "n-c", "T → fixed state", ["s_t = f(s_t-1, x_t),", "no token-level KV"], ["lossy: weak exact", "token retrieval"]),
        ("Hybrid", "2024–25", "n-c", "KV in 1/8–1/4 layers", ["Jamba 1:7, Qwen3-Next", "3:1 recurrent:attention"], ["the ratio is a design", "choice; heavy retraining"]),
    ]
    cw, gap, ch = 228, 16, 172
    for k, (name, year, cls, attacks, idea, cost) in enumerate(cards):
        r, c = divmod(k, 4)
        x, y = 20 + c * (cw + gap), 20 + r * (ch + 56)
        p.rect(x, y, cw, ch, "n", rx=8)
        p.text(x + 12, y + 24, name, "hd")
        p.text(x + cw - 12, y + 22, year, "xs", "end")
        p.rect(x + 12, y + 34, cw - 24, 24, cls, rx=4)
        p.text(x + cw / 2, y + 50, attacks, "xs b", "middle")
        p.lines(x + 12, y + 80, idea, "xs")
        p.add(f'<path d="M{x + 12},{y + 112} L{x + cw - 12},{y + 112}" class="rule"/>')
        p.lines(x + 12, y + 132, cost, "xs f-x")
        if c < 3:
            p.wire([(x + cw + 1, y + ch / 2), (x + cw + gap - 1, y + ch / 2)], "i")
    p.wire([(20 + 3 * (cw + gap) + cw / 2, 20 + ch), (20 + 3 * (cw + gap) + cw / 2, 20 + ch + 28),
            (20 + cw / 2, 20 + ch + 28), (20 + cw / 2, 20 + ch + 54)], "i")
    for k, (cls, lab) in enumerate([("n-s", "share (heads or layers)"), ("n-m", "compress what is stored"),
                                    ("n-c", "replace the cache with a state")]):
        x = 20 + k * 300
        p.rect(x, 448, 14, 14, cls, rx=3)
        p.text(x + 22, 459, lab, "xs")
    return p


def plate_decide() -> Plate:
    p = Plate("decide", 710, "Decision tree for shrinking the KV cache, and what each level of change costs")
    cx = 190
    p.node(cx - 95, 14, 190, 34, "n", "long-context LLM", dot=False)
    steps = [
        ("KV too large?", "keep MHA / GQA", "n-s", "GQA / MQA  (H_kv)"),
        ("still too large?", "stop: GQA is enough", "n-m", "MLA  (H_kv·d_h)"),
        ("still too large?", "stop: MLA is enough", "n-m", "KV quant (S) · window (T)"),
        ("still too large?", "stop: design is enough", "n-s", "CLA / YOCO  (L)"),
        ("still too large?", "stop: design is enough", "n-c", "hybrid SSM / attention"),
    ]
    y = 48
    for q, stop, cls, act in steps:
        dy = y + 40
        p.wire([(cx, y), (cx, dy - 26)], "i")
        p.diamond(cx, dy, 82, 24, q)
        p.wire([(cx + 82, dy), (318, dy)], "i")
        p.text(cx + 96, dy - 6, "no", "xs")
        p.node(320, dy - 16, 180, 32, "n-p", stop, dot=False, tcls="xs")
        p.wire([(cx, dy + 24), (cx, dy + 54)], "i")
        p.text(cx + 8, dy + 44, "yes", "xs")
        p.node(cx - 105, dy + 56, 210, 34, cls, act, dot=False, tcls="sm b")
        y = dy + 90
    # retrofit levels
    p.text(560, 26, "What does adoption cost?", "hd")
    levels = [
        ("n-c", "LEVEL 1 · serving, no retraining", ["PagedAttention, FlashAttention, KV quantization,", "continuous batching, prefix caching"], 1),
        ("n-s", "LEVEL 2 · architecture, uptrain", ["MHA → GQA: mean-pool K, V per group, uptrain ~5%", "MHA → MLA: joint SVD + partial RoPE", "(Llama-2-7B: −92% KV, ~0.5% LongBench drop)"], 2),
        ("n-x", "LEVEL 3 · a new model", ["Transformer → hybrid attention / SSM,", "cross-layer sharing: new computation graph"], 3),
    ]
    y = 42
    for cls, title, rows, cost in levels:
        h = 40 + 15 * len(rows)
        p.rect(560, y, 420, h, cls, rx=7)
        p.text(574, y + 22, title, "sm b")
        for i in range(3):
            p.rect(900 + i * 24, y + 12, 18, 10, "n-x" if i < cost else "n", rx=2)
        p.lines(574, y + 42, rows, "xs")
        y += h + 14
    p.text(560, y + 8, "Start at level 1. Go to level 2 when KV does not fit;", "xs")
    p.text(560, y + 23, "level 3 only when you train a new model.", "xs")
    y += 60
    p.text(560, y, "Decode still slow? Compare arithmetic intensity", "sm b")
    p.text(560, y + 16, "AI = FLOPs ÷ bytes with the ridge (A100 ≈ 200 FLOP/B)", "xs")
    p.node(560, y + 32, 200, 78, "n-m", "", dot=False)
    p.lines(574, y + 52, ["memory-bound (AI < ridge)", "GQA/MQA, KV quant,", "paged KV, FlashAttention"], "xs")
    p.node(780, y + 32, 200, 78, "n-c", "", dot=False)
    p.lines(794, y + 52, ["compute-bound (AI > ridge)", "Tensor Cores, lower", "precision, fewer FLOPs"], "xs")
    p.text(560, y + 132, "GQA decode AI ≈ H_q / H_kv (4 for GQA-8): memory-bound.", "xs")
    p.text(560, y + 147, "MLA absorbed decode ≈ 242 FLOP/B: near the ridge.", "xs")
    return p


def plate_stack() -> Plate:
    p = Plate("stack", 420, "Where attention variants sit in the inference stack")
    p.text(20, 20, "architecture decides HOW MUCH", "xs b f-s")
    for i, (name, elems, _) in enumerate(VARIANTS[:2] + VARIANTS[3:]):
        p.node(20, 30 + i * 30, 170, 24, "n-s", f"{name} → {elems:,}", dot=False, tcls="xs b")
    p.node(230, 50, 150, 46, "n-m", "KV cache", "per token, per layer")
    p.node(420, 50, 150, 46, "n", "paged allocator", "block table · 16 tok")
    p.node(610, 50, 140, 46, "n-m", "GPU HBM", "scattered blocks")
    p.node(790, 50, 190, 46, "n-c", "FlashAttention", "FlashInfer · FlashMLA")
    p.node(405, 128, 180, 40, "n-s", "prefix sharing", "refcount · copy-on-write")
    p.wire([(190, 73), (228, 73)], "s")
    p.wire([(380, 73), (418, 73)], "m")
    p.wire([(570, 73), (608, 73)], "i")
    p.wire([(750, 73), (788, 73)], "m")
    p.wire([(495, 128), (495, 98)], "s", dash=True)
    p.wire([(885, 96), (885, 130)], "c")
    p.text(885, 146, "next token", "sm b", "middle")
    p.text(495, 188, "WHERE it lives", "xs b", "middle")
    p.text(885, 170, "HOW FAST it is read", "xs b", "middle")
    bands = [
        ("n-s", "1 · MODEL ARCHITECTURE", "GQA · MQA · MLA · CLA · YOCO · hybrid SSM", "step 5, variants"),
        ("n-m", "2 · CACHE DESIGN", "paged KV · prefix cache · KV quantization · offloading", "steps 6–8, 11"),
        ("n-c", "3 · KERNEL", "FlashAttention · FlashMLA · fused kernels · Tensor Cores", "step 10"),
        ("n", "4 · RUNTIME / SCHEDULING", "continuous batching · speculative decoding · P/D split · TP/EP", "steps 9, 12"),
        ("n-p", "5 · HARDWARE", "HBM capacity and bandwidth · Tensor Cores · NVLink · PCIe", "step 3"),
    ]
    for i, (cls, name, ex, where) in enumerate(bands):
        y = 206 + i * 42
        p.rect(20, y, 960, 36, cls, rx=5)
        p.text(34, y + 23, name, "sm b")
        p.text(290, y + 23, ex, "xs")
        p.text(966, y + 23, where, "xs", "end")
    return p


# ---- story plates: the whole site, one request from arrival to the state of the art ----------
# Distilled from explain, memory, vram_budget_a100, roofline, inference_problems, rope,
# paged_attention, kv_memory_management, serving_scheduler, serving_stack and sota.

A100_TFLOPS, A100_BW = 312.0, 1.555  # BF16 dense TFLOPS, TB/s
H100_TFLOPS, H100_BW = 989.0, 3.35


def plate_pipeline() -> Plate:
    p = Plate("pipeline", 330, "One request: prefill writes the prompt's K/V once, decode reads all of it every step")
    p.group(20, 30, 210, 74, "prompt tokens", lcls="xs b")
    for i in range(8):
        p.rect(32 + i * 24, 58, 20, 30, "n-p", rx=3)
    p.node(270, 34, 210, 66, "n-c", "PREFILL", "all prompt tokens at once")
    p.text(375, 118, "compute-bound · sets TTFT", "xs", "middle")
    p.node(560, 34, 210, 66, "n-s", "DECODE", "one new token per step")
    p.text(665, 118, "memory-bound · sets TPOT", "xs", "middle")
    p.wire([(230, 67), (268, 67)], "i")
    p.wire([(480, 67), (558, 67)], "i")
    for i, t in enumerate(["y1", "y2", "y3", "…"]):
        p.node(810 + i * 44, 50, 38, 34, "n-c", t, dot=False, tcls="xs b")
    p.wire([(770, 67), (808, 67)], "c")
    p.text(810, 40, "output tokens", "xs b")
    # the cache
    p.group(20, 180, 960, 72, "KV cache of this request: one K and one V per token, per layer, per KV head", "grp-m", "xs b")
    for i in range(30):
        cls = "n-m" if i < 8 else ("n-mm" if i < 14 else "n")
        p.rect(36 + i * 31, 206, 27, 30, cls, rx=3)
    p.text(36, 268, "written once by prefill", "xs f-m")
    p.text(36 + 8 * 31, 268, "appended by decode", "xs f-m")
    p.text(36 + 14 * 31, 268, "future tokens: final length unknown at arrival", "xs")
    p.wire([(330, 100), (330, 204)], "m")
    p.text(338, 150, "writes K, V once", "xs b f-m")
    p.wire([(620, 204), (620, 102)], "m")
    p.text(612, 150, "reads ALL K, V every step", "xs b f-m", "end")
    p.wire([(720, 102), (720, 160), (36 + 14 * 31 + 13, 160), (36 + 14 * 31 + 13, 204)], "s", dash=True)
    p.text(728, 156, "append k_t, v_t", "xs b f-s")
    p.text(20, 302, "Without a cache, step t recomputes K and V for all t tokens: O(T²) work per sequence.", "xs")
    p.text(20, 318, "With a cache, each step is O(T) reads. The price is memory, and every step re-reads all of it.", "xs b")
    return p


def plate_ledger() -> Plate:
    p = Plate("ledger", 290, "The memory ledger: from one head's key to a batch, for a GQA model at 100K tokens")
    L, hkv, dh, T, S = 32, 8, 64, 100_000, 2
    k = hkv * dh
    steps = [
        ("① K per token-layer", f"{hkv} × {dh}", f"{k:,} elems"),
        ("② + V", "K + V", f"{2 * k:,} elems"),
        ("③ × 2 bytes", "FP16 / BF16", binary(2 * k * S)),
        (f"④ × {L} layers", "per token", binary(2 * k * S * L)),
        ("⑤ × 100K tokens", "one sequence", f"{2 * k * S * L * T / 1e9:.2f} GB"),
        ("⑥ × batch 8", "eight sequences", f"{2 * k * S * L * T * 8 / 1e9:.1f} GB"),
    ]
    for i, (a, b, v) in enumerate(steps):
        x = 20 + i * 162
        p.rect(x, 16, 150, 84, "n-m" if i >= 3 else "n", rx=7)
        p.text(x + 12, 38, a, "xs b")
        p.text(x + 12, 56, b, "xs")
        p.text(x + 12, 86, v, "lg" if i >= 4 else "hd")
        if i < 5:
            p.wire([(x + 150, 58), (x + 161, 58)], "i")
    p.text(20, 128, f"Same model (L = {L}, H_q = 32, d_h = {dh}, FP16), different KV layout, 100K tokens, B = 1:", "xs b")
    rows = [("MHA (H_kv = 32)", 2 * 32 * dh * S * L, "n-mm"), ("GQA-8", 2 * 8 * dh * S * L, "n-m"),
            ("GQA-8 + FP8 KV", 2 * 8 * dh * 1 * L, "n-m"), ("MLA (d_c 256 + d_R 32)", (256 + 32) * S * L, "n-s"),
            ("MQA (H_kv = 1)", 2 * 1 * dh * S * L, "n-m")]
    full = rows[0][1]
    for i, (name, per_tok, cls) in enumerate(rows):
        y = 140 + i * 28
        p.text(20, y + 15, name, "xs b")
        p.rect(200, y + 2, max(480 * per_tok / full, 3), 18, cls, rx=3)
        p.text(200 + max(480 * per_tok / full, 3) + 8, y + 15,
               f"{binary(per_tok)}/token · {per_tok * T / 1e9:.2f} GB · {full / per_tok:.3g}× smaller" if i else
               f"{binary(per_tok)}/token · {per_tok * T / 1e9:.2f} GB · baseline", "xs")
    return p


def plate_vram() -> Plate:
    p = Plate("vram", 350, "Where an A100's 40 GB goes, and how context length divides the batch")
    kv = 2 * 32 * 8 * 64 * 2 * 100_000 / 1e9
    parts = [("weights 7B × 2 B", 14.0, "n"), ("KV 100K", kv, "n-m"), ("activations", 1.5, "n-p"),
             ("workspace", 1.0, "n-p"), ("runtime", 1.5, "n-p")]
    used = sum(v for _, v, _ in parts)
    parts.append((f"free {40 - used:.2f} GB", 40 - used, "n"))
    scale, x = 960 / 40, 20.0
    p.text(20, 18, "A100-40GB · 7B FP16 · GQA-8 (toy d_h = 64) · 100K-token context · B = 1", "xs b")
    for i, (name, v, cls) in enumerate(parts):
        w = v * scale
        extra = ' stroke-dasharray="5 4"' if i == len(parts) - 1 else ""
        p.rect(x, 30, w, 44, cls, rx=3, extra=extra)
        if w > 70:
            p.text(x + 8, 50, name, "xs b")
            p.text(x + 8, 66, f"{v:.2f} GB", "xs")
        x += w
    p.text(20 + (14 + kv) * scale, 92, "grey: activations 1.5 · CUDA workspace 1.0 · runtime/allocator 1.5 GB", "xs")
    p.text(20, 132, "Common mistake: 40 − 14 GB of weights = 26 GB of KV. Reserve ~4–6 GB for everything that is neither.", "xs f-x")
    p.text(20, 166, "B_max = floor( (VRAM − weights − overhead) ÷ KV per sequence )   with a 22 GB KV pool (40 − 14 − 4):", "sm b")
    per_tok = 2 * 32 * 8 * 64 * 2
    for i, T in enumerate((100_000, 8_192, 4_096)):
        y = 186 + i * 50
        per_seq = per_tok * T / 1e9
        bmax = math.floor(22 / per_seq)
        p.text(20, y + 16, f"{T:,} tokens", "sm b")
        p.text(20, y + 32, f"{per_seq:.3g} GB per sequence", "xs")
        for u in range(bmax):
            p.rect(200 + u * 8.6, y + 6, 6.6, 22, "n-m", rx=1.5)
        p.text(200 + bmax * 8.6 + 10, y + 22, f"{bmax} sequences", "sm b")
    p.text(20, 340, "MHA at 100K needs 26.2 GB per sequence: zero fit. Concurrency is capped by KV, not by weights.", "xs")
    return p


def plate_roofline() -> Plate:
    p = Plate("roofline", 370, "Roofline: attainable TFLOPS = min(peak, AI × bandwidth), on an A100 and an H100")
    x0, x1, y0, y1 = 70, 600, 24, 310
    lx = (math.log10(0.5), math.log10(3000))
    ly = (0, 3.2)

    def px(ai):
        return x0 + (math.log10(ai) - lx[0]) / (lx[1] - lx[0]) * (x1 - x0)

    def py(tf):
        return y1 - (math.log10(tf) - ly[0]) / (ly[1] - ly[0]) * (y1 - y0)

    p.add(f'<path d="M{x0},{y0} L{x0},{y1} L{x1},{y1}" class="w-i"/>')
    for ai in (1, 10, 100, 1000):
        p.add(f'<path d="M{px(ai):g},{y1} L{px(ai):g},{y1 + 5}" class="w-i"/>')
        p.text(px(ai), y1 + 18, f"{ai:,}", "xs", "middle")
    for tf in (1, 10, 100, 1000):
        p.text(x0 - 8, py(tf) + 4, f"{tf:,}", "xs", "end")
    p.text((x0 + x1) / 2, y1 + 36, "arithmetic intensity AI = FLOPs ÷ bytes (log)", "xs", "middle")
    p.add(f'<text x="{x0 - 44}" y="{(y0 + y1) / 2}" class="xs" text-anchor="middle" '
          f'transform="rotate(-90 {x0 - 44} {(y0 + y1) / 2})">attainable TFLOPS (log)</text>')
    for peak, bw, k, name in ((H100_TFLOPS, H100_BW, "i", "H100"), (A100_TFLOPS, A100_BW, "c", "A100")):
        ridge = peak / bw
        pts = [(0.5, 0.5 * bw), (ridge, peak), (3000, peak)]
        d = "M" + " L".join(f"{px(a):g},{py(t):g}" for a, t in pts)
        p.add(f'<path d="{d}" class="w-{k}{" dash" if k == "i" else ""}"/>')
        p.text(px(3000) - 4, py(peak) - 6, f"{name} {peak:g} TFLOPS · ridge ≈ {ridge:.0f}", "xs b", "end")
    pts = [("MHA decode attn", 1), ("GQA-8 decode attn", 4), ("MQA decode attn", 32),
           ("weight GEMV, B = 128", 128), ("prefill GEMM", 500)]
    for i, (name, ai) in enumerate(pts):
        tf = min(A100_TFLOPS, ai * A100_BW)
        cls = "n-m" if ai < A100_TFLOPS / A100_BW else "n-c"
        p.add(f'<circle cx="{px(ai):g}" cy="{py(tf):g}" r="5" class="{cls}"/>')
        dx, anchor = (8, "start") if i != 3 else (-8, "end")
        p.text(px(ai) + dx, py(tf) + ((-8 if i == 0 else 16) if i < 3 else (-10 if i == 3 else 22)), f"{name}: {tf:.3g}", "xs b", anchor)
    p.text(px(1.2), py(400), "← memory-bound", "xs b f-m")
    p.text(px(250), py(12), "compute-bound →", "xs b f-c")
    cards = [
        ("n-m", "Decode attention: AI = H_q / H_kv", ["MHA 1, GQA-8 4, MQA 32: all far", "below the ridge (~200). GQA's win", "is fewer bytes, not a new regime."]),
        ("n-c", "Weight GEMV: AI = batch size", ["batching reuses each weight read;", "B ≈ 200 reaches the A100 ridge.", "KV capacity caps B first."]),
        ("n", "Newer GPUs: the ridge rises", ["H100 adds FLOPs faster than", "bandwidth (ridge ≈ 295), so decode", "gets MORE memory-bound."]),
    ]
    y = 24
    for cls, title, rows in cards:
        h = 30 + 15 * len(rows)
        p.rect(640, y, 340, h, cls, rx=6)
        p.text(654, y + 20, title, "xs b")
        p.lines(654, y + 38, rows, "xs")
        y += h + 12
    p.text(640, y + 14, "Rule: AI < ridge → cut bytes.", "sm b")
    p.text(640, y + 32, "AI > ridge → cut or speed up FLOPs.", "sm b")
    return p


def plate_problems() -> Plate:
    p = Plate("problems", 350, "Four ways the KV cache hurts: capacity, bandwidth, allocation, quality")
    cols = [
        ("n-m", "CAPACITY", "can it fit?", ["KV grows with T and B", "64 KiB/token × 100K = 6.55 GB", "B = 4 → 26.2 GB: OOM"],
         "GQA/MLA, FP8 KV, admission", "concurrency, quality"),
        ("n-x", "BANDWIDTH", "can HBM feed the cores?", ["every token re-reads all K/V", "MHA 26.2 GB ÷ 1.555 TB/s", "= 16.8 ms per token"],
         "fewer bytes: GQA, FP8, window", "information removed"),
        ("n-c", "ALLOCATION", "where do the bytes live?", ["max_len slabs + holes:", "only 20.4–38.2% held tokens", "(vLLM paper, old systems)"],
         "paged blocks, prefix sharing", "block-table indirection"),
        ("n-s", "QUALITY", "can it still retrieve?", ["each byte saved can drop", "information; MQA: 32 heads", "read one K/V"],
         "GQA, MLA, careful quant", "retraining, kernels"),
    ]
    for c, (cls, name, q, why, fix, cost) in enumerate(cols):
        x = 20 + c * 245
        p.rect(x, 16, 225, 54, cls, rx=7)
        p.text(x + 14, 38, name, "hd")
        p.text(x + 14, 57, q, "xs")
        p.rect(x, 80, 225, 196, "n", rx=7)
        p.text(x + 14, 102, "WHY", "xs b f-i")
        p.lines(x + 14, 120, why, "xs")
        p.add(f'<path d="M{x + 14},{170} L{x + 211},{170}" class="rule"/>')
        p.text(x + 14, 192, "FIX", "xs b f-c")
        p.text(x + 14, 210, fix, "xs")
        p.add(f'<path d="M{x + 14},{226} L{x + 211},{226}" class="rule"/>')
        p.text(x + 14, 248, "COST", "xs b f-x")
        p.text(x + 14, 266, cost, "xs")
    p.rect(20, 292, 960, 44, "n-p", rx=7)
    p.text(500, 312, "The KV cache is capacity + bandwidth + latency + concurrency + scheduling + quality at once.", "sm b", "middle")
    p.text(500, 328, "Modern inference engineering controls how information moves through the memory hierarchy.", "xs", "middle")
    return p


def plate_rope() -> Plate:
    p = Plate("rope", 340, "RoPE: rotate each pair by m·θ_i; dot products then depend only on the offset")
    cx, cy, r = 150, 150, 92
    p.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" class="rule"/>')
    p.add(f'<path d="M{cx - r - 12},{cy} L{cx + r + 12},{cy} M{cx},{cy - r - 12} L{cx},{cy + r + 12}" class="rule"/>')
    a0, a1 = math.radians(18), math.radians(18 + 60)
    p.wire([(cx, cy), (cx + r * math.cos(a0), cy - r * math.sin(a0))], "i")
    p.wire([(cx, cy), (cx + r * math.cos(a1), cy - r * math.sin(a1))], "x")
    arc = f"M{cx + 46 * math.cos(a0):g},{cy - 46 * math.sin(a0):g} A46,46 0 0 0 {cx + 46 * math.cos(a1):g},{cy - 46 * math.sin(a1):g}"
    p.add(f'<path d="{arc}" class="w-x dash"/>')
    p.text(cx + 52, cy - 50, "m·θ_i", "xs b f-x")
    p.text(20, 22, "one pair (x_2i, x_2i+1) of q or k", "xs b")
    p.text(20, 268, "rotation keeps |x|; position lives in the angle", "xs")
    p.text(20, 284, "⟨R_m q, R_n k⟩ = ⟨q, R_(n−m) k⟩", "sm b")
    # spectrum
    x0, x1, y0, y1 = 340, 660, 30, 250
    p.add(f'<path d="M{x0},{y0} L{x0},{y1} L{x1},{y1}" class="w-i"/>')

    def py(lam):
        return y1 - (math.log10(lam) - 0) / 7 * (y1 - y0)

    for e in (1, 3, 5, 7):
        p.text(x0 - 6, py(10 ** e) + 4, f"1e{e}", "xs", "end")
    for base, k, name in ((10_000, "c", "base 10,000"), (500_000, "s", "base 500,000 (Llama-3)")):
        d = "M" + " L".join(f"{x0 + i / 63 * (x1 - x0):g},{py(2 * math.pi * base ** (2 * i / 128)):g}" for i in range(64))
        p.add(f'<path d="{d}" class="w-{k}"/>')
        lam = 2 * math.pi * base ** (126 / 128)
        p.text(x1 - 4, py(lam) - 6, f"{name}: up to {lam:,.0f}", "xs b", "end")
    p.wire([(x0, py(8192)), (x1, py(8192))], "x", arrow=False, dash=True)
    p.text(x0 + 6, py(8192) - 6, "8,192 = trained context", "xs f-x")
    p.text((x0 + x1) / 2, y1 + 18, "pair index i = 0 … 63 (d_h = 128)", "xs", "middle")
    p.text(x0, 22, "wavelength λ_i = 2π · base^(2i/d_h), tokens (log)", "xs b")
    p.text(x0, y1 + 40, "pairs above the red line never finished a turn", "xs")
    p.text(x0, y1 + 55, "in training: PI, NTK and YaRN rescale their θ_i.", "xs")
    # cache
    p.text(700, 22, "how RoPE meets the KV cache", "xs b")
    for i, lab in enumerate(["R_0 k_0", "R_1 k_1", "R_2 k_2", "…", "R_t-1 k"]):
        p.node(700, 34 + i * 34, 130, 28, "n-m", lab, dot=False, tcls="xs b")
    p.node(850, 34 + 4 * 34, 130, 28, "n-c", "R_t q_t (new)", dot=False, tcls="xs b")
    p.wire([(848, 48 + 4 * 34), (832, 48 + 4 * 34)], "c")
    p.lines(700, 222, ["keys are cached ALREADY rotated at", "their absolute position n; only the", "new query rotates. V is never rotated.",
                       "Anything that renumbers positions", "(windows, θ changes, MLA absorption)", "needs pre-RoPE K or a re-rotation."], "xs")
    return p


def plate_frag() -> Plate:
    p = Plate("frag", 270, "Contiguous max_len slabs strand memory; paged blocks place any request in any free block")
    cw, pitch = 27, 30
    contiguous = list("AAAAAAaaaaBBbbbb.....CCCCcccc...")
    paged = list("AEC.BA.EEC.A.CEA..B.ECA..A.E....")
    assert len(contiguous) == len(paged) == 32 and paged.count(".") == 14
    for row, (cells, title) in enumerate(((contiguous, "CONTIGUOUS · reserve max_len per request"),
                                          (paged, "PAGED · fixed blocks on demand"))):
        y = 30 + row * 104
        p.text(20, y - 8, title, "xs b")
        for i, c in enumerate(cells):
            x = 20 + i * pitch
            if c == ".":
                p.rect(x, y, cw, 34, "n", rx=3, extra=' stroke-dasharray="3 3"')
            elif c.islower():
                p.rect(x, y, cw, 34, "n-x", rx=3)
                p.text(x + cw / 2, y + 22, c.upper(), "xs", "middle")
            else:
                p.rect(x, y, cw, 34, "n-m", rx=3)
                p.text(x + cw / 2, y + 22, c, "xs b", "middle")
    p.text(20, 84, "E needs 6 cells. 8 are free, but the largest hole is 5: E is refused (external fragmentation).", "xs f-x")
    p.text(20, 99, "Crimson = reserved for a max_len the request never reaches (internal fragmentation): 12 of 32 cells hold tokens.", "xs")
    p.text(20, 188, "A 6, B 2, C 4, E 6 = 18 used, 14 free. Any free block works, so E fits.", "xs f-c")
    p.text(20, 203, "Waste is only each sequence's last partial block: under 4% in vLLM, vs 20.4–38.2% useful before.", "xs")
    for k, (cls, lab) in enumerate([("n-m", "token K/V"), ("n-x", "reserved, never used"), ("n", "free")]):
        x = 20 + k * 220
        p.rect(x, 236, 14, 14, cls, rx=3)
        p.text(x + 22, 247, lab, "xs")
    return p


def plate_blocks() -> Plate:
    p = Plate("blocks", 340, "Logical blocks → block table → scattered physical blocks, with block_size 4 for readability")
    seqs = [("A", 11, [2, 0, 3], 30), ("B", 6, [5, 7], 200)]
    for name, ntok, table, y0 in seqs:
        p.text(20, y0 - 8, f"sequence {name} · {ntok} tokens", "xs b")
        p.text(300, y0 - 8, f"block table {name}", "xs b")
        for j, phys in enumerate(table):
            y = y0 + j * 40
            for s in range(4):
                tok = j * 4 + s
                p.rect(20 + s * 40, y, 36, 30, "n-m" if tok < ntok else "n", rx=3,
                       extra="" if tok < ntok else ' stroke-dasharray="3 3"')
                if tok < ntok:
                    p.text(38 + s * 40, y + 20, str(tok), "xs", "middle")
            p.text(190, y + 20, f"L{j}", "xs b")
            p.node(300, y, 80, 30, "n-s", f"L{j} → P{phys}", dot=False, tcls="xs b")
            p.wire([(214, y + 15), (298, y + 15)], "i")
            bend = 396 + (phys % 4) * 18
            p.wire([(380, y + 15), (bend, y + 15), (bend, 36 + phys * 36), (478, 36 + phys * 36)], "s")
    owner = {2: "A:L0", 0: "A:L1", 3: "A:L2", 5: "B:L0", 7: "B:L1"}
    p.text(480, 18, "physical pool", "xs b")
    for i in range(8):
        y = 22 + i * 36
        o = owner.get(i)
        p.node(480, y, 130, 28, "n-m" if o else "n", f"P{i} {o or 'free'}", dot=False, tcls="xs b" if o else "xs")
    p.rect(650, 20, 330, 150, "n-p", rx=7)
    p.text(664, 42, "address translation (block_size 16)", "xs b")
    p.lines(664, 62, ["token position p = 37", "logical block j = p // 16 = 2", "offset o = p % 16 = 5",
                      "block_table[2] = 3", "physical slot = 3 · 16 + 5 = 53"], "xs", step=18)
    p.lines(650, 196, ["Each sequence sees one contiguous space;", "the table hides where the pieces live.",
                       "Only the LAST block can be partly empty:", "waste ≤ block_size − 1 tokens per sequence.",
                       "Growth appends a table entry: O(1),", "never an O(T) copy of the cache."], "xs", step=16)
    return p


def plate_share() -> Plate:
    p = Plate("share", 320, "Prefix sharing: hash-chained blocks, reference counts and copy-on-write")
    p.text(20, 18, "1,000 users share one 2,000-token system prompt (block 16 tokens, 2 MiB per block on Llama-3-8B)", "xs b")
    labels = ["b0", "b1", "b2", "…", "b124"]
    for i, b in enumerate(labels):
        x = 20 + i * 118
        p.node(x, 32, 104, 30, "n-s", b, dot=False, tcls="xs b")
        h = "h0 = H(∅, b0)" if i == 0 else ("…" if b == "…" else f"h{b[1:]} = H(h{int(b[1:]) - 1}, {b})")
        p.text(x + 52, 80, h, "xs", "middle")
        if i < 4:
            p.wire([(x + 104, 47), (x + 116, 47)], "s")
    blocks = 2000 // 16
    mib = blocks * 2
    p.lines(20, 112, [f"no sharing: 1,000 × {blocks} blocks × 2 MiB = {1000 * mib:,} MiB ≈ {1000 * mib / 1024:.0f} GiB of identical K/V",
                      f"shared:     {blocks} blocks × 2 MiB = {mib} MiB once, refcount = 1,000 (+ small private tails)",
                      "prefill:    users 2 … 1,000 skip the 2,000 shared tokens: TTFT covers only their own suffix"], "xs", step=17)
    p.text(20, 178, "One changed token early (say, a timestamp) breaks the chain for every later block.", "xs f-x")
    p.text(20, 214, "refcount of a shared block over time", "xs b")
    for i, (rc, ev) in enumerate([(3, "A, B, C share it"), (2, "A finishes"), (1, "B finishes"), (0, "C finishes")]):
        x = 20 + i * 132
        p.node(x, 226, 112, 40, "n-s" if rc else "n", f"refcount {rc}" if rc else "free (cached)", dot=False, tcls="xs b")
        p.text(x + 56, 284, ev, "xs", "middle")
        if i < 3:
            p.wire([(x + 112, 246), (x + 130, 246)], "i")
    rules = [("fork / share", "ref += 1"), ("write while ref > 1", "copy block, ref −= 1"),
             ("write while ref = 1", "in place"), ("sequence freed", "ref −= 1; at 0 → LRU pool")]
    p.text(570, 214, "copy-on-write rules", "xs b")
    for i, (a, b) in enumerate(rules):
        y = 226 + i * 22
        p.text(570, y + 12, a, "xs b")
        p.text(740, y + 12, b, "xs")
    return p


def plate_sched() -> Plate:
    p = Plate("sched", 370, "The scheduler: a request's life in blocks, and why continuous batching keeps slots full")
    steps = [("arrive", "n"), ("tokenize", "n"), ("prefix hit?", "d"), ("prefill", "n-c"), ("decode step", "n-s"),
             ("block full?", "d"), ("EOS?", "d"), ("free blocks", "n-m")]
    x = 20
    xs = []
    for name, kind in steps:
        w = 112 if kind == "d" else 98
        if kind == "d":
            p.diamond(x + w / 2, 60, w / 2, 26, name, "xs")
        else:
            p.node(x, 42, w, 36, kind, name, dot=False, tcls="xs b")
        xs.append((x, w))
        x += w + 18
    for (xa, wa), (xb, _) in itertools.pairwise(xs):
        p.wire([(xa + wa, 60), (xb - 2, 60)], "i")
    hx, hw = xs[2]
    p.text(hx + hw / 2, 104, "hit: reuse blocks", "xs f-s", "middle")
    p.text(hx + hw / 2, 118, "miss: allocate", "xs", "middle")
    bx, bw = xs[5]
    p.text(bx + bw / 2, 104, "yes: append 1 block", "xs f-m", "middle")
    dx, dw = xs[4]
    ex, ew = xs[6]
    p.wire([(ex + ew / 2, 84), (ex + ew / 2, 136), (dx + dw / 2, 136), (dx + dw / 2, 80)], "s", dash=True)
    p.text((dx + ex + ew) / 2, 150, "no: next iteration (every running request, one token each)", "xs", "middle")
    p.text(20, 196, "Static vs continuous batching · A = 20, B = 500, C = 50, D = 1,000 tokens (linear scale)", "xs b")
    lens = [("A", 20), ("B", 500), ("C", 50), ("D", 1000)]
    scale = 400 / 1000
    for i, (name, n) in enumerate(lens):
        y = 212 + i * 26
        p.text(20, y + 14, name, "xs b")
        p.rect(40, y, n * scale, 18, "n-s", rx=3)
        if n < 1000:
            p.rect(40 + n * scale, y, (1000 - n) * scale, 18, "n-x", rx=3, extra=' opacity="0.55"')
    useful = sum(n for _, n in lens) / 4000
    p.text(40, 334, f"static: slots idle until D ends → {useful:.0%} useful", "xs b f-x")
    fills = [[("A", 20), ("E", 300), ("G", 400), ("I", 280)], [("B", 500), ("F", 500)], [("C", 50), ("H", 600), ("J", 350)],
             [("D", 1000)]]
    for i, row in enumerate(fills):
        y = 212 + i * 26
        x = 540
        for name, n in row:
            p.rect(x, y, n * scale - 2, 18, "n-s", rx=3)
            if n * scale > 16:
                p.text(x + 6, y + 13, name, "xs b")
            x += n * scale
    p.text(540, 334, "continuous: a finished slot is refilled the next iteration", "xs b f-c")
    p.text(540, 350, "(E … J illustrative; needs per-iteration alloc/free = paged KV)", "xs")
    return p


def plate_flash() -> Plate:
    p = Plate("flash", 360, "FlashAttention: stream K/V tiles through SRAM with an online softmax; never write T×T to HBM")
    T = 100_000
    p.group(20, 20, 600, 96, "HBM · large and slow (A100: 40 GB, 1.555 TB/s)", lcls="xs b")
    for i, (name, cls) in enumerate([("Q  (T × d_h)", "n-c"), ("K  (T × d_h)", "n-m"), ("V  (T × d_h)", "n-m"), ("O  (T × d_h)", "n-c")]):
        p.node(34 + i * 146, 46, 132, 54, cls, name, dot=False, tcls="xs b")
    p.group(20, 160, 600, 96, "SRAM / registers · tiny and fast · the inner loop lives here", lcls="xs b")
    for i, (name, cls) in enumerate([("Q_i  B_r×d", "n-c"), ("K_j  B_c×d", "n-m"), ("V_j  B_c×d", "n-m"), ("S_ij  B_r×B_c", "n"), ("m, l  O_i", "n-s")]):
        p.node(34 + i * 116, 188, 104, 54, cls, name, dot=False, tcls="xs b")
    p.wire([(100, 100), (86, 186)], "c")
    p.text(100, 140, "load Q_i once", "xs")
    p.wire([(246, 100), (202, 186)], "m")
    p.wire([(392, 100), (318, 186)], "m")
    p.text(320, 140, "stream K_j, V_j", "xs", "middle")
    p.wire([(560, 186), (544, 100)], "c")
    p.text(540, 140, "write O_i once", "xs", "end")
    gb = T * T * 2 / 1e9
    p.lines(20, 284, [f"Standard attention writes S = QKᵀ (T × T) to HBM: at T = {T:,} that is {gb:.0f} GB per head per layer,",
                      "read back for the softmax and again for P·V. FlashAttention keeps S in SRAM: extra memory O(T), not O(T²).",
                      "The output is exact. It reduces HBM traffic but does not shrink the KV cache or decide where it lives."], "xs", step=16)
    l1 = math.exp(1 - 3) + math.exp(0)
    a = math.exp(3 - 5)
    l2 = a * l1 + 1
    direct = math.exp(1 - 5) + math.exp(3 - 5) + 1
    p.rect(650, 20, 330, 236, "n-p", rx=7)
    p.text(664, 42, "online softmax, one row", "xs b")
    p.lines(664, 62, ["m' = max(m, rowmax S_ij)", "a  = exp(m − m')", "l  = a·l + Σ exp(S_ij − m')", "O  = a·O + exp(S_ij − m')·V_j",
                      "end: O / l   (exact)"], "xs", step=17)
    p.text(664, 160, "worked: tile 1 scores [1, 3]", "xs b")
    p.lines(664, 178, [f"m = 3, l = e^-2 + 1 = {l1:.4f}", "tile 2 score [5] → m' = 5",
                       f"l = {a:.4f} · {l1:.4f} + 1 = {l2:.4f}", f"direct: e^-4 + e^-2 + 1 = {direct:.4f} ✓"], "xs", step=17)
    return p


def plate_frontier() -> Plate:
    p = Plate("frontier", 400, "What open models cache per token (all layers, BF16), derived from public configs")
    models = [  # name, design, (layers, kv heads, head dim) or ("mla", layers), cls
        ("Llama-3.1-405B", "GQA 128/8", (126, 8, 128), "n-m"),
        ("Llama-3-70B", "GQA 64/8", (80, 8, 128), "n-m"),
        ("Qwen2.5-72B", "GQA 64/8", (80, 8, 128), "n-m"),
        ("Llama-3-8B", "GQA 32/8", (32, 8, 128), "n-m"),
        ("Mistral-7B v0.1", "GQA + 4K window", (32, 8, 128), "n-c"),
        ("DeepSeek-V3 / R1", "MLA 512 + 64", ("mla", 61), "n-s"),
        ("Kimi K2", "MLA (V3 design)", ("mla", 61), "n-s"),
        ("Qwen2.5-7B", "GQA 28/4", (28, 4, 128), "n-m"),
        ("gpt-oss-120b", "GQA, 18 full layers", (18, 8, 64), "n-c"),
        ("Falcon-7B", "MQA 71/1", (32, 1, 64), "n-x"),
    ]
    p.text(20, 18, "model", "xs b")
    p.text(190, 18, "attention", "xs b")
    p.text(370, 18, "KV per token", "xs b")
    p.text(860, 18, "one 128K sequence", "xs b")
    biggest = 2 * 126 * 8 * 128 * 2
    for i, (name, design, cfg, cls) in enumerate(models):
        y = 30 + i * 32
        per_tok = (cfg[1] * (D_C + D_R) * 2) if cfg[0] == "mla" else 2 * cfg[0] * cfg[1] * cfg[2] * 2
        seq = per_tok * 131_072
        if "window" in design:
            seq = per_tok * 4096
        p.text(20, y + 17, name, "xs b")
        p.text(190, y + 17, design, "xs")
        w = max(380 * per_tok / biggest, 3)
        p.rect(370, y + 4, w, 20, cls, rx=3)
        p.text(370 + w + 8, y + 18, f"{per_tok / 1024:.3g} KiB", "xs b")
        size = f"{seq / 2**30:.3g} GiB" if seq >= 2**30 else f"{seq / 2**20:.3g} MiB"
        p.text(860, y + 17, size + (" (window cap)" if "window" in design else ""), "xs")
    p.text(20, 362, "DeepSeek-V3 (671B, MLA) caches less per token than Llama-3-8B (GQA): 68.6 vs 128 KiB. Architecture beats size.", "xs b")
    p.text(20, 380, "gpt-oss-120b: growing part only (18 full-attention layers); its window layers add a small fixed amount. "
                    "Verify a model card before quoting.", "xs")
    return p


STORY_PLATES = [plate_pipeline, plate_ledger, plate_vram, plate_roofline, plate_problems, plate_rope,
                plate_frag, plate_blocks, plate_share, plate_sched, plate_flash, plate_frontier]


PLATES = [plate_terms, plate_heads, plate_cells, plate_mla, plate_quality, plate_bandwidth,
          plate_layers, plate_state, plate_timeline, plate_decide, plate_stack,
          *STORY_PLATES]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for build in PLATES:
        plate = build()
        (OUT / f"{plate.name}.svg").write_text(plate.svg(), encoding="utf-8")
        print("wrote", plate.name)


if __name__ == "__main__":
    main()
