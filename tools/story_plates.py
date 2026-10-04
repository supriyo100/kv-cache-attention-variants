"""Render the "Attention variants compared" story plates to docs/assets/plates/*.svg.

    uv run python tools/story_plates.py

Each plate is a plain SVG drawn with semantic classes. hooks/embeds.py inlines it into the page
(`![title](../assets/plates/x.svg){ .plate }`), where the classes pick up the site's color tokens and
follow light/dark mode. The <style> inside each file carries fallback colors, so the files also
render on their own (GitHub, a browser tab). Every number is computed here, not typed in.
The content is distilled from docs/assets/excalidraw/comparison.excalidraw and
architecture_evolution.excalidraw.
"""

from __future__ import annotations

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

    def diamond(self, cx, cy, hw, hh, label):
        pts = f"{cx - hw:g},{cy:g} {cx:g},{cy - hh:g} {cx + hw:g},{cy:g} {cx:g},{cy + hh:g}"
        self.add(f'<polygon points="{pts}" class="n"/>')
        self.text(cx, cy + 4.5, label, "sm", "middle")

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


# @@STORY@@

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
