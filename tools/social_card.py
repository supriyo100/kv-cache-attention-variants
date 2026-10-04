"""Render docs/assets/social-card.png (1200x630) for OpenGraph / LinkedIn previews.

    uv run python tools/social_card.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

PAPER, INK, INK2, RULE = "#F5F6F3", "#1C232B", "#4A5561", "#D5D9D3"
MEMORY, MEMORY_BG, OTHER = "#B5790A", "#F6E8C8", "#ECEEE9"
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "social-card.png"


def main() -> None:
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200)
    ax.set_ylim(630, 0)
    ax.axis("off")

    ax.text(70, 110, "Inference engineering", fontsize=46, fontweight="bold", color=INK, va="top")
    ax.text(70, 180, "from tokens to GPU systems", fontsize=46, fontweight="bold", color=INK, va="top")
    ax.text(72, 270, "KV cache · attention variants · paged memory · scheduling ·\n"
            "FlashAttention · long context · speculative decoding", fontsize=19, color=INK2,
            va="top", linespacing=1.5)
    ax.text(72, 600, "supriyo100.github.io/kv-cache-attention-variants", fontsize=17, color=INK2)

    # one request's contiguous logical blocks (top) -> scattered physical blocks (bottom)
    other = {0, 1, 4, 5, 9, 13, 14, 17, 20, 22}
    targets = [7, 18, 3, 23]
    px, py, w, h, g = 640, 470, 34, 28, 6
    for i in range(24):
        x, y = px + (i % 12) * (w + g), py + (i // 12) * (h + g)
        mine = i in targets
        face = MEMORY_BG if mine else (OTHER if i in other else "white")
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=MEMORY if mine else RULE,
                               lw=1.6 if mine else 1, zorder=2))
    ax.text(px - 24, py + h + g / 2, "GPU KV pool,\nphysical blocks", fontsize=13, color=INK2,
            ha="right", va="center")
    for j, p in enumerate(targets):
        lx, ly = px + j * 120, 385
        ax.add_patch(Rectangle((lx, ly), 100, 30, facecolor=MEMORY_BG, edgecolor=MEMORY, lw=1.6, zorder=2))
        ax.text(lx + 50, ly + 16, f"A{j}", ha="center", va="center", fontsize=13, color=INK, zorder=3)
        tx, ty = px + (p % 12) * (w + g) + w / 2, py + (p // 12) * (h + g)
        ax.add_patch(FancyArrowPatch((lx + 50, ly + 30), (tx, ty), arrowstyle="-", color=MEMORY,
                                     lw=1.4, zorder=1))
    ax.text(px - 24, 400, "one request,\nlogical blocks", fontsize=13, color=INK2, ha="right", va="center")
    fig.savefig(OUT, facecolor=PAPER)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
