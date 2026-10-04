# Session 9 — Migrating to `torch.nn.functional.scaled_dot_product_attention`

**Recap:** since Notebook 0 we've hand-rolled `softmax(QK^T/sqrt(d))V`.
`F.scaled_dot_product_attention` (SDPA) fuses this into one op that
dispatches to whichever backend (math/flash/efficient/cudnn, Notebook 8) is
fastest on the current hardware -- with **identical output**.


## Why migrate if the math is identical

SDPA avoids materializing the full attention matrix in memory and avoids
Python-level overhead of separate matmul/softmax/matmul calls -- same
numbers, less memory, faster. "A migration that changes outputs is a bug,
not a valid solution" -- so the whole point of this notebook is proving
numerical equivalence, not proving speed (that's Notebook 8).



```python
import math
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

console = Console()


def manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
    """Hand-rolled reference: softmax(QK^T / sqrt(d_k)) V, exactly what SDPA fuses."""
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if is_causal:
        seq = q.shape[-2]
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


torch.manual_seed(0)
table = Table(title="manual softmax(QK^T/sqrt(d))V  vs  F.scaled_dot_product_attention")
table.add_column("shape (batch, heads, seq, head_dim)")
table.add_column("max abs diff", justify="right")
table.add_column("numerically identical (atol=1e-5)?", justify="center")
for shape in [(1, 4, 8, 16), (2, 8, 32, 64), (1, 2, 128, 32)]:
    q, k, v = (torch.randn(shape) for _ in range(3))
    manual_out = manual_attention(q, k, v, is_causal=True)
    sdpa_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    diff = (manual_out - sdpa_out).abs().max().item()
    identical = torch.allclose(manual_out, sdpa_out, atol=1e-5)
    table.add_row(str(shape), f"{diff:.2e}", "yes" if identical else "NO - bug")
console.print(table)

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-style: italic">             manual softmax(QK^T/sqrt(d))V  vs  F.scaled_dot_product_attention             </span>
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> shape (batch, heads, seq, head_dim) </span>┃<span style="font-weight: bold"> max abs diff </span>┃<span style="font-weight: bold"> numerically identical (atol=1e-5)? </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ (1, 4, 8, 16)                       │     2.38e-07 │                yes                 │
│ (2, 8, 32, 64)                      │     5.96e-07 │                yes                 │
│ (1, 2, 128, 32)                     │     3.58e-07 │                yes                 │
└─────────────────────────────────────┴──────────────┴────────────────────────────────────┘
</pre>



## Live migration exercise

Take the Session-3-style manual MHA math and replace it with SDPA, verifying
the migrated version produces the same output as the hand-rolled version.



```python
d_model, num_heads, seq_len = 32, 4, 6
head_dim = d_model // num_heads
q = torch.randn(1, num_heads, seq_len, head_dim)
k = torch.randn(1, num_heads, seq_len, head_dim)
v = torch.randn(1, num_heads, seq_len, head_dim)

before = manual_attention(q, k, v, is_causal=True)
after = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("before/after migration identical:", torch.allclose(before, after, atol=1e-5))

```

    before/after migration identical: True


```mermaid
flowchart LR
    subgraph Before
        A1[matmul QK^T] --> A2[scale by 1/sqrt(d)]
        A2 --> A3[apply causal mask]
        A3 --> A4[softmax]
        A4 --> A5[matmul with V]
    end
    subgraph After
        B1["F.scaled_dot_product_attention(q, k, v, is_causal=True)"]
    end
```


## Recap

SDPA is a drop-in, numerically-identical replacement for hand-rolled
attention math, and every attention variant module in this repo
(`mha.py`/`mqa.py`/`gqa.py`/`gqa.py`-based MQA/`mla.py`) already calls it
internally -- this notebook made that migration explicit and provable.

You are now ready to move to `10_pagedattention_vllm.ipynb`.

