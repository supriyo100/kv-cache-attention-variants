"""Verify every arXiv paper cited on the site and generate docs/references.md.

Each entry is (arXiv id, a word that must appear in the real title, topic, pages that cite it).
The script fetches metadata from the arXiv API with the ``arxiv`` package, fails if a title does
not contain its expected word (a wrong id), and writes the bibliography from the fetched data,
so titles, authors and dates are never typed from memory.

    uv run --extra dev python tools/references.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import arxiv

ROOT = Path(__file__).resolve().parents[1]

PAPERS: list[tuple[str, str, str, str]] = [
    # foundations
    ("1706.03762", "Attention", "Foundations", "3"),
    ("2001.08361", "Scaling", "Foundations", "2"),
    ("2211.05102", "Inference", "Foundations", "4"),
    ("1805.02867", "Softmax", "Kernels", "10"),
    ("2104.09864", "RoFormer", "Architecture", "5"),
    # attention variants and architecture
    ("1911.02150", "Decoding", "Architecture", "3, 5"),
    ("2305.13245", "GQA", "Architecture", "3, 5"),
    ("2405.04434", "DeepSeek-V2", "Architecture", "3, 5"),
    ("2412.19437", "DeepSeek-V3", "Architecture", "5, SOTA"),
    ("2502.14837", "Latent Attention", "Architecture", "5"),
    ("2502.07864", "TransMLA", "Architecture", "5"),
    ("2310.06825", "Mistral", "Architecture", "11"),
    ("2503.19786", "Gemma", "Architecture", "11"),
    ("2403.19887", "Jamba", "Architecture", "11"),
    # memory management and sharing
    ("2309.06180", "PagedAttention", "Memory and sharing", "6, 7, 8, 9"),
    ("2405.04437", "vAttention", "Memory and sharing", "7"),
    ("2312.07104", "SGLang", "Memory and sharing", "8, SOTA"),
    # scheduling and serving
    ("2403.02310", "Sarathi", "Scheduling and serving", "9"),
    ("2401.09670", "DistServe", "Scheduling and serving", "1, 12"),
    ("2311.18677", "Splitwise", "Scheduling and serving", "12"),
    ("2407.00079", "Mooncake", "Scheduling and serving", "12, SOTA"),
    # kernels
    ("2205.14135", "FlashAttention", "Kernels", "10"),
    ("2307.08691", "FlashAttention-2", "Kernels", "10"),
    ("2407.08608", "FlashAttention-3", "Kernels", "10"),
    ("2501.01005", "FlashInfer", "Kernels", "7, 10"),
    # long context and KV compression
    ("2402.02750", "KIVI", "Long context", "11"),
    ("2309.17453", "Streaming", "Long context", "11"),
    ("2306.14048", "H$_2$O", "Long context", "11"),
    ("2404.14469", "SnapKV", "Long context", "11"),
    ("2504.19874", "TurboQuant", "Long context", "11, SOTA"),
    ("2310.01889", "Ring Attention", "Distributed", "12"),
    # quantization
    ("2210.17323", "GPTQ", "Quantization", "12"),
    ("2306.00978", "AWQ", "Quantization", "12"),
    ("2211.10438", "SmoothQuant", "Quantization", "12"),
    ("2209.05433", "FP8", "Quantization", "12"),
    # speculative decoding
    ("2211.17192", "Speculative", "Speculative decoding", "12"),
    ("2302.01318", "Speculative", "Speculative decoding", "12"),
    ("2602.06036", "DFlash", "Speculative decoding", "12, SOTA"),
    ("2607.05147", "DSpark", "Speculative decoding", "12, SOTA"),
    ("2401.10774", "Medusa", "Speculative decoding", "12"),
    ("2401.15077", "EAGLE", "Speculative decoding", "12, SOTA"),
    ("2406.16858", "EAGLE-2", "Speculative decoding", "12"),
    ("2503.01840", "EAGLE-3", "Speculative decoding", "12, SOTA"),
    # distributed
    ("1909.08053", "Megatron", "Distributed", "12"),
]

# Not on arXiv: cited in prose with venue only.
NON_ARXIV = [
    ("Yu et al.", "Orca: A Distributed Serving System for Transformer-Based Generative Models",
     "OSDI 2022", "https://www.usenix.org/conference/osdi22/presentation/yu", "Scheduling and serving", "9"),
    ("Little", "A Proof for the Queuing Formula: L = λW", "Operations Research 9(3), 1961",
     "https://doi.org/10.1287/opre.9.3.383", "Foundations", "1"),
    ("Williams, Waterman, Patterson", "Roofline: An Insightful Visual Performance Model for Multicore Architectures",
     "Communications of the ACM 52(4), 2009", "https://doi.org/10.1145/1498765.1498785", "Foundations", "2"),
]


def fetch() -> list[dict]:
    client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=5)
    ids = [p[0] for p in PAPERS]
    found: dict[str, arxiv.Result] = {}
    for i in range(0, len(ids), 20):
        for r in client.results(arxiv.Search(id_list=ids[i:i + 20], max_results=20)):
            found[r.get_short_id().split("v")[0]] = r
        time.sleep(3)
    out, bad = [], []
    for pid, word, topic, pages in PAPERS:
        r = found.get(pid)
        if r is None:
            bad.append(f"{pid}: not returned by arXiv")
            continue
        if word.lower() not in r.title.lower():
            bad.append(f"{pid}: title {r.title!r} lacks {word!r}")
        authors = [a.name.strip() for a in r.authors]
        out.append({
            "id": pid,
            "title": " ".join(r.title.split()),
            "authors": authors,
            "year": r.published.year,
            "url": f"https://arxiv.org/abs/{pid}",
            "topic": topic,
            "pages": pages,
        })
    if bad:
        print("Reference check FAILED:\n  " + "\n  ".join(bad), file=sys.stderr)
        sys.exit(1)
    return out


def short_authors(a: list[str]) -> str:
    return a[0] if len(a) == 1 else (f"{a[0]} and {a[1]}" if len(a) == 2 else f"{a[0]} et al.")


def render(refs: list[dict]) -> str:
    lines = [
        "---",
        "title: References",
        "description: Every paper cited on this site, with titles, authors and dates fetched from arXiv and checked automatically.",
        "---",
        "",
        "# References",
        "",
        ('<p class="lede">Every paper cited on this site. Titles, authors and dates for arXiv papers '
        "are fetched from the arXiv API by <code>tools/references.py</code>, which fails the build "
        "step if a cited ID does not match its paper. The page is generated; do not edit it by hand.</p>"),
        "",
    ]
    topics: dict[str, list[dict]] = {}
    for r in refs:
        topics.setdefault(r["topic"], []).append(r)
    for n in NON_ARXIV:
        topics.setdefault(n[4], []).append({"nonarxiv": n})
    order = ["Foundations", "Architecture", "Memory and sharing", "Scheduling and serving", "Kernels",
             "Long context", "Quantization", "Speculative decoding", "Distributed"]
    for t in order:
        lines += [f"## {t}", "", "| Paper | Authors | Year | Cited in step |", "|---|---|---|---|"]
        for r in sorted(topics.get(t, []), key=lambda x: x.get("year", 0) if "nonarxiv" not in x else 0):
            if "nonarxiv" in r:
                a, title, venue, url, _, pages = r["nonarxiv"]
                lines.append(f"| [{title}]({url}) ({venue}) | {a} | — | {pages} |")
            else:
                lines.append(f"| [{r['title']}]({r['url']}) | {short_authors(r['authors'])} | {r['year']} | {r['pages']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    refs = fetch()
    data = ROOT / "docs" / "assets" / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "references.json").write_text(json.dumps(refs, indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / "docs" / "references.md").write_text(render(refs), encoding="utf-8")
    print(f"Verified {len(refs)} arXiv papers; wrote docs/references.md")


if __name__ == "__main__":
    main()
