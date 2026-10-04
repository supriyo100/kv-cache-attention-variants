# Implementation plan: Inference Engineering, from tokens to GPU systems

**Status:** Phases 0–14 and 16–18 done; Phase 15 partly done (see Milestones) · **Owner:** Supriyo Chakraborty ·
**Date:** 2026-10-05 · **Process standard:** `claude_repo/` (plan before scaffold, deterministic
gates block, scores are evidence)

This plan turns the `kv-cache-attention-variants` fork into an original, public engineering
notebook on LLM inference. The site explains each idea in the same order: the problem, why the
naive answer fails, the core idea, the math, the system design, an implementation, a measured
experiment, the trade-offs, and what the state of the art does today.

---

## 1. Current repository state (audit, 2026-10-05)

Audited with graphify (`graphify-out/graph.json`, built from commit `a3b40d4`, 1,077 nodes,
1,411 edges, 111 communities), plus a manual read of every file group below.

| Area | What exists | Origin | Size / quality |
|---|---|---|---|
| `docs/sessions/00–10` | 11 course pages (naive decode → PagedAttention) | **Upstream** (`sourangshupal/kv-cache-attention-variants`), rendered from its notebooks | 500–1,100 words each; correct, introductory |
| `src/kv_cache_variants/` | MHA/MQA/GQA/MLA modules, RoPE, SDPA probing, memory formula, 10 rich-terminal lessons | Upstream | Clean, small, **no tests** |
| `teaching_notebooks/` | 9 notebooks (00–07, 10) | Upstream | 08 and 09 missing |
| `docs/assets/diagrams/D01–D12` | Interactive HTML diagrams | Upstream | Good; generic styling |
| `docs/assets/excalidraw/` | 18 drawings + SVGs | **This fork (original)** | Strongest original asset |
| `docs/beyond/` | Variants compared, SOTA map, inference problems, serving | This fork | 150–300 words each: mostly captions for drawings |
| `docs/study/` | Deep-dive explanation (4.8k words), 63 interview questions (11.7k words) | This fork | Substantial, original |
| Site | MkDocs 1.6.1 + Material 9.7.7, MathJax, Mermaid, custom `hooks/embeds.py` | Mixed | Builds clean with `--strict` |
| CI | `.github/workflows/docs.yml`: uv → `mkdocs build --strict` → Pages | This fork | No tests, no link check |
| `claude_repo/` | AI-SDLC playbook (gitignored, local only) | Owner | Used as process reference |

**Reference repository.** `elizabetht/100-days-of-inference` has 26 daily notebooks that follow
Philip Kiely's *Inference Engineering* book, and the repository has **no license**, so all rights
are reserved. We use it only as a coverage checklist. Nothing is copied: no text, structure,
notebooks or figures.

**Owner's public profile (verified via GitHub API).** Supriyo Chakraborty, IIT Kharagpur, India;
public repos include seismic inversion (Orthoseisnet, UnetFFT, thesis), MILP optimization
(manufacturing, inventory), agentic AI. No profile README repo exists. No LinkedIn URL is public,
so it is a config value the owner must fill in (§17).

## 2. Existing architecture

```
mkdocs.yml ──► docs/ (Markdown) ──► hooks/embeds.py (marker → <figure>/<iframe>) ──► site/
                     │
                     ├── assets/excalidraw/*.svg  (exported by tools/excalidraw_export, Node)
                     ├── assets/diagrams/*.html   (self-contained, iframed)
                     └── assets/viewer/excalidraw.html (live pan/zoom)
src/kv_cache_variants/ ── imported by teaching_notebooks/ and lessons/ (not by the site)
```

## 3. Problems identified

1. **Identity.** The site presents itself as "Module 03" of someone else's course. The owner's
   original work (the drawings and study notes) sits under "Beyond the course".
2. **Attribution and licensing risk.** Neither upstream nor the reference repo has a license.
   The fork republishes upstream session text. See Risks and Open questions.
3. **Depth gap.** The course stops at a single-request view. Scheduling, the request lifecycle,
   fragmentation, prefix sharing, quantization, speculative decoding, long context, distributed
   inference and production serving are absent or are only a caption.
4. **No interactive tools.** No calculator or simulator; numbers sit in static drawings.
5. **No tests.** `src/` has no test suite; CI checks only that Markdown builds.
6. **A correctness gap upstream.** `memory_calc.kv_cache_bytes` suggests modelling MLA as
   `num_kv_heads=1, head_dim=d_c`. With the leading factor 2, that double-counts: MLA caches one
   latent `c_KV` (512) plus one shared RoPE key (64) per token per layer, not separate K and V.
   The new material computes MLA as `L·(d_c + d_R)·bytes`.
7. **SEO and sharing.** There's no `site_url`, canonical URL, OpenGraph or Twitter metadata,
   social image, `robots.txt` or author identity.
8. **Platform risk.** Material for MkDocs is in maintenance mode, and MkDocs 2.0 drops plugins
   and theme overrides. Pin `mkdocs<2`.

## 4. Proposed architecture

**Decision (ADR-001): keep MkDocs 1.6 + Material 9.7, pinned, and redesign through `overrides/`
and a custom CSS design system.** Alternatives considered:

| Option | For | Against | Verdict |
|---|---|---|---|
| Keep MkDocs + Material, custom theme layer | Search, math, Mermaid, dark mode, code copy, sitemap, deep links already work; the Pages pipeline already works; the embed hook keeps working | MkDocs 2.0 incompatibility (mitigated by pinning) | **Chosen** |
| Migrate to Astro/Starlight | Full design control, islands for widgets | Rewrite of hook, nav, search; two toolchains (Node + Python notebooks) | Revisit if Material is abandoned |
| Zensical (Material successor) | Same authoring model | Young; override API still moving | Re-evaluate in 2027 |

Interactive tools are **dependency-free vanilla JS** (one file each under
`docs/assets/javascripts/`), mounted by `data-widget` attributes, so pages stay static and no
server is needed.

Original code lives in a **new package, `src/inference_lab/`**, separate from upstream's
`kv_cache_variants`. It's tested with pytest and imported by the new notebooks. One rule
applies: every number on a page is either derived from a formula shown on that page or produced
by code in `inference_lab` that has a test.

```
src/inference_lab/     original simulations + reference implementations (tested)
tests/                 pytest for inference_lab
notebooks/             NEW original notebooks (separate from upstream teaching_notebooks/)
docs/
  index.md             home (custom template overrides/home.html)
  path/                the core track: 12 steps, one request end to end
  sota/                state-of-the-art techniques, mapped to their official repos
  compare/             comparison pages
  lab/                 notebooks index + interactive tools
  about.md, resources.md
  sessions/, beyond/, study/   kept; nav-labelled as upstream course / earlier notes
overrides/             Material theme overrides (home, meta/OG, footer)
```

## 5. Content architecture

### 5.1 The core track: "The path of one request"

The spine of the site follows one request from arrival to its last token, so every topic appears
at the moment it becomes the bottleneck. Each step uses the page template in §5.3.

| Step | Page | Question it answers | Layer |
|---|---|---|---|
| 1 | `path/01-the-request.md` | What actually happens between "send" and the first token? | System |
| 2 | `path/02-prefill-and-decode.md` | Why are prefill and decode different workloads? | GPU |
| 3 | `path/03-attention-cost.md` | What does attention cost in FLOPs and bytes? | Kernel |
| 4 | `path/04-kv-cache.md` | Why cache K and V, and how big does the cache get? | Model + memory |
| 5 | `path/05-attention-variants.md` | How do MHA/MQA/GQA/MLA shrink the cache? (incl. MLA absorption) | Model |
| 6 | `path/06-fragmentation.md` | Why does contiguous allocation waste memory? | Cache mgmt |
| 7 | `path/07-paged-kv.md` | How do block tables make scattered memory usable? | Cache mgmt |
| 8 | `path/08-prefix-sharing.md` | How do requests share KV safely? (refcount, COW, radix tree) | Sharing |
| 9 | `path/09-scheduling.md` | Which requests run each iteration, and what happens when blocks run out? | Scheduling |
| 10 | `path/10-flashattention.md` | Why is attention IO-bound, and how do tiling and online softmax fix it? | Kernel |
| 11 | `path/11-long-context.md` | What breaks first at 128K, and which fix fits which break? | All |
| 12 | `path/12-scaling-out.md` | Quantization, speculative decoding, parallelism, P/D disaggregation | Optimization + distributed |

### 5.2 SOTA section (maps to official repos)

`sota/index.md` (landscape, 2026), plus one page per family. Each page names the official repo
and the file or kernel that implements the idea, and links to a `notebooks/sota_*` notebook that
reimplements the core idea from scratch:

| Technique | Official repo (license) | Our notebook |
|---|---|---|
| Prefix caching (hash chain + radix tree) | vllm-project/vllm (93.2k ★), sgl-project/sglang (36.8k ★), nano-vllm (15.7k ★) | `sota_01_radix_prefix_cache` |
| MLA weight absorption | deepseek-ai/FlashMLA (13.0k ★, MIT) | `sota_02_mla_absorption` |
| Split-KV decoding, LSE merge, cascade attention | Dao-AILab/flash-attention (25.1k ★, BSD-3) | `sota_03_flash_decoding_lse_merge` |
| KV quantization: TurboQuant, KIVI | 0xSero/turboquant (1.8k ★, GPL-3.0, linked only); KIVI for comparison | `sota_04_kv_quant_kivi_turboquant` |
| KV eviction | mit-han-lab/streaming-llm (7.3k ★, MIT) | `sota_05_kv_eviction` |
| Speculative decoding: EAGLE-3, DFlash, DSpark | deepseek-ai/DeepSpec (7.2k ★, MIT) | `sota_06_speculative_decoding` |
| Chunked prefill | vllm-project/vllm; nano-vllm | `sota_07_chunked_prefill` |
| KV transfer, P/D disaggregation | LMCache/LMCache (12.0k ★, Apache-2.0) | `sota_08_pd_disaggregation` |

**Decision (ADR-002, owner request): the reference for each technique is its most-starred
open-source implementation**, found with `gh search repos --sort stars` plus the known reference
projects (keyword search alone misses vLLM). Stars were checked on 2026-10-05. Repositories with
non-standard licenses are linked only.

### 5.3 Page template (every core page)

`Problem → Naive solution → Why it fails → Core idea → Math → System design → Diagram →
Implementation → Experiment (measured, or labelled "theoretical") → Bottleneck and optimization →
Trade-offs → Failure modes → Production notes → SOTA → Interview questions → Further reading`

Every page ends with a provenance block that separates **Source** (papers and projects),
**Derivation** (math shown here), **Implementation** (what `inference_lab` implements),
**Experiment** (what was measured, on which hardware) and **Interpretation** (the author's view).

### 5.4 Comparison pages (`compare/`)

MHA vs MQA vs GQA vs MLA · contiguous vs paged KV · internal vs external fragmentation · naive vs
Flash attention · prefill vs decode · static vs continuous batching · FP16/BF16/FP8/INT8/INT4 ·
compute-bound vs memory-bound · TP vs PP vs CP vs EP · KV quantization vs eviction vs compression
· prefix caching on vs off. Each page has a matrix: problem solved, memory, latency, throughput,
complexity, quality risk, when to choose.

## 6. Technical architecture

- **Build:** `uv sync --extra docs && uv run mkdocs build --strict`. Versions are pinned
  (`mkdocs>=1.6,<2`, `mkdocs-material>=9.7,<10`).
- **Math:** pymdownx.arithmatex (generic) + MathJax 3, with a config script that re-typesets on
  Material's instant navigation.
- **Diagrams:** inline SVG in Markdown for new figures (themeable through CSS variables, works in
  dark mode, no export step), Mermaid for flows, existing Excalidraw and D01–D12 kept.
- **Widgets:** `docs/assets/javascripts/widgets/*.js`: KV calculator, block-allocator simulator,
  prefix-cache simulator, continuous-batching simulator, roofline explorer. Each is under 300
  lines, needs no dependencies, works by keyboard and respects `prefers-reduced-motion`.
- **Code:** `src/inference_lab` in **PyTorch** (ADR-003, owner request), in the upstream course's
  `(batch, heads, seq, head_dim)` conventions. MLA absorption runs on the course's own
  `MultiHeadLatentAttention`. Scheduling and memory modules are plain Python. It's pure Python and
  runs on CPU everywhere. GPU-only experiments are written but labelled *not run here*.

## 7. Design architecture (frontend-design pass)

**Subject:** a lab notebook for GPU inference systems. **Audience:** ML and systems engineers who
read math. **Primary job:** make the mechanism visible: where every byte lives and moves.

**Signature element (where the boldness goes): the memory map.** The home hero is a live block
pool: one request's contiguous logical blocks are drawn landing in scattered physical KV blocks,
once, on load. The same block-strip motif becomes the step marker in the core track (a step's
progress drawn as filled blocks). Everything else stays quiet.

**Color is meaning, not decoration.** One semantic encoding is used in every figure, widget and
table:

| Token | Light | Dark | Means |
|---|---|---|---|
| `--ie-paper` | `#F5F6F3` | `#14181D` | page |
| `--ie-ink` | `#1C232B` | `#E4E7EA` | text |
| `--ie-compute` | `#0E7C72` | `#3CC4B5` | FLOPs, tensor cores, prefill |
| `--ie-memory` | `#B5790A` | `#E8B34A` | HBM bytes, KV blocks, decode |
| `--ie-waste` | `#B42B3E` | `#F06F82` | fragmentation, padding, recompute |
| `--ie-shared` | `#5B47C9` | `#9C8CFF` | shared or prefix blocks, refcount > 1 |
| `--ie-link` | `#1F56B8` | `#7FA8F5` | links and focus |

**Type:** *Literata* for body text (a screen-tuned serif that sits well next to MathJax's serif
math; line height 1.65; measure about 72ch), *Archivo* (semi-condensed width) for headings and
UI, *IBM Plex Mono* for code. Sentence case everywhere; no all-caps eyebrows.

**Layout:** Material's three-column documentation layout for reading pages. Home is a custom
template:

```
┌──────────────────────────────────────────────────────────────┐
│ Inference engineering, from tokens to GPU systems            │
│ one-paragraph thesis                    [ live memory map  ] │
├──────────────────────────────────────────────────────────────┤
│ The path of one request   1 ▸ 2 ▸ … ▸ 12  (block-strip steps)│
├──────────────────────────────────────────────────────────────┤
│ The six layers (model → cache → sharing → sched → kernel → GPU)│
├───────────────────────────┬──────────────────────────────────┤
│ Tools (calculators/sims)  │ SOTA map + notebooks             │
└───────────────────────────┴──────────────────────────────────┘
```

**Review against defaults (changed before building):** a dark page with an acid accent was
rejected as a stock "GPU" look. Identical rounded cards for the 12 steps were replaced by a
numbered rail; numbering is justified because the steps are a real sequence. A gradient hero
headline was dropped; the motion budget goes to the memory map alone.

## 8. Notebook plan (`notebooks/`, original, separate from upstream `teaching_notebooks/`)

Each notebook has these sections: objective, theory, equations, diagram, implementation (imports
`inference_lab`), experiment, results (executed outputs committed), interpretation, limitations,
conclusion.

| # | Notebook | Status |
|---|---|---|
| 01 | `01_token_generation_basics` | planned |
| 02 | `02_attention_memory_math` | planned |
| 03 | `03_mha_mqa_gqa_comparison` | planned |
| 04 | `04_mla_latent_attention_explained` | → covered by `sota_02` |
| 05 | `05_kv_cache_memory_calculator` | **built** |
| 06 | `06_kv_cache_growth_simulation` | planned |
| 07 | `07_external_internal_fragmentation` | **built** |
| 08 | `08_paged_kv_allocator_simulation` | **built** |
| 09 | `09_prefix_cache_block_sharing` | → covered by `sota_01` |
| 10 | `10_reference_counting_simulation` | → covered by `sota_01` / `08` |
| 11 | `11_continuous_batching_simulation` | **built** |
| 12 | `12_prefill_vs_decode` | **built** |
| 13 | `13_flash_attention_io_model` | **built** |
| 14 | `14_kv_quantization_experiment` | → covered by `sota_04` (KIVI + TurboQuant) |
| 15 | `15_long_context_memory_analysis` | planned |
| 16 | `16_inference_benchmarking` | planned (needs GPU; CPU harness first) |
| 17 | `17_gpu_memory_bandwidth_experiment` | planned (needs GPU) |
| 18 | `18_speculative_decoding_analysis` | → covered by `sota_06` |
| S01–S08 | SOTA notebooks (§5.2) | **built** |

Notebooks are generated by `tools/build_notebooks.py` and **committed without outputs** (owner
request: the local machine is not used to run them). Every code cell is syntax-checked. The
same computations are covered by the unit tests. `--execute` runs them on a capable machine.

## 9. Diagram plan

New figures are inline SVG drawn with the semantic palette. Each answers one question:

| Figure | Question | Page |
|---|---|---|
| Request timeline | Where do TTFT, ITL and E2E come from? | 01 |
| Roofline with prefill and decode points | Why is decode bandwidth-bound? | 02 |
| Attention tensor shapes | Which tensor is T×T? | 03 |
| KV growth staircase | How does memory grow per token? | 04 |
| MLA dataflow with absorption | Where does decompression happen? | 05 |
| External vs internal fragmentation strips | Which waste does paging remove? | 06 |
| Logical → block table → physical | Why can scattered memory serve one sequence? | 07 |
| Shared prefix tree with refcounts | Who owns a shared block? | 08 |
| Iteration-level batching timeline | When are requests admitted? | 09 |
| HBM ↔ SRAM tiling | What does FlashAttention avoid writing? | 10 |
| Long-context problem → solution matrix | Which fix for which bottleneck? | 11 |
| TP/PP/CP/EP split + P/D disaggregation | What is split, and what is communicated? | 12 |

## 10. Comparison pages: see §5.4 (phase 13)

## 11. SEO plan

`site_url`, canonical links (Material emits them once `site_url` is set), descriptive
`description` front matter per page, OpenGraph + Twitter card tags in `overrides/main.html`, a
1200×630 social image, `robots.txt`, automatic `sitemap.xml`, and JSON-LD `TechArticle` + `Person`
on pages. Titles describe content (for example "KV cache memory: the equation, the growth, the
calculator"); no keyword stuffing.

## 12. GitHub Pages deployment

Pages source: **GitHub Actions** (Settings → Pages → Source: GitHub Actions). URL:
`https://supriyo100.github.io/kv-cache-attention-variants/`.

## 13. GitHub Actions

`docs.yml` (existing, extended) runs these jobs: `test` (ruff + pytest on `inference_lab`) →
`build` (`mkdocs build --strict`, internal link check through strict mode + a validation script
for anchors and widget mounts) → `deploy`. It is path-filtered to add `src/inference_lab/**`,
`tests/**`, `notebooks/**` and `overrides/**`. The action versions in the working tree were bumped
by the owner (checkout v7, setup-uv v10.2.0, setup-python v7, upload-pages-artifact v5,
deploy-pages v5).

## 14. Testing strategy

| Gate | Tool | Blocks |
|---|---|---|
| Lint | `ruff check src tests` | yes |
| Unit | 37 `pytest` tests: allocator invariants, refcount/COW, hash-cache LRU and token verification, tiled and split-KV attention vs `F.scaled_dot_product_attention`, MLA absorption vs the course module's `forward()`, Lloyd-Max codebook vs known values, rotation quantization, spec-decoding output distribution, confidence scheduling | yes |
| Citations | `tools/references.py` checks all 44 arXiv IDs against their titles | before publishing |
| Site | `mkdocs build --strict` (broken internal links, missing nav files) | yes |
| Notebooks | static syntax check of every code cell; `--execute` optional | no (not run here) |
| Visual | Browser check of home, one path page and each widget in light, dark and mobile | manual, before release |

## 15. Milestones

| Phase | Scope | State |
|---|---|---|
| 0 | Audit + gap analysis (this file) | done |
| 1 | IA and content model | done |
| 2 | Design system (`extra.css` tokens, fonts, figure styles) | done |
| 3 | Site shell: overrides, home, nav, SEO meta, about, resources | done |
| 4–12 | Core track pages 01–12 | done (first full draft) |
| 6–9 | Widgets: KV calculator, allocator, prefix cache, batching | done |
| 13 | Comparison pages | done (11 matrices in `compare/index.md`) |
| 14 | SOTA section + 8 SOTA notebooks | done |
| 15 | Core notebooks 01–18 | 6 written + 8 SOTA; 7 planned (table §8); none executed |
| 16 | SEO, GitHub profile README draft, LinkedIn | OG/Twitter/JSON-LD, social card, robots.txt, sitemap done; profile README drafted in `meta/github-profile-README.md`; LinkedIn URL pending from owner |
| 17 | Pages deployment | workflow (test → strict build → deploy) ready; owner pushes and sets Pages source to GitHub Actions |
| 18 | QA review | done: tests, lint, strict build, headless Edge QA (8 pages × desktop/mobile × light/dark: no overflow, 0 MathJax errors, all widgets mounted, widget numbers equal Python) |

## 16. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Upstream has no license, but the fork republishes its session pages | Copyright exposure; looks derivative | Ask upstream for a license, or remove `sessions/` from the deployed nav and link upstream instead (Open question 1) |
| Material/MkDocs 2.0 break | Site stops building on upgrade | Versions pinned; ADR-001 records the exit path |
| Hardware-dependent claims | Fabricated-looking numbers | Every figure is tagged *measured (hardware)*, *theoretical* or *expected trend*; GPU notebooks are not executed here |
| Fast-moving SOTA | Pages go stale | Each SOTA page carries an "as of" date and links to the repo file, not a copied snippet |
| Scope (30+ notebooks) | Unfinished sections | Core track first; notebooks only where they add a measurement |

## 17. Open questions (for the owner)

1. **Upstream licensing:** keep the `sessions/` pages deployed (and ask `sourangshupal` to add a
   license), or deploy only original content and link upstream?
2. **LinkedIn URL:** set `extra.social` in `mkdocs.yml` (placeholder until provided).
3. **Repository name:** the site now covers far more than KV variants. Rename the repo to
   `inference-engineering` (the URL changes; GitHub redirects the repo but **not** Pages)?
4. **Publications:** list the seismic-inversion work (Orthoseisnet) on About only if it is
   published; give the citation.
5. **GPU access:** is there a GPU for notebooks 16–17 and real TTFT/TPOT measurements?
