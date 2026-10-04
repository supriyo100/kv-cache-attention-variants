/* Interactive widgets for the Inference Engineering site. No dependencies.
   Mount points: <div data-widget="NAME"></div>. Formulas mirror src/inference_lab
   (kv_math.py, paged.py, prefix.py, specdec.py); keep them in sync. */
(function () {
  "use strict";

  const REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const SVGNS = "http://www.w3.org/2000/svg";

  // ---------------------------------------------------------------- helpers
  function el(tag, attrs, children) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "text") n.textContent = v;
      else if (k === "html") n.innerHTML = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
    (children || []).forEach((c) => n.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return n;
  }
  function svg(tag, attrs) {
    const n = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
    return n;
  }
  function bytes(n) {
    const u = ["B", "KiB", "MiB", "GiB", "TiB"];
    let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return (n >= 100 ? n.toFixed(0) : n >= 10 ? n.toFixed(1) : n.toFixed(2)) + " " + u[i];
  }
  function fmt(n, d) { return Number(n).toLocaleString(undefined, { maximumFractionDigits: d === undefined ? 0 : d }); }
  function field(label, input) { return el("label", {}, [label, input]); }
  function num(value, min, max, step) {
    return el("input", { type: "number", value: value, min: min, max: max, step: step || 1, inputmode: "numeric" });
  }
  function select(options, value) {
    const s = el("select");
    options.forEach(([v, t]) => { const o = el("option", { value: v, text: t }); if (v === value) o.selected = true; s.appendChild(o); });
    return s;
  }
  function frame(title, intro) {
    const w = el("div", { class: "ie-widget" });
    w.appendChild(el("h4", { text: title }));
    if (intro) w.appendChild(el("p", { text: intro, style: "margin:0 0 .7rem;color:var(--ie-ink-2)" }));
    return w;
  }
  function stat(label) { const b = el("b", { text: "–" }); return [el("div", {}, [b, label]), b]; }

  // ---------------------------------------------------------------- data (kv_math.py)
  const DTYPE = { fp32: 4, bf16: 2, fp16: 2, fp8: 1, int8: 1, int4: 0.5 };
  const MODELS = {
    "Llama-3-8B": { L: 32, Hq: 32, Hkv: 8, dh: 128, P: 8.0e9, A: 8.0e9, att: "gqa" },
    "Llama-3-70B": { L: 80, Hq: 64, Hkv: 8, dh: 128, P: 70.6e9, A: 70.6e9, att: "gqa" },
    "Llama-2-7B": { L: 32, Hq: 32, Hkv: 32, dh: 128, P: 6.7e9, A: 6.7e9, att: "mha" },
    "DeepSeek-V3": { L: 61, Hq: 128, Hkv: 128, dh: 128, P: 671e9, A: 37e9, att: "mla", dc: 512, dr: 64 },
  };
  const GPUS = {
    "H100 SXM": { hbm: 80, bw: 3.35, tf: 989 },
    "H200 SXM": { hbm: 141, bw: 4.8, tf: 989 },
    "A100 80GB SXM": { hbm: 80, bw: 2.039, tf: 312 },
    "B200": { hbm: 180, bw: 8.0, tf: 2250 },
  };
  function kvPerToken(c, dtype) {
    const s = DTYPE[dtype];
    if (c.att === "mla") return c.L * (c.dc + c.dr) * s;
    const h = c.att === "mha" ? c.Hq : c.att === "mqa" ? 1 : c.Hkv;
    return c.L * 2 * h * c.dh * s;
  }

  // ---------------------------------------------------------------- KV calculator
  function kvcalc(root) {
    const w = frame("KV cache calculator",
      "Bytes per token = L × 2 × H_kv × d_h × s (MLA: L × (d_c + d_R) × s). Pick a preset or edit any field.");
    const preset = select(Object.keys(MODELS).map((k) => [k, k]).concat([["custom", "Custom"]]), "Llama-3-8B");
    const att = select([["mha", "MHA"], ["gqa", "GQA"], ["mqa", "MQA"], ["mla", "MLA"]], "gqa");
    const L = num(32, 1, 256), Hq = num(32, 1, 256), Hkv = num(8, 1, 256), dh = num(128, 16, 512, 8);
    const dc = num(512, 64, 4096, 64), dr = num(64, 0, 256, 16);
    const ctx = num(32768, 1, 2097152, 1024), batch = num(8, 1, 4096);
    const dtype = select([["bf16", "BF16 / FP16"], ["fp8", "FP8"], ["int8", "INT8"], ["int4", "INT4"], ["fp32", "FP32"]], "bf16");
    const wdtype = select([["bf16", "BF16"], ["fp8", "FP8"], ["int4", "INT4"]], "bf16");
    const gpu = select(Object.keys(GPUS).map((k) => [k, k]), "H100 SXM");
    const ngpu = num(1, 1, 64);
    const P = num(8, 0.1, 2000, 0.1), A = num(8, 0.1, 2000, 0.1);
    const mlaRow = el("div", { class: "row" }, [field("Latent width d_c", dc), field("RoPE key width d_R", dr)]);
    w.appendChild(el("div", { class: "row" }, [field("Model", preset), field("Attention", att), field("Layers L", L),
      field("Query heads H_q", Hq), field("KV heads H_kv", Hkv), field("Head dim d_h", dh)]));
    w.appendChild(mlaRow);
    w.appendChild(el("div", { class: "row", style: "margin-top:.6rem" }, [field("Context tokens T", ctx), field("Batch (sequences)", batch),
      field("KV dtype", dtype), field("Weight dtype", wdtype), field("Total params (B)", P), field("Active params (B)", A),
      field("GPU", gpu), field("GPUs", ngpu)]));
    const [s1, o1] = stat("KV per token"), [s2, o2] = stat("KV per sequence"), [s3, o3] = stat("KV for the batch");
    const [s4, o4] = stat("Weights"), [s5, o5] = stat("Share of HBM used"), [s6, o6] = stat("Max sequences at T");
    const [s7, o7] = stat("Decode step ≥ (bandwidth bound)"), [s8, o8] = stat("Decode tokens/s ≤");
    w.appendChild(el("div", { class: "out" }, [s1, s2, s3, s4, s5, s6, s7, s8]));
    const bar = el("div", { class: "bar", role: "img" });
    w.appendChild(bar);
    w.appendChild(el("div", { class: "legend", html:
      '<span><i style="background:var(--ie-compute)"></i>weights</span><span><i style="background:var(--ie-memory)"></i>KV cache</span>' +
      '<span><i style="background:var(--ie-waste)"></i>over capacity</span><span><i></i>free</span>' }));
    const note = el("p", { class: "log" });
    w.appendChild(note);

    function load(name) {
      const m = MODELS[name]; if (!m) return;
      att.value = m.att; L.value = m.L; Hq.value = m.Hq; Hkv.value = m.Hkv; dh.value = m.dh;
      P.value = m.P / 1e9; A.value = m.A / 1e9; if (m.dc) { dc.value = m.dc; dr.value = m.dr; }
    }
    function update() {
      const c = { L: +L.value, Hq: +Hq.value, Hkv: +Hkv.value, dh: +dh.value, att: att.value, dc: +dc.value, dr: +dr.value };
      mlaRow.style.display = c.att === "mla" ? "" : "none";
      const g = GPUS[gpu.value], n = +ngpu.value;
      const per = kvPerToken(c, dtype.value);
      const seq = per * +ctx.value, tot = seq * +batch.value;
      const wts = +P.value * 1e9 * DTYPE[wdtype.value];
      const hbm = g.hbm * 1e9 * n;
      const usable = hbm * 0.9 - wts;
      o1.textContent = bytes(per); o2.textContent = bytes(seq); o3.textContent = bytes(tot); o4.textContent = bytes(wts);
      o5.textContent = fmt(100 * (wts + tot) / hbm, 1) + " %";
      o6.textContent = usable > 0 ? fmt(Math.floor(usable / seq)) : "0 (weights do not fit)";
      // Decode step reads active weights once + every sequence's KV once (bandwidth-bound regime)
      const stepBytes = +A.value * 1e9 * DTYPE[wdtype.value] + tot;
      const t = stepBytes / (g.bw * 1e12 * n);
      o7.textContent = fmt(t * 1e3, 2) + " ms"; o8.textContent = fmt(+batch.value / t);
      const fw = Math.min(100, 100 * wts / hbm), fk = Math.min(100 - fw, 100 * tot / hbm);
      const over = (wts + tot) > hbm * 0.9;
      bar.innerHTML = "";
      bar.appendChild(el("span", { style: `width:${fw}%;background:var(--ie-compute)` }));
      bar.appendChild(el("span", { style: `width:${fk}%;background:var(${over ? "--ie-waste" : "--ie-memory"})` }));
      bar.setAttribute("aria-label", `weights ${fw.toFixed(0)}%, KV ${fk.toFixed(0)}% of HBM`);
      note.textContent = over
        ? "Does not fit: weights + KV exceed 90% of HBM (10% is held back for activations and workspace). Reduce batch or context, quantize KV, or add GPUs."
        : `Assumes ${n} × ${gpu.value}, 10% of HBM reserved for activations. The decode bound counts only HBM reads of weights and KV; real kernels add overhead.`;
    }
    preset.addEventListener("change", () => { load(preset.value); update(); });
    w.querySelectorAll("input,select").forEach((i) => { if (i !== preset) i.addEventListener("input", () => { preset.value = "custom"; update(); }); });
    load("Llama-3-8B"); update();
    root.appendChild(w);
  }

  // ---------------------------------------------------------------- Paged allocator (paged.py)
  function allocator(root) {
    const w = frame("Paged KV allocator",
      "64 physical blocks. Admit requests, run decode steps, finish requests, and watch block tables grow. The filled part of a block is used token slots; the empty part of a last block is internal fragmentation.");
    const bsSel = select([["4", "4 tokens"], ["8", "8 tokens"], ["16", "16 tokens"]], "8");
    const controls = el("div", { class: "controls" });
    const grid = el("div", { class: "grid", role: "img", "aria-label": "Physical KV block pool" });
    const log = el("p", { class: "log", "aria-live": "polite" });
    const tables = el("table", { class: "tbl" });
    const [s1, o1] = stat("Blocks in use"), [s2, o2] = stat("Live tokens"), [s3, o3] = stat("Internal waste (slots)"), [s4, o4] = stat("Slot utilization");
    const NB = 64;
    const hues = ["--ie-memory", "--ie-compute", "--ie-shared", "--ie-link"];
    let B, free, seqs, nextId;
    function reset() {
      B = +bsSel.value; free = []; for (let i = NB - 1; i >= 0; i--) free.push(i);
      // shuffle so the pool looks like a long-running server: free blocks are scattered
      for (let i = free.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [free[i], free[j]] = [free[j], free[i]]; }
      seqs = []; nextId = 0; log.textContent = "Pool empty. Admit a request."; draw();
    }
    function admit() {
      const prompt = 3 + Math.floor(Math.random() * (B * 3));
      const need = Math.ceil(prompt / B);
      if (need > free.length) { log.textContent = `Cannot admit: needs ${need} blocks, ${free.length} free. A scheduler would queue it.`; return; }
      const id = String.fromCharCode(65 + (nextId++ % 26));
      const s = { id, tokens: prompt, table: [], color: hues[seqs.length % hues.length] };
      for (let i = 0; i < need; i++) s.table.push(free.pop());
      seqs.push(s);
      log.textContent = `Admitted ${id}: ${prompt}-token prompt → ${need} blocks [${s.table.join(", ")}]. Physical blocks need not be adjacent.`;
      draw();
    }
    function step() {
      if (!seqs.length) { log.textContent = "Nothing is running."; return; }
      const msgs = [];
      for (const s of [...seqs]) {
        if (s.tokens % B === 0) {
          if (!free.length) {
            const v = seqs.pop(); v.table.forEach((b) => free.push(b));
            msgs.push(`out of blocks: preempted ${v.id} (freed ${v.table.length}); it would be recomputed later`);
            if (v === s) continue;
          }
          const b = free.pop(); s.table.push(b); msgs.push(`${s.id} filled its last block → new block ${b}`);
        }
        s.tokens++;
      }
      log.textContent = "Decode step: every running request +1 token. " + (msgs.join("; ") || "No new blocks needed.");
      draw();
    }
    function finish() {
      if (!seqs.length) { log.textContent = "Nothing to finish."; return; }
      const s = seqs.shift(); s.table.forEach((b) => free.push(b));
      log.textContent = `${s.id} finished: ${s.table.length} blocks returned to the free list.`;
      draw();
    }
    let timer = null;
    function auto() {
      if (timer) { clearInterval(timer); timer = null; autoBtn.textContent = "Auto-run"; return; }
      autoBtn.textContent = "Stop";
      timer = setInterval(() => { const r = Math.random(); if (r < 0.12) admit(); else if (r < 0.18) finish(); else step(); }, REDUCED ? 900 : 350);
    }
    const autoBtn = el("button", { text: "Auto-run", onclick: auto });
    controls.append(field("Block size", bsSel), el("button", { class: "primary", text: "Admit request", onclick: admit }),
      el("button", { text: "Decode step", onclick: step }), el("button", { text: "Finish oldest", onclick: finish }), autoBtn,
      el("button", { text: "Reset", onclick: () => { if (timer) auto(); reset(); } }));
    bsSel.addEventListener("change", reset);
    function draw() {
      const owner = {};
      seqs.forEach((s) => s.table.forEach((b, i) => {
        const used = i < s.table.length - 1 ? B : s.tokens - (s.table.length - 1) * B;
        owner[b] = { s, i, used };
      }));
      grid.innerHTML = "";
      for (let b = 0; b < NB; b++) {
        const c = el("div", { class: "cell", title: owner[b] ? `block ${b}: ${owner[b].s.id} logical ${owner[b].i}, ${owner[b].used}/${B} slots` : `block ${b}: free` });
        if (owner[b]) {
          const o = owner[b];
          c.appendChild(el("div", { class: "fill", style: `height:${100 * o.used / B}%;background:var(${o.s.color})` }));
          c.style.borderColor = `var(${o.s.color})`;
          if (o.used < B) c.style.background = "var(--ie-waste-bg)";
          c.appendChild(el("span", { text: o.s.id + o.i }));
        }
        grid.appendChild(c);
      }
      const used = NB - free.length, live = seqs.reduce((a, s) => a + s.tokens, 0);
      o1.textContent = `${used} / ${NB}`; o2.textContent = fmt(live); o3.textContent = fmt(used * B - live);
      o4.textContent = used ? fmt(100 * live / (used * B), 1) + " %" : "–";
      tables.innerHTML = "<tr><th>Request</th><th>Tokens</th><th>Block table (logical → physical)</th></tr>" +
        seqs.map((s) => `<tr><td>${s.id}</td><td>${s.tokens}</td><td>${s.table.map((b, i) => `${i}→${b}`).join("  ")}</td></tr>`).join("");
    }
    w.append(controls, grid, el("div", { class: "legend", html:
      '<span><i style="background:var(--ie-memory)"></i>used slots (color = request)</span><span><i style="background:var(--ie-waste-bg)"></i>last block, partly empty</span><span><i></i>free block</span>' }),
      el("div", { class: "out" }, [s1, s2, s3, s4]), log, tables);
    reset();
    root.appendChild(w);
  }

  // ---------------------------------------------------------------- Prefix sharing (prefix.py)
  function prefixsim(root) {
    const w = frame("Prefix sharing and reference counts",
      "Users send the same system prompt plus their own question. Only full blocks of the shared prefix can be shared (the hash of a block covers its whole prefix); the partial tail is private.");
    const S = num(200, 1, 4000), U = num(4, 1, 32), Q = num(40, 1, 2000), Bs = select([["16", "16"], ["32", "32"]], "16");
    const share = el("input", { type: "checkbox", checked: "" });
    w.appendChild(el("div", { class: "row" }, [field("System prompt tokens", S), field("Users", U), field("Tokens per question", Q),
      field("Block size", Bs), el("label", { style: "flex-direction:row;align-items:center;gap:.4rem" }, [share, "Share prefix blocks"])]));
    const [s1, o1] = stat("Blocks without sharing"), [s2, o2] = stat("Blocks with sharing"), [s3, o3] = stat("Saved"), [s4, o4] = stat("Refcount of a shared block");
    const grid = el("div", { class: "grid" });
    const log = el("p", { class: "log" });
    w.append(el("div", { class: "out" }, [s1, s2, s3, s4]), grid, el("div", { class: "legend", html:
      '<span><i style="background:var(--ie-shared)"></i>shared prefix block (refcount = users)</span><span><i style="background:var(--ie-memory)"></i>private block</span><span><i style="background:var(--ie-waste-bg)"></i>duplicated prefix (no sharing)</span>' }), log);
    function update() {
      const B = +Bs.value, s = +S.value, u = +U.value, q = +Q.value;
      const full = Math.floor(s / B);
      const perUserNo = Math.ceil((s + q) / B);
      const noShare = u * perUserNo;
      const privPerUser = Math.ceil((s - full * B + q) / B);
      const withShare = full + u * privPerUser;
      const on = share.checked;
      o1.textContent = fmt(noShare); o2.textContent = fmt(withShare);
      o3.textContent = fmt(noShare - withShare) + " blocks (" + fmt(100 * (noShare - withShare) / noShare, 1) + " %)";
      o4.textContent = on ? String(u) : "1";
      grid.innerHTML = "";
      const cap = 192; let drawn = 0;
      const add = (cls, label, title) => { if (drawn++ >= cap) return; const c = el("div", { class: "cell", title }); c.appendChild(el("div", { class: "fill", style: `height:100%;background:${cls}` })); c.appendChild(el("span", { text: label })); grid.appendChild(c); };
      if (on) {
        for (let i = 0; i < full; i++) add("var(--ie-shared)", "×" + u, `shared prefix block ${i}, refcount ${u}`);
        for (let k = 0; k < u; k++) for (let i = 0; i < privPerUser; i++) add("var(--ie-memory)", "U" + (k + 1), `user ${k + 1} private block`);
      } else {
        for (let k = 0; k < u; k++) for (let i = 0; i < perUserNo; i++) add(i < full ? "var(--ie-waste-bg)" : "var(--ie-memory)", "U" + (k + 1), i < full ? `user ${k + 1}: duplicate of prefix block ${i}` : `user ${k + 1} private block`);
      }
      log.textContent = (drawn > cap ? `Showing the first ${cap} blocks. ` : "") +
        (on ? `When a user finishes, each shared block's refcount drops by 1; a block returns to the pool (or stays cached for the next user) only at 0.`
            : `Without sharing, the same ${full} prefix blocks are stored ${u} times. This is duplication, not fragmentation: every slot is used, just redundantly.`);
    }
    w.querySelectorAll("input,select").forEach((i) => i.addEventListener("input", update));
    update();
    root.appendChild(w);
  }

  // ---------------------------------------------------------------- Static vs continuous batching
  function batching(root) {
    const w = frame("Static vs continuous batching",
      "Six batch slots, sixteen requests with random output lengths. Static batching waits for the longest request in a batch; continuous batching refills a slot the iteration after it frees.");
    const mode = select([["static", "Static"], ["continuous", "Continuous"]], "continuous");
    const seedBtn = el("button", { text: "New requests" });
    const cv = el("canvas", { width: 960, height: 220, role: "img", "aria-label": "Slot occupancy over iterations" });
    const [s1, o1] = stat("Iterations to finish all"), [s2, o2] = stat("Slot utilization"), [s3, o3] = stat("Mean wait before start (iterations)");
    w.append(el("div", { class: "controls" }, [field("Policy", mode), seedBtn]), cv, el("div", { class: "out" }, [s1, s2, s3]),
      el("div", { class: "legend", html: '<span><i style="background:var(--ie-memory)"></i>decoding</span><span><i style="background:var(--ie-compute)"></i>prefill (first iteration)</span><span><i style="background:var(--ie-waste-bg)"></i>idle slot</span>' }));
    let lens = [];
    function newReqs() { lens = Array.from({ length: 16 }, () => 2 + Math.floor(Math.random() * 22)); run(); }
    function schedule() {
      const slots = 6, rows = Array.from({ length: slots }, () => []);
      const start = [];
      if (mode.value === "static") {
        let t = 0;
        for (let b = 0; b < lens.length; b += slots) {
          const batch = lens.slice(b, b + slots), span = Math.max(...batch);
          for (let s = 0; s < slots; s++) {
            const L = batch[s];
            for (let k = 0; k < span; k++) rows[s][t + k] = L !== undefined && k < L ? (k === 0 ? "p" : "d") : "i";
            if (L !== undefined) start.push(t);
          }
          t += span;
        }
        return { rows, T: t, start };
      }
      const busyUntil = new Array(slots).fill(0); let next = 0, T = 0;
      while (next < lens.length) {
        let s = 0; for (let i = 1; i < slots; i++) if (busyUntil[i] < busyUntil[s]) s = i;
        const t0 = busyUntil[s];
        for (let k = 0; k < lens[next]; k++) rows[s][t0 + k] = k === 0 ? "p" : "d";
        start.push(t0); busyUntil[s] = t0 + lens[next]; next++;
      }
      T = Math.max(...busyUntil);
      for (let s = 0; s < slots; s++) for (let t = 0; t < T; t++) if (!rows[s][t]) rows[s][t] = "i";
      return { rows, T, start };
    }
    function run() {
      const { rows, T, start } = schedule();
      const ctx = cv.getContext("2d"), css = getComputedStyle(document.body);
      const col = (v) => css.getPropertyValue(v).trim() || "#888";
      const W = cv.width, H = cv.height, pad = 6, rh = (H - 2 * pad) / rows.length, cw = (W - 2 * pad) / Math.max(T, 1);
      ctx.clearRect(0, 0, W, H);
      let busy = 0, total = 0;
      rows.forEach((r, s) => r.forEach((c, t) => {
        total++; if (c !== "i") busy++;
        ctx.fillStyle = c === "p" ? col("--ie-compute") : c === "d" ? col("--ie-memory") : col("--ie-waste-bg");
        ctx.fillRect(pad + t * cw, pad + s * rh + 2, Math.max(1, cw - 1), rh - 4);
      }));
      o1.textContent = String(T); o2.textContent = fmt(100 * busy / total, 1) + " %";
      o3.textContent = fmt(start.reduce((a, b) => a + b, 0) / start.length, 1);
    }
    mode.addEventListener("change", run); seedBtn.addEventListener("click", newReqs);
    newReqs();
    root.appendChild(w);
  }

  // ---------------------------------------------------------------- Speculative decoding (specdec.py)
  function specdec(root) {
    const w = frame("Speculative decoding payoff",
      "E[tokens per target pass] = (1 − α^(γ+1)) / (1 − α); speedup = E / (γ·c + 1), where c is the draft's cost relative to one target pass.");
    const a = el("input", { type: "range", min: 0.3, max: 0.98, step: 0.01, value: 0.8 });
    const c = el("input", { type: "range", min: 0.01, max: 0.5, step: 0.01, value: 0.05 });
    const av = el("span"), cvv = el("span");
    w.appendChild(el("div", { class: "row" }, [field(el("span", {}, ["Acceptance rate α = ", av]), a), field(el("span", {}, ["Draft cost c = ", cvv]), c)]));
    const tbl = el("table", { class: "tbl" });
    const [s1, o1] = stat("Best γ"), [s2, o2] = stat("Speedup at best γ");
    w.append(el("div", { class: "out" }, [s1, s2]), tbl);
    function E(al, g) { return al >= 1 ? g + 1 : (1 - Math.pow(al, g + 1)) / (1 - al); }
    function update() {
      const al = +a.value, cc = +c.value; av.textContent = al.toFixed(2); cvv.textContent = cc.toFixed(2);
      let best = [1, 0];
      let rows = "<tr><th>γ (draft length)</th><th>E[tokens / pass]</th><th>Speedup</th></tr>";
      for (let g = 1; g <= 12; g++) {
        const e = E(al, g), sp = e / (g * cc + 1);
        if (sp > best[1]) best = [g, sp];
        if (g <= 8 || g % 2 === 0) rows += `<tr><td>${g}</td><td>${e.toFixed(2)}</td><td>${sp.toFixed(2)}×</td></tr>`;
      }
      tbl.innerHTML = rows; o1.textContent = String(best[0]); o2.textContent = best[1].toFixed(2) + "×";
    }
    a.addEventListener("input", update); c.addEventListener("input", update); update();
    root.appendChild(w);
  }

  // ---------------------------------------------------------------- Hero memory map
  function memmap(root) {
    const W = 530, H = 250, bw = 36, bh = 26, gap = 6;
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" });
    const lx = 20, ly = 34, cols = 12, px = 20, py = 140;
    // physical pool: 12 x 3, some owned by other requests, request A lands on scattered free ones
    const other = new Set([0, 1, 4, 5, 9, 13, 14, 17, 20, 22, 25, 26, 30, 33, 34]);
    const sharedSet = new Set([2, 15]);
    const targets = [7, 19, 3, 28, 11, 23];
    const pending = [];
    const t1 = svg("text", { x: lx, y: ly - 12, class: "lab" }); t1.textContent = "Request A, logical blocks (contiguous)";
    const t2 = svg("text", { x: px, y: py - 12, class: "lab" }); t2.textContent = "GPU KV pool, physical blocks (scattered)";
    s.append(t1, t2);
    const wires = svg("g");  // appended after the blocks so wires stay visible
    for (let i = 0; i < 36; i++) {
      const x = px + (i % cols) * (bw + gap), y = py + Math.floor(i / cols) * (bh + gap);
      const cls = other.has(i) ? "other" : sharedSet.has(i) ? "shared" : "free";
      const r = svg("rect", { x, y, width: bw, height: bh, class: `blk ${cls}`, "data-p": i });
      s.appendChild(r);
      const tt = svg("text", { x: x + bw / 2, y: y + bh / 2 + 3.5, class: "blk-t" }); tt.textContent = String(i);
      s.appendChild(tt);
    }
    targets.forEach((p, i) => {
      const x = lx + i * (bw + gap + 22), y = ly;
      const r = svg("rect", { x, y, width: bw + 22, height: bh, class: "blk mine" });
      const t = svg("text", { x: x + (bw + 22) / 2, y: y + bh / 2 + 3.5, class: "blk-t" }); t.textContent = `A${i}`;
      s.append(r, t);
      const tx = px + (p % cols) * (bw + gap) + bw / 2, ty = py + Math.floor(p / cols) * (bh + gap);
      const path = svg("path", { d: `M${x + (bw + 22) / 2},${y + bh} C${x + (bw + 22) / 2},${y + bh + 40} ${tx},${ty - 40} ${tx},${ty}`, class: "wire" });
      wires.appendChild(path);
      pending.push([path, i]);
      const tgt = () => s.querySelector(`rect[data-p="${p}"]`);
      if (REDUCED) tgt().setAttribute("class", "blk mine");
      else setTimeout(() => tgt().setAttribute("class", "blk mine"), 800 + i * 180);
    });
    s.appendChild(wires);
    const tbl = svg("text", { x: lx, y: H - 6, class: "tbl-t" });
    tbl.textContent = "block table A: 0→7  1→19  2→3  3→28  4→11  5→23";
    s.appendChild(tbl);
    root.innerHTML = ""; root.appendChild(s);
    if (!REDUCED) {  // lengths are only measurable once the SVG is in the document
      pending.forEach(([path, i]) => {
        const len = Math.ceil(path.getTotalLength()) + 2;
        path.style.strokeDasharray = len; path.style.strokeDashoffset = len;
        path.getBoundingClientRect();  // commit the start state before transitioning
        path.style.transition = `stroke-dashoffset 600ms ease ${250 + i * 180}ms`;
        path.style.strokeDashoffset = 0;
      });
    }
  }

  const WIDGETS = { kvcalc, allocator, prefix: prefixsim, batching, specdec, memmap };
  function mount() {
    document.querySelectorAll("[data-widget]").forEach((n) => {
      if (n.dataset.mounted) return;
      const f = WIDGETS[n.dataset.widget];
      if (!f) return;
      n.dataset.mounted = "1";
      try { f(n); } catch (e) { n.textContent = "This interactive figure failed to load: " + e.message; }
    });
  }
  if (typeof document$ !== "undefined") document$.subscribe(mount);  // no emission on file://
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
})();
