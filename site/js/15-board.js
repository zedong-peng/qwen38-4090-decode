// Figure: decode speed of every engine, one session on one card: Spec-Bench-480 (overall and by task group) and the
// eight single-prompt workloads.
FIGS.board = (fig) => {
  const B = D.board;
  if (!B) return;
  const E = B.engines;
  const SB = [["overall", "All 480"], ["mt_bench", "MT-Bench"], ["translation", "Translation"], ["summarization", "Summaries"],
    ["qa", "QA"], ["math_reasoning", "Math"], ["rag", "RAG"]];
  const WL = D.workloads;
  const ALL = Object.fromEntries([...SB.map(([k, n]) => [k, { name: n, kind: "sb" }]),
    ...WL.map(([k, n, sub]) => [k, { name: n, sub, kind: "wl" }])]);
  let sel = "overall", base = B.base, grown = reduceMotion;
  const W = 760, l = 228, r = 120, t = 8, rowH = 50, H = t + E.length * rowH + 30;
  const svg = svgRoot(W, H, "decode speed by engine");
  const note = h("p", { class: "board-note" });
  const pick = (k) => { sel = k; segA.set(k); segB.set(k); draw(350); };
  const segA = seg(SB, sel, pick, "Spec-Bench task group");
  const segB = seg(WL.map(([k, n]) => [k, n]), sel, pick, "single-prompt workload");
  const baseSeg = seg(B.bases.map((k) => [k, "× " + E.find((e) => e.key === k).short]), base, (k) => { base = k; draw(0); }, "baseline");
  mount(fig,
    h("div", { class: "fig-controls" }, h("span", { class: "ctl-lab" }, "Spec-Bench"), segA),
    h("div", { class: "fig-controls" }, h("span", { class: "ctl-lab" }, "Single prompts"), segB),
    svg, h("div", { class: "fig-controls" }, note, h("div", { class: "spacer" }), baseSeg));
  let cur = E.map(() => 0), stop = () => {};
  const cell = (e, k) => (ALL[k].kind === "sb" ? e.groups[k] : e.suite[k]) || {};

  function draw(ms) {
    const to = E.map((e) => cell(e, sel).tps || 0), from = cur.slice();
    const vmax = Math.max(...to) * 1.06;
    const sx = linScale(0, vmax, l, W - r);
    const a = ALL[sel];
    note.textContent = a.kind === "sb"
      ? `Spec-Bench, ${sel === "overall" ? "all 480 prompts" : "80 prompts"}, 256 tokens each`
      : `one prompt (${a.sub}), median of 3`;
    stop();
    stop = tween(ms, (k) => { cur = from.map((v, i) => lerp(v, to[i], k)); render(sx, vmax); });
  }

  function render(sx, vmax) {
    clear(svg);
    for (const v of niceTicks(0, vmax, 6)) {
      s("line", { x1: sx(v), x2: sx(v), y1: t, y2: H - 26, class: "grid" }, svg);
      s("text", { x: sx(v), y: H - 8, class: "t-axis", "text-anchor": "middle", text: fmt(v, 0) }, svg);
    }
    s("text", { x: W - r + 8, y: H - 8, class: "t-axis", text: "tok/s" }, svg);
    const b = E.find((e) => e.key === base), bv = cell(b, sel).tps;
    E.forEach((e, i) => {
      const y = t + i * rowH, q = cell(e, sel);
      if (i && e.ours !== E[i - 1].ours) s("line", { x1: 8, x2: W - 8, y1: y - 1, y2: y - 1, class: "board-sep" }, svg);
      s("text", { x: l - 12, y: y + 21, class: "t-label t-strong", "text-anchor": "end", text: e.name }, svg);
      s("text", { x: l - 12, y: y + 37, class: "t-small", "text-anchor": "end", text: e.sub }, svg);
      const g = s("g", { class: "board-bar" }, svg);
      s("rect", { x: l, y: y + 9, width: Math.max(0, sx(cur[i]) - l), height: 28, rx: 3, class: "e-" + e.key }, g);
      const lab = s("text", { x: sx(cur[i]) + 8, y: y + 28, class: "t-label t-strong", text: q.tps ? fmt(cur[i], 0) : "–" }, svg);
      if (bv && e.key !== base && q.tps) s("tspan", { class: "t-small", dx: 6, text: fmt(q.tps / bv, 2) + "×" }, lab);
      if (q.tps) hover(g, () => `<b>${escH(e.name)}</b> · ${escH(e.long)}<br>${escH(ALL[sel].name)}: <b>${fmt(q.tps, 1)} tok/s</b>` +
        (q.tau ? `<br>${fmt(q.tau, 2)} tokens/round · ${fmt((1000 * q.tau) / q.tps, 1)} ms/round` : "") +
        (bv && e.key !== base ? `<br>${fmt(q.tps / bv, 2)}× ${escH(b.short)}` : ""));
    });
  }

  draw(0);
  if (!grown) {
    const vmax = Math.max(...cur) * 1.06;
    cur = E.map(() => 0);
    render(linScale(0, vmax, l, W - r), vmax);
    onVisible(fig, () => { grown = true; draw(900); }, 0.3);
  }
};
