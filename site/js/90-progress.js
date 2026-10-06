// Figure: every adopted change in order (one Spec-Bench-480 run each at adoption time).
FIGS.progress = (fig) => {
  const P = D.progress;
  const W = 760, H = 300, l = 56, r = 20, t = 18, b = 62;
  const y0 = 220, y1 = 330;
  const sx = (i) => l + ((W - l - r) * (i + 0.5)) / P.length, sy = linScale(y0, y1, H - b, t);
  const svg = svgRoot(W, H, "progress over four days");
  const KC = { engine: "k-other", quant: "k-gemm", drafter: "k-norm", kernel: "k-head" };
  const KN = { quant: "quantization", drafter: "drafter and tree", kernel: "kernels and engine" };
  const leg = h("div", { class: "legend-row" });
  mount(fig, svg, leg);
  let focus = null;
  for (const [k, name] of Object.entries(KN)) {
    const n = P.filter((p) => p[2] === k).length;
    let gain = 0;
    P.forEach((p, i) => { if (i && p[2] === k) gain += Math.log(p[1] / P[i - 1][1]); });
    const it = h("button", { type: "button", class: "lg-item" }, h("span", { class: "sw round " + KC[k] }), h("span", {}, name),
      h("span", { class: "lg-val" }, `${n} steps, ${pct(Math.exp(gain) - 1, 1)}`));
    it.addEventListener("mouseenter", () => { focus = k; draw(); });
    it.addEventListener("mouseleave", () => { focus = null; draw(); });
    leg.append(it);
  }
  function draw() {
    clear(svg);
    let cur = null;
    P.forEach((p, i) => {
      if (p[0] !== cur) {
        cur = p[0];
        const j = P.findLastIndex ? P.findLastIndex((q) => q[0] === cur) : P.map((q) => q[0]).lastIndexOf(cur);
        const xa = sx(i) - (W - l - r) / P.length / 2, xb = sx(j) + (W - l - r) / P.length / 2;
        if (["10-03", "10-05"].includes(cur)) s("rect", { x: xa, y: t, width: xb - xa, height: H - t - b, class: "band" }, svg);
        s("text", { x: (xa + xb) / 2, y: H - b + 18, class: "t-axis", "text-anchor": "middle", text: "Oct " + cur.split("-")[1].replace(/^0/, "") }, svg);
      }
    });
    for (let v = y0; v <= y1; v += 20) {
      s("line", { x1: l, x2: W - r, y1: sy(v), y2: sy(v), class: "grid" }, svg);
      s("text", { x: l - 8, y: sy(v) + 4, class: "t-axis", "text-anchor": "end", text: v }, svg);
    }
    s("text", { x: 14, y: (t + H - b) / 2, class: "t-axis", "text-anchor": "middle", transform: `rotate(-90 14 ${(t + H - b) / 2})`, text: "Spec-Bench tok/s" }, svg);
    const d = P.map((p, i) => (i ? `L${sx(i)},${sy(P[i - 1][1])} ` : "") + `${i ? "L" : "M"}${sx(i)},${sy(p[1])}`).join(" ");
    s("path", { d, class: "step" }, svg);
    P.forEach((p, i) => {
      if (i && p[2] === focus) {
        s("line", { x1: sx(i), x2: sx(i), y1: sy(P[i - 1][1]), y2: sy(p[1]), class: "gain " + KC[p[2]] }, svg);
      }
      const dot = s("circle", { cx: sx(i), cy: sy(p[1]), r: focus && focus !== p[2] ? 3.5 : 6, class: KC[p[2]] + (focus && focus !== p[2] ? " dim" : "") }, svg);
      hover(dot, () => `<b>${escH(p[3])}</b><br>${fmt(p[1], 1)} tok/s${i ? ` (${pct(p[1] / P[i - 1][1] - 1)})` : ""}<br><span class="mut">Oct ${p[0].split("-")[1].replace(/^0/, "")}</span>`);
    });
    s("text", { x: sx(0) + 10, y: sy(P[0][1]) + 20, class: "t-strong t-label", text: fmt(P[0][1], 1) }, svg);
    s("text", { x: sx(P.length - 1) - 10, y: sy(P[P.length - 1][1]) - 10, class: "t-strong t-label", "text-anchor": "end", text: fmt(P[P.length - 1][1], 1) }, svg);
  }
  draw();
};

// Figure: the eight-workload suite as grouped bars.
FIGS.suite = (fig) => {
  const S = D.suite, O = D.suite_0310;
  const ENG = [["llama", "llama.cpp recipe*", O.llama], ["vllm", "vLLM recipe*", O.vllm], ["stock", "stock Cinference", S.stock],
    ["rel_def", "our engine, official weights†", S.rel_def], ["final", "our engine + our weights", S.final]];
  const WL = D.workloads;
  let metric = "tps";
  const W = 760, l = 150, r = 70, rowH = 16 * ENG.length + 14, t = 10;
  const H = t + WL.length * rowH + 30;
  const svg = svgRoot(W, H, "workload suite");
  const leg = h("div", { class: "legend-row" });
  ENG.forEach(([k, name]) => leg.append(h("span", { class: "lg-item static" }, h("span", { class: "sw e-" + k }), h("span", {}, name))));
  mount(fig, h("div", { class: "fig-controls" }, seg([["tps", "tok/s"], ["x", "× stock Cinference"]], metric, (m) => { metric = m; draw(); }, "metric")), svg, leg);
  function draw() {
    clear(svg);
    const val = (d, k) => (metric === "tps" ? d[k] : d[k] / S.stock[k]);
    const vmax = Math.max(...WL.map(([k]) => Math.max(...ENG.map(([, , d]) => val(d, k))))) * 1.04;
    const sx = linScale(0, vmax, l, W - r);
    for (const v of niceTicks(0, vmax, 6)) {
      s("line", { x1: sx(v), x2: sx(v), y1: t, y2: H - 26, class: "grid" }, svg);
      s("text", { x: sx(v), y: H - 10, class: "t-axis", "text-anchor": "middle", text: metric === "tps" ? fmt(v, 0) : fmt(v, 1) + "×" }, svg);
    }
    if (metric === "x") s("line", { x1: sx(1), x2: sx(1), y1: t, y2: H - 26, class: "bar-line" }, svg);
    WL.forEach(([k, name, sub], i) => {
      const y = t + i * rowH;
      s("text", { x: l - 10, y: y + rowH / 2 - 2, class: "t-label", "text-anchor": "end", text: name }, svg);
      s("text", { x: l - 10, y: y + rowH / 2 + 13, class: "t-small", "text-anchor": "end", text: sub }, svg);
      ENG.forEach(([ek, en, d], j) => {
        const v = val(d, k), yy = y + 4 + j * 16;
        const bar = s("rect", { x: l, y: yy, width: Math.max(0, sx(v) - l), height: 13, rx: 2, class: "e-" + ek }, svg);
        hover(bar, () => `<b>${escH(name)}</b> · ${escH(en)}<br>${fmt(d[k], 1)} tok/s · ${fmt(d[k] / S.stock[k], 2)}× stock`);
        if (ek === "final") s("text", { x: sx(v) + 5, y: yy + 11, class: "t-small t-strong", text: metric === "tps" ? fmt(v, 0) : fmt(v, 2) + "×" }, svg);
      });
    });
  }
  draw();
};
