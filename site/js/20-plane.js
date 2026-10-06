// Figure: the throughput plane. tok/s = tokens per round / time per round, so iso-throughput lines are rays from
// the origin. Each configuration is a point; the steps between them are either "up" (more tokens) or "left" (faster).
FIGS.plane = (fig) => {
  const P = D.plane;
  if (!P) return;
  const W = 760, H = 460, l = 62, r = 116, t = 18, b = 52;
  const svg = svgRoot(W, H, "tokens per round versus milliseconds per round");
  let mode = "overall", dom = null, stop = () => {};
  const DOMS = { overall: [15.5, 24, 3.6, 5.9], tasks: [15.5, 24, 2.8, 9.6] };
  const ctl = h("div", { class: "fig-controls" },
    seg([["overall", "Spec-Bench overall"], ["tasks", "By task type"]], mode, (m) => go(m), "view"));
  mount(fig, ctl, svg);
  const NAMES = { mt_bench: "MT-Bench", translation: "translation", summarization: "summarization", qa: "QA",
    math_reasoning: "math reasoning", rag: "RAG" };

  function draw(d) {
    clear(svg);
    const [x0, x1, y0, y1] = d;
    const sx = linScale(x0, x1, l, W - r), sy = linScale(y0, y1, H - b, t);
    const g = s("g", {}, svg);
    // clip for iso lines
    const defs = s("defs", {}, svg);
    const cp = s("clipPath", { id: "plane-clip" }, defs);
    s("rect", { x: l, y: t, width: W - l - r, height: H - t - b }, cp);
    s("marker", { id: "pl-arr", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs)
      .append(s("path", { d: "M0,0 L10,5 L0,10 z", class: "c-ink" }));
    for (const v of niceTicks(x0, x1, 8)) {
      s("line", { x1: sx(v), x2: sx(v), y1: t, y2: H - b, class: "grid" }, g);
      s("text", { x: sx(v), y: H - b + 18, class: "t-axis", "text-anchor": "middle", text: fmt(v, 0) }, g);
    }
    for (const v of niceTicks(y0, y1, 6)) {
      s("line", { x1: l, x2: W - r, y1: sy(v), y2: sy(v), class: "grid" }, g);
      s("text", { x: l - 8, y: sy(v) + 4, class: "t-axis", "text-anchor": "end", text: fmt(v, v % 1 ? 1 : 0) }, g);
    }
    s("text", { x: (l + W - r) / 2, y: H - 12, class: "t-axis", "text-anchor": "middle", text: "milliseconds per verify round  (← faster rounds)" }, g);
    s("text", { x: 16, y: (t + H - b) / 2, class: "t-axis", "text-anchor": "middle", transform: `rotate(-90 16 ${(t + H - b) / 2})`, text: "tokens accepted per round  (more ↑)" }, g);
    // iso-throughput rays: tau = tps * ms / 1000
    const iso = s("g", { "clip-path": "url(#plane-clip)" }, svg);
    const isoStep = y1 - y0 > 4 ? 100 : 50;
    for (let v = isoStep; v <= 800; v += isoStep) {
      const ya = (v * x0) / 1000, yb = (v * x1) / 1000;
      if (yb < y0 || ya > y1) continue;
      s("line", { x1: sx(x0), y1: sy(ya), x2: sx(x1), y2: sy(yb), class: "iso" }, iso);
      // label where the ray leaves the plot (top edge or right edge)
      let lx = x1, ly = yb;
      if (yb > y1) { lx = (y1 * 1000) / v; ly = y1; }
      if (lx > x0 && ly > y0) {
        const tx = ly >= y1 ? sx(lx) + 3 : W - r + 6, ty = ly >= y1 ? t + 12 : sy(ly) + 4;
        s("text", { x: tx, y: ty, class: "t-iso", text: `${v} tok/s` }, svg);
      }
    }
    const pts = P.configs;
    const pos = (c, k) => [sx(k ? c.groups[k].ms : c.ms), sy(k ? c.groups[k].tau : c.tau)];
    if (mode === "tasks") {
      const groups = Object.keys(pts[0].groups).filter((k) => k !== "overall");
      for (const k of groups) {
        const pl = pts.map((c) => pos(c, k).join(",")).join(" ");
        s("polyline", { points: pl, class: "task-line" }, svg);
      }
      pts.forEach((c, i) => {
        for (const k of groups) {
          const [cx, cy] = pos(c, k), q = c.groups[k];
          const dot = s("circle", { cx, cy, r: 5.5, class: "pt-" + c.key }, svg);
          hover(dot, () => `<b>${escH(c.label)}</b><br>${NAMES[k] || k} (${q.n} prompts)<br>${fmt(q.tau, 2)} tokens/round · ${fmt(q.ms, 1)} ms/round<br><b>${fmt(q.tps, 1)} tok/s</b>`);
          if (i === pts.length - 1) s("text", { x: cx + 9, y: cy + 4, class: "t-small", text: NAMES[k] || k }, svg);
        }
      });
    }
    // steps between configurations
    for (let i = 0; i + 1 < pts.length; i++) {
      const [ax, ay] = pos(pts[i]), [bx, by] = pos(pts[i + 1]);
      const dd = Math.hypot(bx - ax, by - ay) || 1, sh = 13 / dd;
      const ln = s("line", { x1: ax + (bx - ax) * sh, y1: ay + (by - ay) * sh, x2: bx - (bx - ax) * sh, y2: by - (by - ay) * sh,
        class: "step-arrow" + (mode === "tasks" ? " faint" : ""), "marker-end": "url(#pl-arr)" }, svg);
      const st = P.steps[i];
      const hit = s("line", { x1: ax, y1: ay, x2: bx, y2: by, class: "hit" }, svg);
      hover(hit, () => `<b>${escH(st.title)}</b><br>${escH(st.text)}<br>tokens/round ${pct(pts[i + 1].tau / pts[i].tau - 1)}, round time ${pct(pts[i + 1].ms / pts[i].ms - 1)}`);
      if (mode === "overall") {
        const mx = (ax + bx) / 2, my = (ay + by) / 2;
        const lab = s("text", { x: mx + st.dx, y: my + st.dy, class: "t-step", "text-anchor": st.anchor || "middle" }, svg);
        st.lines.forEach((tx, j) => s("tspan", { x: mx + st.dx, dy: j ? 15 : 0, text: tx }, lab));
      }
      void ln;
    }
    pts.forEach((c) => {
      const [cx, cy] = pos(c);
      const big = mode === "overall";
      const dot = s("circle", { cx, cy, r: big ? 10 : 7, class: "pt-" + c.key + " pt-main" }, svg);
      hover(dot, () => `<b>${escH(c.label)}</b><br>${fmt(c.tau, 2)} tokens/round · ${fmt(c.ms, 1)} ms/round<br><b>${fmt(c.tps, 1)} tok/s</b> on Spec-Bench-480`);
      if (big) {
        const lab = s("text", { x: cx + c.lx, y: cy + c.ly, class: "t-label", "text-anchor": c.anchor || "start" }, svg);
        s("tspan", { x: cx + c.lx, dy: 0, class: "t-strong", text: `${fmt(c.tps, 1)} tok/s` }, lab);
        s("tspan", { x: cx + c.lx, dy: 16, class: "t-small", text: c.short }, lab);
      }
    });
  }

  function go(m) {
    mode = m;
    const from = dom || DOMS[m], to = DOMS[m];
    stop();
    stop = tween(dom ? 450 : 0, (k) => { dom = from.map((v, i) => lerp(v, to[i], k)); draw(dom); });
  }
  go("overall");
};
