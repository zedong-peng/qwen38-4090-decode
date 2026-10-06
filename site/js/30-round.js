// Figure: one verify round, kernel by kernel, from nsys traces (median round of a code prompt).
FIGS.round = (fig) => {
  const RD = D.rounds;
  if (!RD) return;
  const CLS = [
    ["gemm", "weight GEMMs"], ["head", "LM-head GEMM / screens"], ["attention", "attention"],
    ["gdn", "GDN recurrence, conv"], ["norm", "norms"], ["other", "other"]];
  const CN = Object.fromEntries(CLS);
  const T = RD.traces;
  const W = 900, L = 196, R = 14, rowH = 78, top = 8, stripH = 30;
  const H = top + T.length * rowH + 26;
  const svg = svgRoot(W, H, "verify round kernel timeline");
  const maxSpan = Math.max(...T.map((tr) => tr.span));
  let doms = T.map(() => [0, maxSpan]), stop = () => {}, focus = null;

  const presets = [
    ["round", "Whole round"], ["drafter", "Drafter"], ["block", "4 target layers"], ["tail", "Verify head"]];
  const pre = seg(presets, "round", (v) => zoomTo(v), "zoom");
  const ctl = h("div", { class: "fig-controls" }, pre, h("div", { class: "spacer" }),
    h("span", { class: "hint" }, "drag across a strip to zoom · double-click to reset"));
  const legend = h("div", { class: "legend-row" });
  mount(fig, ctl, svg, legend);

  for (const [k, name] of CLS) {
    const tot = T.map((tr) => (tr.class_us[k] || 0) / 1000);
    const item = h("button", { type: "button", class: "lg-item", "data-k": k },
      h("span", { class: "sw k-" + k }), h("span", {}, name),
      h("span", { class: "lg-val" }, tot.map((v) => fmt(v, 2)).join(" → ") + " ms"));
    item.addEventListener("mouseenter", () => { focus = k; draw(); });
    item.addEventListener("mouseleave", () => { focus = null; draw(); });
    item.addEventListener("focus", () => { focus = k; draw(); });
    item.addEventListener("blur", () => { focus = null; draw(); });
    legend.append(item);
  }
  const idleTot = T.map((tr) => tr.idle_us / 1000);
  legend.append(h("span", { class: "lg-item static" }, h("span", { class: "sw k-idle" }), h("span", {}, "GPU idle"),
    h("span", { class: "lg-val" }, idleTot.map((v) => fmt(v, 2)).join(" → ") + " ms")));

  function draw() {
    clear(svg);
    T.forEach((tr, i) => {
      const y = top + i * rowH, ys = y + 30;
      const [a, b] = doms[i];
      const sx = linScale(a, b, L, W - R);
      const g = s("g", {}, svg);
      s("text", { x: 4, y: ys + 12, class: "t-label t-strong", text: `${tr.label} · ${fmt(tr.span / 1000, 1)} ms` }, g);
      s("text", { x: 4, y: ys + 28, class: "t-small", text: tr.sub }, g);
      const clipId = "rc" + i;
      s("clipPath", { id: clipId }, s("defs", {}, g)).append(s("rect", { x: L, y: y, width: W - L - R, height: rowH - 8 }));
      const inner = s("g", { "clip-path": `url(#${clipId})` }, g);
      s("rect", { x: sx(0), y: ys, width: Math.max(0, sx(tr.span) - sx(0)), height: stripH, class: "k-idle" }, inner);
      // phases
      for (const [name, pa, pb] of tr.phases) {
        if (pb < a || pa > b) continue;
        const xa = Math.max(L, sx(pa)), xb = Math.min(W - R, sx(pb));
        s("line", { x1: xa + 1, x2: xb - 1, y1: ys - 6, y2: ys - 6, class: "phase" }, inner);
        s("line", { x1: xa + 1, x2: xa + 1, y1: ys - 10, y2: ys - 2, class: "phase" }, inner);
        s("line", { x1: xb - 1, x2: xb - 1, y1: ys - 10, y2: ys - 2, class: "phase" }, inner);
        if (xb - xa > 7 * name.length) s("text", { x: (xa + xb) / 2, y: ys - 11, class: "t-phase", "text-anchor": "middle", text: name }, inner);
      }
      const kg = s("g", {}, inner);
      for (const [kt, kd, kc] of tr.k) {
        if (kt + kd < a || kt > b) continue;
        const x = sx(kt), w = Math.max(0.7, sx(kt + kd) - x);
        s("rect", { x, y: ys, width: w, height: stripH, class: "k-" + kc + (focus && focus !== kc ? " dim" : "") }, kg);
      }
      // axis
      for (const v of niceTicks(a, b, 6)) {
        s("line", { x1: sx(v), x2: sx(v), y1: ys + stripH, y2: ys + stripH + 4, class: "tick" }, g);
        s("text", { x: sx(v), y: ys + stripH + 16, class: "t-axis", "text-anchor": "middle", text: v === 0 ? "0" : v >= 1000 && b - a > 3000 ? fmt(v / 1000, v % 1000 ? 1 : 0) + " ms" : fmt(v, 0) + " µs" }, g);
      }
      // interaction surface
      const hit = s("rect", { x: L, y: ys - 4, width: W - L - R, height: stripH + 8, class: "hit", "data-i": i }, g);
      hover(hit, (ev) => {
        const p = svgPoint(svg, ev), us = sx.inv(p.x);
        const tol = (b - a) / (W - L - R) * 1.5;
        const kk = tr.k.find(([kt, kd]) => us >= kt - tol && us <= kt + kd + tol);
        if (!kk) return `<b>GPU idle</b><br>${fmt(us, 0)} µs into the round`;
        const ph = tr.phases.find(([, pa, pb]) => kk[0] >= pa && kk[0] < pb);
        return `<b>${escH(tr.names[kk[3]])}</b><br>${CN[kk[2]]} · ${fmt(kk[1], 1)} µs<br><span class="mut">at ${fmt(kk[0], 0)} µs${ph ? " · " + escH(ph[0]) : ""}</span>`;
      });
      attachDrag(hit, i);
    });
  }

  function animateTo(target) {
    const from = doms.map((d) => d.slice());
    stop();
    stop = tween(500, (k) => { doms = from.map((d, i) => [lerp(d[0], target[i][0], k), lerp(d[1], target[i][1], k)]); draw(); });
  }

  function zoomTo(v) {
    if (v === "round") return animateTo(T.map(() => [0, maxSpan]));
    animateTo(T.map((tr) => {
      const [x0, x1] = tr.ranges[v];
      const pad = (x1 - x0) * 0.04;
      return [x0 - pad, x1 + pad];
    }));
  }

  function attachDrag(hit, i) {
    let x0 = null, band = null;
    hit.addEventListener("pointerdown", (ev) => {
      x0 = svgPoint(svg, ev).x;
      band = s("rect", { x: x0, y: top + i * rowH + 26, width: 0, height: stripH + 8, class: "brush" }, svg);
      hit.setPointerCapture(ev.pointerId);
    });
    hit.addEventListener("pointermove", (ev) => {
      if (x0 == null) return;
      const x = svgPoint(svg, ev).x;
      band.setAttribute("x", Math.min(x, x0));
      band.setAttribute("width", Math.abs(x - x0));
    });
    hit.addEventListener("pointerup", (ev) => {
      if (x0 == null) return;
      const x = svgPoint(svg, ev).x, [a, b] = doms[i];
      const sx = linScale(a, b, L, W - R);
      const u0 = sx.inv(Math.min(x, x0)), u1 = sx.inv(Math.max(x, x0));
      x0 = null;
      if (band) band.remove();
      if (u1 - u0 < (b - a) / 60) return;
      pre.set(null);
      animateTo(T.map(() => [u0, u1]));
    });
    hit.addEventListener("dblclick", () => { pre.set("round"); zoomTo("round"); });
  }

  draw();
};
