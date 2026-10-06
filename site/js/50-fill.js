// Figure: L2 fills (schematic). Toggle between the two schedules; a cursor sweeps time.
FIGS.fill = (fig) => {
  const W = 760, H = 210, L = 118, sc = 1.32;
  const svg = svgRoot(W, H, "L2 fill schematic");
  let on = false, timer = 0, stop = () => {};
  const tog = seg([[false, "Without fill"], [true, "With fill"]], on, (v) => { clearInterval(timer); set(v); }, "schedule");
  mount(fig, h("div", { class: "fig-controls" }, tog), svg);
  const X = (v) => L + v * sc;
  const y1 = 44, y2 = 112;
  s("text", { x: L - 12, y: y1 + 18, class: "t-label", "text-anchor": "end", text: "SMs" }, svg);
  s("text", { x: L - 12, y: y1 + 33, class: "t-small", "text-anchor": "end", text: "kernels" }, svg);
  s("text", { x: L - 12, y: y2 + 14, class: "t-label", "text-anchor": "end", text: "DRAM" }, svg);
  s("text", { x: L - 12, y: y2 + 29, class: "t-small", "text-anchor": "end", text: "weight stream" }, svg);
  const mk = (cls, y, hgt) => s("rect", { y, height: hgt, class: cls, rx: 2 }, svg);
  const gA = mk("k-gemm", y1, 28), norm = mk("k-norm", y1, 28), gB = mk("k-gemm", y1, 28);
  const dA = mk("k-gemm op", y2, 18), dIdle = mk("k-idle", y2, 18), dFill = mk("c-fill", y2, 18), dB = mk("k-gemm op", y2, 18);
  const tA = s("text", { y: y1 + 19, class: "t-in", "text-anchor": "middle", text: "GEMM A" }, svg);
  const tN = s("text", { y: y1 + 19, class: "t-in", "text-anchor": "middle", text: "norm" }, svg);
  const tB = s("text", { y: y1 + 19, class: "t-in", "text-anchor": "middle" }, svg);
  const tD = s("text", { y: y2 + 40, class: "t-small", "text-anchor": "middle" }, svg);
  const endLine = s("line", { y1: 30, y2: y2 + 26, class: "end-line" }, svg);
  const endTxt = s("text", { y: 24, class: "t-small", "text-anchor": "middle" }, svg);
  const cursor = s("line", { y1: 34, y2: y2 + 24, class: "cursor" }, svg);
  const geom = (f) => ({ a: 200, n: 60 + 2 * f, b: 200 - 28 * f });
  function place(f) {
    const g = geom(f), xA = X(0), xN = X(g.a), xB = X(g.a + g.n), xE = X(g.a + g.n + g.b);
    gA.setAttribute("x", xA); gA.setAttribute("width", xN - xA);
    norm.setAttribute("x", xN); norm.setAttribute("width", xB - xN);
    gB.setAttribute("x", xB); gB.setAttribute("width", xE - xB);
    dA.setAttribute("x", xA); dA.setAttribute("width", xN - xA);
    dIdle.setAttribute("x", xN); dIdle.setAttribute("width", xB - xN);
    dFill.setAttribute("x", xN); dFill.setAttribute("width", (xB - xN) * f); dFill.setAttribute("opacity", f);
    dB.setAttribute("x", xB); dB.setAttribute("width", xE - xB);
    tA.setAttribute("x", (xA + xN) / 2); tN.setAttribute("x", (xN + xB) / 2); tB.setAttribute("x", (xB + xE) / 2);
    tB.textContent = f > 0.5 ? "GEMM B (first MBs already in L2)" : "GEMM B (starts cold)";
    tD.setAttribute("x", (xN + xB) / 2);
    tD.textContent = f > 0.5 ? "extra CTAs of the norm pull B's first MBs into L2" : "DRAM idle while the norm runs";
    endLine.setAttribute("x1", xE); endLine.setAttribute("x2", xE);
    endTxt.setAttribute("x", xE); endTxt.textContent = f > 0.5 ? "ends earlier" : "end";
    return xE;
  }
  function set(v) {
    on = v; tog.set(v);
    const f0 = on ? 0 : 1, f1 = on ? 1 : 0;
    stop();
    stop = tween(600, (k) => place(lerp(f0, f1, k)), () => {
      const xE = place(f1);
      const t0 = performance.now();
      const sweep = (now) => {
        const p = clamp((now - t0) / 1600, 0, 1);
        const x = lerp(X(0), xE, p);
        cursor.setAttribute("x1", x); cursor.setAttribute("x2", x);
        if (p < 1 && on === v) requestAnimationFrame(sweep);
      };
      if (!reduceMotion) requestAnimationFrame(sweep);
    });
  }
  place(0);
  onVisible(fig, () => {
    set(true);
    let n = 0;
    timer = setInterval(() => { if (++n > 6) return clearInterval(timer); set(!on); }, 3200);
  });
};
