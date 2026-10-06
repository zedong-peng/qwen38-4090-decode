// Figure: measuring a 0.16% effect. Eight alternating runs (A B B A B A A B) with drift; fitting the drift away.
FIGS.drift = (fig) => {
  const R = D.drift;
  if (!R) return;
  const n = R.runs.length, ys = R.runs.map((r) => r[1]), isB = R.runs.map((r) => (r[0] === "B" ? 1 : 0));
  const tc = R.runs.map((_, i) => i - (n - 1) / 2);
  // least squares y = c0 + c1*B + c2*t + c3*t^2
  const X = R.runs.map((_, i) => [1, isB[i], tc[i], tc[i] * tc[i]]);
  const solve = (A, v) => {
    const m = A.length, M = A.map((row, i) => [...row, v[i]]);
    for (let c = 0; c < m; c++) {
      let p = c;
      for (let r = c + 1; r < m; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = 0; r < m; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= m; k++) M[r][k] -= f * M[c][k]; }
    }
    return M.map((row, i) => row[m] / M[i][i]);
  };
  const XtX = [0, 1, 2, 3].map((i) => [0, 1, 2, 3].map((j) => X.reduce((s_, row) => s_ + row[i] * row[j], 0)));
  const Xty = [0, 1, 2, 3].map((i) => X.reduce((s_, row, k) => s_ + row[i] * ys[k], 0));
  const c = solve(XtX, Xty);
  const fit = X.map((row) => row.reduce((s_, v, i) => s_ + v * c[i], 0));
  const res = ys.map((y, i) => y - fit[i]);
  const df = n - 4, s2 = res.reduce((a, v) => a + v * v, 0) / df;
  const inv11 = solve(XtX, [0, 1, 0, 0])[1];
  const seFit = Math.sqrt(s2 * inv11) / c[0], estFit = c[1] / c[0];
  const tq = { 4: 2.776, 6: 2.447 };
  const A = ys.filter((_, i) => !isB[i]), B = ys.filter((_, i) => isB[i]);
  const mean = (v) => v.reduce((a, x) => a + x, 0) / v.length, vr = (v) => { const m = mean(v); return v.reduce((a, x) => a + (x - m) ** 2, 0) / (v.length - 1); };
  const estN = (mean(B) - mean(A)) / mean(A), seN = Math.sqrt(vr(A) / A.length + vr(B) / B.length) / mean(A);

  const W = 760, H = 330, l = 70, r = 220, t = 20, b = 46;
  const svg = svgRoot(W, H, "drift-balanced A/B");
  let mode = "raw", stop = () => {}, k = 0;
  const ctl = h("div", { class: "fig-controls" }, seg([["raw", "Runs as measured"], ["fit", "Drift removed"]], mode, (m) => go(m), "view"));
  mount(fig, ctl, svg);
  const lo = Math.min(...ys) - 0.012, hi = Math.max(...ys) + 0.012;
  const sx = linScale(0, n - 1, l + 20, W - r - 20), sy = linScale(lo, hi, H - b, t);
  function draw() {
    clear(svg);
    for (const v of niceTicks(lo, hi, 5)) {
      s("line", { x1: l, x2: W - r, y1: sy(v), y2: sy(v), class: "grid" }, svg);
      s("text", { x: l - 8, y: sy(v) + 4, class: "t-axis", "text-anchor": "end", text: fmt(v, 2) }, svg);
    }
    s("text", { x: 16, y: (t + H - b) / 2, class: "t-axis", "text-anchor": "middle", transform: `rotate(-90 16 ${(t + H - b) / 2})`, text: "ms per round" }, svg);
    R.runs.forEach((_, i) => s("text", { x: sx(i), y: H - b + 18, class: "t-axis", "text-anchor": "middle", text: `run ${i + 1}` }, svg));
    s("text", { x: (l + W - r) / 2, y: H - 8, class: "t-axis", "text-anchor": "middle", text: "time order, one forced-text Spec-Bench run each (≈1 min)" }, svg);
    // drift curves
    const curve = (bb) => {
      const pts = [];
      for (let u = 0; u <= 60; u++) {
        const x = (u / 60) * (n - 1), tt = x - (n - 1) / 2;
        const y = c[0] + c[1] * bb + c[2] * tt + c[3] * tt * tt;
        const yy = lerp(y, c[0] + c[1] * bb, k);
        pts.push(`${sx(x).toFixed(1)},${sy(yy).toFixed(1)}`);
      }
      return pts.join(" ");
    };
    s("polyline", { points: curve(0), class: "drift a" }, svg);
    s("polyline", { points: curve(1), class: "drift b" }, svg);
    R.runs.forEach(([cfg, y], i) => {
      const yy = lerp(y, y - (fit[i] - (c[0] + c[1] * isB[i])), k);
      const dot = s("circle", { cx: sx(i), cy: sy(yy), r: 7, class: "run-" + cfg.toLowerCase() }, svg);
      hover(dot, () => `<b>run ${i + 1}: ${cfg === "A" ? escH(R.a_label) : escH(R.b_label)}</b><br>${fmt(y, 4)} ms per round${k > 0.5 ? `<br>after removing drift: ${fmt(yy, 4)}` : ""}`);
    });
    // legend + estimates
    const lx = W - r + 18;
    s("circle", { cx: lx, cy: t + 8, r: 6, class: "run-a" }, svg);
    s("text", { x: lx + 12, y: t + 12, class: "t-small", text: "A: " + R.a_label }, svg);
    s("circle", { cx: lx, cy: t + 28, r: 6, class: "run-b" }, svg);
    s("text", { x: lx + 12, y: t + 32, class: "t-small", text: "B: " + R.b_label }, svg);
    const ci = (e, se, q) => `${pct(e, 2)}  [${pct(e - q * se, 2)}, ${pct(e + q * se, 2)}]`;
    const box = s("text", { x: lx - 6, y: t + 72, class: "t-small" }, svg);
    s("tspan", { x: lx - 6, dy: 0, class: "t-strong", text: "B vs A, 95% interval" }, box);
    s("tspan", { x: lx - 6, dy: 20, text: "two-sample (ignores drift):" }, box);
    s("tspan", { x: lx - 6, dy: 16, class: k < 0.5 ? "t-strong" : "", text: ci(estN, seN, tq[6]) }, box);
    s("tspan", { x: lx - 6, dy: 22, text: "drift fit (linear + quadratic):" }, box);
    s("tspan", { x: lx - 6, dy: 16, class: k >= 0.5 ? "t-strong" : "", text: ci(estFit, seFit, tq[4]) }, box);
    s("tspan", { x: lx - 6, dy: 22, text: `residual sd ${fmt((100 * Math.sqrt(s2)) / c[0], 3)}% per run` }, box);
  }
  function go(m) {
    mode = m;
    const from = k, to = m === "fit" ? 1 : 0;
    stop();
    stop = tween(700, (u) => { k = lerp(from, to, u); draw(); });
  }
  draw();
};
