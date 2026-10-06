// Figure: KL vs target bytes per round, step by step from the release to the final model.
FIGS.quality = (fig) => {
  const P = D.quality_path;
  const W = 760, H = 400, l = 70, r = 24, t = 20, b = 50;
  const x0 = 11.3, x1 = 14.7, y0 = 0.036, y1 = 0.072;
  const sx = linScale(x0, x1, l, W - r), sy = linScale(y0, y1, H - b, t);
  const svg = svgRoot(W, H, "KL versus bytes per round");
  const list = h("ol", { class: "qp-list" });
  mount(fig, svg, list);
  for (const v of [11.5, 12, 12.5, 13, 13.5, 14, 14.5]) {
    s("line", { x1: sx(v), x2: sx(v), y1: t, y2: H - b, class: "grid" }, svg);
    s("text", { x: sx(v), y: H - b + 18, class: "t-axis", "text-anchor": "middle", text: fmt(v, 1) }, svg);
  }
  for (const v of [0.04, 0.05, 0.06, 0.07]) {
    s("line", { x1: l, x2: W - r, y1: sy(v), y2: sy(v), class: "grid" }, svg);
    s("text", { x: l - 8, y: sy(v) + 4, class: "t-axis", "text-anchor": "end", text: fmt(v, 2) }, svg);
  }
  s("text", { x: (l + W - r) / 2, y: H - 10, class: "t-axis", "text-anchor": "middle", text: "target weight bytes read per verify round (GB)  ← fewer bytes" }, svg);
  s("text", { x: 16, y: (t + H - b) / 2, class: "t-axis", "text-anchor": "middle", transform: `rotate(-90 16 ${(t + H - b) / 2})`, text: "KL₆₄ to FP32  (lower is better)" }, svg);
  const rel = P[0];
  s("rect", { x: l, y: sy(rel.kl), width: W - l - r, height: sy(y0) - sy(rel.kl), class: "ok-zone" }, svg);
  s("line", { x1: l, x2: W - r, y1: sy(rel.kl), y2: sy(rel.kl), class: "bar-line" }, svg);
  s("text", { x: l + 8, y: sy(rel.kl) - 7, class: "t-small", text: "the release's KL: everything below this line ships" }, svg);
  s("marker", { id: "qa-arr", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, s("defs", {}, svg))
    .append(s("path", { d: "M0,0 L10,5 L0,10 z", class: "c-ink" }));
  const pts = [];
  P.forEach((p, i) => {
    const last = pts[pts.length - 1];
    if (last && p.gb === last.gb && Math.abs(p.kl - last.kl) < 0.0005) {
      Object.assign(last, { label: last.label + "; " + p.label, kl: p.kl, top1: p.top1 || last.top1 });
      return;
    }
    pts.push({ ...p, n: pts.length + 1 });
  });
  const segs = [];
  for (let i = 0; i + 1 < pts.length; i++) {
    const a = pts[i], c = pts[i + 1];
    const ax = sx(a.gb), ay = sy(a.kl), bx = sx(c.gb), by = sy(c.kl), d = Math.hypot(bx - ax, by - ay) || 1, k = 12 / d;
    segs.push(s("line", { x1: ax + (bx - ax) * k, y1: ay + (by - ay) * k, x2: bx - (bx - ax) * k, y2: by - (by - ay) * k, class: "path", "marker-end": "url(#qa-arr)", opacity: 0 }, svg));
  }
  const dots = pts.map((p, i) => {
    const g = s("g", { class: "qp", opacity: i ? 0 : 1 }, svg);
    s("circle", { cx: sx(p.gb), cy: sy(p.kl), r: 10, class: i === 0 ? "c-gray" : i === pts.length - 1 ? "c-acc3" : "c-acc2" }, g);
    s("text", { x: sx(p.gb), y: sy(p.kl) + 4, class: "t-num", "text-anchor": "middle", text: p.n }, g);
    hover(g, () => `<b>${p.n}. ${escH(p.label)}</b><br>${fmt(p.gb, 2)} GB per round<br>KL₆₄ ${fmt(p.kl, 4)}${p.top1 ? "<br>top-1 " + fmt(p.top1, 4) : ""}`);
    const li = h("li", {}, h("b", {}, p.label), h("span", { class: "mut" }, ` — ${fmt(p.gb, 2)} GB, KL ${fmt(p.kl, 4)}`));
    li.addEventListener("mouseenter", () => g.classList.add("hl"));
    li.addEventListener("mouseleave", () => g.classList.remove("hl"));
    list.append(li);
    return g;
  });
  onVisible(fig, () => {
    pts.forEach((p, i) => {
      if (!i) return;
      setTimeout(() => {
        segs[i - 1].setAttribute("opacity", 1);
        dots[i].setAttribute("opacity", 1);
      }, reduceMotion ? 0 : 350 * i);
    });
  });
};
