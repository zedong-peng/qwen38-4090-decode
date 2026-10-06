// Figure: where 3 bits are cheap. Measured Q4 -> Q3 cost per projection group and layer range; click cells to build
// a bit allocation and watch the KL budget.
FIGS.sens = (fig) => {
  const S = D.sens;
  if (!S) return;
  const W = 760, L = 128, T = 34, cw = (W - L - 10) / 4, ch = 52;
  const rows = S.rows, H = T + rows.length * ch + 8;
  const svg = svgRoot(W, H, "Q3 sensitivity map");
  const sel = new Set(S.cells.filter((c) => c.adopted).map((c) => c.id));
  const meter = h("div", { class: "meter" });
  const reset = button("Reset to our choice", () => { sel.clear(); S.cells.filter((c) => c.adopted).forEach((c) => sel.add(c.id)); draw(); });
  const none = button("Clear", () => { sel.clear(); draw(); });
  mount(fig, h("div", { class: "fig-controls" }, h("span", { class: "hint" }, "click cells to move them to 3 bits"), h("div", { class: "spacer" }), none, reset), svg, meter);
  const maxEff = Math.max(...S.cells.map((c) => c.mb / (c.dkl * 1000)));

  function draw() {
    clear(svg);
    ["layers 0–15", "16–31", "32–47", "48–63"].forEach((t, j) =>
      s("text", { x: L + cw * (j + 0.5), y: 20, class: "t-axis", "text-anchor": "middle", text: t }, svg));
    rows.forEach((r, i) => {
      s("text", { x: L - 10, y: T + i * ch + ch / 2 + 1, class: "t-label", "text-anchor": "end", text: r.label }, svg);
      s("text", { x: L - 10, y: T + i * ch + ch / 2 + 16, class: "t-small", "text-anchor": "end", text: r.sub }, svg);
      for (let j = 0; j < 4; j++) {
        if (!S.cells.some((c) => c.row === r.key && c.c0 <= j && j <= c.c1))
          s("rect", { x: L + j * cw + 2, y: T + i * ch + 2, width: cw - 4, height: ch - 4, class: "cell-na", rx: 4 }, svg);
      }
    });
    for (const c of S.cells) {
      const i = rows.findIndex((r) => r.key === c.row);
      const x = L + c.c0 * cw + 2, y = T + i * ch + 2, w = (c.c1 - c.c0 + 1) * cw - 4;
      const eff = c.mb / (c.dkl * 1000);
      const on = sel.has(c.id);
      const g = s("g", { class: "cell" + (on ? " on" : ""), tabindex: 0, role: "button", "aria-pressed": String(on) }, svg);
      s("rect", { x, y, width: w, height: ch - 4, rx: 4, class: "cell-bg", style: `fill-opacity:${(0.12 + 0.88 * Math.sqrt(eff / maxEff)).toFixed(3)}` }, g);
      s("text", { x: x + w / 2, y: y + 20, class: "t-cell", "text-anchor": "middle", text: `KL +${fmt(c.dkl, c.dkl < 0.001 ? 5 : 4)}` }, g);
      s("text", { x: x + w / 2, y: y + 36, class: "t-cell2", "text-anchor": "middle", text: `${c.mb} MB saved` }, g);
      if (c.adopted) s("text", { x: x + w - 8, y: y + 14, class: "t-cell2", "text-anchor": "end", text: "★" }, g);
      const toggle = () => { if (sel.has(c.id)) sel.delete(c.id); else sel.add(c.id); draw(); };
      g.addEventListener("click", toggle);
      g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
      hover(g, () => `<b>${escH(c.name)} → 3 bits</b><br>simulated KL<sub>64</sub> +${fmt(c.dkl, 4)}<br>${c.mb} MB less per round (${fmt(eff, 0)} MB per 0.001 KL)<br>${escH(c.time)}${c.adopted ? "<br>★ in the final model" : ""}`);
    }
    // meter
    let dkl = 0, mb = 0, us = 0, slow = false;
    for (const c of S.cells) if (sel.has(c.id)) { dkl += c.dkl; mb += c.mb; us += c.us || 0; slow = slow || c.slow; }
    const bud = S.budget, frac = dkl / bud;
    meter.innerHTML = `
      <div class="m-row"><span class="m-lab">simulated ΔKL<sub>64</sub></span>
        <div class="m-bar"><div class="m-fill ${frac > 1 ? "over" : ""}" style="width:${Math.min(100, (100 * frac) / 2.2).toFixed(1)}%"></div>
        <div class="m-mark" style="left:${(100 / 2.2).toFixed(1)}%"><span>release bar</span></div></div>
        <span class="m-val">+${fmt(dkl, 4)}</span></div>
      <div class="m-row"><span class="m-lab">bytes saved per round</span>
        <div class="m-bar"><div class="m-fill b" style="width:${Math.min(100, (100 * mb) / 1600).toFixed(1)}%"></div></div>
        <span class="m-val">${fmt(mb, 0)} MB</span></div>
      <div class="m-row"><span class="m-lab">kernel time saved</span>
        <div class="m-bar"><div class="m-fill t" style="width:${Math.min(100, (100 * us) / 1000).toFixed(1)}%"></div></div>
        <span class="m-val">${fmt(us / 1000, 2)} ms${slow ? "*" : ""}</span></div>
      <p class="m-note">${frac > 1
        ? "Over the budget the all-Q4 model leaves under the release's KL. Our final set is over it too in this simulation: end-to-end scale tuning and the INT8 V cache (Part II) brought the final model back to " + fmt(S.final_kl, 4) + " against the release's " + fmt(S.release_kl, 4) + "."
        : "Within the budget the all-Q4 model leaves under the release's KL."}
        ${slow ? " *GDN Q3 routes lost bandwidth on Ada: those bytes buy no time." : ""}</p>`;
  }
  draw();
};
