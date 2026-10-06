// Figure: real decoding streams replayed with their recorded timing. One SSE chunk = one verify round.
FIGS.race = (fig) => {
  const R = D.race;
  if (!R) return;
  let pi = 0, speed = 1, raf = 0, t0 = 0, started = false;
  const MAXTOK = R.max_tokens || 256;

  const prompt = h("p", { class: "race-prompt" });
  const replay = button("↻ Replay", () => { reset(); play(); }, "primary");
  const tabs = seg(R.prompts.map((p, i) => [i, p.label]), 0, (i) => { pi = i; reset(); play(); }, "prompt");
  const speedSeg = seg([[1, "real time"], [0.2, "⅕ speed"]], 1, (v) => { speed = v; reset(); play(); }, "playback speed");
  const controls = h("div", { class: "fig-controls" }, tabs, h("div", { class: "spacer" }), speedSeg, replay);

  const tapes = h("div", { class: "race-tapes" });
  const axis = h("div", { class: "tape-axis" }, h("span", {}, "0"), h("span", {}, `${MAXTOK / 2} tokens`), h("span", {}, `${MAXTOK}`));
  const grid = h("div", { class: "race-grid" });
  const lanes = R.lanes.map((L) => {
    const name = h("div", { class: "tape-name" }, h("b", {}, L.title), h("span", {}, L.sub));
    const tsvg = s("svg", { class: "tape", viewBox: `0 0 ${2 * MAXTOK} 20`, preserveAspectRatio: "none" });
    const stat = h("div", { class: "tape-stat" });
    tapes.append(h("div", { class: "tape-row lane-" + L.key }, name, tsvg, stat));
    const box = h("div", { class: "race-text", tabindex: "0", "aria-live": "off" });
    grid.append(h("div", { class: "race-card lane-" + L.key },
      h("div", { class: "race-head" }, h("span", { class: "dot" }), h("b", {}, L.title)), box));
    const lane = { L, tsvg, stat, box, i: 0, tok: 0, rounds: [] };
    hover(tsvg, (ev) => {
      const p = svgPoint(tsvg, ev), x = p.x / 2;
      const r = lane.rounds.find((q) => x >= q.a && x < q.a + q.k);
      if (!r) return null;
      return `<b>${escH(L.title)}</b><br>round ${r.n}: ${r.k} token${r.k > 1 ? "s" : ""}<br><span class="tip-quote">${escH(r.txt.slice(0, 80))}</span>`;
    });
    return lane;
  });
  tapes.append(axis);
  mount(fig, controls, prompt, tapes, grid);

  function reset() {
    cancelAnimationFrame(raf);
    const P = R.prompts[pi];
    prompt.innerHTML = `<span class="mut">Spec-Bench #${P.qid}, ${escH(P.cat)}:</span> ${escH(P.prompt.length > 240 ? P.prompt.slice(0, 240) + "…" : P.prompt)}`;
    for (const ln of lanes) {
      ln.i = 0; ln.tok = 0; ln.rounds = [];
      clear(ln.tsvg); clear(ln.box);
      ln.stat.innerHTML = '<span class="big">–</span><span>tok/s</span>';
      ln.run = P.runs[ln.L.key];
    }
  }

  function frame(now) {
    const el = ((now - t0) / 1000) * speed;
    let alive = false;
    for (const ln of lanes) {
      const run = ln.run;
      if (!run) continue;
      let added = false;
      while (ln.i < run.t.length && run.t[ln.i] <= el) {
        const k = run.k[ln.i], txt = run.x[ln.i];
        const r = { n: ln.i + 1, a: ln.tok, k, txt };
        ln.rounds.push(r);
        s("rect", { x: 2 * r.a + 0.25, y: 2, width: Math.max(0.5, 2 * k - 0.5), height: 16, class: "tape-seg" + (r.n % 2 ? "" : " alt") }, ln.tsvg);
        ln.tok += k;
        const span = h("span", { class: "fresh" }, txt);
        ln.box.append(span);
        setTimeout(() => span.classList.remove("fresh"), 700 / Math.max(speed, 0.4));
        ln.i++;
        added = true;
      }
      if (added) ln.box.scrollTop = ln.box.scrollHeight;
      const done = ln.i >= run.t.length;
      if (!done) alive = true;
      if (ln.i > 1) {
        const tEnd = done ? run.t[run.t.length - 1] : el;
        const rate = (ln.tok - run.k[0]) / Math.max(1e-6, tEnd - run.t[0]);
        ln.stat.innerHTML = `<span class="big">${fmt(rate, 0)}</span><span>tok/s${done ? " · done " + fmt(tEnd, 2) + " s" : ""}</span>`;
      }
    }
    if (alive) raf = requestAnimationFrame(frame);
  }

  function play() {
    started = true;
    t0 = performance.now();
    raf = requestAnimationFrame(frame);
  }

  reset();
  onVisible(fig, () => { if (!started) play(); }, 0.25);
};
