// Figure: real verify trees. Each round the drafter proposes a 15-node tree; the target scores all 16 positions in
// one pass, keeps the longest branch that matches its own greedy choices, and adds one token of its own.
FIGS.tree = (fig) => {
  const T = D.trees;
  if (!T) return;
  let qi = 0, ri = 0, timer = 0;
  const vis = (t) => (t == null ? "∅" : t.replace(/\n/g, "↵").replace(/\t/g, "⇥").replace(/ /g, "␣"));
  const tabs = seg(T.map((q, i) => [i, q.label]), 0, (i) => { qi = i; ri = q0(i); stopPlay(); draw(); }, "prompt");
  const prev = button("‹", () => { stopPlay(); step(-1); }, "icon");
  const next = button("›", () => { stopPlay(); step(1); }, "icon");
  const playB = button("▶ Play", () => (timer ? stopPlay() : startPlay()));
  const ctl = h("div", { class: "fig-controls" }, tabs, h("div", { class: "spacer" }), prev, playB, next);
  const strip = h("div", { class: "tree-strip" });
  const info = h("p", { class: "tree-info" });
  const W = 900;
  let H = 260;
  const svg = svgRoot(W, H, "verify tree");
  const ctx = h("div", { class: "tree-ctx" });
  mount(fig, ctl, strip, info, svg, ctx);

  function q0(i) {
    // start on a round that shows branching and a long accepted path
    const rs = T[i].rounds;
    let best = 0, bs = -1;
    rs.forEach((r, j) => {
      const branches = r.nodes.filter((n, k) => r.nodes.findIndex((m) => m[0] === n[0]) !== k).length;
      const sc = Math.min(r.acc.length, 9) + Math.min(branches, 4) - (j < 2 ? 5 : 0);
      if (sc > bs) { bs = sc; best = j; }
    });
    return best;
  }
  function step(d) { ri = clamp(ri + d, 0, T[qi].rounds.length - 1); draw(); }
  function startPlay() {
    playB.textContent = "❚❚ Pause";
    timer = setInterval(() => { if (ri >= T[qi].rounds.length - 1) return stopPlay(); step(1); }, 1600);
  }
  function stopPlay() { clearInterval(timer); timer = 0; playB.textContent = "▶ Play"; }

  function drawStrip() {
    clear(strip);
    const rs = T[qi].rounds;
    rs.forEach((r, j) => {
      const n = r.acc.length + 1;
      const b = h("button", { type: "button", class: "ts-bar" + (j === ri ? " on" : ""), title: `round ${j + 1}: ${n} tokens`, style: `--h:${((100 * n) / 16).toFixed(1)}%` });
      b.addEventListener("click", () => { stopPlay(); ri = j; draw(); });
      strip.append(b);
    });
  }

  function draw() {
    drawStrip();
    clear(svg);
    const q = T[qi], r = q.rounds[ri], N = r.nodes;
    const acc = new Set(r.acc);
    const nAcc = r.acc.length;
    info.innerHTML = `<b>Round ${ri + 1} of ${q.rounds.length}.</b> The drafter proposed ${N.length} tokens; the target accepted ` +
      `<b>${nAcc}</b>${nAcc === N.length ? " (all of them)" : ""} and added its own, so this round emitted <b>${nAcc + 1}</b> token${nAcc ? "s" : ""}.` +
      ` <span class="mut">Prompt mean: ${fmt(q.rounds.reduce((a, x) => a + x.acc.length + 1, 0) / q.rounds.length, 2)} tokens per round.</span>`;
    // layout: depth -> x; DFS leaf order -> y
    const kids = N.map(() => []), rootKids = [];
    N.forEach((n, i) => (n[0] === 0 ? rootKids : kids[n[0] - 1]).push(i + 1));
    const depthMax = Math.max(0, ...N.map((n) => n[1])) + 1;
    const ypos = {}; let leaf = 0;
    const lay = (id, ch) => {
      if (!ch.length) { ypos[id] = leaf++; return ypos[id]; }
      // accepted child first so the main path runs straight
      ch.sort((a, b) => (acc.has(b) - acc.has(a)) || (N[b - 1][3] - N[a - 1][3]));
      const ys = ch.map((c) => lay(c, kids[c - 1]));
      ypos[id] = ys[0];
      return ypos[id];
    };
    lay(0, rootKids);
    const rows = Math.max(1, leaf);
    const L = 100, colW = Math.min(64, (W - L - 90) / Math.max(depthMax, 1)), rowH = 40;
    H = Math.max(200, 52 + rows * rowH);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const X = (d) => L + 20 + d * colW, Y = (y) => 26 + y * rowH + rowH / 2;
    const pos = (id) => (id === 0 ? [L - 54, Y(ypos[0])] : [X(N[id - 1][1]), Y(ypos[id])]);
    // edges
    N.forEach((n, i) => {
      const [x1, y1] = pos(n[0]), [x2, y2] = pos(i + 1);
      const on = acc.has(i + 1) && (n[0] === 0 || acc.has(n[0]));
      s("path", { d: `M${x1 + (n[0] ? colW / 2 - 4 : 30)},${y1} C${x1 + colW * 0.7},${y1} ${x2 - colW * 0.6},${y2} ${x2 - colW / 2 + 4},${y2}`, class: "tr-edge" + (on ? " on" : "") }, svg);
    });
    // root
    const [rx, ry] = pos(0);
    s("rect", { x: rx - 32, y: ry - 12, width: 62, height: 24, rx: 5, class: "tr-root" }, svg);
    s("text", { x: rx - 1, y: ry + 4, class: "t-tok", "text-anchor": "middle", text: clip(vis(r.anchor), 8) }, svg);
    s("text", { x: rx - 1, y: ry - 17, class: "t-small", "text-anchor": "middle", text: "last token" }, svg);
    N.forEach((n, i) => {
      const id = i + 1, [x, y] = pos(id), on = acc.has(id);
      const g = s("g", { class: "tr-node" + (on ? " on" : "") }, svg);
      s("rect", { x: x - colW / 2 + 4, y: y - 12, width: colW - 8, height: 24, rx: 5 }, g);
      s("text", { x, y: y + 4, class: "t-tok", "text-anchor": "middle", text: clip(vis(n[2]), Math.floor((colW - 12) / 6.4)) }, g);
      hover(g, () => `<b>${escH(vis(n[2]))}</b><br>depth ${n[1] + 1} · drafter log-score ${fmt(n[3], 2)}${n[4] ? "<br>from prompt lookup" : ""}<br>${on ? "accepted: matches the target's greedy choice" : "rejected"}`);
    });
    // bonus token
    const lastId = nAcc ? r.acc[nAcc - 1] : 0;
    const [bx0, by] = pos(lastId);
    const bx = lastId ? bx0 + colW : X(0);
    s("path", { d: `M${bx0 + (lastId ? colW / 2 - 4 : 30)},${by} L${bx - colW / 2 + 4},${by}`, class: "tr-edge bonus" }, svg);
    const bg = s("g", { class: "tr-node bonus" }, svg);
    s("rect", { x: bx - colW / 2 + 4, y: by - 12, width: colW - 8, height: 24, rx: 5 }, bg);
    s("text", { x: bx, y: by + 4, class: "t-tok", "text-anchor": "middle", text: clip(vis(r.bonus), Math.floor((colW - 12) / 6.4)) }, bg);
    s("text", { x: bx, y: by + 26, class: "t-small", "text-anchor": "middle", text: "target's own" }, bg);
    hover(bg, () => `<b>${escH(vis(r.bonus))}</b><br>the target's own greedy token after the accepted branch: every round emits at least one token`);
    // depth axis
    for (let d = 0; d < depthMax; d++) s("text", { x: X(d), y: 12, class: "t-axis", "text-anchor": "middle", text: d + 1 }, svg);
    s("text", { x: L - 54, y: 12, class: "t-axis", "text-anchor": "middle", text: "depth" }, svg);
    // context
    const accText = r.acc.map((id) => N[id - 1][2]).join("");
    const before = r.ctx.length > 420 ? "…" + r.ctx.slice(-420) : r.ctx;
    ctx.innerHTML = `<span class="mut">${escH(before)}</span><span class="acc">${escH(accText)}</span><span class="bon">${escH(r.bonus || "")}</span>`;
    ctx.scrollTop = ctx.scrollHeight;
  }
  function lastTok(t) { const m = t.match(/(\s*\S+|\s+)$/); return m ? m[0] : t.slice(-6); }
  function clip(t, n) { return t.length > n ? t.slice(0, Math.max(1, n - 1)) + "…" : t; }
  ri = q0(0);
  draw();
};
