// Shared helpers for the interactive figures. Everything is vanilla JS drawing SVG; data comes from window.__DATA__.
const D = window.__DATA__;
const NS = "http://www.w3.org/2000/svg";
const FIGS = {};

function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null) continue;
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of kids) if (c != null) e.append(c);
  return e;
}

function s(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null) continue;
    if (k === "text") e.textContent = v;
    else e.setAttribute(k, v);
  }
  if (parent) parent.appendChild(e);
  return e;
}

function svgRoot(w, hgt, label) {
  return s("svg", { viewBox: `0 0 ${w} ${hgt}`, role: "img", "aria-label": label, preserveAspectRatio: "xMidYMid meet" });
}

function clear(e) { while (e.firstChild) e.removeChild(e.firstChild); return e; }
const fmt = (v, d = 1) => Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "") + fmt(Math.abs(v) * 100, d) + "%";
const escH = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// One floating tooltip for the whole page.
const tip = h("div", { class: "tip", role: "tooltip" });
document.body.append(tip);
function showTip(html, ev) {
  tip.innerHTML = html;
  tip.style.display = "block";
  const pad = 14, r = tip.getBoundingClientRect();
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + r.width > innerWidth - 8) x = Math.max(8, ev.clientX - r.width - pad);
  if (y + r.height > innerHeight - 8) y = Math.max(8, ev.clientY - r.height - pad);
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}
function hideTip() { tip.style.display = "none"; }
function hover(el, fn) {
  el.addEventListener("pointermove", (e) => { const t = fn(e); if (t) showTip(t, e); else hideTip(); });
  el.addEventListener("pointerleave", hideTip);
}
addEventListener("scroll", hideTip, { passive: true });

// Segmented control: opts = [[value, label], ...]
function seg(opts, val, onchange, label) {
  const wrap = h("div", { class: "seg", role: "group", "aria-label": label || "" });
  const btns = opts.map(([v, lab]) => {
    const b = h("button", { type: "button", "aria-pressed": String(v === val) }, lab);
    b.addEventListener("click", () => {
      btns.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      onchange(v);
    });
    wrap.append(b);
    return b;
  });
  wrap.set = (v) => btns.forEach((b, i) => b.setAttribute("aria-pressed", String(opts[i][0] === v)));
  return wrap;
}

function button(label, onclick, cls = "") {
  const b = h("button", { type: "button", class: "btn " + cls }, label);
  b.addEventListener("click", onclick);
  return b;
}

// Tween: calls fn(t in [0,1]) over ms milliseconds.
function tween(ms, fn, done) {
  if (reduceMotion || ms <= 0) { fn(1); if (done) done(); return () => {}; }
  const t0 = performance.now();
  let raf = 0, live = true;
  const finish = () => { if (!live) return; live = false; cancelAnimationFrame(raf); fn(1); if (done) done(); };
  const step = (now) => {
    if (!live) return;
    const t = clamp((now - t0) / ms, 0, 1);
    if (t >= 1) return finish();
    fn(ease(t));
    raf = requestAnimationFrame(step);
  };
  raf = requestAnimationFrame(step);
  // rAF can stall (background tabs, throttled or headless renderers): always land on the end state
  const timer = setTimeout(finish, ms + 120);
  return () => { live = false; cancelAnimationFrame(raf); clearTimeout(timer); };
}

// Run fn once when the element first scrolls into view (or at once without IntersectionObserver).
function onVisible(el, fn, threshold = 0.35) {
  if (!("IntersectionObserver" in window)) { fn(); return; }
  const io = new IntersectionObserver((ents) => {
    for (const e of ents) if (e.isIntersecting) { io.disconnect(); fn(); break; }
  }, { threshold });
  io.observe(el);
}

function mount(fig, ...nodes) {
  const cap = fig.querySelector("figcaption");
  for (const n of nodes) {
    // SVG figures keep a readable minimum width on phones and scroll sideways instead of shrinking
    fig.insertBefore(n instanceof SVGElement ? h("div", { class: "svg-wrap" }, n) : n, cap);
  }
}

function linScale(d0, d1, r0, r1) {
  const f = (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
  f.inv = (p) => d0 + ((p - r0) / (r1 - r0)) * (d1 - d0);
  return f;
}

function niceTicks(a, b, n = 5) {
  const span = b - a, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const err = step0 / mag, step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

// Pointer position in SVG user units.
function svgPoint(svg, ev) {
  const p = svg.createSVGPoint();
  p.x = ev.clientX; p.y = ev.clientY;
  return p.matrixTransform(svg.getScreenCTM().inverse());
}
