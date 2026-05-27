"""Build a standalone HTML page comparing plant point assignments.

For every detected plant it shows the 3D region around it, coloured by
which points the old radius rule and the Point Transformer assign to that
plant (or by the network's probability), next to a real photo with both
assignments drawn on it.  A whole-yard view shows every plant at once.

    python -m plant_seg.compare_view --artifacts ../artifacts_yard_avg \
        --overlays <dir from plant_seg.overlay> --out ../plant_seg_compare.html
"""
import argparse
import base64
import json
import os

import cv2
import numpy as np

from view_map3d import WEB, MapScene
from .yard import Yard


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def quantize(xyz):
    lo = xyz.min(0)
    step = np.maximum(xyz.max(0) - lo, 1e-6) / 65535.0
    q = np.round((xyz - lo) / step).astype(np.uint16)
    return q, lo.tolist(), step.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--overlays", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--region_points", type=int, default=25000)
    ap.add_argument("--yard_points", type=int, default=400000)
    args = ap.parse_args()

    s = MapScene(args.artifacts, use_seg=False)
    y = Yard(s)
    inv = np.empty_like(s.order)
    inv[s.order] = np.arange(len(s.order))
    seg = np.load(os.path.join(args.artifacts, "plant_seg.npz"))
    new_tags = seg["tags"][s.order]                  # scene order
    old_tags = s.tags
    reg = np.load(os.path.join(args.artifacts, "plant_seg_regions.npz"))
    ev_path = os.path.join(args.artifacts, "plant_seg_eval.json")
    ev = {}
    if os.path.exists(ev_path):
        with open(ev_path) as f:
            ev = {r["id"]: r for r in json.load(f)["plants"]}
    rng = np.random.default_rng(0)

    plants = []
    for i, p in enumerate(s.plants):
        pid = p["id"]
        if f"idx_{pid}" not in reg:
            continue
        idx = inv[reg[f"idx_{pid}"]]                  # scene order
        prob = reg[f"prob_{pid}"].astype(np.float32)
        if len(idx) > args.region_points:
            # Keep every point either method assigns, fill up with the rest.
            mine = (old_tags[idx] == i) | (new_tags[idx] == i)
            rest = np.flatnonzero(~mine)
            take = np.flatnonzero(mine)
            if len(take) > args.region_points:
                take = rng.choice(take, args.region_points, replace=False)
            room = args.region_points - len(take)
            take = np.concatenate([take, rng.choice(rest, min(room, len(rest)), replace=False)])
            idx, prob = idx[take], prob[take]
        q, lo, step = quantize(y.xyz[idx])
        flags = ((old_tags[idx] == i).astype(np.uint8)
                 | ((new_tags[idx] == i).astype(np.uint8) << 1)
                 | (((new_tags[idx] >= 0) & (new_tags[idx] != i)).astype(np.uint8) << 2))
        n_old = int((old_tags == i).sum())
        n_new = int((new_tags == i).sum())
        both = int(((old_tags == i) & (new_tags == i)).sum())
        img = os.path.join(args.overlays, f"plant_{pid:02d}.jpg")
        photo = ""
        if os.path.exists(img):
            im = cv2.imread(img)
            im = cv2.resize(im, (1400, int(im.shape[0] * 1400 / im.shape[1])),
                            interpolation=cv2.INTER_AREA)
            photo = base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode()
        e = ev.get(pid, {})
        plants.append({
            "id": pid, "category": p["category"],
            "species": p.get("species_common") or p.get("species") or "",
            "views": len(p.get("frames", [])),
            "n_old": n_old, "n_new": n_new, "both": both,
            "agree": both / max(n_old + n_new - both, 1),
            "metrics": {k: {m: e[k][m] for m in ("box_precision", "seed_recall", "ground_leak")}
                        for k in ("old", "new") if k in e},
            "lo": lo, "step": step, "n": len(idx),
            "pos": b64(q), "rgb": b64(y.rgb[idx]), "flags": b64(flags),
            "prob": b64(np.clip(prob * 255, 0, 255).astype(np.uint8)),
            "photo": photo,
        })

    # Whole yard: every assigned point (up to a cap) plus a background sample.
    assigned = np.flatnonzero((old_tags >= 0) | (new_tags >= 0))
    if len(assigned) > args.yard_points // 2:
        assigned = rng.choice(assigned, args.yard_points // 2, replace=False)
    others = np.flatnonzero((old_tags < 0) & (new_tags < 0))
    others = rng.choice(others, min(len(others), args.yard_points - len(assigned)), replace=False)
    yi = np.concatenate([assigned, others])
    q, lo, step = quantize(y.xyz[yi])
    yard = {"lo": lo, "step": step, "n": len(yi), "pos": b64(q), "rgb": b64(y.rgb[yi]),
            "old": b64(old_tags[yi].astype(np.int8)), "new": b64(new_tags[yi].astype(np.int8))}

    three = open(os.path.join(WEB, "vendor", "three.module.js"), "rb").read()
    data = {"plants": plants, "yard": yard, "n_plants": len(s.plants),
            "weights": os.path.basename(os.path.dirname(str(seg["weights"]))),
            "total_old": int((old_tags >= 0).sum()), "total_new": int((new_tags >= 0).sum())}
    html = (TEMPLATE
            .replace("__THREE__", "data:text/javascript;base64," + base64.b64encode(three).decode())
            .replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, {len(plants)} plants)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plant Assignment Comparison</title>
<style>
:root {
  --bg: #f6f6f3; --panel: #ffffff; --ink: #1d1f1c; --muted: #6a6e66; --line: #e2e3dd;
  --accent: #2f6f3e; --sel: #e8f1ea; --canvas: #20231f;
  --old: #e0483e; --new: #3fbf5a; --both: #f0c419;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #141613; --panel: #1c1f1b; --ink: #e9ebe5; --muted: #9aa094; --line: #2d312b;
    --accent: #7fd18f; --sel: #24342a; --canvas: #0d0f0c;
  }
}
:root[data-theme="dark"] {
  --bg: #141613; --panel: #1c1f1b; --ink: #e9ebe5; --muted: #9aa094; --line: #2d312b;
  --accent: #7fd18f; --sel: #24342a; --canvas: #0d0f0c;
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--ink);
  font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
.app { display: grid; grid-template-columns: 300px 1fr; height: 100vh; }
aside { border-right: 1px solid var(--line); background: var(--panel); display: flex;
  flex-direction: column; min-height: 0; }
aside header { padding: 14px 16px 10px; border-bottom: 1px solid var(--line); }
aside h1 { font-size: 16px; margin: 0 0 4px; }
aside p { margin: 0; color: var(--muted); font-size: 12px; }
.sort { display: flex; gap: 6px; padding: 8px 16px; border-bottom: 1px solid var(--line);
  font-size: 12px; color: var(--muted); align-items: center; }
.sort select { font: inherit; color: var(--ink); background: var(--bg); border: 1px solid var(--line);
  border-radius: 6px; padding: 2px 4px; }
ul { list-style: none; margin: 0; padding: 0; overflow-y: auto; flex: 1; }
li { padding: 9px 16px; border-bottom: 1px solid var(--line); cursor: pointer; }
li:hover { background: var(--sel); }
li.on { background: var(--sel); box-shadow: inset 3px 0 0 var(--accent); }
li .t { display: flex; justify-content: space-between; font-weight: 600; }
li .s { color: var(--muted); font-size: 12px; display: flex; justify-content: space-between; }
.bar { height: 4px; border-radius: 2px; background: var(--line); margin-top: 5px; overflow: hidden; }
.bar i { display: block; height: 100%; background: var(--accent); }
main { display: grid; grid-template-rows: auto 1fr auto; min-width: 0; min-height: 0; }
.tools { display: flex; flex-wrap: wrap; gap: 8px; padding: 10px 14px; border-bottom: 1px solid var(--line);
  background: var(--panel); align-items: center; }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.seg button { font: inherit; border: 0; background: transparent; color: var(--ink); padding: 5px 10px; cursor: pointer; }
.seg button.on { background: var(--accent); color: var(--panel); }
label.chk { display: inline-flex; gap: 5px; align-items: center; color: var(--muted); font-size: 13px; }
.legend { display: flex; gap: 12px; font-size: 12px; color: var(--muted); margin-left: auto; flex-wrap: wrap; }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%;
  margin-right: 5px; vertical-align: -1px; background: var(--c); }
.ramp { display: inline-block; width: 90px; height: 10px; border-radius: 3px; vertical-align: -1px;
  background: linear-gradient(90deg, #2b2f6b, #3b8bc2, #7fd18f, #f0c419, #e0483e); }
.stage { position: relative; min-height: 0; background: var(--canvas); }
canvas { display: block; width: 100%; height: 100%; touch-action: none; }
.hint { position: absolute; left: 12px; bottom: 10px; color: #c9cdc3; font-size: 12px; opacity: .8; pointer-events: none; }
.info { position: absolute; right: 12px; top: 10px; background: rgba(20,22,19,.82); color: #e9ebe5;
  border-radius: 8px; padding: 10px 12px; font-size: 12px; max-width: 320px; }
.info b { font-size: 14px; }
.info table { border-collapse: collapse; margin-top: 6px; }
.info td { padding: 1px 8px 1px 0; }
.info td:not(:first-child) { text-align: right; font-variant-numeric: tabular-nums; }
.photo { border-top: 1px solid var(--line); background: var(--panel); padding: 8px 14px; }
.photo img { width: 100%; max-height: 34vh; object-fit: contain; display: block; }
.photo .cap { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
.photo.hidden { display: none; }
@media (max-width: 760px) {
  .app { grid-template-columns: 1fr; grid-template-rows: 40vh 1fr; height: auto; min-height: 100vh; }
  aside { border-right: 0; border-bottom: 1px solid var(--line); }
  main { min-height: 80vh; }
  .stage { height: 55vh; }
  .legend { margin-left: 0; }
}
</style>
<script type="importmap">{"imports": {"three": "__THREE__"}}</script>
</head>
<body>
<div class="app">
  <aside>
    <header>
      <h1>Which points belong to which plant</h1>
      <p id="summary"></p>
    </header>
    <div class="sort">Sort
      <select id="sort">
        <option value="id">plant id</option>
        <option value="agree">agreement (low first)</option>
        <option value="n_new">new size</option>
        <option value="n_old">old size</option>
      </select>
    </div>
    <ul id="list"></ul>
  </aside>
  <main>
    <div class="tools">
      <div class="seg" id="scope"><button data-v="plant" class="on">Plant</button><button data-v="yard">Whole yard</button></div>
      <div class="seg" id="mode"><button data-v="compare" class="on">Old vs new</button><button data-v="prob">Probability</button><button data-v="new">New only</button><button data-v="old">Old only</button><button data-v="rgb">Photo colours</button></div>
      <label class="chk"><input type="checkbox" id="dim" checked> dim unassigned</label>
      <label class="chk"><input type="checkbox" id="showphoto" checked> photo</label>
      <div class="legend" id="legend"></div>
    </div>
    <div class="stage"><canvas id="c"></canvas>
      <div class="info" id="info"></div>
      <div class="hint">drag to orbit · right-drag or shift-drag to pan · scroll to zoom · double-click to re-centre</div>
    </div>
    <div class="photo" id="photo"><div class="cap">Detection photo: the white box is the 2D detection. Left, old radius rule (red); right, Point Transformer (green).</div><img id="img" alt="Plant photo with assignments overlaid"></div>
  </main>
</div>
<script type="application/json" id="data">__DATA__</script>
<script type="module">
import * as THREE from "three";
const D = JSON.parse(document.getElementById("data").textContent);
const bytes = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const u16 = s => new Uint16Array(bytes(s).buffer);
const i8 = s => new Int8Array(bytes(s).buffer);
const $ = id => document.getElementById(id);

const css = getComputedStyle(document.documentElement);
const OLD = [0.88, 0.28, 0.24], NEW = [0.25, 0.75, 0.35], BOTH = [0.94, 0.77, 0.10];
const pal = n => Array.from({length: n}, (_, i) => { const c = new THREE.Color(); c.setHSL((i * 0.618034) % 1, 0.65, 0.55); return [c.r, c.g, c.b]; });
const PAL = pal(D.n_plants);
function ramp(t) {
  const s = [[0.17,0.18,0.42],[0.23,0.55,0.76],[0.5,0.82,0.56],[0.94,0.77,0.1],[0.88,0.28,0.24]];
  const x = Math.min(Math.max(t, 0), 1) * 4, i = Math.min(Math.floor(x), 3), f = x - i;
  return s[i].map((v, k) => v + (s[i + 1][k] - v) * f);
}

$("summary").textContent = `${D.plants.length} plants · model ${D.weights} · points assigned: old ${D.total_old.toLocaleString()}, new ${D.total_new.toLocaleString()}`;

// --- three.js scene -------------------------------------------------------
const canvas = $("c");
const renderer = new THREE.WebGLRenderer({canvas, antialias: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.001, 100);
camera.up.set(0, 0, 1);
let pts = null;
const mat = new THREE.PointsMaterial({size: 2.2, sizeAttenuation: false, vertexColors: true});

const orbit = {target: new THREE.Vector3(), r: 1, az: 0.6, el: 0.45};
function placeCamera() {
  const {target: t, r, az, el} = orbit;
  camera.position.set(t.x + r * Math.cos(el) * Math.cos(az), t.y + r * Math.cos(el) * Math.sin(az), t.z + r * Math.sin(el));
  camera.lookAt(t);
  camera.near = r / 500; camera.far = r * 50; camera.updateProjectionMatrix();
}
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false); camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
}
new ResizeObserver(() => { resize(); draw(); }).observe(canvas);
let drag = null;
canvas.addEventListener("pointerdown", e => { drag = {x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey}; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener("pointerup", () => drag = null);
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("pointermove", e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) {
    const k = orbit.r / canvas.clientHeight;
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
    orbit.target.addScaledVector(right, -dx * k).addScaledVector(up, dy * k);
  } else {
    orbit.az -= dx * 0.008; orbit.el = Math.min(1.5, Math.max(-0.3, orbit.el + dy * 0.008));
  }
  placeCamera(); draw();
});
canvas.addEventListener("wheel", e => { e.preventDefault(); orbit.r *= Math.exp(e.deltaY * 0.001); placeCamera(); draw(); }, {passive: false});
canvas.addEventListener("dblclick", () => { frame(); draw(); });
function draw() { renderer.render(scene, camera); }

// --- data -> geometry -----------------------------------------------------
const cache = new Map();
function decode(key, o, extra) {
  if (cache.has(key)) return cache.get(key);
  const q = u16(o.pos), n = o.n, pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) pos[i * 3 + k] = o.lo[k] + q[i * 3 + k] * o.step[k];
  const out = {pos, rgb: bytes(o.rgb), n, ...extra(o)};
  cache.set(key, out); return out;
}
let state = {scope: "plant", mode: "compare", plant: null};

function colours(g) {
  const n = g.n, col = new Float32Array(n * 3), dim = $("dim").checked;
  const mode = state.mode;
  for (let i = 0; i < n; i++) {
    let c = null;
    const rgb = [g.rgb[i * 3] / 255, g.rgb[i * 3 + 1] / 255, g.rgb[i * 3 + 2] / 255];
    if (state.scope === "plant") {
      const f = g.flags[i], o = f & 1, nw = (f >> 1) & 1, other = (f >> 2) & 1;
      if (mode === "compare") c = o && nw ? BOTH : nw ? NEW : o ? OLD : null;
      else if (mode === "new") c = nw ? NEW : null;
      else if (mode === "old") c = o ? OLD : null;
      else if (mode === "prob") c = ramp(g.prob[i] / 255);
      if (!c && other && mode !== "rgb" && mode !== "prob") c = [0.45, 0.45, 0.62];
    } else {
      const o = g.old[i], nw = g.new[i];
      if (mode === "compare") c = o >= 0 && nw === o ? BOTH : nw >= 0 ? NEW : o >= 0 ? OLD : null;
      else if (mode === "new" || mode === "prob") c = nw >= 0 ? PAL[nw] : null;
      else if (mode === "old") c = o >= 0 ? PAL[o] : null;
    }
    if (!c) c = (dim && mode !== "rgb") ? rgb.map(v => 0.18 + v * 0.22) : rgb;
    col[i * 3] = c[0]; col[i * 3 + 1] = c[1]; col[i * 3 + 2] = c[2];
  }
  return col;
}

function current() {
  if (state.scope === "yard") return decode("yard", D.yard, o => ({old: i8(o.old), new: i8(o.new)}));
  const p = D.plants[state.plant];
  return decode("p" + p.id, p, o => ({flags: bytes(o.flags), prob: bytes(o.prob)}));
}

function frame() {
  const g = current();
  // Centre on the plant's own points when there are any, else on the region.
  let sel = [];
  if (state.scope === "plant") for (let i = 0; i < g.n; i++) if (g.flags[i] & 3) sel.push(i);
  if (sel.length < 20) sel = null;
  const box = new THREE.Box3(), v = new THREE.Vector3();
  const n = sel ? sel.length : g.n;
  for (let j = 0; j < n; j++) { const i = sel ? sel[j] : j; v.set(g.pos[i * 3], g.pos[i * 3 + 1], g.pos[i * 3 + 2]); box.expandByPoint(v); }
  box.getCenter(orbit.target);
  orbit.r = Math.max(box.getSize(v).length() * (state.scope === "yard" ? 0.7 : 1.4), 0.05);
  placeCamera();
}

function rebuild(reframe) {
  const g = current();
  if (pts) { scene.remove(pts); pts.geometry.dispose(); }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(g.pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colours(g), 3));
  pts = new THREE.Points(geo, mat);
  mat.size = state.scope === "yard" ? 1.6 : 2.6;
  scene.add(pts);
  if (reframe) frame();
  legend(); info(); draw();
}

function legend() {
  const L = $("legend"), m = state.mode;
  const sw = (c, t) => `<span style="--c:rgb(${c.map(v => Math.round(v * 255))})">${t}</span>`;
  if (m === "prob" && state.scope === "plant") L.innerHTML = `network probability <i class="ramp"></i> 0 → 1`;
  else if (m === "compare") L.innerHTML = sw(BOTH, "both") + sw(NEW, "new only") + sw(OLD, "old only") + (state.scope === "plant" ? sw([0.45,0.45,0.62], "another plant (new)") : "");
  else if (m === "rgb") L.innerHTML = "reconstructed colours";
  else L.innerHTML = state.scope === "yard" ? "one colour per plant" : sw(m === "old" ? OLD : NEW, m === "old" ? "old radius rule" : "Point Transformer");
}

const fmt = x => x == null ? "–" : (typeof x === "number" ? (Number.isInteger(x) ? x.toLocaleString() : x.toFixed(2)) : x);
function info() {
  const I = $("info");
  if (state.scope === "yard") {
    I.innerHTML = `<b>Whole yard</b><br>${D.yard.n.toLocaleString()} points shown<br>assigned old ${D.total_old.toLocaleString()} · new ${D.total_new.toLocaleString()}`;
    return;
  }
  const p = D.plants[state.plant], m = p.metrics;
  const row = (k, lab) => m.old ? `<tr><td>${lab}</td><td>${fmt(m.old[k])}</td><td>${fmt(m.new[k])}</td></tr>` : "";
  I.innerHTML = `<b>#${p.id} ${p.category}</b>${p.species ? `<br>${p.species}` : ""}<br>${p.views} detection frames
    <table><tr><td></td><td>old</td><td>new</td></tr>
    <tr><td>points</td><td>${fmt(p.n_old)}</td><td>${fmt(p.n_new)}</td></tr>
    ${row("box_precision", "inside its boxes")}${row("seed_recall", "covers lifted pts")}${row("ground_leak", "on the ground")}
    <tr><td>agreement</td><td colspan="2">${(p.agree * 100).toFixed(0)}% (IoU)</td></tr></table>`;
}

function photo() {
  const ph = $("photo"), p = state.scope === "plant" ? D.plants[state.plant] : null;
  const show = $("showphoto").checked && p && p.photo;
  ph.classList.toggle("hidden", !show);
  if (show) $("img").src = "data:image/jpeg;base64," + p.photo;
  resize();
}

function list() {
  const key = $("sort").value, items = D.plants.map((p, i) => [p, i]);
  items.sort((a, b) => key === "id" ? a[0].id - b[0].id : key === "agree" ? a[0].agree - b[0].agree : b[0][key] - a[0][key]);
  $("list").innerHTML = items.map(([p, i]) => `<li data-i="${i}" class="${i === state.plant ? "on" : ""}">
    <div class="t"><span>#${p.id} ${p.category}</span><span>${(p.agree * 100).toFixed(0)}%</span></div>
    <div class="s"><span>old ${p.n_old.toLocaleString()} → new ${p.n_new.toLocaleString()}</span><span>${p.views} views</span></div>
    <div class="bar"><i style="width:${(p.agree * 100).toFixed(0)}%"></i></div></li>`).join("");
}
$("list").addEventListener("click", e => {
  const li = e.target.closest("li"); if (!li) return;
  state.plant = +li.dataset.i; state.scope = "plant";
  for (const b of $("scope").children) b.classList.toggle("on", b.dataset.v === "plant");
  list(); photo(); rebuild(true);
});
$("sort").addEventListener("change", list);
for (const id of ["scope", "mode"]) $(id).addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  for (const x of $(id).children) x.classList.toggle("on", x === b);
  const was = state.scope; state[id] = b.dataset.v;
  if (id === "scope") photo();
  rebuild(id === "scope" && was !== state.scope);
});
$("dim").addEventListener("change", () => rebuild(false));
$("showphoto").addEventListener("change", () => { photo(); draw(); });

state.plant = 0;
list(); photo(); resize(); rebuild(true);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
