"""Build a standalone HTML page for hand-labelling which points belong to
each plant.  The labels are the ground truth for plant_seg.score_labels.

    python -m plant_seg.label_tool --artifacts ../artifacts_yard_avg --out ../plant_label.html

For every plant the page shows a region around it (wider than the
network's search region, so the true plant is inside even when the prompt
is off), two photos with the detection box, and lasso add/remove tools.
Labels are exported as cloud.npz point indices, independent of plant ids,
and the sampled region indices are written next to the page
(<artifacts>/plant_label_regions.npz) so unlabelled points count as
negatives when scoring.
"""
import argparse
import base64
import json
import os

import cv2
import numpy as np

from view_map3d import WEB, MapScene
from . import prompt as P
from .compare_view import b64, quantize
from .views import plant_views
from .yard import Yard


def photo_with_box(jpg, box, width=640):
    im = cv2.imread(jpg)
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(im, (x0, y0), (x1, y1), (255, 255, 255), 3)
    cv2.rectangle(im, (x0, y0), (x1, y1), (40, 40, 40), 1)
    im = cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA)
    return base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 82])[1]).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--region_points", type=int, default=50000)
    ap.add_argument("--seg", default="plant_seg.npz", help="network result offered as a starting point")
    args = ap.parse_args()

    s = MapScene(args.artifacts, use_seg=False)
    y = Yard(s)
    views, seeds = y.views()
    pv = plant_views(s.art, s.plants, {}, {}, y.model_hw)
    new_tags = np.load(os.path.join(args.artifacts, args.seg))["tags"][s.order]
    rng = np.random.default_rng(0)
    plants, regions = [], {}
    for i, p in enumerate(s.plants):
        pid = p["id"]
        sd = seeds.get(pid)
        old = s.member.get(pid, np.zeros(0, np.uint32)).astype(np.int64)
        new = np.flatnonzero(new_tags == i)
        parts = [q for q in (sd, y.xyz[old], y.xyz[new]) if q is not None and len(q)]
        if not parts:
            continue
        lo, hi, diag = P.crop_region(np.vstack(parts), pad=0.8)
        pool = np.flatnonzero(np.all((y.xyz >= lo) & (y.xyz <= hi), 1))
        mine = np.union1d(np.intersect1d(pool, old), np.intersect1d(pool, new))
        rest = np.setdiff1d(pool, mine)
        n_mine = min(len(mine), args.region_points * 2 // 3)
        take = np.concatenate([rng.choice(mine, n_mine, replace=False),
                               rng.choice(rest, min(len(rest), args.region_points - n_mine), replace=False)])
        take.sort()
        q, qlo, step = quantize(y.xyz[take])
        flags = (np.isin(take, old).astype(np.uint8) | (np.isin(take, new).astype(np.uint8) << 1))
        vs = sorted(pv.get(pid, []), key=lambda r: -(r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]))
        # the widest view plus the tightest one, which usually shows the plant best
        picks = vs[:1] + (vs[-1:] if len(vs) > 1 else [])
        photos = [photo_with_box(s.frame_by_id[v["frame"]]["jpg"], v["box"]) for v in picks]
        regions[f"idx_{pid}"] = s.order[take].astype(np.int32)        # cloud.npz order
        plants.append({"id": pid, "category": p["category"],
                       "species": p.get("species_common") or "", "views": len(vs),
                       "n": len(take), "lo": qlo, "step": step, "pos": b64(q),
                       "rgb": b64(y.rgb[take]), "flags": b64(flags),
                       "cloud": b64(s.order[take].astype(np.int32)), "photos": photos})
        print(f"  plant {pid}: region {len(pool):,} -> {len(take):,} shown", flush=True)
    np.savez_compressed(os.path.join(args.artifacts, "plant_label_regions.npz"), **regions)

    three = open(os.path.join(WEB, "vendor", "three.module.js"), "rb").read()
    data = {"plants": plants, "artifacts": os.path.basename(os.path.abspath(args.artifacts))}
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
<title>Plant Point Labelling</title>
<style>
:root { --bg:#f6f6f3; --panel:#fff; --ink:#1d1f1c; --muted:#6a6e66; --line:#e2e3dd; --accent:#2f6f3e;
  --sel:#e8f1ea; --canvas:#20231f; --lab:#ff3df2; --warn:#b5481f; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#141613; --panel:#1c1f1b; --ink:#e9ebe5; --muted:#9aa094; --line:#2d312b; --accent:#7fd18f; --sel:#24342a; --canvas:#0d0f0c; --warn:#f08a5d; } }
:root[data-theme="dark"] { --bg:#141613; --panel:#1c1f1b; --ink:#e9ebe5; --muted:#9aa094; --line:#2d312b; --accent:#7fd18f; --sel:#24342a; --canvas:#0d0f0c; --warn:#f08a5d; }
* { box-sizing: border-box; }
html, body { margin:0; height:100%; background:var(--bg); color:var(--ink); font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif; }
.app { display:grid; grid-template-columns:260px 1fr 340px; height:100vh; }
aside, .side { background:var(--panel); display:flex; flex-direction:column; min-height:0; }
aside { border-right:1px solid var(--line); }
.side { border-left:1px solid var(--line); overflow-y:auto; padding:12px 14px; gap:10px; }
header { padding:14px 16px 10px; border-bottom:1px solid var(--line); }
h1 { font-size:16px; margin:0 0 4px; } h2 { font-size:13px; margin:6px 0 2px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
header p, .note { margin:0; color:var(--muted); font-size:12px; }
ul { list-style:none; margin:0; padding:0; overflow-y:auto; flex:1; }
li { padding:8px 16px; border-bottom:1px solid var(--line); cursor:pointer; display:flex; justify-content:space-between; gap:8px; }
li:hover { background:var(--sel); } li.on { background:var(--sel); box-shadow:inset 3px 0 0 var(--accent); }
li .st { font-size:12px; color:var(--muted); white-space:nowrap; } li .st.done { color:var(--accent); font-weight:600; }
li .st.prog { color:var(--warn); }
.stage { position:relative; background:var(--canvas); min-width:0; }
canvas { position:absolute; inset:0; width:100%; height:100%; display:block; touch-action:none; }
#lasso { pointer-events:none; }
.hud { position:absolute; left:12px; top:10px; color:#e9ebe5; background:rgba(20,22,19,.8); border-radius:8px; padding:8px 10px; font-size:12px; }
.hud b { font-size:14px; }
.tools { display:flex; flex-wrap:wrap; gap:6px; }
button { font:inherit; border:1px solid var(--line); background:var(--bg); color:var(--ink); border-radius:7px; padding:5px 10px; cursor:pointer; }
button:hover { border-color:var(--accent); }
button.on { background:var(--accent); color:var(--panel); border-color:var(--accent); }
button.primary { background:var(--accent); color:var(--panel); border-color:var(--accent); }
label.chk { display:flex; gap:6px; align-items:center; font-size:13px; color:var(--muted); }
.side img { width:100%; border-radius:6px; display:block; }
kbd { font:11px ui-monospace,monospace; border:1px solid var(--line); border-bottom-width:2px; border-radius:4px; padding:0 4px; }
.keys { font-size:12px; color:var(--muted); line-height:1.8; }
input[type=range] { width:100%; }
@media (max-width: 900px) { .app { grid-template-columns:1fr; grid-template-rows:30vh 60vh auto; height:auto; } .stage { min-height:60vh; } }
</style>
<script type="importmap">{"imports": {"three": "__THREE__"}}</script>
</head>
<body>
<div class="app">
  <aside>
    <header><h1>Label plant points</h1><p id="progress"></p></header>
    <ul id="list"></ul>
  </aside>
  <div class="stage">
    <canvas id="c"></canvas><canvas id="lasso"></canvas>
    <div class="hud" id="hud"></div>
  </div>
  <div class="side">
    <h2>Tool</h2>
    <div class="tools" id="tool">
      <button data-v="orbit" class="on">Orbit <kbd>O</kbd></button>
      <button data-v="add">Lasso add <kbd>A</kbd></button>
      <button data-v="remove">Lasso remove <kbd>R</kbd></button>
    </div>
    <div class="tools">
      <button id="undo">Undo <kbd>Ctrl+Z</kbd></button>
      <button id="done">Mark done <kbd>D</kbd></button>
    </div>
    <h2>Start from</h2>
    <div class="tools">
      <button id="fromNew">Network's points</button><button id="fromOld">Old rule's points</button><button id="clear">Nothing</button>
    </div>
    <p class="note">Starting from an existing answer is faster, but check every edge: the point is to catch where it is wrong.</p>
    <h2>Show</h2>
    <label class="chk"><input type="checkbox" id="onlyLab"> only labelled points</label>
    <label class="chk"><input type="checkbox" id="hideLab"> only unlabelled points</label>
    <label class="chk"><input type="checkbox" id="ref"> outline network (green) / old rule (red)</label>
    <label class="chk">point size <input type="range" id="size" min="1" max="6" step="0.5" value="2.5"></label>
    <h2>Photos</h2>
    <p class="note">The white box is what the detector found. Label the plant it refers to, the whole plant, and nothing else (no ground, wall or neighbouring plant).</p>
    <div id="photos"></div>
    <h2>Save</h2>
    <button class="primary" id="save">Download plant_labels.json</button>
    <p class="note">Progress is also kept in this browser. Put the downloaded file in the artifacts folder.</p>
    <div class="keys">drag: orbit · right-drag / shift-drag: pan · scroll: zoom · double-click: re-centre</div>
  </div>
</div>
<script type="application/json" id="data">__DATA__</script>
<script type="module">
import * as THREE from "three";
const D = JSON.parse(document.getElementById("data").textContent);
const bytes = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const $ = id => document.getElementById(id);
const KEY = "plant_labels_" + D.artifacts;

// state: per plant a Uint8Array of labels over its shown points
let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { store = {}; }
const labels = {}, done = {}, undoStack = {};
for (const p of D.plants) {
  labels[p.id] = new Uint8Array(p.n);
  const s = store[p.id];
  if (s && s.n === p.n) { for (const i of s.pos) labels[p.id][i] = 1; done[p.id] = !!s.done; }
  undoStack[p.id] = [];
}
function persist() {
  const out = {};
  for (const p of D.plants) {
    const pos = []; const L = labels[p.id];
    for (let i = 0; i < L.length; i++) if (L[i]) pos.push(i);
    if (pos.length || done[p.id]) out[p.id] = {n: p.n, pos, done: !!done[p.id]};
  }
  try { localStorage.setItem(KEY, JSON.stringify(out)); } catch (e) {}
}

// three.js
const canvas = $("c"), lc = $("lasso"), lctx = lc.getContext("2d");
const renderer = new THREE.WebGLRenderer({canvas, antialias: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.001, 100); camera.up.set(0, 0, 1);
const mat = new THREE.PointsMaterial({size: 2.5, sizeAttenuation: false, vertexColors: true});
let pts = null, cur = null, tool = "orbit";
const orbit = {target: new THREE.Vector3(), r: 1, az: 0.6, el: 0.45};
function place() {
  const {target: t, r, az, el} = orbit;
  camera.position.set(t.x + r * Math.cos(el) * Math.cos(az), t.y + r * Math.cos(el) * Math.sin(az), t.z + r * Math.sin(el));
  camera.lookAt(t); camera.near = r / 500; camera.far = r * 50; camera.updateProjectionMatrix();
}
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false); camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
  lc.width = w * devicePixelRatio; lc.height = h * devicePixelRatio;
}
new ResizeObserver(() => { resize(); draw(); }).observe(canvas);
const draw = () => renderer.render(scene, camera);

const cache = new Map();
function geom(p) {
  if (cache.has(p.id)) return cache.get(p.id);
  const q = new Uint16Array(bytes(p.pos).buffer), pos = new Float32Array(p.n * 3);
  for (let i = 0; i < p.n; i++) for (let k = 0; k < 3; k++) pos[i * 3 + k] = p.lo[k] + q[i * 3 + k] * p.step[k];
  const g = {pos, rgb: bytes(p.rgb), flags: bytes(p.flags)};
  cache.set(p.id, g); return g;
}
function colours() {
  const g = geom(cur), L = labels[cur.id], n = cur.n, col = new Float32Array(n * 3);
  const only = $("onlyLab").checked, hide = $("hideLab").checked, ref = $("ref").checked;
  for (let i = 0; i < n; i++) {
    let r = g.rgb[i * 3] / 255, gg = g.rgb[i * 3 + 1] / 255, b = g.rgb[i * 3 + 2] / 255;
    if (L[i]) { r = 0.55 * r + 0.45; gg = 0.55 * gg + 0.24 * 0.45; b = 0.55 * b + 0.95 * 0.45; }
    else if (ref) {
      const f = g.flags[i];
      if (f & 2) { r *= 0.5; gg = 0.5 * gg + 0.4; b *= 0.5; } else if (f & 1) { r = 0.5 * r + 0.4; gg *= 0.5; b *= 0.5; }
    }
    if ((only && !L[i]) || (hide && L[i])) { r = 0.12; gg = 0.12; b = 0.12; }
    col[i * 3] = r; col[i * 3 + 1] = gg; col[i * 3 + 2] = b;
  }
  return col;
}
function rebuild(reframe) {
  const g = geom(cur);
  if (pts) { scene.remove(pts); pts.geometry.dispose(); }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(g.pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colours(), 3));
  pts = new THREE.Points(geo, mat); scene.add(pts);
  if (reframe) {
    const box = new THREE.Box3(), v = new THREE.Vector3();
    for (let i = 0; i < cur.n; i++) if (g.flags[i]) box.expandByPoint(v.set(g.pos[i*3], g.pos[i*3+1], g.pos[i*3+2]));
    if (box.isEmpty()) for (let i = 0; i < cur.n; i++) box.expandByPoint(v.set(g.pos[i*3], g.pos[i*3+1], g.pos[i*3+2]));
    box.getCenter(orbit.target); orbit.r = Math.max(box.getSize(v).length() * 1.3, 0.05); place();
  }
  hud(); draw();
}
function recolour() { pts.geometry.setAttribute("color", new THREE.BufferAttribute(colours(), 3)); hud(); draw(); list(); persist(); }
function hud() {
  const L = labels[cur.id]; let n = 0; for (const v of L) n += v;
  $("hud").innerHTML = `<b>#${cur.id} ${cur.category}</b>${cur.species ? " · " + cur.species : ""}<br>${n.toLocaleString()} of ${cur.n.toLocaleString()} shown points labelled · ${done[cur.id] ? "done" : "not done"}`;
}
function list() {
  let nd = 0;
  $("list").innerHTML = D.plants.map((p, i) => {
    let n = 0; for (const v of labels[p.id]) n += v;
    if (done[p.id]) nd++;
    const st = done[p.id] ? `<span class="st done">done</span>` : n ? `<span class="st prog">${n.toLocaleString()} pts</span>` : `<span class="st">–</span>`;
    return `<li data-i="${i}" class="${p === cur ? "on" : ""}"><span>#${p.id} ${p.category}</span>${st}</li>`;
  }).join("");
  $("progress").textContent = `${nd} of ${D.plants.length} plants done · aim for 8–10, mixing easy and hard ones`;
}
function select(i) {
  cur = D.plants[i];
  $("photos").innerHTML = cur.photos.map((b, k) => `<img alt="Detection photo ${k + 1}" src="data:image/jpeg;base64,${b}">`).join("");
  $("done").textContent = done[cur.id] ? "Mark not done" : "Mark done";
  list(); rebuild(true);
}
$("list").addEventListener("click", e => { const li = e.target.closest("li"); if (li) select(+li.dataset.i); });

// camera interaction + lasso
let drag = null, poly = null;
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("pointerdown", e => {
  canvas.setPointerCapture(e.pointerId);
  if (tool !== "orbit" && e.button === 0 && !e.shiftKey) { poly = [[e.offsetX, e.offsetY]]; return; }
  drag = {x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey};
});
canvas.addEventListener("pointermove", e => {
  if (poly) { poly.push([e.offsetX, e.offsetY]); drawLasso(); return; }
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) {
    const k = orbit.r / canvas.clientHeight;
    orbit.target.addScaledVector(new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0), -dx * k)
                .addScaledVector(new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1), dy * k);
  } else { orbit.az -= dx * 0.008; orbit.el = Math.min(1.5, Math.max(-0.3, orbit.el + dy * 0.008)); }
  place(); draw();
});
canvas.addEventListener("pointerup", () => {
  if (poly) { if (poly.length > 2) applyLasso(poly, tool === "add" ? 1 : 0); poly = null; drawLasso(); }
  drag = null;
});
canvas.addEventListener("wheel", e => { e.preventDefault(); orbit.r *= Math.exp(e.deltaY * 0.001); place(); draw(); }, {passive: false});
canvas.addEventListener("dblclick", () => rebuild(true));
function drawLasso() {
  lctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  lctx.clearRect(0, 0, lc.width, lc.height);
  if (!poly) return;
  lctx.beginPath(); poly.forEach(([x, y], i) => i ? lctx.lineTo(x, y) : lctx.moveTo(x, y)); lctx.closePath();
  lctx.fillStyle = tool === "add" ? "rgba(255,61,242,.15)" : "rgba(255,255,255,.12)"; lctx.fill();
  lctx.strokeStyle = tool === "add" ? "#ff3df2" : "#ffffff"; lctx.lineWidth = 1.5; lctx.setLineDash([5, 4]); lctx.stroke();
}
function inside(poly, x, y) {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
  }
  return c;
}
function applyLasso(poly, value) {
  const g = geom(cur), L = labels[cur.id], w = canvas.clientWidth, h = canvas.clientHeight, v = new THREE.Vector3();
  const only = $("onlyLab").checked, hide = $("hideLab").checked;
  const changed = [];
  for (let i = 0; i < cur.n; i++) {
    if (L[i] === value) continue;
    if ((only && !L[i]) || (hide && L[i])) continue;      // hidden points are not edited
    v.set(g.pos[i*3], g.pos[i*3+1], g.pos[i*3+2]).project(camera);
    if (v.z < -1 || v.z > 1) continue;
    const x = (v.x + 1) / 2 * w, y = (1 - v.y) / 2 * h;
    if (inside(poly, x, y)) changed.push(i);
  }
  if (!changed.length) return;
  undoStack[cur.id].push({idx: changed, before: changed.map(i => L[i])});
  for (const i of changed) L[i] = value;
  recolour();
}
function setTool(t) { tool = t; for (const b of $("tool").children) b.classList.toggle("on", b.dataset.v === t); canvas.style.cursor = t === "orbit" ? "grab" : "crosshair"; }
$("tool").addEventListener("click", e => { const b = e.target.closest("button"); if (b) setTool(b.dataset.v); });
// every edit records the indices it changed and their previous values
function undo() {
  const u = undoStack[cur.id].pop(); if (!u) return;
  u.idx.forEach((i, k) => labels[cur.id][i] = u.before[k]); recolour();
}
$("undo").onclick = undo;
function fill(bit) {
  const g = geom(cur), L = labels[cur.id], idx = [], before = [];
  for (let i = 0; i < cur.n; i++) {
    const v = bit ? ((g.flags[i] & bit) ? 1 : 0) : 0;
    if (L[i] !== v) { idx.push(i); before.push(L[i]); L[i] = v; }
  }
  if (idx.length) undoStack[cur.id].push({idx, before});
  recolour();
}
$("fromNew").onclick = () => fill(2);
$("fromOld").onclick = () => fill(1);
$("clear").onclick = () => fill(0);
$("done").onclick = () => { done[cur.id] = !done[cur.id]; $("done").textContent = done[cur.id] ? "Mark not done" : "Mark done"; recolour(); };
for (const id of ["onlyLab", "hideLab", "ref"]) $(id).addEventListener("change", () => { if (id === "onlyLab" && $("onlyLab").checked) $("hideLab").checked = false; if (id === "hideLab" && $("hideLab").checked) $("onlyLab").checked = false; recolour(); });
$("size").addEventListener("input", e => { mat.size = +e.target.value; draw(); });
window.addEventListener("keydown", e => {
  if (e.target.type === "range") return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); $("undo").onclick(); return; }
  const k = e.key.toLowerCase();
  if (k === "a") setTool("add"); else if (k === "r") setTool("remove"); else if (k === "o" || k === "escape") setTool("orbit");
  else if (k === "d") $("done").onclick();
});
$("save").onclick = () => {
  const out = {format: "plant_labels/1", artifacts: D.artifacts, plants: {}};
  for (const p of D.plants) {
    const L = labels[p.id]; if (!done[p.id] && !L.some(v => v)) continue;
    const cloud = new Int32Array(bytes(p.cloud).buffer), pos = [];
    for (let i = 0; i < L.length; i++) if (L[i]) pos.push(cloud[i]);
    out.plants[p.id] = {category: p.category, done: !!done[p.id], points: pos};
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out)], {type: "application/json"}));
  a.download = "plant_labels.json"; a.click(); URL.revokeObjectURL(a.href);
};
resize(); setTool("orbit"); select(0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
