
import io
import sys

path = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\viewer\app.js"
s = io.open(path, encoding="utf-8").read()
subs = []

subs.append((
    '''const _vp = new THREE.Matrix4();

function viewProj() {
  return _vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
}''',
    '''const _vp = new THREE.Matrix4();

function viewProj() {
  return _vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
}

/** Where the 3D viewport actually sits on the canvas, in CSS pixels.
 *
 * Standing in a recorded frame the render is letterboxed into the sub-rectangle
 * matching that camera's aspect ratio, so the whole canvas is not the image.
 * Every screen-space operation -- rectangle select, point pick, box pick --
 * has to go through this or it silently addresses the wrong pixels, which is
 * how a selection can come back empty over a cloud that is plainly there.
 */
function activeRect() {
  if (S.frameLock) return frameRect();
  return { x: 0, y: 0, w: dom.canvas.clientWidth, h: dom.canvas.clientHeight };
}

const screenX = (sx, r) => r.x + (sx * 0.5 + 0.5) * r.w;
const screenY = (sy, r) => r.y + (0.5 - sy * 0.5) * r.h;'''))

subs.append((
    '''  const m = viewProj().elements;
  const P = S.positions;
  const cw = dom.canvas.clientWidth, ch = dom.canvas.clientHeight;
  const n = Math.floor(S.N * S.drawFrac);
  const out = [], depth = [];''',
    '''  const m = viewProj().elements;
  const P = S.positions;
  const r = activeRect();
  const n = Math.floor(S.N * S.drawFrac);
  const out = [], depth = [];'''))

subs.append((
    '''    const ux = (sx * 0.5 + 0.5) * cw, uy = (0.5 - sy * 0.5) * ch;
    if (ux < ax0 || ux > ax1 || uy < ay0 || uy > ay1) continue;''',
    '''    const ux = screenX(sx, r), uy = screenY(sy, r);
    if (ux < ax0 || ux > ax1 || uy < ay0 || uy > ay1) continue;'''))

subs.append((
    '''  const m = viewProj().elements;
  const P = S.positions;
  const cw = dom.canvas.clientWidth, ch = dom.canvas.clientHeight;
  const n = Math.floor(S.N * S.drawFrac);
  let best = -1, bestD = Infinity;''',
    '''  const m = viewProj().elements;
  const P = S.positions;
  const r = activeRect();
  const n = Math.floor(S.N * S.drawFrac);
  let best = -1, bestD = Infinity;'''))

subs.append((
    '''    const dx = (sx * 0.5 + 0.5) * cw - px, dy = (0.5 - sy * 0.5) * ch - py;''',
    '''    const dx = screenX(sx, r) - px, dy = screenY(sy, r) - py;'''))

subs.append((
    '''function pickBox(x, y) {
  const r = dom.canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    (x / dom.canvas.clientWidth) * 2 - 1, -(y / dom.canvas.clientHeight) * 2 + 1);''',
    '''function pickBox(x, y) {
  const r = activeRect();
  const ndc = new THREE.Vector2(
    ((x - r.x) / r.w) * 2 - 1, -((y - r.y) / r.h) * 2 + 1);'''))

subs.append((
    '''  const m = viewProj().elements;
  const cw = dom.canvas.clientWidth, ch = dom.canvas.clientHeight;
  const withSpecies = $("cbSpecies").checked;''',
    '''  const m = viewProj().elements;
  const rect = activeRect();
  const withSpecies = $("cbSpecies").checked;'''))

subs.append((
    '''    cands.push({ i, p, px: (sx * 0.5 + 0.5) * cw, py: (0.5 - sy * 0.5) * ch, w });''',
    '''    cands.push({ i, p, px: screenX(sx, rect), py: screenY(sy, rect), w });'''))

subs.append((
    '''addEventListener("keydown", (e) => {
  if (e.target.matches("input,select,textarea")) return;
  keys.add(e.code);
  const k = e.key.toLowerCase();
  if (k === "escape") { leaveFrame(); clearSelection(); dom.help.classList.add("hidden"); }''',
    '''addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  // Escape is the way out of any state, so it must not be swallowed by whatever
  // widget happens to hold focus -- which after clicking a checkbox is a
  // checkbox. Everything else still yields to text entry.
  if (k === "escape") {
    leaveFrame(); clearSelection(); dom.help.classList.add("hidden");
    if (e.target.blur) e.target.blur();
    return;
  }
  if (e.target.matches("input,select,textarea")) return;
  keys.add(e.code);'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:240])
    s = s.replace(old, new, 1)

io.open(path, "w", encoding="utf-8").write(s)
print("viewport mapping + escape fixed")
