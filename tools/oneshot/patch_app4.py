
import io
import sys

path = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\viewer\app.js"
s = io.open(path, encoding="utf-8").read()
subs = []

subs.append((
    '''let last = performance.now();

function tick(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;

  if (!S.frameLock) {''',
    '''let last = performance.now();

// Frames still owed. Two rather than one, so a change that lands mid-frame is
// never missed, and interactions stay responsive without a permanent redraw.
let dirty = 3;
const invalidate = (n = 2) => { dirty = Math.max(dirty, n); };

// Blanket capture: every control in the sidebar goes through one of these, so
// no handler has to remember to ask for a redraw and none can be forgotten.
for (const ev of ["input", "change", "click", "pointerdown", "pointerup"]) {
  document.addEventListener(ev, () => invalidate(3), true);
}
addEventListener("resize", () => invalidate(3));

const MOVE_KEYS = ["KeyW", "KeyA", "KeyS", "KeyD", "Space",
                   "ShiftLeft", "ShiftRight"];

function tick(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;

  if (MOVE_KEYS.some((k) => keys.has(k))) invalidate(2);
  if (dirty <= 0) { requestAnimationFrame(tick); return; }
  dirty -= 1;

  if (!S.frameLock) {'''))

subs.append((
    '''  cam.yaw += dx * 0.0032;
  cam.pitch = Math.max(-1.553, Math.min(1.553, cam.pitch - dy * 0.0032));''',
    '''  cam.yaw += dx * 0.0032;
  cam.pitch = Math.max(-1.553, Math.min(1.553, cam.pitch - dy * 0.0032));
  invalidate(2);'''))

subs.append((
    '''  const step = -Math.sign(e.deltaY) * S.speed * 0.35;
  const fwd = forwardVec();
  cam.pos.addScaledVector(fwd, step);''',
    '''  const step = -Math.sign(e.deltaY) * S.speed * 0.35;
  const fwd = forwardVec();
  cam.pos.addScaledVector(fwd, step);
  invalidate(2);'''))

subs.append((
    '''  overlay.frustumCulled = false;
  world.add(overlay);
}''',
    '''  overlay.frustumCulled = false;
  world.add(overlay);
  invalidate(2);
}'''))

subs.append((
    '''    ownPoints.frustumCulled = false;
    world.add(ownPoints);''',
    '''    ownPoints.frustumCulled = false;
    world.add(ownPoints);
    invalidate(2);'''))

subs.append((
    '''  dom.photo.src = `/api/frame/${f.id}/image.jpg`;
  dom.photo.style.display = "block";''',
    '''  dom.photo.src = `/api/frame/${f.id}/image.jpg`;
  dom.photo.onload = () => invalidate(2);
  dom.photo.style.display = "block";
  invalidate(3);'''))

subs.append((
    '''  refreshUnits();
  dom.loading.classList.add("hidden");''',
    '''  refreshUnits();
  invalidate(4);
  dom.loading.classList.add("hidden");'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:240])
    s = s.replace(old, new, 1)

io.open(path, "w", encoding="utf-8").write(s)
print("render-on-demand added")
