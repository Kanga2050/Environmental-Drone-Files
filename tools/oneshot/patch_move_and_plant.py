
import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\viewer\app.js"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''    const v = S.speed * (keys.has("ControlLeft") ? 0.25 : 1) * dt;
    if (v > 0) {
      const f = forwardVec();''',
    '''    const v = S.speed * (keys.has("ControlLeft") ? 0.25 : 1) * dt;
    if (v > 0) {
      // Any translation invalidates the pinned photo: it is a flat image that
      // only lines up from the exact point it was taken from.
      if (S.anchored >= 0 && MOVE_KEYS.some((c) => keys.has(c)))
        clearAnchorPhoto();
      const f = forwardVec();'''))

subs.append((
    '''  const step = -Math.sign(e.deltaY) * S.speed * 0.35;
  const fwd = forwardVec();
  cam.pos.addScaledVector(fwd, step);
  invalidate(2);''',
    '''  const step = -Math.sign(e.deltaY) * S.speed * 0.35;
  const fwd = forwardVec();
  cam.pos.addScaledVector(fwd, step);
  clearAnchorPhoto();
  invalidate(2);'''))

subs.append((
    '''function setLook(target, dist) {''',
    '''/** Keys that move the camera through the world, as opposed to turning it. */
const MOVE_KEYS = ["KeyW", "KeyS", "KeyA", "KeyD", "Space",
                   "ShiftLeft", "ShiftRight"];

function setLook(target, dist) {
  clearAnchorPhoto();'''))

subs.append((
    '''function flyToPlant(i) {''',
    '''/** Stand in the first recorded frame that saw this plant.
 *
 * The detail panel lists that same frame first, so this is the view the user
 * would have clicked through to anyway -- and the one where the plant's 3D box
 * can be checked against the photo it was detected in. Frames are only saved
 * every --artifact_every, so a detection's frame is normally on disk, but fall
 * back to flying to the instance when none of them is.
 */
function standInPlantFrame(i) {
  const p = S.scene.plants[i];
  for (const fid of p.frames || []) {
    const k = S.scene.frames.findIndex((f) => f.id === fid);
    if (k >= 0) {
      setFrame(k);
      enterFrame(k);
      return true;
    }
  }
  return false;
}

function flyToPlant(i) {'''))

subs.append((
    '''    el.querySelector(".head").onclick = () => {
      const i = +el.dataset.i;
      selectPlant(i);
      flyToPlant(i);
    };''',
    '''    el.querySelector(".head").onclick = () => {
      const i = +el.dataset.i;
      selectPlant(i);
      if (!standInPlantFrame(i)) flyToPlant(i);
    };'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:240])
    s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("app.js: translation clears the pinned photo; plant select stands in its frame")
