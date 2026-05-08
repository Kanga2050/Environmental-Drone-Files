
import io
import sys

path = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\viewer\app.js"
s = io.open(path, encoding="utf-8").read()
subs = []

subs.append((
    '''  // A window of thumbnails around the current frame: all 264 at once is a lot
  // of requests for a strip you scroll past.
  const half = 11;
  const a = Math.max(0, S.frameIdx - half);
  const b = Math.min(S.scene.frames.length, a + 2 * half + 1);
  dom.frameStrip.innerHTML = S.scene.frames.slice(a, b).map((fr, k) => `
    <img src="/api/frame/${fr.id}/thumb.jpg" loading="lazy" data-i="${a + k}"
         class="${a + k === S.frameIdx ? "on" : ""}" title="frame ${fr.id}">`).join("");
  dom.frameStrip.querySelectorAll("img").forEach((el) => {
    el.onclick = () => { setFrame(+el.dataset.i); enterFrame(+el.dataset.i); };
  });''',
    '''  updateFrameStrip();'''))

subs.append((
    '''async function refreshFrameLayers() {''',
    '''/** The thumbnail strip, rebuilt only when its window actually moves.
 *
 * Playback steps the frame several times a second, and re-creating twenty-three
 * <img> elements each time re-issues twenty-three requests and stalls the very
 * thing being played. Re-pointing the highlight is almost always all that is
 * needed.
 */
let stripFrom = -1;
function updateFrameStrip() {
  const half = 11;
  const a = Math.max(0, Math.min(S.frameIdx - half,
                                 S.scene.frames.length - (2 * half + 1)));
  const lo = Math.max(0, a);
  if (lo !== stripFrom) {
    stripFrom = lo;
    const hi = Math.min(S.scene.frames.length, lo + 2 * half + 1);
    dom.frameStrip.innerHTML = S.scene.frames.slice(lo, hi).map((fr, k) => `
      <img src="/api/frame/${fr.id}/thumb.jpg" loading="lazy" data-i="${lo + k}"
           title="frame ${fr.id}">`).join("");
    dom.frameStrip.querySelectorAll("img").forEach((el) => {
      el.onclick = () => { setFrame(+el.dataset.i); enterFrame(+el.dataset.i); };
    });
  }
  dom.frameStrip.querySelectorAll("img").forEach((el) => {
    el.classList.toggle("on", +el.dataset.i === S.frameIdx);
  });
}

async function refreshFrameLayers() {'''))

subs.append((
    '''function updateFrameUI() {
  const f = S.scene.frames[S.frameIdx];
  if (!f) return;''',
    '''function updateFrameUI(light = false) {
  const f = S.scene.frames[S.frameIdx];
  if (!f) return;
  if (light) { updateCurrentCamera(f); updateFrameStrip(); return; }'''))

subs.append((
    '''  updateFrameStrip();

  const cur = helperGroup.getObjectByName("curFrustum");''',
    '''  updateCurrentCamera(f);
}

function updateCurrentCamera(f) {
  const cur = helperGroup.getObjectByName("curFrustum");'''))

subs.append((
    '''  if (S.playing) {
    S.playAcc = (S.playAcc || 0) + dt * (S.rate || 6);''',
    '''  if (S.playing) {
    S.playAcc = (S.playAcc || 0) + Math.min(dt, 0.2) * (S.rate || 6);'''))

subs.append((
    '''  updateFrameUI();
  applyBuildUp();
  if (!S.playing) refreshFrameLayers();''',
    '''  updateFrameUI(S.playing);
  applyBuildUp();
  if (!S.playing) refreshFrameLayers();'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:260])
    s = s.replace(old, new, 1)

io.open(path, "w", encoding="utf-8").write(s)
print("playback no longer rebuilds the thumbnail strip")
