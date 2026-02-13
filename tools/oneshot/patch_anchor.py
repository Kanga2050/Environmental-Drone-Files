
import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\viewer\app.js"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''function buildPhotoRibbon() {''',
    '''/** Pin a frame's photo into the world where the screen overlay was.
 *
 * Sized to fill that camera's frustum exactly, so from the frame's own position
 * it is pixel-for-pixel what the glued overlay showed. Depth testing is off and
 * it draws last, matching the overlay it replaces, which sat above everything.
 */
function anchorPhoto(f, opacity) {
  clearAnchorPhoto();
  if (!f || !f.has_image || opacity <= 0) return;
  const [H, W] = S.scene.meta.model_hw || [420, 518];
  const fb = frameBasis(f);
  const K = f.intrinsic;
  const d = camScale(0.55);
  const geo = new THREE.PlaneGeometry(W * d / K[0], H * d / K[4]);
  const mat = new THREE.MeshBasicMaterial({
    map: new THREE.TextureLoader().load(`/api/frame/${f.id}/image.jpg`,
                                        () => invalidate(2)),
    side: THREE.DoubleSide, transparent: true, opacity: opacity / 100,
    depthTest: false, depthWrite: false,
  });
  const quad = new THREE.Mesh(geo, mat);
  quad.name = "standPhoto";
  quad.renderOrder = 10;
  quad.applyMatrix4(fb.basis);
  quad.position.copy(camPoint(fb, 0, 0, d));
  helperGroup.add(quad);
  S.anchored = f.id;
}

function clearAnchorPhoto() {
  const q = helperGroup.getObjectByName("standPhoto");
  if (!q) return;
  helperGroup.remove(q);
  q.geometry.dispose();
  if (q.material.map) q.material.map.dispose();
  q.material.dispose();
  S.anchored = -1;
  invalidate(2);
}

function buildPhotoRibbon() {'''))

subs.append((
    '''  S.frameLock = null;
  dom.photo.style.display = "none";''',
    '''  const leaving = S.frameLock.f;
  S.frameLock = null;
  dom.photo.style.display = "none";
  // Hand the picture over to the world rather than dropping it.
  anchorPhoto(leaving, S.overlayOpacity);'''))

subs.append((
    '''function enterFrame(i) {
  const f = S.scene.frames[i];
  if (!f) return;''',
    '''function enterFrame(i) {
  const f = S.scene.frames[i];
  if (!f) return;
  clearAnchorPhoto();'''))

subs.append((
    '''  if (k === "o" && S.frameLock) setOpacity(S.overlayOpacity > 0 ? 0 : 70);''',
    '''  if (k === "o" && S.frameLock) setOpacity(S.overlayOpacity > 0 ? 0 : 70);
  else if (k === "o" && S.anchored >= 0) clearAnchorPhoto();
  else if (k === "o" && S.frameIdx >= 0 && !S.frameLock)
    anchorPhoto(S.scene.frames[S.frameIdx], S.overlayOpacity || 70);'''))

subs.append((
    '''  if (k === "r") resetView();''',
    '''  if (k === "r") { clearAnchorPhoto(); resetView(); }'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:240])
    s = s.replace(old, new, 1)

if "anchored:" not in s:
    import re
    m = re.search(r"const S = \{", s)
    if not m:
        sys.exit("S object not found")
    s = s[:m.end()] + " anchored: -1," + s[m.end():]

io.open(p, "w", encoding="utf-8").write(s)
print("app.js: photo is pinned in the world on leaving, not deleted")
