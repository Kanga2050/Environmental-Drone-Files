
import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\viewer\app.js"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''function applyCamera() {
  if (S.frameLock) return;
  // Build a basis from gravity so "up" is the scene's up and not the world's Y.
  const fwd = new THREE.Vector3();
  const right = new THREE.Vector3();
  const ref = Math.abs(UP.y) > 0.9 ? new THREE.Vector3(0, 0, 1)
                                   : new THREE.Vector3(0, 1, 0);
  const e0 = new THREE.Vector3().crossVectors(ref, UP).normalize();
  const e1 = new THREE.Vector3().crossVectors(UP, e0).normalize();
  fwd.copy(e0).multiplyScalar(Math.cos(cam.yaw) * Math.cos(cam.pitch))
     .addScaledVector(e1, Math.sin(cam.yaw) * Math.cos(cam.pitch))
     .addScaledVector(UP, Math.sin(cam.pitch)).normalize();
  right.crossVectors(fwd, UP).normalize();
  const upv = new THREE.Vector3().crossVectors(right, fwd).normalize();
  const m = new THREE.Matrix4().makeBasis(right, upv, fwd.clone().negate());''',
    '''/** The gravity-aligned horizontal basis (e0, e1) that yaw is measured in. */
function gravityBasis() {
  const ref = Math.abs(UP.y) > 0.9 ? new THREE.Vector3(0, 0, 1)
                                   : new THREE.Vector3(0, 1, 0);
  const e0 = new THREE.Vector3().crossVectors(ref, UP).normalize();
  const e1 = new THREE.Vector3().crossVectors(UP, e0).normalize();
  return { e0, e1 };
}

function applyCamera() {
  if (S.frameLock) return;
  // Build a basis from gravity so "up" is the scene's up and not the world's Y.
  const fwd = new THREE.Vector3();
  const right = new THREE.Vector3();
  const { e0, e1 } = gravityBasis();
  fwd.copy(e0).multiplyScalar(Math.cos(cam.yaw) * Math.cos(cam.pitch))
     .addScaledVector(e1, Math.sin(cam.yaw) * Math.cos(cam.pitch))
     .addScaledVector(UP, Math.sin(cam.pitch)).normalize();
  right.crossVectors(fwd, UP).normalize();
  // Roll about the view axis. Without it the camera can only ever be
  // horizon-level, so stepping out of a rolled frame snapped the horizon flat.
  if (cam.roll) {
    const up0 = new THREE.Vector3().crossVectors(right, fwd).normalize();
    right.multiplyScalar(Math.cos(cam.roll))
         .addScaledVector(up0, Math.sin(cam.roll)).normalize();
  }
  const upv = new THREE.Vector3().crossVectors(right, fwd).normalize();
  const m = new THREE.Matrix4().makeBasis(right, upv, fwd.clone().negate());'''))

subs.append((
    '''function faceToward(target) {
  const d = new THREE.Vector3().subVectors(target, cam.pos).normalize();
  const ref = Math.abs(UP.y) > 0.9 ? new THREE.Vector3(0, 0, 1)
                                   : new THREE.Vector3(0, 1, 0);
  const e0 = new THREE.Vector3().crossVectors(ref, UP).normalize();
  const e1 = new THREE.Vector3().crossVectors(UP, e0).normalize();
  const h = new THREE.Vector3().copy(d).addScaledVector(UP, -d.dot(UP));
  cam.yaw = Math.atan2(h.dot(e1), h.dot(e0));
  cam.pitch = Math.asin(Math.max(-1, Math.min(1, d.dot(UP))));
}''',
    '''function faceToward(target, keepRoll = false) {
  const d = new THREE.Vector3().subVectors(target, cam.pos).normalize();
  const { e0, e1 } = gravityBasis();
  const h = new THREE.Vector3().copy(d).addScaledVector(UP, -d.dot(UP));
  cam.yaw = Math.atan2(h.dot(e1), h.dot(e0));
  cam.pitch = Math.asin(Math.max(-1, Math.min(1, d.dot(UP))));
  if (!keepRoll) cam.roll = 0;
}

/** Set yaw/pitch/roll so the fly camera reproduces `basis` exactly.
 *
 * This is what makes standing in a frame and stepping out of it continuous:
 * the recorded pose becomes a state the fly camera can actually hold, instead
 * of something approximated the moment the user touches a key.
 */
function adoptBasis(basis) {
  const right = new THREE.Vector3().setFromMatrixColumn(basis, 0);
  const fwd = new THREE.Vector3().setFromMatrixColumn(basis, 2).negate();
  const { e0, e1 } = gravityBasis();
  const h = new THREE.Vector3().copy(fwd).addScaledVector(UP, -fwd.dot(UP));
  cam.yaw = Math.atan2(h.dot(e1), h.dot(e0));
  cam.pitch = Math.asin(Math.max(-1, Math.min(1, fwd.dot(UP))));
  // Roll is the angle from the horizon-level right vector to the real one,
  // measured in the plane perpendicular to the view axis.
  const right0 = new THREE.Vector3().crossVectors(fwd, UP).normalize();
  const up0 = new THREE.Vector3().crossVectors(right0, fwd).normalize();
  cam.roll = Math.atan2(right.dot(up0), right.dot(right0));
}'''))

subs.append((
    '''  cam.pos.copy(pos);
  const fwd = new THREE.Vector3(0, 0, 1).transformDirection(basis);
  faceToward(pos.clone().add(fwd));''',
    '''  cam.pos.copy(pos);
  // Adopt the recorded pose outright. The old code aimed at the basis's +Z
  // column, which is the direction the camera looks *away* from, so leaving
  // the frame spun the view through 180 degrees.
  adoptBasis(basis);'''))

subs.append((
    '''function leaveFrame() {
  if (!S.frameLock) return;
  S.frameLock = null;
  dom.photo.style.display = "none";
  camera.updateProjectionMatrix();''',
    '''function leaveFrame() {
  if (!S.frameLock) return;
  // cam.yaw/pitch/roll already hold this frame's exact pose, so the handover is
  // silent: same view, now flyable. Only the projection changes, from the
  // recorded camera's intrinsics back to the window's.
  S.frameLock = null;
  dom.photo.style.display = "none";
  camera.updateProjectionMatrix();'''))

subs.append((
    '''  if (k === "r") resetView();''',
    '''  if (k === "r") resetView();
  if (k === "l" && !S.frameLock && cam.roll) {   // level the horizon
    cam.roll = 0; applyCamera(); invalidate(2);
  }'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:260])
    s = s.replace(old, new, 1)

if "roll: 0" not in s:
    import re
    m = re.search(r"const cam = \{([^}]*)\}", s)
    if not m:
        sys.exit("cam object not found")
    s = s[:m.start(1)] + m.group(1).rstrip().rstrip(",") + ", roll: 0" + s[m.end(1):]

io.open(p, "w", encoding="utf-8").write(s)
print("app.js: adoptBasis + roll; enterFrame no longer aims backwards")
