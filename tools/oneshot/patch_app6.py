
import io
import sys

path = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\viewer\app.js"
s = io.open(path, encoding="utf-8").read()
subs = []

subs.append((
    '''function projFromK(K, W, H, near, far) {''',
    '''/** A point given in the camera's own frame (x right, y down, z forward),
 *  expressed in world coordinates. */
function camPoint(fb, x, y, z) {
  return fb.pos.clone()
    .addScaledVector(fb.r0, x)
    .addScaledVector(fb.r1, y)
    .addScaledVector(fb.r2, z);
}

/** The four image corners of a frame's frustum, at depth d. */
function frustumCorners(f, d) {
  const fb = frameBasis(f);
  const K = f.intrinsic;
  const [H, W] = S.scene.meta.model_hw || [420, 518];
  return [[0, 0], [W, 0], [W, H], [0, H]].map(([u, v]) =>
    camPoint(fb, (u - K[2]) * d / K[0], (v - K[5]) * d / K[4], d));
}

function projFromK(K, W, H, near, far) {'''))

subs.append((
    '''  const segs = [];
  for (const f of S.scene.frames) {
    const { basis, pos } = frameBasis(f);
    const K = f.intrinsic, [H, W] = S.scene.meta.model_hw || [420, 518];
    const d = 0.06;
    const corners = [[0, 0], [W, 0], [W, H], [0, H]].map(([u, v]) => {
      const x = (u - K[2]) * d / K[0], y = (v - K[5]) * d / K[4];
      return new THREE.Vector3(x, y, d).applyMatrix4(basis).add(pos);
    });
    for (let i = 0; i < 4; i++) {
      segs.push(pos, corners[i]);
      segs.push(corners[i], corners[(i + 1) % 4]);
    }
  }''',
    '''  const segs = [];
  for (const f of S.scene.frames) {
    const pos = frameBasis(f).pos;
    const corners = frustumCorners(f, 0.06);
    for (let i = 0; i < 4; i++) {
      segs.push(pos, corners[i]);
      segs.push(corners[i], corners[(i + 1) % 4]);
    }
  }'''))

subs.append((
    '''  const cur = helperGroup.getObjectByName("curFrustum");
  const { basis, pos } = frameBasis(f);
  const K = f.intrinsic, [H, W] = S.scene.meta.model_hw || [420, 518];
  const d = 0.12;
  const corners = [[0, 0], [W, 0], [W, H], [0, H]].map(([u, v]) => {
    const x = (u - K[2]) * d / K[0], y = (v - K[5]) * d / K[4];
    return new THREE.Vector3(x, y, d).applyMatrix4(basis).add(pos);
  });
  const segs = [];''',
    '''  const cur = helperGroup.getObjectByName("curFrustum");
  const { pos, r0, r1, r2 } = frameBasis(f);
  const corners = frustumCorners(f, 0.12);
  const segs = [];'''))

subs.append((
    '''  const ax = helperGroup.getObjectByName("curAxes");
  const L = 0.09;
  const e = basis.elements;
  const axpts = [];
  for (let k = 0; k < 3; k++) {
    const dir = new THREE.Vector3(e[k * 4], e[k * 4 + 1], e[k * 4 + 2]).normalize();
    axpts.push(pos, pos.clone().addScaledVector(dir, L));
  }
  ax.geometry.setFromPoints(axpts);''',
    '''  const ax = helperGroup.getObjectByName("curAxes");
  const L = 0.09;
  const axpts = [];
  for (const dir of [r0, r1, r2]) {
    axpts.push(pos, pos.clone().addScaledVector(dir, L));
  }
  ax.geometry.setFromPoints(axpts);'''))

subs.append((
    '''    const quad = new THREE.Mesh(geo, mat);
    // The plane is built in the xy plane facing +z; the camera basis has y
    // pointing down, so the image needs flipping to read the right way up.
    quad.scale.y = -1;
    quad.applyMatrix4(basis);
    quad.position.copy(pos).add(
      new THREE.Vector3(0, 0, d).applyMatrix4(basis).sub(pos));
    group.add(quad);''',
    '''    const quad = new THREE.Mesh(geo, mat);
    // `basis` maps the plane's local +x/+y/+z to the camera's right/up/backward,
    // so a plane built the usual way already hangs the image the right way up.
    // It only has to be pushed out along the viewing direction.
    quad.applyMatrix4(basis);
    quad.position.copy(camPoint(frameBasis(f), 0, 0, d));
    group.add(quad);'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:260])
    s = s.replace(old, new, 1)

io.open(path, "w", encoding="utf-8").write(s)
print("frusta, axes and photo quads now use the camera's own axes")
