
import io
import sys

base = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\viewer"

def patch(fname, subs):
    path = base + "\\" + fname
    s = io.open(path, encoding="utf-8").read()
    for old, new in subs:
        if old not in s:
            sys.exit(f"[{fname}] NOT FOUND:\n" + old[:240])
        s = s.replace(old, new, 1)
    io.open(path, "w", encoding="utf-8").write(s)
    print("patched", fname)

patch("index.html", [
    ('''    <div class="ctl">
      <label>Point size <b id="psVal">0.0040</b></label>''',
     '''    <div class="ctl">
      <label>Colour by</label>
      <select id="colorMode">
        <option value="0">photo colour</option>
        <option value="1">height above ground</option>
        <option value="2">times observed</option>
        <option value="3">last seen (time in video)</option>
      </select>
    </div>
    <div class="ctl">
      <label>Point size <b id="psVal">0.0040</b></label>'''),
    ('''      <label><input id="showOwn" type="checkbox"> points it produced</label>
      <label><input id="showFrusta" type="checkbox" checked> cameras</label>''',
     '''      <label><input id="showOwn" type="checkbox"> its own depth map</label>
      <label><input id="showFrusta" type="checkbox" checked> cameras</label>
    </div>
    <div class="ctl row tight">
      <label><input id="showCreated" type="checkbox"> points it created</label>'''),
])

patch("app.js", [

    ('''  uHighlight: { value: 1.0 },
  uFade: { value: 0.0 },          // dim everything not in the selected plant
  uSelTag: { value: -1.0 },
};''',
     '''  uHighlight: { value: 1.0 },
  uFade: { value: 0.0 },          // dim everything not in the selected plant
  uSelTag: { value: -1.0 },
  uMode: { value: 0.0 },          // 0 photo, 1 height, 2 observations, 3 recency
  uGroundN: { value: new THREE.Vector3(0, 1, 0) },
  uGroundD: { value: 0.0 },
  uSpanLo: { value: 0.0 },
  uSpanHi: { value: 1.0 },
  uMaxPx: { value: 26.0 },
};'''),

    ('''    attribute float tag;
    uniform float uSize, uPx, uTint, uBright, uPalN, uHighlight, uFade, uSelTag;
    uniform sampler2D uPal;
    uniform sampler2D uConfMask;
    varying vec3 vColor;
    varying float vAlpha;
    void main() {
      vec3 c = color;
      float a = 1.0;''',
     '''    attribute float tag;
    attribute float obs;
    attribute float fidx;
    uniform float uSize, uPx, uTint, uBright, uPalN, uHighlight, uFade, uSelTag;
    uniform float uMode, uGroundD, uSpanLo, uSpanHi, uMaxPx;
    uniform vec3 uGroundN;
    uniform sampler2D uPal;
    uniform sampler2D uConfMask;
    varying vec3 vColor;
    varying float vAlpha;

    // Cheap cosine palette (Inigo Quilez). A ramp is only ever read
    // qualitatively here -- is this patch better supported than that one -- so
    // it does not need to be perceptually uniform, only monotone and legible.
    vec3 ramp(float t) {
      t = clamp(t, 0.0, 1.0);
      return 0.5 + 0.5 * cos(6.28318 * (vec3(1.0) * t + vec3(0.0, 0.33, 0.67)));
    }

    void main() {
      vec3 c = color;
      float a = 1.0;
      if (uMode > 0.5) {
        float v = 0.0;
        if (uMode < 1.5)      v = dot(position, uGroundN) + uGroundD;
        else if (uMode < 2.5) v = obs;
        else                  v = fidx;
        c = ramp((v - uSpanLo) / max(uSpanHi - uSpanLo, 1e-6));
      }'''),

    ('''      vColor = c * uBright;
      vAlpha = a;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      gl_PointSize = max(1.0, uSize * uPx / max(-mv.z, 0.001));
    }`,''',
     '''      vColor = c * uBright;
      vAlpha = a;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      // World-sized splats, but capped: standing inside the cloud, a point a
      // few centimetres away would otherwise cover a quarter of the screen and
      // the scene becomes a wall of discs.
      gl_PointSize = clamp(uSize * uPx / max(-mv.z, 0.001), 1.0, uMaxPx);
    }`,'''),

    ('''  const n = S.N = S.scene.n_points;
  S.positions = new Float32Array(buf, 0, n * 3);
  S.tags = new Int16Array(buf, n * 12, n);
  S.colors = new Uint8Array(buf, n * 14, n * 3);

  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(S.positions, 3));
  g.setAttribute("color", new THREE.BufferAttribute(S.colors, 3, true));
  g.setAttribute("tag", new THREE.BufferAttribute(S.tags, 1));''',
     '''  const n = S.N = S.scene.n_points;
  // The server names the blocks it wrote, so an older artifacts directory with
  // no provenance and a new one with it both parse from the same code.
  let off = 0;
  const take = (Type, per) => {
    const a = new Type(buf, off, n * per);
    off += n * per * Type.BYTES_PER_ELEMENT;
    return a;
  };
  for (const ch of S.scene.channels || ["pos", "tag", "color"]) {
    if (ch === "pos") S.positions = take(Float32Array, 3);
    else if (ch === "tag") S.tags = take(Int16Array, 1);
    else if (ch === "obs") S.obs = take(Uint16Array, 1);
    else if (ch === "frame") S.lastFrame = take(Uint16Array, 1);
    else if (ch === "color") S.colors = take(Uint8Array, 3);
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(S.positions, 3));
  g.setAttribute("color", new THREE.BufferAttribute(S.colors, 3, true));
  g.setAttribute("tag", new THREE.BufferAttribute(S.tags, 1));
  if (S.obs) g.setAttribute("obs", new THREE.BufferAttribute(S.obs, 1));
  if (S.lastFrame) g.setAttribute("fidx", new THREE.BufferAttribute(S.lastFrame, 1));'''),

    ('''  leaveFrame();
  setLook(c, Math.max(0.3, lo.distanceTo(hi) * 1.6));''',
     '''  leaveFrame();
  // Far enough that the instance fills a comfortable part of a 60-degree view
  // rather than filling all of it -- context is most of what you are checking.
  setLook(c, Math.max(0.25, lo.distanceTo(hi) * 2.2));'''),

    ('''  bind("psSlider", (e) => {''',
     '''  bind("colorMode", (e) => setColorMode(+e.target.value), "change");
  if (!S.scene.provenance) {
    [...$("colorMode").options].forEach((o) => {
      if (o.value === "2" || o.value === "3") {
        o.disabled = true;
        o.textContent += "  (needs --fuse average)";
      }
    });
    $("showCreated").disabled = true;
    $("showCreated").parentElement.title =
      "Rebuild with --fuse average to record which frame created each point";
  }
  bind("showCreated", refreshFrameLayers, "change");
  bind("psSlider", (e) => {'''),

    ('''function syncView() {
  cloudUniforms.uHighlight.value = $("cbHighlight").checked ? 1.0 : 0.0;
}''',
     '''function syncView() {
  cloudUniforms.uHighlight.value = $("cbHighlight").checked ? 1.0 : 0.0;
}

/** Switch what the cloud's colour encodes, and set the ramp's range to match. */
function setColorMode(mode) {
  cloudUniforms.uMode.value = mode;
  const gn = S.scene.ground;
  cloudUniforms.uGroundN.value.fromArray(gn.normal);
  cloudUniforms.uGroundD.value = gn.offset;
  if (mode === 1) {
    // 2nd..98th percentile of height would need a scan; the ground fit already
    // gives the floor, and the interesting range is the metre above it.
    cloudUniforms.uSpanLo.value = 0;
    cloudUniforms.uSpanHi.value = S.scene.framing.radius * 0.6;
    hudInfo(`colour: height, 0 to ${fmtLen(S.scene.framing.radius * 0.6)}`);
  } else if (mode === 2) {
    cloudUniforms.uSpanLo.value = 1;
    cloudUniforms.uSpanHi.value = Math.max(2, S.scene.obs_p99);
    hudInfo(`colour: observations, 1 to ${Math.round(S.scene.obs_p99)} ` +
            `(dark red = seen once, so least trustworthy)`);
  } else if (mode === 3) {
    cloudUniforms.uSpanLo.value = 0;
    cloudUniforms.uSpanHi.value = S.scene.num_frames || 1;
    hudInfo("colour: the frame that last saw each point");
  } else {
    hudInfo("");
  }
}'''),

    ('''  if ($("showOwn").checked) {
    const b = await (await fetch(`/api/frame/${f.id}/own.bin?stride=2`)).arrayBuffer();''',
     '''  // Standing at the frame's own camera, its depth map projects onto exactly
  // the same pixels as the fused cloud, so with both drawn you see nothing new.
  // Hiding the fusion is what makes the comparison legible.
  cloud.visible = !$("showOwn").checked;

  if ($("showCreated").checked) {
    const b = await (await fetch(`/api/frame/${f.id}/created.bin`)).arrayBuffer();
    const idx = new Uint32Array(b);
    setOverlay(idx, 0xffc857);
    hudInfo(`frame ${f.id} created ${idx.length.toLocaleString()} of the ` +
            `cloud's points (first observation of that voxel)`);
  }

  if ($("showOwn").checked) {
    const b = await (await fetch(`/api/frame/${f.id}/own.bin?stride=2`)).arrayBuffer();'''),
])
print("done")
