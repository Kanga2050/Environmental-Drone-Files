
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
    ('''    <div class="ctl row tight">
      <button id="framePrev">◀</button>
      <button id="frameEnter">Stand here (G)</button>
      <button id="frameNext">▶</button>
    </div>''',
     '''    <div class="ctl row tight">
      <button id="framePrev" title="Previous frame">◀</button>
      <button id="framePlay" title="Step through the frames (K)">▶ Play</button>
      <button id="frameNext" title="Next frame">▶</button>
    </div>
    <div class="ctl row tight">
      <button id="frameEnter" title="Put the camera exactly where this one was">
        Stand here (G)</button>
    </div>
    <div class="ctl">
      <label>Playback <b id="rateVal">6</b> frames/s</label>
      <input id="rateSlider" type="range" min="1" max="30" step="1" value="6">
    </div>
    <div class="ctl row tight">
      <label><input id="buildUp" type="checkbox"> reveal the map as it was built</label>
    </div>'''),
    ('''      <label><input id="showFrusta" type="checkbox" checked> cameras</label>''',
     '''      <label><input id="showFrusta" type="checkbox" checked> cameras</label>
    </div>
    <div class="ctl row tight">
      <label><input id="showAxes" type="checkbox" checked> orientation axes</label>
      <label><input id="showPhotos" type="checkbox"> photos on cameras</label>'''),
])

patch("app.js", [

    ('''    attribute float tag;
    attribute float obs;
    attribute float fidx;''',
     '''    attribute float tag;
    attribute float obs;
    attribute float fidx;
    attribute float ffirst;'''),

    ('''    uniform float uMode, uGroundD, uSpanLo, uSpanHi, uMaxPx;''',
     '''    uniform float uMode, uGroundD, uSpanLo, uSpanHi, uMaxPx, uBuildTo;'''),

    ('''    void main() {
      vec3 c = color;
      float a = 1.0;
      if (uMode > 0.5) {''',
     '''    void main() {
      vec3 c = color;
      float a = 1.0;
      // Replaying the build: a point exists from the frame that first saw it.
      if (uBuildTo >= 0.0 && ffirst > uBuildTo) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);   // off-clip, never rasterised
        gl_PointSize = 0.0;
        vColor = c; vAlpha = 0.0;
        return;
      }
      if (uMode > 0.5) {'''),

    ('''  uSpanHi: { value: 1.0 },
  uMaxPx: { value: 26.0 },
};''',
     '''  uSpanHi: { value: 1.0 },
  uMaxPx: { value: 26.0 },
  uBuildTo: { value: -1.0 },
};'''),

    ('''    else if (ch === "frame") S.lastFrame = take(Uint16Array, 1);''',
     '''    else if (ch === "frame") S.lastFrame = take(Uint16Array, 1);
    else if (ch === "first") S.firstFrame = take(Uint16Array, 1);'''),

    ('''  if (S.lastFrame) g.setAttribute("fidx", new THREE.BufferAttribute(S.lastFrame, 1));''',
     '''  if (S.lastFrame) g.setAttribute("fidx", new THREE.BufferAttribute(S.lastFrame, 1));
  if (S.firstFrame)
    g.setAttribute("ffirst", new THREE.BufferAttribute(S.firstFrame, 1));'''),

    ('''  const cur = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(new Array(16).fill(0)
      .map(() => new THREE.Vector3())),
    new THREE.LineBasicMaterial({ color: 0xffc857 }));
  cur.name = "curFrustum"; cur.visible = false;
  helperGroup.add(cur);''',
     '''  const cur = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(new Array(16).fill(0)
      .map(() => new THREE.Vector3())),
    new THREE.LineBasicMaterial({ color: 0xffc857 }));
  cur.name = "curFrustum"; cur.visible = false;
  helperGroup.add(cur);

  // Orientation triad for the current camera, in the usual convention: red is
  // the camera's right, green its down, blue the way it is looking. A frustum
  // alone is symmetric enough that you cannot tell a camera from its mirror.
  const axes = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(new Array(6).fill(0)
      .map(() => new THREE.Vector3())),
    new THREE.LineBasicMaterial({ vertexColors: true }));
  axes.geometry.setAttribute("color", new THREE.Float32BufferAttribute([
    1, 0.25, 0.25, 1, 0.25, 0.25,
    0.35, 1, 0.35, 0.35, 1, 0.35,
    0.4, 0.55, 1, 0.4, 0.55, 1], 3));
  axes.name = "curAxes"; axes.visible = false;
  helperGroup.add(axes);

  // One quad per sampled camera, textured with that frame's photo. This is the
  // "see where it flew" view: the path as a ribbon of what was actually seen,
  // rather than a line. Sampled, because 264 textures is a lot of uploads for
  // something you read at a glance.
  const photoGroup = new THREE.Group();
  photoGroup.name = "photos";
  photoGroup.visible = false;
  helperGroup.add(photoGroup);'''),

    ('''  cur.geometry.setFromPoints(segs);
  cur.geometry.computeBoundingSphere();
  cur.visible = $("showFrusta").checked;''',
     '''  cur.geometry.setFromPoints(segs);
  cur.geometry.computeBoundingSphere();
  cur.visible = $("showFrusta").checked;

  const ax = helperGroup.getObjectByName("curAxes");
  const L = 0.09;
  const e = basis.elements;
  const axpts = [];
  for (let k = 0; k < 3; k++) {
    const dir = new THREE.Vector3(e[k * 4], e[k * 4 + 1], e[k * 4 + 2]).normalize();
    axpts.push(pos, pos.clone().addScaledVector(dir, L));
  }
  ax.geometry.setFromPoints(axpts);
  ax.geometry.computeBoundingSphere();
  ax.visible = $("showAxes").checked;'''),

    ('''  bind("showFrusta", () => {
    helperGroup.getObjectByName("frusta").visible = $("showFrusta").checked;
    helperGroup.getObjectByName("curFrustum").visible = $("showFrusta").checked;
  }, "change");''',
     '''  bind("showFrusta", () => {
    helperGroup.getObjectByName("frusta").visible = $("showFrusta").checked;
    helperGroup.getObjectByName("curFrustum").visible = $("showFrusta").checked;
  }, "change");
  bind("showAxes", () => {
    helperGroup.getObjectByName("curAxes").visible = $("showAxes").checked;
  }, "change");
  bind("showPhotos", () => {
    if ($("showPhotos").checked) buildPhotoRibbon();
    helperGroup.getObjectByName("photos").visible = $("showPhotos").checked;
  }, "change");
  bind("rateSlider", (e) => {
    S.rate = +e.target.value;
    $("rateVal").textContent = e.target.value;
  });
  bind("buildUp", () => {
    applyBuildUp();
    if (!S.scene.provenance) {
      hudInfo("rebuild with --fuse average to replay how the map was built");
    }
  }, "change");
  $("framePlay").onclick = togglePlay;
  if (!S.scene.provenance) $("buildUp").disabled = true;'''),

    ('''function setFrame(i) {
  S.frameIdx = Math.max(0, Math.min(S.scene.frames.length - 1, i));
  $("frameSlider").value = S.frameIdx;
  updateFrameUI();
  refreshFrameLayers();
  if (S.frameLock) enterFrame(S.frameIdx);
}''',
     '''function setFrame(i) {
  S.frameIdx = Math.max(0, Math.min(S.scene.frames.length - 1, i));
  $("frameSlider").value = S.frameIdx;
  updateFrameUI();
  applyBuildUp();
  if (!S.playing) refreshFrameLayers();
  if (S.frameLock) enterFrame(S.frameIdx);
}

/** Show only the points that existed by the current frame. */
function applyBuildUp() {
  const on = $("buildUp").checked && S.scene.provenance;
  cloudUniforms.uBuildTo.value = on ? S.scene.frames[S.frameIdx].id : -1;
}

function togglePlay() {
  S.playing = !S.playing;
  $("framePlay").textContent = S.playing ? "\\u275a\\u275a Pause" : "\\u25b6 Play";
  S.playAcc = 0;
  if (S.playing) {
    // Stepping is about watching the reconstruction, so leave the locked
    // photo-overlay view -- otherwise every step reloads a full-size JPEG.
    leaveFrame();
    hudInfo("stepping through the recorded frames");
  }
}

/** Textured quads at a sample of the recorded cameras. */
function buildPhotoRibbon() {
  const group = helperGroup.getObjectByName("photos");
  if (group.children.length) return;
  const frames = S.scene.frames;
  const step = Math.max(1, Math.ceil(frames.length / 40));
  const loader = new THREE.TextureLoader();
  const [H, W] = S.scene.meta.model_hw || [420, 518];
  for (let i = 0; i < frames.length; i += step) {
    const f = frames[i];
    if (!f.has_image) continue;
    const { basis, pos } = frameBasis(f);
    const K = f.intrinsic;
    const d = 0.10;
    const w = W * d / K[0], h = H * d / K[4];
    const geo = new THREE.PlaneGeometry(w, h);
    const mat = new THREE.MeshBasicMaterial({
      map: loader.load(`/api/frame/${f.id}/thumb.jpg`, () => invalidate(2)),
      side: THREE.DoubleSide, transparent: true, opacity: 0.92,
    });
    const quad = new THREE.Mesh(geo, mat);
    // The plane is built in the xy plane facing +z; the camera basis has y
    // pointing down, so the image needs flipping to read the right way up.
    quad.scale.y = -1;
    quad.applyMatrix4(basis);
    quad.position.copy(pos).add(
      new THREE.Vector3(0, 0, d).applyMatrix4(basis).sub(pos));
    group.add(quad);
  }
  hudInfo(`${group.children.length} camera photos placed along the path`);
}'''),

    ('''  if (MOVE_KEYS.some((k) => keys.has(k))) invalidate(2);''',
     '''  if (MOVE_KEYS.some((k) => keys.has(k))) invalidate(2);
  if (S.playing) {
    S.playAcc = (S.playAcc || 0) + dt * (S.rate || 6);
    if (S.playAcc >= 1) {
      const step = Math.floor(S.playAcc);
      S.playAcc -= step;
      const next = S.frameIdx + step;
      if (next >= S.scene.frames.length) { setFrame(0); }
      else setFrame(next);
    }
    invalidate(2);
  }'''),

    ('''  if (k === "m") setTool(S.tool === "measure" ? "fly" : "measure");''',
     '''  if (k === "m") setTool(S.tool === "measure" ? "fly" : "measure");
  if (k === "k") togglePlay();
  if (k === "," ) setFrame(S.frameIdx - 1);
  if (k === "." ) setFrame(S.frameIdx + 1);'''),

    ('''  traj.name = "traj"; traj.visible = false;''',
     '''  traj.name = "traj"; traj.visible = true;'''),
])

patch("index.html", [
    ('''      <label><input id="cbTraj" type="checkbox"> flight path</label>''',
     '''      <label><input id="cbTraj" type="checkbox" checked> flight path</label>'''),
])
print("done")
