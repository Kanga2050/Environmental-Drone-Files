

import * as THREE from "three";

const $ = (id) => document.getElementById(id);
const v3 = (a) => new THREE.Vector3().fromArray(a);
const dom = {
  stage: $("stage"), canvas: $("gl"), labels: $("labels"), svg: $("overlaySvg"),
  photo: $("framePhoto"), rubber: $("rubber"), crosshair: $("crosshair"),
  hudMode: $("hudMode"), hudSpeed: $("hudSpeed"), hudInfo: $("hudInfo"),
  help: $("help"), loading: $("loading"), loadBar: $("loading").querySelector("i"),
  loadText: $("loading").querySelector("span"),
  plantList: $("plantList"), frameStrip: $("frameStrip"), frameInfo: $("frameInfo"),
  selBody: $("selBody"), sceneInfo: $("sceneInfo"),
};

const S = { anchored: -1,
  scene: null,
  N: 0,
  positions: null, colors: null, tags: null,
  tool: "fly",
  selectedPlant: -1,
  selection: null,
  frameIdx: 0,
  frameLock: null,
  overlayOpacity: 0,
  measure: [],
  speed: 1.0,
  conf: 0.15,
  drawFrac: 1.0,
  pendingSeen: 0,
};

const renderer = new THREE.WebGLRenderer({
  canvas: dom.canvas, antialias: false, powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
renderer.autoClear = false;
const BG = new THREE.Color(0xffffff);

const world = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 500);
camera.matrixAutoUpdate = false;

const cam = { pos: new THREE.Vector3(), yaw: 0, pitch: 0, roll: 0};
let UP = new THREE.Vector3(0, -1, 0);

function upBasis() {
  const ref = Math.abs(UP.y) > 0.9 ? new THREE.Vector3(0, 0, 1)
                                   : new THREE.Vector3(0, 1, 0);
  const e0 = new THREE.Vector3().crossVectors(ref, UP).normalize();
  const e1 = new THREE.Vector3().crossVectors(UP, e0).normalize();
  return { e0, e1 };
}

function applyCamera() {
  if (S.frameLock) return;

  const fwd = new THREE.Vector3();
  const right = new THREE.Vector3();
  const { e0, e1 } = upBasis();
  fwd.copy(e0).multiplyScalar(Math.cos(cam.yaw) * Math.cos(cam.pitch))
     .addScaledVector(e1, Math.sin(cam.yaw) * Math.cos(cam.pitch))
     .addScaledVector(UP, Math.sin(cam.pitch)).normalize();
  right.crossVectors(fwd, UP).normalize();

  if (cam.roll) {
    const up0 = new THREE.Vector3().crossVectors(right, fwd).normalize();
    right.multiplyScalar(Math.cos(cam.roll))
         .addScaledVector(up0, Math.sin(cam.roll)).normalize();
  }
  const upv = new THREE.Vector3().crossVectors(right, fwd).normalize();
  const m = new THREE.Matrix4().makeBasis(right, upv, fwd.clone().negate());
  m.setPosition(cam.pos);
  camera.matrixWorld.copy(m);
  camera.matrixWorldInverse.copy(m).invert();
  camera.matrix.copy(m);
}

function setLook(target, dist) {
  dropPin();

  const dir = new THREE.Vector3().subVectors(cam.pos, target);
  if (dir.lengthSq() < 1e-9) dir.set(1, 0, 1);
  dir.normalize();
  cam.pos.copy(target).addScaledVector(dir, dist);
  faceToward(target);
}

function faceToward(target, keepRoll = false) {
  const d = new THREE.Vector3().subVectors(target, cam.pos).normalize();
  const { e0, e1 } = upBasis();
  const h = new THREE.Vector3().copy(d).addScaledVector(UP, -d.dot(UP));
  cam.yaw = Math.atan2(h.dot(e1), h.dot(e0));
  cam.pitch = Math.asin(Math.max(-1, Math.min(1, d.dot(UP))));
  if (!keepRoll) cam.roll = 0;
}

function adoptPose(basis) {
  const right = new THREE.Vector3().setFromMatrixColumn(basis, 0);
  const fwd = new THREE.Vector3().setFromMatrixColumn(basis, 2).negate();
  const { e0, e1 } = upBasis();
  const h = new THREE.Vector3().copy(fwd).addScaledVector(UP, -fwd.dot(UP));
  cam.yaw = Math.atan2(h.dot(e1), h.dot(e0));
  cam.pitch = Math.asin(Math.max(-1, Math.min(1, fwd.dot(UP))));

  const right0 = new THREE.Vector3().crossVectors(fwd, UP).normalize();
  const up0 = new THREE.Vector3().crossVectors(right0, fwd).normalize();
  cam.roll = Math.atan2(right.dot(up0), right.dot(right0));
}

const cloudUniforms = {
  uSize: { value: 0.04 },
  uPx: { value: 500.0 },
  uTint: { value: 1.0 },
  uBright: { value: 1.0 },
  uPal: { value: null },
  uPalN: { value: 1.0 },
  uConfMask: { value: null },
  uHighlight: { value: 1.0 },
  uFade: { value: 0.0 },
  uSelTag: { value: -1.0 },
  uMode: { value: 0.0 },
  uGroundN: { value: new THREE.Vector3(0, 1, 0) },
  uGroundD: { value: 0.0 },
  uSpanLo: { value: 0.0 },
  uSpanHi: { value: 1.0 },
  uMaxPx: { value: 26.0 },
  uBuildTo: { value: -1.0 },
};

const cloudMat = new THREE.ShaderMaterial({
  uniforms: cloudUniforms,
  vertexShader: `
    attribute float tag;
    attribute float obs;
    attribute float fidx;
    attribute float ffirst;
    uniform float uSize, uPx, uTint, uBright, uPalN, uHighlight, uFade, uSelTag;
    uniform float uMode, uGroundD, uSpanLo, uSpanHi, uMaxPx, uBuildTo;
    uniform vec3 uGroundN;
    uniform sampler2D uPal;
    uniform sampler2D uConfMask;
    varying vec3 vColor;
    varying float vAlpha;
    vec3 ramp(float t) {
      t = clamp(t, 0.0, 1.0);
      return 0.5 + 0.5 * cos(6.28318 * (vec3(1.0) * t + vec3(0.0, 0.33, 0.67)));
    }
    void main() {
      vec3 c = color;
      float a = 1.0;
      if (uBuildTo >= 0.0 && ffirst > uBuildTo) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        vColor = c; vAlpha = 0.0;
        return;
      }
      if (uMode > 0.5) {
        float v = 0.0;
        if (uMode < 1.5)      v = dot(position, uGroundN) + uGroundD;
        else if (uMode < 2.5) v = log2(obs + 1.0);
        else                  v = fidx;
        c = ramp((v - uSpanLo) / max(uSpanHi - uSpanLo, 1e-6));
      }
      if (tag >= 0.0) {
        float uu = (tag + 0.5) / uPalN;
        float vis = texture2D(uConfMask, vec2(uu, 0.5)).r;
        if (uHighlight > 0.5 && vis > 0.5) {
          c = mix(c, texture2D(uPal, vec2(uu, 0.5)).rgb, uTint);
        }
        if (uSelTag >= 0.0 && abs(tag - uSelTag) > 0.5) a = 1.0 - uFade;
      } else if (uSelTag >= 0.0) {
        a = 1.0 - uFade;
      }
      vColor = c * uBright;
      vAlpha = a;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      gl_PointSize = clamp(uSize * uPx / max(-mv.z, 0.001), 1.0, uMaxPx);
    }`,
  fragmentShader: `
    varying vec3 vColor;
    varying float vAlpha;
    void main() {
      vec2 d = gl_PointCoord - vec2(0.5);
      if (dot(d, d) > 0.25) discard;
      if (vAlpha < 0.02) discard;
      gl_FragColor = vec4(vColor, vAlpha);
    }`,
  vertexColors: true,
  transparent: true,
  depthWrite: true,
});

let cloud = null;

const overlayUniforms = {
  uSize: { value: 0.007 }, uPx: { value: 500.0 },
  uColor: { value: new THREE.Color(0x4a9eff) }, uUseOwn: { value: 0.0 },
};
const overlayMat = new THREE.ShaderMaterial({
  uniforms: overlayUniforms,
  vertexShader: `
    uniform float uSize, uPx, uUseOwn;
    uniform vec3 uColor;
    varying vec3 vColor;
    void main() {
      vColor = mix(uColor, color, uUseOwn);
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      gl_PointSize = max(1.5, uSize * uPx / max(-mv.z, 0.001));
    }`,
  fragmentShader: `
    varying vec3 vColor;
    void main() {
      vec2 d = gl_PointCoord - vec2(0.5);
      if (dot(d, d) > 0.25) discard;
      gl_FragColor = vec4(vColor, 1.0);
    }`,
  vertexColors: true,
  depthTest: true,
});
let overlay = null, ownPoints = null;

const boxGroup = new THREE.Group();
const helperGroup = new THREE.Group();
world.add(boxGroup, helperGroup);

async function boot() {
  dom.loadText.textContent = "loading scene…";
  S.scene = await (await fetch("/api/scene.json")).json();
  UP = v3(S.scene.up).normalize();

  dom.loadText.textContent = "loading points…";
  const buf = await fetchWithProgress("/api/points.bin");
  const n = S.N = S.scene.n_points;

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
    else if (ch === "first") S.firstFrame = take(Uint16Array, 1);
    else if (ch === "color") S.colors = take(Uint8Array, 3);
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(S.positions, 3));
  g.setAttribute("color", new THREE.BufferAttribute(S.colors, 3, true));
  g.setAttribute("tag", new THREE.BufferAttribute(S.tags, 1));
  if (S.obs) g.setAttribute("obs", new THREE.BufferAttribute(S.obs, 1));
  if (S.lastFrame) g.setAttribute("fidx", new THREE.BufferAttribute(S.lastFrame, 1));
  if (S.firstFrame)
    g.setAttribute("ffirst", new THREE.BufferAttribute(S.firstFrame, 1));
  g.setDrawRange(0, n);
  g.boundingSphere = new THREE.Sphere(
    v3(S.scene.bbox_min)
      .add(v3(S.scene.bbox_max)).multiplyScalar(0.5),
    v3(S.scene.bbox_max)
      .sub(v3(S.scene.bbox_min)).length());
  cloud = new THREE.Points(g, cloudMat);
  cloud.frustumCulled = false;
  world.add(cloud);

  buildPalette();
  buildBoxes();
  buildHelpers();
  buildUI();
  resetView();

  window.viewer = { S, world, camera, cam, renderer, get cloud() { return cloud; },
                    get overlay() { return overlay; }, THREE,
                    selectRect, pickPoint, applySelection, viewProj };

  setSpeed(S.speed);
  invalidate(4);
  dom.loading.classList.add("hidden");
  requestAnimationFrame(tick);
}

async function fetchWithProgress(url) {
  const res = await fetch(url);
  const total = +res.headers.get("Content-Length") || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) {
      dom.loadBar.style.width = (100 * got / total).toFixed(1) + "%";
      dom.loadText.textContent =
        `loading points… ${(got / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`;
    }
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out.buffer;
}

function buildPalette() {
  const pal = S.scene.palette;
  const n = Math.max(pal.length, 1);
  const data = new Uint8Array(n * 4);
  pal.forEach((c, i) => {
    data[i * 4] = c[0]; data[i * 4 + 1] = c[1]; data[i * 4 + 2] = c[2];
    data[i * 4 + 3] = 255;
  });
  const tex = new THREE.DataTexture(data, n, 1, THREE.RGBAFormat);
  tex.needsUpdate = true;
  tex.magFilter = tex.minFilter = THREE.NearestFilter;
  cloudUniforms.uPal.value = tex;
  cloudUniforms.uPalN.value = n;

  S.maskData = new Uint8Array(n * 4).fill(255);
  S.maskTex = new THREE.DataTexture(S.maskData, n, 1, THREE.RGBAFormat);
  S.maskTex.magFilter = S.maskTex.minFilter = THREE.NearestFilter;
  S.maskTex.needsUpdate = true;
  cloudUniforms.uConfMask.value = S.maskTex;
}

function updateConfMask() {
  S.scene.plants.forEach((p, i) => {
    const on = p.detection_score >= S.conf ? 255 : 0;
    S.maskData[i * 4] = on;
  });
  S.maskTex.needsUpdate = true;
  boxGroup.children.forEach((o) => {
    o.visible = $("cbBoxes").checked &&
      S.scene.plants[o.userData.i].detection_score >= S.conf;
  });
}

function buildBoxes() {
  S.scene.plants.forEach((p, i) => {
    const lo = v3(p.bbox_min);
    const hi = v3(p.bbox_max);
    const box = new THREE.Box3(lo, hi);
    const size = new THREE.Vector3(); box.getSize(size);
    const ctr = new THREE.Vector3(); box.getCenter(ctr);
    const geo = new THREE.BoxGeometry(size.x, size.y, size.z);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({
        color: new THREE.Color(...p3(S.scene.palette[i])), transparent: true,
      }));
    edges.position.copy(ctr);
    edges.userData = { i, box };
    boxGroup.add(edges);
  });
}

const p3 = (c) => [c[0] / 255, c[1] / 255, c[2] / 255];

function buildHelpers() {
  const pts = S.scene.frames.map((f) => v3(f.position));
  const traj = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: 0x4a9eff, transparent: true, opacity: 0.55 }));
  traj.name = "traj"; traj.visible = true;
  helperGroup.add(traj);

  const segs = [];
  for (const f of S.scene.frames) {
    const pos = frameBasis(f).pos;
    const corners = frustumCorners(f, camScale(0.05));
    for (let i = 0; i < 4; i++) {
      segs.push(pos, corners[i]);
      segs.push(corners[i], corners[(i + 1) % 4]);
    }
  }
  const fr = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(segs),
    new THREE.LineBasicMaterial({ color: 0x6b7784, transparent: true, opacity: 0.35 }));
  fr.name = "frusta";
  helperGroup.add(fr);

  const cur = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(new Array(16).fill(0)
      .map(() => new THREE.Vector3())),
    new THREE.LineBasicMaterial({ color: 0xffc857 }));
  cur.name = "curFrustum"; cur.visible = false;
  helperGroup.add(cur);

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

  const photoGroup = new THREE.Group();
  photoGroup.name = "photos";
  photoGroup.visible = false;
  helperGroup.add(photoGroup);

  const gn = v3(S.scene.ground.normal).normalize();
  const gd = S.scene.ground.offset;
  const a = new THREE.Vector3(1, 0, 0);
  if (Math.abs(gn.dot(a)) > 0.9) a.set(0, 1, 0);
  const e0 = new THREE.Vector3().crossVectors(a, gn).normalize();
  const e1 = new THREE.Vector3().crossVectors(gn, e0).normalize();
  const origin = gn.clone().multiplyScalar(-gd);
  const glines = [];
  const R = 4, step = 0.25;
  for (let t = -R; t <= R + 1e-6; t += step) {
    glines.push(origin.clone().addScaledVector(e0, t).addScaledVector(e1, -R),
                origin.clone().addScaledVector(e0, t).addScaledVector(e1, R));
    glines.push(origin.clone().addScaledVector(e1, t).addScaledVector(e0, -R),
                origin.clone().addScaledVector(e1, t).addScaledVector(e0, R));
  }
  const grid = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(glines),
    new THREE.LineBasicMaterial({ color: 0x39404a, transparent: true, opacity: 0.5 }));
  grid.name = "grid"; grid.visible = false;
  helperGroup.add(grid);

  const meas = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
    new THREE.LineBasicMaterial({ color: 0xffc857 }));
  meas.name = "measure"; meas.visible = false;
  helperGroup.add(meas);
}

function frameBasis(f) {
  const E = f.extrinsic;
  const r0 = new THREE.Vector3(E[0], E[1], E[2]);
  const r1 = new THREE.Vector3(E[4], E[5], E[6]);
  const r2 = new THREE.Vector3(E[8], E[9], E[10]);
  const t = new THREE.Vector3(E[3], E[7], E[11]);
  const pos = new THREE.Vector3(
    -(r0.x * t.x + r1.x * t.y + r2.x * t.z),
    -(r0.y * t.x + r1.y * t.y + r2.y * t.z),
    -(r0.z * t.x + r1.z * t.y + r2.z * t.z));
  const basis = new THREE.Matrix4().makeBasis(
    r0, r1.clone().negate(), r2.clone().negate());
  return { basis, pos, r0, r1, r2 };
}

function camPoint(fb, x, y, z) {
  return fb.pos.clone()
    .addScaledVector(fb.r0, x)
    .addScaledVector(fb.r1, y)
    .addScaledVector(fb.r2, z);
}

function camScale(frac) {
  return Math.max(1e-3, (S.scene.framing.radius || 1) * frac);
}

function frustumCorners(f, d) {
  const fb = frameBasis(f);
  const K = f.intrinsic;
  const [H, W] = S.scene.meta.model_hw || [420, 518];
  return [[0, 0], [W, 0], [W, H], [0, H]].map(([u, v]) =>
    camPoint(fb, (u - K[2]) * d / K[0], (v - K[5]) * d / K[4], d));
}

function projFromK(K, W, H, near, far) {
  const [fx, , cx, , fy, cy] = [K[0], K[1], K[2], K[3], K[4], K[5]];
  return new THREE.Matrix4().set(
    2 * fx / W, 0, (W - 2 * cx) / W, 0,
    0, 2 * fy / H, -(H - 2 * cy) / H, 0,
    0, 0, -(far + near) / (far - near), -2 * far * near / (far - near),
    0, 0, -1, 0);
}

function enterFrame(i) {
  const f = S.scene.frames[i];
  if (!f) return;
  dropPin();
  S.frameIdx = i;
  const { basis, pos } = frameBasis(f);
  const m = basis.clone(); m.setPosition(pos);
  camera.matrixWorld.copy(m);
  camera.matrixWorldInverse.copy(m).invert();
  S.frameLock = { f };
  cam.pos.copy(pos);

  adoptPose(basis);
  if (S.overlayOpacity === 0) setOpacity(70);
  dom.photo.src = `/api/frame/${f.id}/image.jpg`;
  dom.photo.onload = () => invalidate(2);
  dom.photo.style.display = "block";
  invalidate(3);
  setMode(`frame ${f.id}`);
  updateFrameUI();
}

function leaveFrame() {
  if (!S.frameLock) return;

  const K = S.frameLock.f.intrinsic;
  const mh = (S.scene.meta.model_hw || [420, 518])[0];
  if (K && K[4]) {
    const r = frameRect();
    const half = mh / (2 * K[4]);
    const ch = dom.canvas.clientHeight || r.h;
    const fov = 2 * Math.atan(half * ch / r.h) * 180 / Math.PI;
    camera.fov = Math.max(15, Math.min(110, fov));
  }
  const leaving = S.frameLock.f;
  S.frameLock = null;
  dom.photo.style.display = "none";

  pinPhoto(leaving, S.overlayOpacity);
  camera.updateProjectionMatrix();
  setMode(S.tool);
  applyCamera();
  drawFrameBoxes();

  invalidate(3);
}

function frameRect() {
  const [H, W] = S.scene.meta.model_hw || [420, 518];
  const cw = dom.canvas.clientWidth, ch = dom.canvas.clientHeight;
  const ar = W / H;
  let vw = cw, vh = cw / ar;
  if (vh > ch) { vh = ch; vw = ch * ar; }
  return { x: (cw - vw) / 2, y: (ch - vh) / 2, w: vw, h: vh, W, H };
}

function setOpacity(v) {
  S.overlayOpacity = v;
  dom.photo.style.opacity = v / 100;
  $("opacitySlider").value = v;
  $("opacityVal").textContent = v + "%";
}

function drawFrameBoxes() {
  const svg = dom.svg;
  svg.innerHTML = "";
  if (!S.frameLock) return;
  const r = frameRect();
  const f = S.frameLock.f;
  const K = f.intrinsic;
  const E = f.extrinsic;
  const R = [[E[0], E[1], E[2]], [E[4], E[5], E[6]], [E[8], E[9], E[10]]];
  const t = [E[3], E[7], E[11]];
  const project = (p) => {
    const z = R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + t[2];
    if (z <= 0.02) return null;
    const x = R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + t[0];
    const y = R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + t[1];
    return [(x * K[0] / z + K[2]) / r.W * r.w + r.x,
            (y * K[4] / z + K[5]) / r.H * r.h + r.y];
  };
  S.scene.plants.forEach((p, i) => {
    if (p.detection_score < S.conf) return;
    if (!$("cbBoxes").checked) return;
    const lo = p.bbox_min, hi = p.bbox_max;
    const pts = [];
    for (const X of [lo[0], hi[0]]) for (const Y of [lo[1], hi[1]])
      for (const Z of [lo[2], hi[2]]) {
        const q = project([X, Y, Z]);
        if (!q) return;
        pts.push(q);
      }
    const xs = pts.map((q) => q[0]), ys = pts.map((q) => q[1]);
    const x0 = Math.max(r.x, Math.min(...xs)), x1 = Math.min(r.x + r.w, Math.max(...xs));
    const y0 = Math.max(r.y, Math.min(...ys)), y1 = Math.min(r.y + r.h, Math.max(...ys));
    if (x1 - x0 < 4 || y1 - y0 < 4) return;
    const c = S.scene.palette[i];
    const el = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    el.setAttribute("x", x0); el.setAttribute("y", y0);
    el.setAttribute("width", x1 - x0); el.setAttribute("height", y1 - y0);
    el.setAttribute("fill", "none");
    el.setAttribute("stroke", `rgb(${c[0]},${c[1]},${c[2]})`);
    el.setAttribute("stroke-width", i === S.selectedPlant ? 2.5 : 1.2);
    el.setAttribute("opacity", i === S.selectedPlant ? 1 : 0.75);
    svg.appendChild(el);
    const tx = document.createElementNS("http://www.w3.org/2000/svg", "text");
    tx.setAttribute("x", x0 + 2); tx.setAttribute("y", Math.max(r.y + 10, y0 - 3));
    tx.setAttribute("fill", `rgb(${c[0]},${c[1]},${c[2]})`);
    tx.setAttribute("font-size", "10");
    tx.textContent = `#${p.id}`;
    svg.appendChild(tx);
  });
}

const _vp = new THREE.Matrix4();

function viewProj() {
  return _vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
}

function activeRect() {
  if (S.frameLock) return frameRect();
  return { x: 0, y: 0, w: dom.canvas.clientWidth, h: dom.canvas.clientHeight };
}

const screenX = (sx, r) => r.x + (sx * 0.5 + 0.5) * r.w;
const screenY = (sy, r) => r.y + (0.5 - sy * 0.5) * r.h;

function selectRect(x0, y0, x1, y1, frontOnly = true) {
  const m = viewProj().elements;
  const P = S.positions;
  const r = activeRect();
  const n = Math.floor(S.N * S.drawFrac);
  const out = [], depth = [];
  const [ax0, ax1] = [Math.min(x0, x1), Math.max(x0, x1)];
  const [ay0, ay1] = [Math.min(y0, y1), Math.max(y0, y1)];
  for (let i = 0; i < n; i++) {
    const px = P[i * 3], py = P[i * 3 + 1], pz = P[i * 3 + 2];
    const w = m[3] * px + m[7] * py + m[11] * pz + m[15];
    if (w <= 0) continue;
    const sx = (m[0] * px + m[4] * py + m[8] * pz + m[12]) / w;
    if (sx < -1 || sx > 1) continue;
    const sy = (m[1] * px + m[5] * py + m[9] * pz + m[13]) / w;
    if (sy < -1 || sy > 1) continue;
    const ux = screenX(sx, r), uy = screenY(sy, r);
    if (ux < ax0 || ux > ax1 || uy < ay0 || uy > ay1) continue;
    out.push(i); depth.push(w);
  }
  if (!out.length) return new Uint32Array(0);
  if (!frontOnly || out.length < 200) return Uint32Array.from(out);

  const sorted = Float64Array.from(depth).sort();
  const near = sorted[Math.floor(sorted.length * 0.05)];
  const band = Math.max(0.15, near * 0.35);
  const keep = [];
  for (let i = 0; i < out.length; i++) if (depth[i] <= near + band) keep.push(out[i]);
  return Uint32Array.from(keep);
}

function pickPoint(px, py, radius = 9) {
  const m = viewProj().elements;
  const P = S.positions;
  const r = activeRect();
  const n = Math.floor(S.N * S.drawFrac);
  let best = -1, bestD = Infinity;
  const r2 = radius * radius;
  for (let i = 0; i < n; i++) {
    const x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2];
    const w = m[3] * x + m[7] * y + m[11] * z + m[15];
    if (w <= 0 || w >= bestD) continue;
    const sx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w;
    if (sx < -1 || sx > 1) continue;
    const sy = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
    if (sy < -1 || sy > 1) continue;
    const dx = screenX(sx, r) - px, dy = screenY(sy, r) - py;
    if (dx * dx + dy * dy > r2) continue;
    best = i; bestD = w;
  }
  return best;
}

function setOverlay(idx, color, useOwnColors = false) {
  if (overlay) { world.remove(overlay); overlay.geometry.dispose(); overlay = null; }
  if (!idx || !idx.length) return;
  const pos = new Float32Array(idx.length * 3);
  const col = new Uint8Array(idx.length * 3);
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k];
    pos[k * 3] = S.positions[i * 3];
    pos[k * 3 + 1] = S.positions[i * 3 + 1];
    pos[k * 3 + 2] = S.positions[i * 3 + 2];
    col[k * 3] = S.colors[i * 3];
    col[k * 3 + 1] = S.colors[i * 3 + 1];
    col[k * 3 + 2] = S.colors[i * 3 + 2];
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3, true));
  overlay = new THREE.Points(g, overlayMat.clone());
  overlay.material.uniforms.uColor.value = new THREE.Color(color);
  overlay.material.uniforms.uUseOwn.value = useOwnColors ? 1.0 : 0.0;
  overlay.frustumCulled = false;
  world.add(overlay);
  invalidate(2);
}

function clearSelection() {
  S.selection = null;
  S.selectedPlant = -1;
  cloudUniforms.uSelTag.value = -1;
  cloudUniforms.uFade.value = 0;
  setOverlay(null);
  dom.selBody.innerHTML = "Nothing selected.";
  dom.selBody.classList.add("dim");
  renderPlantList();
  hudInfo("");
}

const keys = new Set();
let dragging = false, dragMoved = false, rubberFrom = null, lastXY = null;

addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();

  if (k === "escape") {
    leaveFrame(); clearSelection(); dom.help.classList.add("hidden");
    if (e.target.blur) e.target.blur();
    return;
  }
  if (e.target.matches("input,select,textarea")) return;
  keys.add(e.code);
  if (k === "f" && S.selectedPlant >= 0) flyToPlant(S.selectedPlant);
  if (k === "h") { $("cbHighlight").checked = !$("cbHighlight").checked; syncView(); }
  if (k === "i") toggleIsolate();
  if (k === "g") { S.frameLock ? leaveFrame() : enterFrame(S.frameIdx); }
  if (k === "o" && S.frameLock) setOpacity(S.overlayOpacity > 0 ? 0 : 70);
  else if (k === "o" && S.anchored >= 0) dropPin();
  else if (k === "o" && S.frameIdx >= 0 && !S.frameLock)
    pinPhoto(S.scene.frames[S.frameIdx], S.overlayOpacity || 70);
  if (k === "p") { $("showOwn").checked = !$("showOwn").checked; refreshFrameLayers(); }
  if (k === "m") setTool(S.tool === "measure" ? "fly" : "measure");
  if (k === "k") togglePlay();
  if (k === "," ) setFrame(S.frameIdx - 1);
  if (k === "." ) setFrame(S.frameIdx + 1);
  if (k === "r") { dropPin(); resetView(); }
  if (k === "l" && !S.frameLock && cam.roll) {
    cam.roll = 0; applyCamera(); invalidate(2);
  }
  if (k === "-") setSpeed(S.speed / 1.3);
  if (k === "=" || k === "+") setSpeed(S.speed * 1.3);
  if (["KeyW", "KeyA", "KeyS", "KeyD", "Space", "ShiftLeft", "ShiftRight"]
      .includes(e.code)) {
    if (S.frameLock) leaveFrame();
    if (e.code === "Space") e.preventDefault();
  }
});
addEventListener("keyup", (e) => keys.delete(e.code));
addEventListener("blur", () => keys.clear());

dom.canvas.addEventListener("contextmenu", (e) => e.preventDefault());

dom.canvas.addEventListener("pointerdown", (e) => {
  dom.canvas.setPointerCapture(e.pointerId);
  dragMoved = false;
  lastXY = [e.clientX, e.clientY];
  if (e.ctrlKey || S.tool === "select") {
    const r = dom.canvas.getBoundingClientRect();
    rubberFrom = [e.clientX - r.left, e.clientY - r.top];
    dom.rubber.style.display = "block";
  } else {
    dragging = true;
  }
});

addEventListener("pointermove", (e) => {
  if (rubberFrom) {
    const r = dom.canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    dom.rubber.style.left = Math.min(x, rubberFrom[0]) + "px";
    dom.rubber.style.top = Math.min(y, rubberFrom[1]) + "px";
    dom.rubber.style.width = Math.abs(x - rubberFrom[0]) + "px";
    dom.rubber.style.height = Math.abs(y - rubberFrom[1]) + "px";
    dragMoved = true;
    return;
  }
  if (!dragging || !lastXY) return;
  const dx = e.clientX - lastXY[0], dy = e.clientY - lastXY[1];
  lastXY = [e.clientX, e.clientY];
  if (Math.abs(dx) + Math.abs(dy) > 2) dragMoved = true;
  if (S.frameLock) leaveFrame();
  cam.yaw += dx * 0.0032;
  cam.pitch = Math.max(-1.553, Math.min(1.553, cam.pitch - dy * 0.0032));
  invalidate(2);
});

addEventListener("pointerup", (e) => {
  const r = dom.canvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  if (rubberFrom) {
    dom.rubber.style.display = "none";
    const [sx, sy] = rubberFrom;
    rubberFrom = null;
    if (Math.abs(x - sx) > 4 && Math.abs(y - sy) > 4) {
      applySelection(selectRect(sx, sy, x, y));
    }
  } else if (dragging && !dragMoved) {
    handleClick(x, y);
  }
  dragging = false;
});

dom.canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (S.frameLock) leaveFrame();
  const step = -Math.sign(e.deltaY) * S.speed * 0.35;
  const fwd = forwardVec();
  cam.pos.addScaledVector(fwd, step);
  dropPin();
  invalidate(2);
}, { passive: false });

function forwardVec() {
  const d = new THREE.Vector3();
  camera.matrixWorld.extractBasis(new THREE.Vector3(), new THREE.Vector3(), d);
  return d.negate();
}

function handleClick(x, y) {
  if (S.tool === "measure") {
    const i = pickPoint(x, y);
    if (i < 0) return;
    const p = new THREE.Vector3(
      S.positions[i * 3], S.positions[i * 3 + 1], S.positions[i * 3 + 2]);
    S.measure.push(p);
    if (S.measure.length > 2) S.measure = [p];
    updateMeasure();
    return;
  }
  if (S.tool === "pick") { inspectPoint(x, y); return; }

  const hit = pickBox(x, y);
  if (hit >= 0) { selectPlant(hit); return; }
  const i = pickPoint(x, y, 7);
  if (i >= 0 && S.tags[i] >= 0) selectPlant(S.tags[i]);
  else clearSelection();
}

function pickBox(x, y) {
  const r = activeRect();
  const ndc = new THREE.Vector2(
    ((x - r.x) / r.w) * 2 - 1, -((y - r.y) / r.h) * 2 + 1);
  const ray = new THREE.Raycaster();
  ray.setFromCamera(ndc, camera);
  let best = -1, bestT = Infinity;
  for (const o of boxGroup.children) {
    if (!o.visible) continue;
    const t = new THREE.Vector3();
    if (ray.ray.intersectBox(o.userData.box, t)) {
      const d = t.distanceTo(ray.ray.origin);
      if (d < bestT) { bestT = d; best = o.userData.i; }
    }
  }
  return best;
}

function updateMeasure() {
  const line = helperGroup.getObjectByName("measure");
  if (S.measure.length < 2) { line.visible = false; hudInfo(""); return; }
  const [a, b] = S.measure;
  line.geometry.setFromPoints([a, b]);
  line.geometry.computeBoundingSphere();
  line.visible = true;
  const d = a.distanceTo(b);
  const gn = v3(S.scene.ground.normal);
  const dv = Math.abs(b.clone().sub(a).dot(gn));
  const dh = Math.sqrt(Math.max(0, d * d - dv * dv));
  hudInfo(`${fmtLen(d)}   vertical ${fmtLen(dv)} · horizontal ${fmtLen(dh)}`);
}

function inspectPoint(x, y) {
  const i = pickPoint(x, y);
  if (i < 0) return;
  const p = [S.positions[i * 3], S.positions[i * 3 + 1], S.positions[i * 3 + 2]];
  const gn = S.scene.ground.normal;
  const h = p[0] * gn[0] + p[1] * gn[1] + p[2] * gn[2] + S.scene.ground.offset;
  const c = [S.colors[i * 3], S.colors[i * 3 + 1], S.colors[i * 3 + 2]];
  const tag = S.tags[i];
  hudInfo(`point ${i} · ${fmtLen(h)} above ground · rgb(${c.join(",")})` +
          (tag >= 0 ? ` · plant #${S.scene.plants[tag].id}` : " · unassigned"));
  setOverlay(Uint32Array.of(i), 0xffc857);
  switchTab("sel");
  dom.selBody.classList.remove("dim");
  dom.selBody.innerHTML = `
    <h4>Point ${i}</h4>
    <div class="kv">
      <div class="k">position</div><div class="v">${p.map((v) => v.toFixed(3)).join(", ")}</div>
      <div class="k">height above ground</div><div class="v">${fmtLen(h)}</div>
      <div class="k">colour</div><div class="v">rgb(${c.join(", ")})</div>
      <div class="k">excess green</div><div class="v">${(2 * c[1] - c[0] - c[2]).toFixed(0)}</div>
      <div class="k">assigned to</div><div class="v">${
        tag >= 0 ? `#${S.scene.plants[tag].id} ${S.scene.plants[tag].category}`
                 : "—"}</div>
    </div>`;
}

async function applySelection(idx) {
  if (!idx.length) { clearSelection(); return; }
  S.selection = { idx };
  S.selectedPlant = -1;
  cloudUniforms.uSelTag.value = -1;
  setOverlay(idx, 0x4a9eff);
  switchTab("sel");
  dom.selBody.classList.remove("dim");
  dom.selBody.innerHTML = `<p class="dim">analysing ${idx.length.toLocaleString()} points…</p>`;
  const res = await fetch("/api/select", { method: "POST", body: idx.buffer });
  const a = await res.json();
  S.selection.analysis = a;
  renderSelection(a);
}

function renderSelection(a) {
  if (!a || !a.n) { dom.selBody.innerHTML = "<p class='dim'>Empty selection.</p>"; return; }
  const m2 = fmtLen;
  const rgb = a.mean_rgb.map((v) => Math.round(v));
  const named = S.scene.live_classifier !== "none";
  dom.selBody.innerHTML = `
    <h4>Selection</h4>
    <div class="kv">
      <div class="k">points</div><div class="v">${a.n.toLocaleString()}</div>
      <div class="k">height</div><div class="v">${m2(a.height_m)}</div>
      <div class="k">base above ground</div><div class="v">${m2(a.base_height_m)}</div>
      <div class="k">crown width</div><div class="v">${m2(a.crown_w_m)}</div>
      <div class="k">crown depth</div><div class="v">${m2(a.crown_d_m)}</div>
      <div class="k">occupied volume</div><div class="v">${fmtVol(a.volume_m3)}</div>
      <div class="k">mean colour</div><div class="v">
        <span style="display:inline-block;width:9px;height:9px;border-radius:2px;
        background:rgb(${rgb.join(",")})"></span> ${rgb.join(", ")}</div>
      <div class="k">excess green</div><div class="v">${a.green_excess.toFixed(1)}</div>
    </div>
    <div class="btnrow">
      <button class="act" id="selIso">Isolate</button>
      <button class="act" id="selFly">Fly to</button>
      <button class="act primary" id="selId" ${named ? "" : "disabled"}>Identify</button>
      <button class="act" id="selClear">Clear</button>
      <button class="act" id="selPly">Save .ply</button>
    </div>
    ${a.overlap && a.overlap.length ? `
      <h4>Overlaps detected plants</h4>
      <table class="res">${a.overlap.map((o) => `
        <tr><td>#${o.id} ${o.category}</td><td class="s">${o.n.toLocaleString()} pts</td></tr>`
      ).join("")}</table>` : `
      <h4>Overlaps detected plants</h4>
      <p>none</p>`}
    <h4>Best views of this region</h4>
    <div class="viewgrid">${(a.views || []).slice(0, 6).map((v) => `
      <figure data-frame="${v.frame}" class="vthumb">
        <img src="/api/frame/${v.frame}/thumb.jpg" loading="lazy">
        <figcaption>#${v.frame} · ${(v.coverage * 100).toFixed(0)}% · ${
          fmtLen(v.distance)}</figcaption>
      </figure>`).join("")}</div>
    <div id="idResult"></div>`;

  $("selIso").onclick = toggleIsolate;
  $("selFly").onclick = () => {
    const c = v3(a.centroid);
    const r = v3(a.bbox_max)
      .sub(v3(a.bbox_min)).length();
    setLook(c, Math.max(0.35, r * 1.3));
  };
  $("selClear").onclick = clearSelection;
  $("selPly").onclick = () => exportPly(S.selection.idx);
  if (named) $("selId").onclick = identifySelection;
  dom.selBody.querySelectorAll(".vthumb").forEach((el) => {
    el.style.cursor = "pointer";
    el.onclick = () => {
      const i = S.scene.frames.findIndex((f) => f.id === +el.dataset.frame);
      if (i >= 0) { switchTab("frames"); setFrame(i); enterFrame(i); }
    };
  });
}

async function identifySelection() {
  const btn = $("selId");
  btn.disabled = true; btn.textContent = "identifying…";
  const out = $("idResult");
  out.innerHTML = "<p class='dim'>cropping the best views and classifying…</p>";
  try {
    const res = await fetch("/api/identify",
      { method: "POST", body: S.selection.idx.buffer });
    const r = await res.json();
    if (r.error) { out.innerHTML = `<div class="warnbox">${r.error}</div>`; return; }
    const top = r.results && r.results[0];
    out.innerHTML = `
      <h4>Identification · ${r.classifier}</h4>
      <table class="res">${(r.results || []).map((x) => `
        <tr><td>${x.name}${x.common ? `<br><span class="dim" style="font-size:10.5px">${
          x.common}</span>` : ""}</td>
        <td class="s">${x.score.toFixed(3)}${
          x.views ? `<br><span style="font-size:10px">${x.views} views</span>` : ""}
        <div class="bar2"><i style="width:${Math.min(100, x.score * 100)}%"></i></div>
        </td></tr>`).join("")}</table>
      <h4>What was sent</h4>
      <div class="viewgrid">${(r.views || []).map((v) => `
        <figure><img src="data:image/jpeg;base64,${v.jpeg}">
        <figcaption>#${v.frame} · ${fmtLen(v.distance)}</figcaption></figure>`
      ).join("")}</div>`;
  } catch (e) {
    out.innerHTML = `<div class="warnbox">${e}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = "Identify";
  }
}

function download(data, name, mime) {
  const url = URL.createObjectURL(new Blob([data], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

function exportPly(idx) {
  if (!idx || !idx.length) return;
  const k = 1;
  const header =
    "ply\nformat binary_little_endian 1.0\n" +
    `comment exported from ${S.scene.artifacts}\n` +
    `element vertex ${idx.length}\n` +
    "property float x\nproperty float y\nproperty float z\n" +
    "property uchar red\nproperty uchar green\nproperty uchar blue\n" +
    "end_header\n";
  const head = new TextEncoder().encode(header);
  const body = new ArrayBuffer(idx.length * 15);
  const dv = new DataView(body);
  for (let j = 0; j < idx.length; j++) {
    const i = idx[j], o = j * 15;
    dv.setFloat32(o, S.positions[i * 3] * k, true);
    dv.setFloat32(o + 4, S.positions[i * 3 + 1] * k, true);
    dv.setFloat32(o + 8, S.positions[i * 3 + 2] * k, true);
    dv.setUint8(o + 12, S.colors[i * 3]);
    dv.setUint8(o + 13, S.colors[i * 3 + 1]);
    dv.setUint8(o + 14, S.colors[i * 3 + 2]);
  }
  const out = new Uint8Array(head.length + body.byteLength);
  out.set(head, 0);
  out.set(new Uint8Array(body), head.length);
  download(out, `selection_${idx.length}pts.ply`, "application/octet-stream");
  hudInfo(`saved ${idx.length.toLocaleString()} points as PLY`);
}

function toggleIsolate() {
  const on = cloud.visible;
  if (on && (S.selection || S.selectedPlant >= 0)) {
    const idx = S.selection ? S.selection.idx
      : null;
    cloud.visible = false;
    if (idx) setOverlay(idx, 0, true);
    else if (overlay) overlay.material.uniforms.uUseOwn.value = 1.0;
    setMode("isolated");
  } else {
    cloud.visible = true;
    if (overlay) overlay.material.uniforms.uUseOwn.value = 0.0;
    setMode(S.tool);
  }
}

function selectPlant(i) {
  S.selectedPlant = i;
  S.selection = null;
  const p = S.scene.plants[i];
  cloudUniforms.uSelTag.value = i;
  cloudUniforms.uFade.value = 0.0;
  fetch(`/api/plant/${p.id}/points.bin`).then((r) => r.arrayBuffer()).then((b) => {
    if (S.selectedPlant !== i) return;
    const idx = new Uint32Array(b);
    S.selection = { idx, plant: i };
    setOverlay(idx, rgbHex(S.scene.palette[i]));
  });
  renderPlantList();
  renderPlantDetail(i);
  switchTab("sel");
  drawFrameBoxes();
}

const rgbHex = (c) => (c[0] << 16) | (c[1] << 8) | c[2];

function fmtLen(v, digits = 1) {
  if (v == null || !isFinite(v)) return "—";
  return v.toFixed(3) + " u";
}

function fmtLenShort(v) {
  if (v == null || !isFinite(v)) return "";
  return v.toFixed(2) + "u";
}

function fmtVol(v) {
  if (v == null || !isFinite(v)) return "—";
  return v.toExponential(2) + " u³";
}

function frameConf(p, fid) {
  const b = (p.boxes || []).filter((x) => x.frame === fid && x.score != null);
  if (b.length) return Math.max(...b.map((x) => x.score));
  const c = p.best_frame != null ? p.best_frame : (p.crop ? parseInt(p.crop, 10) : NaN);
  return fid === c ? p.detection_score : null;
}

function rankedFrames(p) {
  const c = p.best_frame != null ? p.best_frame : (p.crop ? parseInt(p.crop, 10) : NaN);
  return (p.frames || [])
    .filter((fid) => S.scene.frames.some((f) => f.id === fid))
    .map((fid) => [fid, frameConf(p, fid)])
    .sort((a, b) => ((b[1] ?? -1) - (a[1] ?? -1)) || ((b[0] === c) - (a[0] === c)))
    .map((x) => x[0]);
}

function standAtPlant(i) {
  const fid = rankedFrames(S.scene.plants[i])[0];
  const k = S.scene.frames.findIndex((f) => f.id === fid);
  if (k < 0) return false;
  setFrame(k);
  enterFrame(k);
  return true;
}

function flyToPlant(i) {
  const p = S.scene.plants[i];
  const lo = v3(p.bbox_min);
  const hi = v3(p.bbox_max);
  const c = lo.clone().add(hi).multiplyScalar(0.5);
  leaveFrame();

  setLook(c, Math.max(0.25, lo.distanceTo(hi) * 2.2));
}

function hierLadder(h) {
  // form -> family -> genus -> species; ranks past the stopping level are dimmed
  const order = ["form", "family", "genus", "species"];
  const depth = { plant: -1, form: 0, family: 1, genus: 2, species: 3 }[h.level];
  return `<div class="dist"><div class="distlab">hierarchy &mdash; named down to
    <b>${h.level}</b></div>` + order.map((r, k) => {
      const x = h[r] || {};
      const on = k <= depth || (r === "form" && depth >= 0);
      return `<div class="bar" title="${r}">
        <span class="bn">${r}: ${on ? `<b>${x.name || "-"}</b>` :
          `<span class="dim">${x.name || "-"}</span>`}</span>
        <span class="bt"><i class="${on ? "ok" : "no"}"
          style="width:${Math.max(1, (x.conf || 0) * 100).toFixed(1)}%"></i></span>
        <span class="bv">${(x.conf || 0).toFixed(2)}</span></div>`;
    }).join("") + `</div>`;
}

function speciesLine(p) {
  if (p.hierarchy) {
    const h = p.hierarchy;
    if (h.level === "plant") return `<span class="dim">plant (no rank confident)</span>`;
    if (h.level === "form") return `${p.category}`;
    const name = h.level === "family" ? p.species : `<i>${p.species}</i>`;
    return `${name} <span class="dim">(${h.level})</span>${
      p.species_common ? ` — ${p.species_common}` : ""}`;
  }
  if (!p.species) return `<span class="dim">no species candidate</span>`;
  if (!p.species_identified) {
    return `<span class="dim">unidentified — best guess <i>${p.species}</i>
            (${(p.species_raw_score || 0).toFixed(2)})</span>`;
  }
  return `<i>${p.species}</i>${p.species_common ? ` — ${p.species_common}` : ""}`;
}

function distBars(p) {
  const cand = p.species_candidates || [];
  if (!cand.length) return "";
  const top = p.species || "";
  let html = "";
  if (cand.length) {
    html += `<div class="dist"><div class="distlab">across species</div>` +
      cand.map((c) => `<div class="bar" title="${c.name} ${c.score.toFixed(3)}">
        <span class="bn">${c.name === top ? `<b>${c.name}</b>` : c.name}</span>
        <span class="bt"><i style="width:${Math.max(1, c.score * 100).toFixed(1)}%"></i></span>
        <span class="bv">${c.score.toFixed(3)}</span></div>`).join("") + `</div>`;
  }
  return html;
}

function renderPlantDetail(i) {
  const p = S.scene.plants[i];
  const m = p.metrics || {};
  const cm = fmtLen;
  dom.selBody.classList.remove("dim");
  dom.selBody.innerHTML = `
    <h4>#${p.id} ${p.category}</h4>
    ${p.crop ? `<img class="crop" src="/api/crop/${p.crop}">` : ""}
    <div>${speciesLine(p)}</div>
    ${p.hierarchy ? hierLadder(p.hierarchy) : ""}
    <div class="kv" style="margin-top:8px">
      <div class="k">detection</div><div class="v">${p.detection_score.toFixed(2)}</div>
      <div class="k">category vote</div><div class="v">${p.category_confidence.toFixed(2)}</div>
      <div class="k">classifier raw score</div><div class="v">${
        (p.species_raw_score || 0).toFixed(3)}</div>
      <div class="k">vote share</div><div class="v">${
        (p.species_confidence || 0).toFixed(2)}</div>
      ${p.species_agreement == null ? "" : `<div class="k">views agreeing</div>
        <div class="v">${(p.species_agreement * 100).toFixed(0)}%</div>`}
      ${p.species_spread == null ? "" : `<div class="k">per-view spread</div>
        <div class="v">&plusmn;${p.species_spread.toFixed(3)}</div>`}
      ${p.species_sharpening == null ? "" : `<div class="k" title="fused score
        divided by the best single view's score for the same name">from fusing</div>
        <div class="v">${p.species_sharpening.toFixed(2)}&times;</div>`}
      ${p.species_entropy == null ? "" : `<div class="k" title="0 = one species,
        ${(p.species_entropy_max || 0).toFixed(2)} = no preference at all">entropy</div>
        <div class="v">${p.species_entropy.toFixed(2)} / ${
          (p.species_entropy_max || 0).toFixed(2)} bits</div>`}
      <div class="k">views</div><div class="v">${p.views}</div>
      <div class="k">cloud points</div><div class="v">${
        (m.n_cloud_points || 0).toLocaleString()}</div>
      <div class="k">height</div><div class="v">${cm(m.height_m)}</div>
      <div class="k">base above ground</div><div class="v">${cm(m.base_height_m)}</div>
      <div class="k">crown width</div><div class="v">${cm(m.crown_w_m)}</div>
      <div class="k">crown depth</div><div class="v">${cm(m.crown_d_m)}</div>
      <div class="k">occupied volume</div><div class="v">${fmtVol(m.volume_m3)}</div>
      <div class="k">mean colour</div><div class="v">${
        (m.mean_rgb || []).map((v) => Math.round(v)).join(", ")}</div>
    </div>
    ${distBars(p)}
    <div class="btnrow">
      <button class="act" id="pdFly">Fly to (F)</button>
      <button class="act" id="pdIso">Isolate (I)</button>
      <button class="act primary" id="pdId" ${
        S.scene.live_classifier === "none" ? "disabled" : ""}>Re-identify</button>
    </div>
    <h4>Seen in ${p.frames.length} recorded frames</h4>
    <div class="viewgrid">${rankedFrames(p).slice(0, 8).map((fid) => `
      <figure class="vthumb" data-frame="${fid}">
        <img src="/api/frame/${fid}/thumb.jpg" loading="lazy">
        <figcaption>#${fid}</figcaption></figure>`).join("")}</div>
    <div id="idResult"></div>`;
  $("pdFly").onclick = () => flyToPlant(i);
  $("pdIso").onclick = toggleIsolate;
  if (S.scene.live_classifier !== "none") $("pdId").onclick = identifySelection;
  dom.selBody.querySelectorAll(".vthumb").forEach((el) => {
    el.style.cursor = "pointer";
    el.onclick = () => {
      const k = S.scene.frames.findIndex((f) => f.id === +el.dataset.frame);
      if (k >= 0) { switchTab("frames"); setFrame(k); enterFrame(k); }
      else hudInfo(`frame ${el.dataset.frame} was not saved to disk`);
    };
  });
}

function plantVisible(p) {
  const c = v3(p.centroid);
  const f = new THREE.Frustum().setFromProjectionMatrix(viewProj().clone());
  return f.containsPoint(c);
}

function renderPlantList() {
  const q = $("plantSearch").value.trim().toLowerCase();
  const sort = $("plantSort").value;
  const onlyNamed = $("onlyNamed").checked;
  const onlyVis = $("onlyVisible").checked;
  let rows = S.scene.plants.map((p, i) => ({ p, i }))
    .filter(({ p }) => p.detection_score >= S.conf)
    .filter(({ p }) => !onlyNamed || p.species_identified)
    .filter(({ p }) => !q || (
      `#${p.id} ${p.category} ${p.species || ""} ${p.species_common || ""}`
        .toLowerCase().includes(q)))
    .filter(({ p }) => !onlyVis || plantVisible(p));
  const key = {
    id: ({ p }) => p.id,
    conf: ({ p }) => -p.detection_score,
    points: ({ p }) => -(p.metrics?.n_cloud_points || 0),
    height: ({ p }) => -(p.metrics?.height_m || 0),
    volume: ({ p }) => -(p.metrics?.volume_m3 || 0),
    views: ({ p }) => -p.views,
  }[sort];
  rows.sort((a, b) => key(a) - key(b));

  $("plantCount").textContent =
    `${rows.length} of ${S.scene.plants.length} shown`;
  dom.plantList.innerHTML = rows.map(({ p, i }) => {
    const c = S.scene.palette[i];
    const m = p.metrics || {};
    return `<div class="card ${i === S.selectedPlant ? "sel" : ""}" data-i="${i}">
      <div class="head">
        <span class="swatch" style="background:rgb(${c.join(",")})"></span>
        <span class="t1">#${p.id} ${p.category}
          ${p.species_identified ? `<span class="sp">${p.species}</span>` : ""}</span>
        <span class="t2">${m.height_m ? fmtLenShort(m.height_m) + " · " : ""}${
          p.detection_score.toFixed(2)}</span>
      </div></div>`;
  }).join("") || `<p class="note">Nothing matches.</p>`;

  dom.plantList.querySelectorAll(".card").forEach((el) => {
    el.querySelector(".head").onclick = () => {
      const i = +el.dataset.i;
      selectPlant(i);
      if (!standAtPlant(i)) flyToPlant(i);
    };
  });
}

function setFrame(i) {
  S.frameIdx = Math.max(0, Math.min(S.scene.frames.length - 1, i));
  $("frameSlider").value = S.frameIdx;
  updateFrameUI(S.playing);
  applyBuildUp();
  if (!S.playing) refreshFrameLayers();
  if (S.frameLock) enterFrame(S.frameIdx);
}

function applyBuildUp() {
  const on = $("buildUp").checked && S.scene.provenance;
  cloudUniforms.uBuildTo.value = on ? S.scene.frames[S.frameIdx].id : -1;
}

function togglePlay() {
  S.playing = !S.playing;
  $("framePlay").textContent = S.playing ? "\u275a\u275a Pause" : "\u25b6 Play";
  S.playAcc = 0;
  if (S.playing) {
    leaveFrame();
  }
}

function pinPhoto(f, opacity) {
  dropPin();
  if (!f || !f.has_image || opacity <= 0) return;
  const [H, W] = S.scene.meta.model_hw || [420, 518];
  const fb = frameBasis(f);
  const K = f.intrinsic;
  const d = camScale(0.55);
  const geo = new THREE.PlaneGeometry(W * d / K[0], H * d / K[4]);

  const tex = new THREE.TextureLoader().load(`/api/frame/${f.id}/image.jpg`,
                                             () => invalidate(2));
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.MeshBasicMaterial({
    map: tex, side: THREE.DoubleSide, transparent: true,
    opacity: opacity / 100, depthTest: false, depthWrite: false,
  });
  const quad = new THREE.Mesh(geo, mat);
  quad.name = "pin";
  quad.renderOrder = 10;
  quad.applyMatrix4(fb.basis);
  quad.position.copy(camPoint(fb, 0, 0, d));
  helperGroup.add(quad);
  S.anchored = f.id;
}

function dropPin() {
  const q = helperGroup.getObjectByName("pin");
  if (!q) return;
  helperGroup.remove(q);
  q.geometry.dispose();
  if (q.material.map) q.material.map.dispose();
  q.material.dispose();
  S.anchored = -1;
  invalidate(2);
}

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
    const d = camScale(0.09);
    const w = W * d / K[0], h = H * d / K[4];
    const geo = new THREE.PlaneGeometry(w, h);
    const tex = loader.load(`/api/frame/${f.id}/thumb.jpg`, () => invalidate(2));
    tex.colorSpace = THREE.SRGBColorSpace;
    const mat = new THREE.MeshBasicMaterial({
      map: tex, side: THREE.DoubleSide, transparent: true, opacity: 0.92,
    });
    const quad = new THREE.Mesh(geo, mat);

    quad.applyMatrix4(basis);
    quad.position.copy(camPoint(frameBasis(f), 0, 0, d));
    group.add(quad);
  }
  hudInfo(`${group.children.length} camera photos placed along the path`);
}

function updateFrameUI(light = false) {
  const f = S.scene.frames[S.frameIdx];
  if (!f) return;
  $("frameVal").textContent = `#${f.id}`;
  if (light) { updateCurrentCamera(f); updateFrameStrip(); return; }
  const fps = S.scene.meta.fps || 30;
  const t = f.id / fps;
  const seen = S.frameSeenCount != null ? S.frameSeenCount : null;
  dom.frameInfo.innerHTML = `
    <div class="kv">
      <div class="k">time in video</div><div class="v">${
        Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}</div>
      <div class="k">camera position</div><div class="v">${
        f.position.map((v) => v.toFixed(2)).join(", ")}</div>
      <div class="k">plants in frame</div><div class="v">${
        S.scene.plants.filter((p) => p.frames.includes(f.id)).length}</div>
      ${frameConfRow(f)}
      ${seen != null ? `<div class="k">cloud points supported</div>
        <div class="v">${seen.toLocaleString()}</div>` : ""}
    </div>`;

  updateCurrentCamera(f);
}

function frameConfRow(f) {
  const ps = S.selectedPlant >= 0 ? [S.scene.plants[S.selectedPlant]]
    : S.scene.plants.filter((p) => p.frames.includes(f.id));
  const v = ps.map((p) => frameConf(p, f.id)).filter((x) => x != null);
  if (!v.length) return "";
  return `<div class="k">confidence</div><div class="v">${Math.max(...v).toFixed(2)}</div>`;
}

function updateCurrentCamera(f) {
  const cur = helperGroup.getObjectByName("curFrustum");
  const { pos, r0, r1, r2 } = frameBasis(f);
  const corners = frustumCorners(f, camScale(0.10));
  const segs = [];
  for (let i = 0; i < 4; i++) {
    segs.push(pos, corners[i], corners[i], corners[(i + 1) % 4]);
  }
  cur.geometry.setFromPoints(segs);
  cur.geometry.computeBoundingSphere();
  cur.visible = $("showFrusta").checked;

  const ax = helperGroup.getObjectByName("curAxes");
  const L = camScale(0.075);
  const axpts = [];
  for (const dir of [r0, r1, r2]) {
    axpts.push(pos, pos.clone().addScaledVector(dir, L));
  }
  ax.geometry.setFromPoints(axpts);
  ax.geometry.computeBoundingSphere();
  ax.visible = $("showAxes").checked;
}

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

async function refreshFrameLayers() {
  const f = S.scene.frames[S.frameIdx];
  if (!f) return;
  if (ownPoints) { world.remove(ownPoints); ownPoints.geometry.dispose(); ownPoints = null; }

  cloud.visible = !$("showOwn").checked;

  if ($("showCreated").checked) {
    const b = await (await fetch(`/api/frame/${f.id}/created.bin`)).arrayBuffer();
    const idx = new Uint32Array(b);
    setOverlay(idx, 0xffc857);
    hudInfo(`frame ${f.id} created ${idx.length.toLocaleString()} of the ` +
            `cloud's points (first observation of that voxel)`);
  }

  if ($("showOwn").checked) {
    const b = await (await fetch(`/api/frame/${f.id}/own.bin?stride=2`)).arrayBuffer();
    const n = new Uint32Array(b, 0, 1)[0];
    const pos = new Float32Array(b, 4, n * 3);
    const col = new Uint8Array(b, 4 + n * 12, n * 3);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3, true));
    const mat = overlayMat.clone();
    mat.uniforms.uUseOwn.value = 1.0;
    mat.uniforms.uSize.value = 0.006;
    ownPoints = new THREE.Points(g, mat);
    ownPoints.frustumCulled = false;
    world.add(ownPoints);
    invalidate(2);
    hudInfo(`frame ${f.id} produced ${n.toLocaleString()} points at stride 2`);
  }
  if ($("showSeen").checked) {
    const b = await (await fetch(`/api/frame/${f.id}/seen.bin`)).arrayBuffer();
    const idx = new Uint32Array(b);
    S.frameSeenCount = idx.length;
    setOverlay(idx, 0x4a9eff);
    updateFrameUI();
  } else if (S.frameSeenCount != null && !S.selection) {
    S.frameSeenCount = null;
    setOverlay(null);
  }
}

function buildUI() {
  document.querySelectorAll("#tabs button").forEach((b) => {
    b.onclick = () => switchTab(b.dataset.tab);
  });
  document.querySelectorAll("#toolbar button[data-tool]").forEach((b) => {
    b.onclick = () => setTool(b.dataset.tool);
  });
  $("btnHelp").onclick = () => dom.help.classList.toggle("hidden");
  dom.help.onclick = () => dom.help.classList.add("hidden");

  const bind = (id, fn, ev = "input") => $(id).addEventListener(ev, fn);
  bind("confSlider", (e) => {
    S.conf = +e.target.value;
    $("confVal").textContent = S.conf.toFixed(2);
    updateConfMask(); renderPlantList(); drawFrameBoxes();
  });
  bind("plantSearch", renderPlantList);
  bind("plantSort", renderPlantList, "change");
  bind("onlyNamed", renderPlantList, "change");
  bind("onlyVisible", renderPlantList, "change");

  $("frameSlider").max = S.scene.frames.length - 1;
  $("frameCount").textContent = S.scene.frames.length;
  bind("frameSlider", (e) => setFrame(+e.target.value));
  $("framePrev").onclick = () => setFrame(S.frameIdx - 1);
  $("frameNext").onclick = () => setFrame(S.frameIdx + 1);
  $("frameEnter").onclick = () => (S.frameLock ? leaveFrame() : enterFrame(S.frameIdx));
  bind("opacitySlider", (e) => setOpacity(+e.target.value));
  bind("showSeen", refreshFrameLayers, "change");
  bind("showOwn", refreshFrameLayers, "change");
  bind("showFrusta", () => {
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
  bind("buildUp", applyBuildUp, "change");
  $("framePlay").onclick = togglePlay;
  if (!S.scene.provenance) $("buildUp").disabled = true;

  bind("colorMode", (e) => setColorMode(+e.target.value), "change");
  if (!S.scene.provenance) {
    [...$("colorMode").options].forEach((o) => {
      if (o.value === "2" || o.value === "3") o.disabled = true;
    });
    $("showCreated").disabled = true;
  }
  bind("showCreated", refreshFrameLayers, "change");
  bind("psSlider", (e) => {
    cloudUniforms.uSize.value = +e.target.value;
    $("psVal").textContent = (+e.target.value).toFixed(4);
  });
  bind("denSlider", (e) => {
    S.drawFrac = +e.target.value / 100;
    cloud.geometry.setDrawRange(0, Math.floor(S.N * S.drawFrac));
    $("denVal").textContent = e.target.value + "%";
  });
  bind("tintSlider", (e) => {
    cloudUniforms.uTint.value = +e.target.value / 100;
    $("tintVal").textContent = e.target.value + "%";
  });
  bind("briSlider", (e) => {
    cloudUniforms.uBright.value = +e.target.value;
    $("briVal").textContent = (+e.target.value).toFixed(2);
  });
  bind("spdSlider", (e) => setSpeed(+e.target.value));
  bind("cbHighlight", syncView, "change");
  bind("cbBoxes", () => { updateConfMask(); drawFrameBoxes(); }, "change");
  bind("cbLabels", () => {}, "change");
  bind("cbSpecies", () => {}, "change");
  bind("cbGround", () => {
    helperGroup.getObjectByName("grid").visible = $("cbGround").checked;
  }, "change");
  bind("cbTraj", () => {
    helperGroup.getObjectByName("traj").visible = $("cbTraj").checked;
  }, "change");

  helperGroup.getObjectByName("frusta").visible = true;

  $("exportCsv").onclick = () => {
    const u = "u";
    const k = 1;
    const head = ["id", "category", "category_conf", "detection", "species",
                  "common", "species_raw", "species_vote", "agreement",
                  "view_spread", "sharpening", "entropy", "identified", "views",
                  "cloud_points", `height_${u}`, `crown_w_${u}`, `crown_d_${u}`,
                  `volume_${u}3`, "mean_r", "mean_g", "mean_b",
                  "x", "y", "z"];
    const rows = S.scene.plants
      .filter((p) => p.detection_score >= S.conf)
      .map((p) => {
        const m = p.metrics || {};
        const c = m.mean_rgb || [0, 0, 0];
        return [p.id, p.category, p.category_confidence, p.detection_score,
                p.species || "", p.species_common || "", p.species_raw_score,
                p.species_confidence, p.species_agreement, p.species_spread,
                p.species_sharpening, p.species_entropy,
                p.species_identified ? 1 : 0, p.views,
                m.n_cloud_points || 0,
                ((m.height_m || 0) * k).toFixed(4),
                ((m.crown_w_m || 0) * k).toFixed(4),
                ((m.crown_d_m || 0) * k).toFixed(4),
                ((m.volume_m3 || 0) * k ** 3).toFixed(6),
                ...c.map((v) => Math.round(v)),
                ...p.centroid.map((v) => v.toFixed(4))];
      });
    const esc = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v);
    download([head, ...rows].map((r) => r.map(esc).join(",")).join("\n"),
             `plants_${S.scene.artifacts}.csv`, "text/csv");
  };

  renderSceneInfo();

  updateConfMask();
  renderPlantList();
  setFrame(0);
  syncView();
}

function renderSceneInfo() {
  const s = S.scene;
  const named = s.plants.filter((p) => p.species_identified).length;
  dom.sceneInfo.innerHTML = `
    <h4>Scene</h4>
    <div class="kv">
      <div class="k">artifacts</div><div class="v">${s.artifacts}</div>
      <div class="k">points</div><div class="v">${s.n_points.toLocaleString()}</div>
      <div class="k">voxel</div><div class="v">${(s.voxel * 1000).toFixed(1)} mm</div>
      <div class="k">frames in video</div><div class="v">${
        (s.meta.num_frames || 0).toLocaleString()}</div>
      <div class="k">frames saved</div><div class="v">${s.frames.length}</div>
      <div class="k">plants</div><div class="v">${s.plants.length}</div>
      <div class="k">named</div><div class="v">${named}</div>
      <div class="k">pipeline classifier</div><div class="v">${s.classifier}</div>
      <div class="k">viewer classifier</div><div class="v">${s.live_classifier}</div>
    </div>`;
}

function syncView() {
  cloudUniforms.uHighlight.value = $("cbHighlight").checked ? 1.0 : 0.0;
}

function setColorMode(mode) {
  cloudUniforms.uMode.value = mode;
  const gn = S.scene.ground;
  cloudUniforms.uGroundN.value.fromArray(gn.normal);
  cloudUniforms.uGroundD.value = gn.offset;
  if (mode === 1) {
    cloudUniforms.uSpanLo.value = 0;
    cloudUniforms.uSpanHi.value = S.scene.framing.radius * 0.6;
    hudInfo(`colour: height, 0 to ${fmtLen(S.scene.framing.radius * 0.6)}`);
  } else if (mode === 2) {
    cloudUniforms.uSpanLo.value = 1;
    cloudUniforms.uSpanHi.value = Math.log2(Math.max(4, S.scene.obs_p99) + 1);
    hudInfo(`colour: pixels per voxel, 1 to ${Math.round(S.scene.obs_p99)} (log)`);
  } else if (mode === 3) {
    cloudUniforms.uSpanLo.value = 0;
    cloudUniforms.uSpanHi.value = S.scene.num_frames || 1;
    hudInfo("colour: the frame that last saw each point");
  } else {
    hudInfo("");
  }
}

function switchTab(name) {
  document.querySelectorAll("#tabs button").forEach(
    (b) => b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach(
    (t) => t.classList.toggle("on", t.id === `tab-${name}`));
}

function setTool(t) {
  S.tool = t;
  document.querySelectorAll("#toolbar button[data-tool]").forEach(
    (b) => b.classList.toggle("on", b.dataset.tool === t));
  dom.crosshair.style.display = (t === "measure" || t === "pick") ? "block" : "none";
  if (t !== "measure") { S.measure = []; updateMeasure(); }
  setMode(t);
}

function setMode(m) { dom.hudMode.textContent = m; }
function hudInfo(s) { dom.hudInfo.textContent = s; }
function setSpeed(v) {
  S.speed = Math.max(0.05, Math.min(8, v));
  $("spdSlider").value = S.speed;
  $("spdVal").textContent = S.speed.toFixed(2);
  dom.hudSpeed.textContent = S.speed.toFixed(2) + " u/s";
}

function resetView() {
  leaveFrame();

  const fr = S.scene.framing;
  const focus = v3(fr.focus);
  const fwd = v3(fr.forward).normalize();
  cam.pos.copy(focus)
    .addScaledVector(fwd, -fr.radius * 1.35)
    .addScaledVector(UP, fr.radius * 0.45);
  faceToward(focus);
  setSpeed(Math.max(0.05, fr.radius * 0.35));
  applyCamera();
  invalidate(3);
}

let last = performance.now();

let dirty = 3;
const invalidate = (n = 2) => { dirty = Math.max(dirty, n); };

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
  if (S.playing) {
    S.playAcc = (S.playAcc || 0) + Math.min(dt, 0.2) * (S.rate || 6);
    if (S.playAcc >= 1) {
      const step = Math.floor(S.playAcc);
      S.playAcc -= step;
      const next = S.frameIdx + step;
      if (next >= S.scene.frames.length) { setFrame(0); }
      else setFrame(next);
    }
    invalidate(2);
  }
  if (dirty <= 0) { requestAnimationFrame(tick); return; }
  dirty -= 1;

  if (!S.frameLock) {
    const v = S.speed * (keys.has("ControlLeft") ? 0.25 : 1) * dt;
    if (v > 0) {
      if (S.anchored >= 0 && MOVE_KEYS.some((c) => keys.has(c)))
        dropPin();
      const f = forwardVec();
      const right = new THREE.Vector3().crossVectors(f, UP).normalize();
      if (keys.has("KeyW")) cam.pos.addScaledVector(f, v);
      if (keys.has("KeyS")) cam.pos.addScaledVector(f, -v);
      if (keys.has("KeyD")) cam.pos.addScaledVector(right, v);
      if (keys.has("KeyA")) cam.pos.addScaledVector(right, -v);
      if (keys.has("Space")) cam.pos.addScaledVector(UP, v);
      if (keys.has("ShiftLeft") || keys.has("ShiftRight"))
        cam.pos.addScaledVector(UP, -v);
    }
    applyCamera();
  }

  resize();

  const cw = dom.canvas.clientWidth, ch = dom.canvas.clientHeight;
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, cw, ch);
  renderer.setClearColor(BG, 1);
  renderer.clear();

  if (S.frameLock) {
    const r = frameRect();
    camera.projectionMatrix.copy(
      projFromK(S.frameLock.f.intrinsic, r.W, r.H, 0.01, 500));
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    renderer.setViewport(r.x, ch - r.y - r.h, r.w, r.h);
    renderer.setScissor(r.x, ch - r.y - r.h, r.w, r.h);
    renderer.setScissorTest(true);
    Object.assign(dom.photo.style,
      { left: r.x + "px", top: r.y + "px", width: r.w + "px", height: r.h + "px" });
    cloudUniforms.uPx.value = camera.projectionMatrix.elements[5] * r.h * 0.5;
  } else {
    cloudUniforms.uPx.value = camera.projectionMatrix.elements[5] * ch * 0.5;
  }
  overlayUniforms.uPx.value = cloudUniforms.uPx.value;
  if (overlay) overlay.material.uniforms.uPx.value = cloudUniforms.uPx.value;
  if (ownPoints) ownPoints.material.uniforms.uPx.value = cloudUniforms.uPx.value;

  renderer.render(world, camera);
  drawLabels();
  if (S.frameLock) drawFrameBoxes();
  requestAnimationFrame(tick);
}

let lastW = 0, lastH = 0;
function resize() {
  const w = dom.stage.clientWidth, h = dom.stage.clientHeight;
  if (w === lastW && h === lastH) return;
  lastW = w; lastH = h;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  if (!S.frameLock) camera.updateProjectionMatrix();
  dom.svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
}

const labelPool = [];
function drawLabels() {
  const show = $("cbLabels").checked && !S.frameLock;
  if (!show) {
    labelPool.forEach((l) => (l.style.display = "none"));
    return;
  }
  const m = viewProj().elements;
  const rect = activeRect();
  const withSpecies = $("cbSpecies").checked;
  let k = 0;
  const cands = [];
  S.scene.plants.forEach((p, i) => {
    if (p.detection_score < S.conf) return;
    const x = (p.bbox_min[0] + p.bbox_max[0]) / 2;
    const y = (p.bbox_min[1] + p.bbox_max[1]) / 2;
    const z = (p.bbox_min[2] + p.bbox_max[2]) / 2;
    const top = new THREE.Vector3(x, y, z).addScaledVector(
      UP, v3(p.bbox_max)
        .sub(v3(p.bbox_min)).length() * 0.5);
    const w = m[3] * top.x + m[7] * top.y + m[11] * top.z + m[15];
    if (w <= 0) return;
    const sx = (m[0] * top.x + m[4] * top.y + m[8] * top.z + m[12]) / w;
    const sy = (m[1] * top.x + m[5] * top.y + m[9] * top.z + m[13]) / w;
    if (sx < -1.1 || sx > 1.1 || sy < -1.1 || sy > 1.1) return;
    cands.push({ i, p, px: screenX(sx, rect), py: screenY(sy, rect), w });
  });

  cands.sort((a, b) => a.w - b.w);
  const placed = [];
  const fits = (c) => {
    const w = 130, h = 17;
    for (const q of placed) {
      if (Math.abs(q.px - c.px) < w && Math.abs(q.py - c.py) < h) return false;
    }
    placed.push(c);
    return true;
  };
  for (const c of cands.filter(fits).slice(0, 34)) {
    let el = labelPool[k];
    if (!el) {
      el = document.createElement("div");
      el.className = "lbl";
      dom.labels.appendChild(el);
      labelPool.push(el);
    }
    const col = S.scene.palette[c.i];
    const sp = withSpecies && c.p.species_identified
      ? ` <span style="opacity:.8"><i>${c.p.species}</i></span>`
      : ` <span style="opacity:.6">${c.p.detection_score.toFixed(2)}</span>`;
    el.innerHTML = `<b style="color:rgb(${col.join(",")})">#${c.p.id}</b> ${
      c.p.category}${sp}`;
    el.style.display = "block";
    el.style.left = c.px + "px";
    el.style.top = c.py + "px";
    el.classList.toggle("sel", c.i === S.selectedPlant);
    el.onclick = ((i) => () => { selectPlant(i); })(c.i);
    k++;
  }
  for (let j = k; j < labelPool.length; j++) labelPool[j].style.display = "none";
}

window.__dbg = () => ({
  lock: !!S.frameLock,
  anchored: S.anchored,
  N: S.N, drawFrac: S.drawFrac, up: UP.toArray(),
  framing: S.scene && S.scene.framing,
  p0: S.positions ? [S.positions[0], S.positions[1], S.positions[2]] : null,
  fov: camera.fov,
  pos: cam.pos.toArray(),
  yaw: cam.yaw, pitch: cam.pitch, roll: cam.roll,
  world: camera.matrixWorld.toArray(),
  proj: camera.projectionMatrix.toArray(),
  rect: activeRect(),
  canvas: [dom.canvas.clientWidth, dom.canvas.clientHeight],
});

boot().catch((e) => {
  dom.loading.innerHTML = `<div style="color:#e0a458;max-width:460px;
    text-align:center">Failed to start: ${e}</div>`;
  console.error(e);
});
