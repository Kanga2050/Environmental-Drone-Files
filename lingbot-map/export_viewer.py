"""Bake the 3D map viewer and one scene into a single HTML file that opens
from disk with no server. Per-frame depth layers and identification stay
server-only; everything else runs in the browser."""

import argparse
import base64
import gzip
import json
import os
import re
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from view_map3d import WEB, MapScene

SHIM = r"""
(() => {
  const b64bytes = (s) => {
    s = s.replace(/\s+/g, "");
    const out = new Uint8Array(Math.floor(s.length * 3 / 4) -
                               (s.endsWith("==") ? 2 : s.endsWith("=") ? 1 : 0));
    const CH = 1 << 22;
    let o = 0;
    for (let i = 0; i < s.length; i += CH) {
      const bin = atob(s.slice(i, i + CH));
      for (let k = 0; k < bin.length; k++) out[o++] = bin.charCodeAt(k);
    }
    return out;
  };
  const gunzip = async (id) => {
    const el = document.getElementById(id);
    const raw = b64bytes(el.textContent);
    el.textContent = "";
    const ds = new Blob([raw]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Uint8Array(await new Response(ds).arrayBuffer());
  };
  const json = (id) => JSON.parse(document.getElementById(id).textContent);

  const SCENE = json("d_scene");
  const X = json("d_extra");
  const N = SCENE.n_points;
  let D = null;

  async function data() {
    if (D) return D;
    const q = new Uint16Array((await gunzip("d_pos")).buffer);
    const pos = new Float32Array(N * 3);
    for (let i = 0; i < N * 3; i++) {
      const a = i % 3;
      pos[i] = X.qmin[a] + q[i] * X.qstep[a];
    }
    const tags = new Int16Array((await gunzip("d_tag")).buffer);
    const prov = SCENE.provenance ? new Uint16Array((await gunzip("d_prov")).buffer) : null;
    const col = await gunzip("d_col");
    const member = new Uint32Array((await gunzip("d_member")).buffer);
    const parts = [pos, tags].concat(prov ? [prov] : [], [col]);
    const blob = new Uint8Array(parts.reduce((s, p) => s + p.byteLength, 0));
    let o = 0;
    for (const p of parts) {
      blob.set(new Uint8Array(p.buffer, p.byteOffset, p.byteLength), o);
      o += p.byteLength;
    }
    D = { pos, tags, col, member, blob,
          first: prov ? prov.subarray(2 * N, 3 * N) : null };
    return D;
  }

  const bin = (a, len) => new Response(a, { headers: {
    "Content-Type": "application/octet-stream", "Content-Length": String(len) } });
  const jres = (o) => new Response(JSON.stringify(o),
                                   { headers: { "Content-Type": "application/json" } });

  const pct = (sorted, p) => {
    const x = (sorted.length - 1) * p / 100, lo = Math.floor(x), hi = Math.ceil(x);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (x - lo);
  };

  function eig3(A) {
    const a = A.map((r) => r.slice());
    const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    for (let sweep = 0; sweep < 50; sweep++) {
      let off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
      if (off < 1e-15) break;
      for (const [p, r] of [[0, 1], [0, 2], [1, 2]]) {
        if (Math.abs(a[p][r]) < 1e-300) continue;
        const th = (a[r][r] - a[p][p]) / (2 * a[p][r]);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < 3; k++) {
          const akp = a[k][p], akr = a[k][r];
          a[k][p] = c * akp - s * akr; a[k][r] = s * akp + c * akr;
        }
        for (let k = 0; k < 3; k++) {
          const apk = a[p][k], ark = a[r][k];
          a[p][k] = c * apk - s * ark; a[r][k] = s * apk + c * ark;
        }
        for (let k = 0; k < 3; k++) {
          const vkp = V[k][p], vkr = V[k][r];
          V[k][p] = c * vkp - s * vkr; V[k][r] = s * vkp + c * vkr;
        }
      }
    }
    return [0, 1, 2].map((i) => ({ val: a[i][i], vec: [V[0][i], V[1][i], V[2][i]] }))
      .sort((x, y) => y.val - x.val);
  }

  function shape(d, idx) {
    const n = idx.length, gn = SCENE.ground.normal, gd = SCENE.ground.offset;
    const v = SCENE.voxel, P = d.pos, C = d.col;
    const hs = new Float64Array(n);
    const flat = new Float64Array(n * 3);
    const mean = [0, 0, 0], rgb = [0, 0, 0];
    const vox = new Set();
    let ge = 0;
    for (let k = 0; k < n; k++) {
      const i = idx[k], x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
      const h = x * gn[0] + y * gn[1] + z * gn[2] + gd;
      hs[k] = h;
      for (let a = 0; a < 3; a++) {
        const f = P[3 * i + a] - h * gn[a];
        flat[3 * k + a] = f; mean[a] += f;
        rgb[a] += C[3 * i + a];
      }
      ge += 2 * C[3 * i + 1] - C[3 * i] - C[3 * i + 2];
      vox.add(((Math.floor(x / v) + 65536) * 131072 + (Math.floor(y / v) + 65536))
              * 131072 + (Math.floor(z / v) + 65536));
    }
    const sorted = Float64Array.from(hs).sort();
    const out = {
      height_m: sorted[n - 1] - Math.max(sorted[0], 0),
      base_height_m: pct(sorted, 2),
      top_height_m: sorted[n - 1],
      volume_m3: vox.size * v ** 3,
      mean_rgb: rgb.map((s) => s / n),
      green_excess: ge / n,
    };
    if (n > 3) {
      for (let a = 0; a < 3; a++) mean[a] /= n;
      const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let k = 0; k < n; k++) {
        const f = [0, 1, 2].map((a) => flat[3 * k + a] - mean[a]);
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) cov[a][b] += f[a] * f[b];
      }
      const [e0, e1] = eig3(cov);
      const span = (e) => {
        let lo = Infinity, hi = -Infinity;
        for (let k = 0; k < n; k++) {
          const p = (flat[3 * k] - mean[0]) * e.vec[0] + (flat[3 * k + 1] - mean[1]) * e.vec[1]
                  + (flat[3 * k + 2] - mean[2]) * e.vec[2];
          if (p < lo) lo = p; if (p > hi) hi = p;
        }
        return hi - lo;
      };
      out.crown_w_m = span(e0);
      out.crown_d_m = span(e1);
    }
    return out;
  }

  function bestViews(d, idx, topk = 6, minPoints = 40) {
    let sub = idx;
    if (idx.length > 20000) {
      sub = new Uint32Array(20000);
      for (let k = 0; k < 20000; k++) sub[k] = idx[Math.floor(Math.random() * idx.length)];
    }
    const [H, W] = SCENE.meta.model_hw || [420, 518];
    const out = [];
    for (const f of SCENE.frames) {
      const E = f.extrinsic, K = f.intrinsic;
      const us = [], vs = [], zs = [];
      let front = 0;
      for (const i of sub) {
        const x = d.pos[3 * i], y = d.pos[3 * i + 1], z = d.pos[3 * i + 2];
        const qz = E[8] * x + E[9] * y + E[10] * z + E[11];
        if (qz <= 0.05) continue;
        front++;
        const qx = E[0] * x + E[1] * y + E[2] * z + E[3];
        const qy = E[4] * x + E[5] * y + E[6] * z + E[7];
        const u = qx * K[0] / qz + K[2], v = qy * K[4] / qz + K[5];
        if (u >= 0 && u < W && v >= 0 && v < H) { us.push(u); vs.push(v); zs.push(qz); }
      }
      if (front < minPoints || us.length < minPoints) continue;
      const su = Float64Array.from(us).sort(), sv = Float64Array.from(vs).sort();
      const x0 = pct(su, 3), x1 = pct(su, 97), y0 = pct(sv, 3), y1 = pct(sv, 97);
      const cov = us.length / sub.length, fill = (x1 - x0) * (y1 - y0) / (W * H);
      out.push({ frame: f.id, coverage: cov, fill,
                 score: cov * Math.min(fill, 0.6) ** 0.5,
                 box: [x0, y0, x1, y1], n_visible: us.length,
                 distance: pct(Float64Array.from(zs).sort(), 50) });
    }
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, topk);
  }

  async function analyse(raw) {
    const d = await data();
    const idx = new Uint32Array(raw).filter((i) => i < N);
    if (!idx.length) return { n: 0 };
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    const c = [0, 0, 0];
    for (const i of idx) for (let a = 0; a < 3; a++) {
      const p = d.pos[3 * i + a];
      c[a] += p; if (p < lo[a]) lo[a] = p; if (p > hi[a]) hi[a] = p;
    }
    const out = { n: idx.length, centroid: c.map((s) => s / idx.length),
                  bbox_min: lo, bbox_max: hi, ...shape(d, idx) };
    const hits = new Map();
    for (const i of idx) { const t = d.tags[i]; if (t >= 0) hits.set(t, (hits.get(t) || 0) + 1); }
    out.overlap = [...hits].sort((a, b) => b[1] - a[1]).slice(0, 6).map(([t, n]) => (
      { id: SCENE.plants[t].id, n, category: SCENE.plants[t].category }));
    out.views = bestViews(d, idx, 8);
    return out;
  }

  const realFetch = window.fetch.bind(window);
  window.fetch = async (url, opts = {}) => {
    const path = typeof url === "string" ? url.split("?")[0] : "";
    if (!path.startsWith("/api/")) return realFetch(url, opts);
    if (path === "/api/scene.json") return jres(SCENE);
    if (path === "/api/points.bin") { const d = await data(); return bin(d.blob, d.blob.length); }
    if (path === "/api/select") return jres(await analyse(await new Response(opts.body).arrayBuffer()));
    if (path === "/api/identify")
      return jres({ error: "Identification needs the live viewer (view_map3d.py --classify ...)." });
    const p = path.split("/");
    if (p[2] === "plant") {
      const d = await data(), [s, n] = X.member[p[3]] || [0, 0];
      return bin(d.member.slice(s, s + n), n * 4);
    }
    if (p[2] === "frame") {
      const fid = +p[3];
      if (p[4] === "created.bin") {
        const d = await data(), hit = [];
        if (d.first) for (let i = 0; i < N; i++) if (d.first[i] === fid) hit.push(i);
        return bin(Uint32Array.from(hit), hit.length * 4);
      }
      if (p[4] === "own.bin") return bin(new Uint32Array(1), 4);
      return bin(new Uint32Array(0), 0);
    }
    return new Response("not in this export", { status: 404 });
  };

  let IMGS = null;
  const BLANK = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";
  window.__img = (kind, id) => {
    IMGS = IMGS || json("d_imgs");
    const b = (IMGS[kind] || {})[id];
    return b ? "data:image/jpeg;base64," + b : BLANK;
  };
})();
"""

def b64gz(a):
    return base64.b64encode(gzip.compress(np.ascontiguousarray(a).tobytes(), 6)).decode()

def jpeg(img, side, q):
    h, w = img.shape[:2]
    s = min(1.0, side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""

def patch(text, old, new, what):
    if old not in text:
        print(f"warning: could not patch {what}")
    return text.replace(old, new)

def main():
    ap = argparse.ArgumentParser(description="Export the map viewer as one offline HTML file")
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--plants", default=None)
    ap.add_argument("--out", default=None, help="Default: <artifacts name>_viewer.html")
    ap.add_argument("--max_points", type=int, default=0, help="Subsample the cloud (0 = all)")
    ap.add_argument("--photo_side", type=int, default=640,
                    help="Longest side of embedded frame photos (0 = no photos)")
    ap.add_argument("--highlight_radius", type=float, default=2.5)
    args = ap.parse_args()

    t0 = time.time()
    s = MapScene(args.artifacts, args.plants, args.highlight_radius, args.max_points)
    scene = s.scene_json()
    scene["live_classifier"] = "none"
    n = len(s.points)

    qmin = s.points.min(0).astype(np.float64)
    qstep = np.maximum(s.points.max(0) - qmin, 1e-6) / 65535.0
    q = np.round((s.points - qmin) / qstep).clip(0, 65535).astype(np.uint16)

    blob = s.points_blob()
    prov = blob[n * 14:n * 20] if s.has_prov else b""
    member, table, off = [], {}, 0
    for pid, idx in s.member.items():
        table[str(pid)] = [off, int(len(idx))]
        member.append(idx.astype(np.uint32))
        off += len(idx)
    member = np.concatenate(member) if member else np.zeros(0, np.uint32)

    imgs = {"i": {}, "t": {}, "c": {}}
    for f in s.frames:
        if not f["jpg"]:
            continue
        img = cv2.imread(f["jpg"])
        imgs["t"][f["id"]] = jpeg(img, 160, 80)
        if args.photo_side:
            imgs["i"][f["id"]] = jpeg(img, args.photo_side, 82)
    for p in s.plants:
        c = p.get("crop")
        path = os.path.join(s.art, "crops", c or "")
        if c and os.path.exists(path):
            imgs["c"][c] = jpeg(cv2.imread(path), 360, 85)

    js = open(os.path.join(WEB, "app.js"), encoding="utf-8").read()
    js = re.sub(r"/api/frame/\$\{([^}]+)\}/(image|thumb)\.jpg",
                lambda m: "${__img('%s', %s)}" % (m[2][0], m[1]), js)
    js = re.sub(r"/api/crop/\$\{([^}]+)\}", lambda m: "${__img('c', %s)}" % m[1], js)
    js = patch(js, '$("showOwn").checked = !$("showOwn").checked;',
               'if (!$("showOwn").disabled) $("showOwn").checked = !$("showOwn").checked;',
               "the depth-map hotkey")
    js = re.sub(r"Start the viewer with\s*<code>--classify bioclip</code> to enable identification\.",
                "Identification needs the live viewer.", js)

    html = open(os.path.join(WEB, "index.html"), encoding="utf-8").read()
    css = open(os.path.join(WEB, "style.css"), encoding="utf-8").read()
    three = open(os.path.join(WEB, "vendor", "three.module.js"), "rb").read()
    three_url = "data:text/javascript;base64," + base64.b64encode(three).decode()
    off_title = ' disabled title="Needs the live viewer (view_map3d.py)"'
    for el in ("showSeen", "showOwn"):
        html = patch(html, f'<input id="{el}" type="checkbox">',
                     f'<input id="{el}" type="checkbox"{off_title}>', el)
    html = patch(html, '<link rel="stylesheet" href="/static/style.css">',
                 f"<style>\n{css}\n</style>", "the stylesheet")
    html = re.sub(r'<script type="importmap">.*?</script>',
                  lambda _: '<script type="importmap">' + json.dumps(
                      {"imports": {"three": three_url}}) + "</script>",
                  html, flags=re.S)

    def tag(i, body, kind="application/octet-stream"):
        return f'<script type="{kind}" id="{i}">{body}</script>\n'

    safe = lambda o: json.dumps(o, separators=(",", ":")).replace("</", "<\\/")
    js = js.replace("</script", "<\\/script")
    payload = (
        tag("d_scene", safe(scene), "application/json")
        + tag("d_extra", safe({"qmin": qmin.tolist(), "qstep": qstep.tolist(),
                                "member": table}), "application/json")
        + tag("d_imgs", safe(imgs), "application/json")
        + tag("d_pos", b64gz(q)) + tag("d_tag", b64gz(s.tags))
        + (tag("d_prov", base64.b64encode(gzip.compress(prov, 6)).decode()) if prov else "")
        + tag("d_col", b64gz(s.colors)) + tag("d_member", b64gz(member))
        + f"<script>{SHIM}</script>\n"
        + f'<script type="module">\n{js}\n</script>'
    )
    html = patch(html, '<script type="module" src="/static/app.js"></script>',
                 payload, "the app script")

    out = args.out or os.path.basename(os.path.abspath(s.art)) + "_viewer.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out}: {os.path.getsize(out) / 1e6:.1f} MB, {n:,} points, "
          f"{len(imgs['i'])} photos, {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
