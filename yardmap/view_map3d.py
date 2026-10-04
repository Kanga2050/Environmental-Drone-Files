

import argparse
import base64
import io
import json
import mimetypes
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import colorsys
import traceback
from collections import defaultdict
from urllib.parse import parse_qs

import cv2
import numpy as np
from scipy.spatial import cKDTree

PLANT_FIELDS = {
    "id": None, "category": None, "category_confidence": 0.0,
    "detection_score": 0.0, "species": None, "species_common": None,
    "species_confidence": 0.0, "species_raw_score": 0.0,
    "species_identified": False, "species_candidates": [],
    "species_view_scores": [], "species_view_top1": [],
    "species_agreement": None, "species_spread": None,
    "species_sharpening": None, "species_entropy": None,
    "species_entropy_max": None, "species_unsupported": False,
    "views": 0, "frames": [], "crop": "", "bbox_min": None, "bbox_max": None,
    "centroid": None, "metrics": {}, "hierarchy": None,
}

MEAN_CONF = 2.43

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")

def instance_colors(n):
    return [[int(255 * c)
             for c in colorsys.hsv_to_rgb((i * 0.618033988749895) % 1, .85, 1)]
            for i in range(max(n, 1))]

def fit_plane(points, iters=200, tol=0.02, seed=0):

    rng = np.random.default_rng(seed)
    if len(points) < 3:
        return np.array([0.0, 1.0, 0.0]), 0.0, np.zeros(len(points), bool)
    sub = points if len(points) <= 60000 else points[
        rng.choice(len(points), 60000, replace=False)]
    best_n, best_d, best_in = None, 0.0, -1
    for _ in range(iters):
        idx = rng.choice(len(sub), 3, replace=False)
        a, b, c = sub[idx]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-8:
            continue
        n = n / ln
        d = -float(n @ a)
        cnt = int((np.abs(sub @ n + d) < tol).sum())
        if cnt > best_in:
            best_n, best_d, best_in = n, d, cnt
    if best_n is None:
        return np.array([0.0, 1.0, 0.0]), 0.0, np.zeros(len(points), bool)
    mask = np.abs(points @ best_n + best_d) < tol
    return best_n, best_d, mask

def camera_up(frames, mount="forward"):
    down = np.array([0.0, 1.0, 0.0])
    if frames:
        m = np.mean([f["extrinsic"][2 if mount == "down" else 1, :3] for f in frames], axis=0)
        if np.linalg.norm(m) > 1e-6:
            down = m / np.linalg.norm(m)
    return (-down).astype(np.float32)

class MapScene:

    def __init__(self, art, plants_file=None, highlight_radius=2.5,
                 max_points=0, conf_threshold=None, use_seg=True, mount="forward"):
        self.art = art
        self.use_seg = use_seg
        t0 = time.time()

        cloud = np.load(os.path.join(art, "cloud.npz"))
        pts = cloud["points"].astype(np.float32)
        cols = cloud["colors"].astype(np.uint8)

        rng = np.random.default_rng(0)
        if max_points and len(pts) > max_points:
            order = rng.choice(len(pts), max_points, replace=False)
        else:
            order = rng.permutation(len(pts))
        self.order = order
        self.n_cloud = len(pts)
        self.points = np.ascontiguousarray(pts[order])
        self.colors = np.ascontiguousarray(cols[order])
        del pts, cols

        self.has_prov = "first_frame" in cloud
        if self.has_prov:
            self.first_frame = cloud["first_frame"].astype(np.int32)[order]
            self.last_frame = cloud["last_frame"].astype(np.int32)[order]

            self.obs_exact = "obs_count" in cloud
            key = "obs_count" if self.obs_exact else "obs_weight"
            self.obs = cloud[key].astype(np.float32)[order]
            if not self.obs_exact:
                self.obs /= MEAN_CONF
            self.obs_p99 = float(np.percentile(self.obs, 99))
        else:
            self.first_frame = self.last_frame = self.obs = None
            self.obs_p99 = 1.0
            self.obs_exact = False

        self.meta = self._load_json("meta.json", {})
        self.voxel = float(self.meta.get("voxel_size", 0.005))
        self.conf_threshold = (conf_threshold if conf_threshold is not None
                               else float(self.meta.get("conf_threshold", 1.5)))
        self.model_hw = tuple(self.meta.get("model_hw", (420, 518)))

        pf = plants_file or os.path.join(art, "plants.json")
        pdata = self._load_json(pf, {"plants": []}, absolute=True)
        self.plants = pdata.get("plants", [])
        self.classifier_name = pdata.get("classifier", "?")
        self.palette = instance_colors(len(self.plants))

        self._load_frames()
        self.up = camera_up(self.frames, mount)

        self.tree = cKDTree(self.points)
        self._assign_plant_points(highlight_radius)
        self._ground()
        self._metrics()

        self._own_cache = {}
        self._seen_cache = {}
        self._lock = threading.Lock()
        self._classifier = None

        print(f"Scene ready in {time.time() - t0:.1f}s: "
              f"{len(self.points):,} points, {len(self.plants)} plants, "
              f"{len(self.frames)} frames")

    def _load_json(self, name, default, absolute=False):
        path = name if absolute else os.path.join(self.art, name)
        if not os.path.exists(path):
            return default
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _load_frames(self):
        fdir = os.path.join(self.art, "frames")
        paths = ((int(fid), os.path.join(fdir, f"{fid:06d}.npz"),
                  os.path.join(fdir, f"{fid:06d}.jpg"))
                 for fid in self.meta.get("saved_frame_ids", []))
        self.frames = [
            dict(id=fid, npz=npz, jpg=jpg if os.path.exists(jpg) else None,
                 extrinsic=d["extrinsic"].astype(np.float64),
                 intrinsic=d["intrinsic"].astype(np.float64))
            for fid, npz, jpg in paths if os.path.exists(npz)
            for d in [np.load(npz, allow_pickle=True)]]
        self.frame_by_id = {f["id"]: f for f in self.frames}

    def _frame_arrays(self, fid):
        f = self.frame_by_id[fid]
        d = np.load(f["npz"], allow_pickle=True)
        depth = d["depth"].astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth, d["conf"].astype(np.float32), f["extrinsic"], f["intrinsic"]

    def _assign_plant_points(self, radius_voxels):

        n = len(self.points)
        self.tags = np.full(n, -1, dtype=np.int16)
        self.member = {}
        self.seg_source = "radius"
        seg = os.path.join(self.art, "plant_seg.npz")
        if self.use_seg and os.path.exists(seg):
            d = np.load(seg)
            if (len(d["tags"]) == self.n_cloud
                    and list(d["plant_ids"]) == [p["id"] for p in self.plants]):
                # Learned assignment from plant_seg/segment.py.
                self.tags = d["tags"][self.order].astype(np.int16)
                for i, p in enumerate(self.plants):
                    self.member[p["id"]] = np.flatnonzero(self.tags == i).astype(np.uint32)
                self.seg_source = "point_transformer"
                return
            print("  plant_seg.npz does not match this cloud/plants.json; "
                  "using the radius rule")
        path = os.path.join(self.art, "plant_points.npz")
        inst = np.load(path) if os.path.exists(path) else None
        radius = self.voxel * radius_voxels

        order = sorted(range(len(self.plants)),
                       key=lambda i: -float(np.prod(
                           np.array(self.plants[i]["bbox_max"])
                           - np.array(self.plants[i]["bbox_min"]) + 1e-6)))
        for i in order:
            p = self.plants[i]
            idx = np.zeros(0, dtype=np.int64)
            key = f"pts_{p['id']}"
            if inst is not None and key in inst and len(inst[key]):
                hits = self.tree.query_ball_point(
                    inst[key].astype(np.float32), r=radius, workers=-1)
                flat = [h for h in hits if h]
                if flat:
                    idx = np.unique(np.concatenate(
                        [np.asarray(h, dtype=np.int64) for h in flat]))
            lo = np.array(p["bbox_min"], dtype=np.float32)
            hi = np.array(p["bbox_max"], dtype=np.float32)
            box = np.flatnonzero(np.all((self.points >= lo) & (self.points <= hi), axis=1))
            if len(idx) < 0.02 * max(len(box), 1):
                idx = box
            self.member[p["id"]] = idx.astype(np.uint32)
            self.tags[idx] = i

    def _ground(self):

        h = self.points @ self.up
        low = self.points[h < np.percentile(h, 45)]
        best = None
        for s in range(6):
            n, d, mask = fit_plane(low, iters=120, tol=2.5 * self.voxel, seed=s)
            if n @ self.up < 0:
                n, d = -n, -d
            align = float(n @ self.up)
            score = align * mask.mean()
            if align > 0.75 and (best is None or score > best[0]):
                best = (score, n, d)
        if best is None:
            self.ground_n = self.up.astype(np.float64)
            self.ground_d = -float(np.percentile(h, 2))
        else:
            _, self.ground_n, self.ground_d = best
        self.ground_n = np.asarray(self.ground_n, dtype=np.float64)

    def height_above_ground(self, xyz):
        return np.asarray(xyz, dtype=np.float64) @ self.ground_n + self.ground_d

    def _shape(self, q, c):
        hh = self.height_above_ground(q)
        flat = q - np.outer(hh, self.ground_n)
        flat = flat - flat.mean(0)
        voxels = np.unique(np.floor(q / self.voxel).astype(np.int64), axis=0)
        out = {
            "height_m": float(hh.max() - max(hh.min(), 0.0)),
            "base_height_m": float(np.percentile(hh, 2)),
            "top_height_m": float(hh.max()),
            "volume_m3": float(len(voxels) * self.voxel ** 3),
            "mean_rgb": c.mean(0).tolist(),
            "green_excess": float((2 * c[:, 1] - c[:, 0] - c[:, 2]).mean()),
        }
        if len(flat) > 3:
            proj = flat @ np.linalg.svd(flat, full_matrices=False)[2][:2].T
            out["crown_w_m"] = float(np.ptp(proj[:, 0]))
            out["crown_d_m"] = float(np.ptp(proj[:, 1]))
        return out

    def _metrics(self):
        for p in self.plants:
            idx = self.member.get(p["id"], np.zeros(0, np.uint32))
            p["metrics"] = {"n_cloud_points": int(len(idx))}
            if len(idx):
                p["metrics"] |= self._shape(
                    self.points[idx], self.colors[idx].astype(np.float32))

    def frame_seen(self, fid):

        with self._lock:
            if fid in self._seen_cache:
                return self._seen_cache[fid]
        depth, conf, E, K = self._frame_arrays(fid)
        H, W = depth.shape
        R, t = E[:, :3], E[:, 3]
        Q = self.points @ R.T.astype(np.float32) + t.astype(np.float32)
        z = Q[:, 2]
        ok = z > 0.05
        qi = Q[ok]
        u = (qi[:, 0] * K[0, 0] / qi[:, 2] + K[0, 2])
        v = (qi[:, 1] * K[1, 1] / qi[:, 2] + K[1, 2])
        inb = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        gi = np.flatnonzero(ok)[inb]
        ui = u[inb].astype(np.int32)
        vi = v[inb].astype(np.int32)
        dz = depth[vi, ui]
        cz = conf[vi, ui]
        zz = z[gi]
        tol = np.maximum(3.0 * self.voxel, 0.03 * zz)
        keep = (np.abs(zz - dz) < tol) & (cz >= self.conf_threshold) & (dz > 0)
        out = gi[keep].astype(np.uint32)
        with self._lock:
            self._seen_cache[fid] = out
            if len(self._seen_cache) > 48:
                self._seen_cache.pop(next(iter(self._seen_cache)))
        return out

    def frame_own(self, fid, stride=2):

        key = (fid, stride)
        with self._lock:
            if key in self._own_cache:
                return self._own_cache[key]
        depth, conf, E, K = self._frame_arrays(fid)
        H, W = depth.shape
        ys, xs = np.mgrid[0:H:stride, 0:W:stride]
        d = depth[ys, xs]
        c = conf[ys, xs]
        good = (d > 0) & (c >= self.conf_threshold)
        xs, ys, d = xs[good], ys[good], d[good]
        x = (xs - K[0, 2]) * d / K[0, 0]
        y = (ys - K[1, 2]) * d / K[1, 1]
        cam = np.stack([x, y, d], axis=1)
        R, t = E[:, :3], E[:, 3]
        world = (cam - t) @ R
        rgb = None
        f = self.frame_by_id[fid]
        if f["jpg"]:
            img = cv2.cvtColor(cv2.imread(f["jpg"]), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            rgb = img[ys, xs]
        else:
            rgb = np.full((len(world), 3), 200, np.uint8)
        out = (world.astype(np.float32), rgb.astype(np.uint8))
        with self._lock:
            self._own_cache[key] = out
            if len(self._own_cache) > 24:
                self._own_cache.pop(next(iter(self._own_cache)))
        return out

    def point_info(self, i):

        i = int(i)
        out = {"index": i,
               "position": [float(x) for x in self.points[i]],
               "color": [int(x) for x in self.colors[i]],
               "height": float(self.height_above_ground(self.points[i])),
               "tag": int(self.tags[i])}
        if self.has_prov:
            out["first_frame"] = int(self.first_frame[i])
            out["last_frame"] = int(self.last_frame[i])
            out["observations"] = float(self.obs[i])
            out["observations_exact"] = bool(self.obs_exact)
        return out

    def analyse(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < len(self.points))]
        if not len(idx):
            return {"n": 0}
        q = self.points[idx]
        out = {"n": int(len(idx)), "centroid": q.mean(0).tolist(),
               "bbox_min": q.min(0).tolist(), "bbox_max": q.max(0).tolist()}
        out |= self._shape(q, self.colors[idx].astype(np.float32))
        tags = self.tags[idx]
        hits = np.bincount(tags[tags >= 0], minlength=len(self.plants))
        out["overlap"] = [
            {"id": self.plants[i]["id"], "n": int(hits[i]),
             "category": self.plants[i]["category"]}
            for i in np.argsort(-hits)[:6] if hits[i]]
        out["views"] = self.best_views(idx, topk=8)
        return out

    def best_views(self, idx, topk=6, min_points=40):

        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) > 20000:
            idx = np.random.default_rng(0).choice(idx, 20000, replace=False)
        P = self.points[idx]
        H, W = self.model_hw
        out = []
        for f in self.frames:
            E, K = f["extrinsic"], f["intrinsic"]
            Q = P @ E[:, :3].T + E[:, 3]
            qi = Q[Q[:, 2] > 0.05]
            if len(qi) < min_points:
                continue
            u = qi[:, 0] * K[0, 0] / qi[:, 2] + K[0, 2]
            v = qi[:, 1] * K[1, 1] / qi[:, 2] + K[1, 2]
            inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            n = int(inb.sum())
            if n < min_points:
                continue
            x0, x1 = np.percentile(u[inb], [3, 97])
            y0, y1 = np.percentile(v[inb], [3, 97])
            cov = n / len(P)
            fill = (x1 - x0) * (y1 - y0) / float(W * H)
            out.append({
                "frame": f["id"], "coverage": float(cov), "fill": float(fill),
                "score": float(cov * min(fill, 0.6) ** 0.5),
                "box": [float(x0), float(y0), float(x1), float(y1)],
                "n_visible": n,
                "distance": float(np.median(qi[inb][:, 2])),
            })
        out.sort(key=lambda r: -r["score"])
        return out[:topk]

    def crop_for_view(self, view, pad=0.12, min_px=48):

        f = self.frame_by_id[view["frame"]]
        if not f["jpg"]:
            return None
        img = cv2.cvtColor(cv2.imread(f["jpg"]), cv2.COLOR_BGR2RGB)
        H, W = self.model_hw
        sy, sx = img.shape[0] / H, img.shape[1] / W
        x0, y0, x1, y1 = view["box"]
        w, h = x1 - x0, y1 - y0
        x0 -= w * pad; x1 += w * pad
        y0 -= h * pad; y1 += h * pad
        x0 = int(max(0, x0 * sx)); x1 = int(min(img.shape[1], x1 * sx))
        y0 = int(max(0, y0 * sy)); y1 = int(min(img.shape[0], y1 * sy))
        if x1 - x0 < min_px or y1 - y0 < min_px:
            return None
        return img[y0:y1, x0:x1]

    def identify(self, idx, topk=5, max_views=5):

        picks = [(v, c) for v in self.best_views(idx, topk=max_views * 2)
                 if (c := self.crop_for_view(v)) is not None][:max_views]
        if not picks:
            return {"error": "no view of this selection is large enough to crop"}
        used, crops = zip(*picks)
        res = self._classify(list(crops), topk=topk)
        return {"results": res, "classifier": self.classifier_kind,
                "views": [self._thumb_of(c, v) for c, v in zip(crops, used)]}

    def _thumb_of(self, crop, view, side=240):
        h, w = crop.shape[:2]
        s = side / max(h, w)
        small = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 88])
        return {"frame": view["frame"], "coverage": view["coverage"],
                "distance": view["distance"],
                "jpeg": base64.b64encode(buf.tobytes()).decode() if ok else ""}

    classifier_kind = "none"
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

    def enable_classifier(self, kind, plantnet_key=None, models_dir=None,
                          plantnet_project=None):
        self.classifier_kind = kind
        self._plantnet_key = plantnet_key
        self._plantnet_project = plantnet_project
        if models_dir:
            self.models_dir = models_dir

    def _classify(self, crops, topk=5):
        if self.classifier_kind == "none":
            return [{"name": "classifier disabled",
                     "common": "start with --classify bioclip", "score": 0.0}]
        sys.path.insert(0, HERE)
        import detect_plants as dp
        if self._classifier is None:
            if self.classifier_kind == "plantnet":
                self._classifier = dp.PlantNetClassifier(
                    self._plantnet_key, project=self._plantnet_project)
            else:
                self._classifier = dp.BioClipClassifier(
                    os.path.join(self.models_dir, "bioclip"), "cuda",
                    dp.DEFAULT_SPECIES)
        if hasattr(self._classifier, "classify_many"):

            fused = self._classifier.classify_many(crops, topk=topk)
            res = [{"name": a, "common": b, "score": float(c),
                    "views": len(crops)} for a, b, c in fused]
        else:
            pool = defaultdict(list)
            for c in crops:
                for name, common, score in self._classifier(c, topk=topk):
                    pool[name, common].append(score)
            res = sorted(({"name": n, "common": cm, "score": float(np.mean(v)),
                           "views": len(v)} for (n, cm), v in pool.items()),
                         key=lambda r: -r["score"])[:topk]
        return res

    _blob = None

    def channels(self):

        base = ["pos", "tag"]
        if self.has_prov:
            base += ["obs", "frame", "first"]
        return base + ["color"]

    def points_blob(self):

        if self._blob is None:
            u16 = lambda a: np.clip(a, 0, 65535).astype(np.uint16).tobytes()
            parts = [self.points.tobytes(), self.tags.tobytes()]
            if self.has_prov:
                parts += [u16(self.obs), u16(self.last_frame),
                          u16(self.first_frame)]
            self._blob = b"".join(parts + [self.colors.tobytes()])
        return self._blob

    def frame_created(self, fid):

        if not self.has_prov:
            return np.zeros(0, np.uint32)
        return np.flatnonzero(self.first_frame == fid).astype(np.uint32)

    def framing(self):

        sub = self.points[::max(1, len(self.points) // 200000)]
        focus = np.median(sub, axis=0)
        radius = float(np.percentile(np.linalg.norm(sub - focus, axis=1), 90))
        fwd = np.sum([f["extrinsic"][2, :3] for f in self.frames] or [(0, 0, 1)],
                     axis=0)
        fwd = fwd / n if (n := np.linalg.norm(fwd)) > 1e-6 else np.array([0., 0., 1.])
        flat = fwd - self.up * float(fwd @ self.up)
        flat = flat / n if (n := np.linalg.norm(flat)) > 1e-3 else fwd
        return {"focus": focus.tolist(), "radius": max(radius, 0.2),
                "forward": [float(x) for x in flat]}

    def scene_json(self):
        gmin = self.height_above_ground(self.points).min()
        return {
            "framing": self.framing(),
            "artifacts": os.path.basename(os.path.abspath(self.art)),
            "n_points": int(len(self.points)),
            "voxel": self.voxel,
            "up": self.up.tolist(),
            "ground": {"normal": self.ground_n.tolist(),
                       "offset": float(self.ground_d), "min": float(gmin)},
            "bbox_min": self.points.min(0).tolist(),
            "bbox_max": self.points.max(0).tolist(),
            "classifier": self.classifier_name,
            "live_classifier": self.classifier_kind,
            "channels": self.channels(),
            "provenance": bool(self.has_prov),
            "created_frames": (sorted(int(x) for x in
                                      np.unique(self.first_frame))[:20000]
                               if self.has_prov else []),
            "obs_p99": self.obs_p99,
            "obs_exact": bool(getattr(self, "obs_exact", False)),
            "num_frames": int(self.meta.get("num_frames", 0)),
            "meta": {k: v for k, v in self.meta.items()
                     if k in ("num_frames", "fps", "source", "model_hw",
                              "conf_threshold", "crop_margins")},
            "palette": self.palette,
            "plants": [{k: p.get(k, d) for k, d in PLANT_FIELDS.items()}
                       for p in self.plants],
            "frames": [{
                "id": f["id"],
                "extrinsic": f["extrinsic"].reshape(-1).tolist(),
                "intrinsic": f["intrinsic"].reshape(-1).tolist(),
                "position": (-f["extrinsic"][:, :3].T
                             @ f["extrinsic"][:, 3]).tolist(),
                "has_image": bool(f["jpg"]),
            } for f in self.frames],
        }

class Handler(BaseHTTPRequestHandler):
    scene: MapScene = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, body, ctype="application/octet-stream", code=200,
              cache=False, etag=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=3600")
        elif etag:

            self.send_header("Cache-Control", "no-cache")
            self.send_header("ETag", etag)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(json.dumps(obj), "application/json", code)

    def _file(self, path, ctype=None):
        if not os.path.exists(path):
            return self._send("not found", "text/plain", 404)
        st = os.stat(path)

        hard = path.endswith(".jpg")
        etag = None
        if not hard:
            etag = f'"{int(st.st_mtime):x}-{st.st_size:x}"'
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", "0")
                return self.end_headers()
        with open(path, "rb") as f:
            body = f.read()
        ctype = ctype or (mimetypes.guess_type(path)[0] or "application/octet-stream")
        self._send(body, ctype, cache=hard, etag=etag)

    def do_GET(self):
        s = self.scene
        path = unquote(urlparse(self.path).path)
        try:
            if path in ("/", "/index.html"):
                return self._file(os.path.join(WEB, "index.html"), "text/html")
            if path.startswith("/static/"):
                rel = path[len("/static/"):]
                full = os.path.normpath(os.path.join(WEB, rel))
                if not full.startswith(WEB):
                    return self._send("no", "text/plain", 403)
                return self._file(full)

            if path == "/api/scene.json":
                return self._json(s.scene_json())
            if path == "/api/points.bin":
                return self._send(s.points_blob())

            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[1] == "plant" and parts[3] == "points.bin":
                idx = s.member.get(int(parts[2]), np.zeros(0, np.uint32))
                return self._send(idx.tobytes())
            if len(parts) == 4 and parts[1] == "frame":
                fid = int(parts[2])
                if fid not in s.frame_by_id:
                    return self._send("no such frame", "text/plain", 404)
                if parts[3] == "seen.bin":
                    return self._send(s.frame_seen(fid).tobytes())
                if parts[3] == "created.bin":
                    return self._send(s.frame_created(fid).tobytes())
                if parts[3] == "own.bin":
                    stride = int(self._q("stride", 2))
                    xyz, rgb = s.frame_own(fid, stride)
                    head = np.array([len(xyz)], dtype=np.uint32).tobytes()
                    return self._send(head + xyz.tobytes() + rgb.tobytes())
                if parts[3] == "image.jpg":
                    return self._file(s.frame_by_id[fid]["jpg"] or "", "image/jpeg")
                if parts[3] == "thumb.jpg":
                    return self._thumb(s.frame_by_id[fid]["jpg"])
            if len(parts) == 3 and parts[1] == "crop":
                return self._file(os.path.join(s.art, "crops", parts[2]), "image/jpeg")
        except Exception as exc:
            traceback.print_exc()
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        return self._send("not found", "text/plain", 404)

    def _q(self, key, default):
        return parse_qs(urlparse(self.path).query).get(key, [default])[0]

    def _thumb(self, jpg):
        if not jpg or not os.path.exists(jpg):
            return self._send("no image", "text/plain", 404)
        img = cv2.imread(jpg)
        h, w = img.shape[:2]
        s = 160.0 / max(w, 1)
        img = cv2.resize(img, (160, max(1, int(h * s))))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return self._send(buf.tobytes(), "image/jpeg", cache=True)

    def do_POST(self):
        s = self.scene
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        try:
            if path in ("/api/select", "/api/identify"):
                idx = np.frombuffer(raw, dtype=np.uint32)
                if path == "/api/select":
                    return self._json(s.analyse(idx))
                return self._json(s.identify(idx))
        except Exception as exc:
            traceback.print_exc()
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        return self._send("not found", "text/plain", 404)

def main():
    ap = argparse.ArgumentParser(
        description="Navigable point-cloud viewer with plant analysis tools")
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--plants", default=None, help="Default: <artifacts>/plants.json")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--highlight_radius", type=float, default=2.5,
                    help="Cloud points within this many map-voxels of a detected "
                         "plant point are tagged as that plant")
    ap.add_argument("--seg", action=argparse.BooleanOptionalAction, default=True,
                    help="Use the learned plant assignment in <artifacts>/plant_seg.npz "
                         "when present (--no-seg falls back to the radius rule)")
    ap.add_argument("--max_points", type=int, default=0,
                    help="0 keeps every point; the browser subsamples with a slider")
    ap.add_argument("--classify", default="none",
                    choices=["none", "bioclip", "plantnet"],
                    help="Enable the viewer's 'identify selection' button. plantnet "
                         "uploads the selected crops to an external service")
    ap.add_argument("--plantnet_api_key", default=None)
    ap.add_argument("--plantnet_project", default="k-world-flora")
    ap.add_argument("--models_dir",
                    default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
    args = ap.parse_args()

    scene = MapScene(args.artifacts, args.plants, args.highlight_radius,
                     args.max_points, use_seg=args.seg)
    if args.classify != "none":
        if args.classify == "plantnet" and not args.plantnet_api_key:
            ap.error("--classify plantnet requires --plantnet_api_key")
        if args.classify == "plantnet":
            print("  NOTE: Pl@ntNet uploads every crop you identify to an "
                  "external service.")
        scene.enable_classifier(args.classify, args.plantnet_api_key,
                                args.models_dir, args.plantnet_project)

    Handler.scene = scene
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"\n  Viewer at http://localhost:{args.port}   (Ctrl-C to exit)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
