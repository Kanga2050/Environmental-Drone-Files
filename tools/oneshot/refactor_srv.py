import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\view_map3d.py"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''import cv2
import numpy as np
from scipy.spatial import cKDTree''',
    '''import colorsys
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
    "centroid": None, "metrics": {},
}'''))

subs.append((
    '''def instance_colors(n):

    import colorsys
    out = []
    for i in range(max(n, 1)):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.85, 1.0)
        out.append([int(r * 255), int(g * 255), int(b * 255)])
    return out''',
    '''def instance_colors(n):
    return [[int(255 * c)
             for c in colorsys.hsv_to_rgb((i * 0.618033988749895) % 1, .85, 1)]
            for i in range(max(n, 1))]'''))

subs.append((
    '''def camera_up(frames):

    if not frames:
        return np.array([0.0, -1.0, 0.0], dtype=np.float32)
    downs = np.array([f["extrinsic"][1, :3] for f in frames], dtype=np.float64)
    m = downs.mean(0)
    n = np.linalg.norm(m)
    if n < 1e-6:
        return np.array([0.0, -1.0, 0.0], dtype=np.float32)
    return (-m / n).astype(np.float32)''',
    '''def camera_up(frames):
    down = np.array([0.0, 1.0, 0.0])
    if frames:
        m = np.mean([f["extrinsic"][1, :3] for f in frames], axis=0)
        if np.linalg.norm(m) > 1e-6:
            down = m / np.linalg.norm(m)
    return (-down).astype(np.float32)'''))

subs.append((
    '''            if "obs_count" in cloud:
                self.obs = cloud["obs_count"].astype(np.float32)[order]
                self.obs_exact = True
            else:
                self.obs = (cloud["obs_weight"].astype(np.float32)[order]
                            / MEAN_CONF)
                self.obs_exact = False
            self.obs_p99 = float(np.percentile(self.obs, 99))''',
    '''            self.obs_exact = "obs_count" in cloud
            key = "obs_count" if self.obs_exact else "obs_weight"
            self.obs = cloud[key].astype(np.float32)[order]
            if not self.obs_exact:
                self.obs /= MEAN_CONF
            self.obs_p99 = float(np.percentile(self.obs, 99))'''))

subs.append((
    '''    def _load_frames(self):
        self.frames = []
        ids = self.meta.get("saved_frame_ids", [])
        fdir = os.path.join(self.art, "frames")
        for fid in ids:
            npz = os.path.join(fdir, f"{fid:06d}.npz")
            jpg = os.path.join(fdir, f"{fid:06d}.jpg")
            if not os.path.exists(npz):
                continue
            d = np.load(npz, allow_pickle=True)
            self.frames.append({
                "id": int(fid),
                "extrinsic": d["extrinsic"].astype(np.float64),
                "intrinsic": d["intrinsic"].astype(np.float64),
                "npz": npz,
                "jpg": jpg if os.path.exists(jpg) else None,
            })
        self.frame_by_id = {f["id"]: f for f in self.frames}''',
    '''    def _load_frames(self):
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
        self.plants_by_id = {p["id"]: p for p in self.plants}'''))

subs.append((
    '''    @property
    def plants_by_id(self):
        return {p["id"]: p for p in self.plants}

''', ''))

subs.append((
    '''    def _metrics(self):

        v3 = self.voxel ** 3
        for i, p in enumerate(self.plants):
            idx = self.member.get(p["id"], np.zeros(0, np.uint32))
            m = {"n_cloud_points": int(len(idx))}
            if len(idx):
                q = self.points[idx]
                hh = self.height_above_ground(q)
                m["height_m"] = float(hh.max() - max(hh.min(), 0.0))
                m["base_height_m"] = float(np.percentile(hh, 2))
                m["top_height_m"] = float(hh.max())
                flat = q - np.outer(hh, self.ground_n)
                flat = flat - flat.mean(0)
                if len(flat) > 3:

                    u, s, vt = np.linalg.svd(flat - flat.mean(0), full_matrices=False)
                    proj = flat @ vt[:2].T
                    m["crown_w_m"] = float(proj[:, 0].max() - proj[:, 0].min())
                    m["crown_d_m"] = float(proj[:, 1].max() - proj[:, 1].min())
                keys = np.unique(np.floor(q / self.voxel).astype(np.int64), axis=0)
                m["volume_m3"] = float(len(keys) * v3)
                c = self.colors[idx].astype(np.float32)
                m["mean_rgb"] = [float(x) for x in c.mean(0)]
                m["green_excess"] = float((2 * c[:, 1] - c[:, 0] - c[:, 2]).mean())
            p["metrics"] = m''',
    '''    def _shape(self, q, c):
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
                    self.points[idx], self.colors[idx].astype(np.float32))'''))

subs.append((
    '''    def analyse(self, idx):

        idx = np.asarray(idx, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < len(self.points))]
        if not len(idx):
            return {"n": 0}
        q = self.points[idx]
        c = self.colors[idx].astype(np.float32)
        hh = self.height_above_ground(q)
        flat = q - np.outer(hh, self.ground_n)
        flat = flat - flat.mean(0)
        out = {
            "n": int(len(idx)),
            "centroid": [float(x) for x in q.mean(0)],
            "bbox_min": [float(x) for x in q.min(0)],
            "bbox_max": [float(x) for x in q.max(0)],
            "height_m": float(hh.max() - max(hh.min(), 0.0)),
            "base_height_m": float(np.percentile(hh, 2)),
            "top_height_m": float(hh.max()),
            "mean_rgb": [float(x) for x in c.mean(0)],
            "green_excess": float((2 * c[:, 1] - c[:, 0] - c[:, 2]).mean()),
            "volume_m3": float(len(np.unique(
                np.floor(q / self.voxel).astype(np.int64), axis=0)) * self.voxel ** 3),
        }
        if len(flat) > 3:
            _, _, vt = np.linalg.svd(flat, full_matrices=False)
            proj = flat @ vt[:2].T
            out["crown_w_m"] = float(proj[:, 0].max() - proj[:, 0].min())
            out["crown_d_m"] = float(proj[:, 1].max() - proj[:, 1].min())
        overlap = {}
        for i, p in enumerate(self.plants):
            t = self.tags[idx] == i
            if t.any():
                overlap[p["id"]] = int(t.sum())
        out["overlap"] = [{"id": k, "n": v, "category": self.plants_by_id[k]["category"]}
                          for k, v in sorted(overlap.items(), key=lambda kv: -kv[1])[:6]]
        out["views"] = self.best_views(idx, topk=8)
        return out''',
    '''    def analyse(self, idx):
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
        return out'''))

subs.append((
    '''        P = self.points[idx]
        out = []
        for f in self.frames:
            E, K = f["extrinsic"], f["intrinsic"]
            R, t = E[:, :3], E[:, 3]
            Q = P @ R.T + t
            z = Q[:, 2]
            ok = z > 0.05
            if ok.sum() < min_points:
                continue
            qi = Q[ok]
            u = qi[:, 0] * K[0, 0] / qi[:, 2] + K[0, 2]
            v = qi[:, 1] * K[1, 1] / qi[:, 2] + K[1, 2]
            H, W = self.model_hw
            inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if inb.sum() < min_points:
                continue
            u, v = u[inb], v[inb]
            cov = inb.sum() / len(P)

            x0, x1 = np.percentile(u, [3, 97])
            y0, y1 = np.percentile(v, [3, 97])
            fill = ((x1 - x0) * (y1 - y0)) / float(W * H)
            out.append({
                "frame": f["id"], "coverage": float(cov), "fill": float(fill),
                "score": float(cov * min(fill, 0.6) ** 0.5),
                "box": [float(x0), float(y0), float(x1), float(y1)],
                "n_visible": int(inb.sum()),
                "distance": float(np.median(qi[inb][:, 2])),
            })
        out.sort(key=lambda r: -r["score"])
        return out[:topk]''',
    '''        P = self.points[idx]
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
        return out[:topk]'''))

subs.append((
    '''        views = self.best_views(idx, topk=max_views * 2)
        crops, used = [], []
        for v in views:
            c = self.crop_for_view(v)
            if c is None:
                continue
            crops.append(c)
            used.append(v)
            if len(crops) >= max_views:
                break
        if not crops:''',
    '''        pairs = ((v, self.crop_for_view(v))
                 for v in self.best_views(idx, topk=max_views * 2))
        used, crops = zip(*[(v, c) for v, c in pairs if c is not None][:max_views]) \\
            or ((), ())
        if not crops:'''))

subs.append((
    '''        thumbs = []
        for c, v in zip(crops, used):
            h, w = c.shape[:2]
            s = 240.0 / max(h, w)
            t = cv2.resize(c, (max(1, int(w * s)), max(1, int(h * s))))
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(t, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 88])
            thumbs.append({
                "frame": v["frame"],
                "jpeg": base64.b64encode(buf.tobytes()).decode() if ok else "",
                "coverage": v["coverage"], "distance": v["distance"],
            })
        return {"results": res, "views": thumbs, "classifier": self.classifier_kind}''',
    '''        return {"results": res, "classifier": self.classifier_kind,
                "views": [self._thumb_of(c, v) for c, v in zip(crops, used)]}

    def _thumb_of(self, crop, view, side=240):
        h, w = crop.shape[:2]
        s = side / max(h, w)
        small = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 88])
        return {"frame": view["frame"], "coverage": view["coverage"],
                "distance": view["distance"],
                "jpeg": base64.b64encode(buf.tobytes()).decode() if ok else ""}'''))

subs.append((
    '''            pool = {}
            for c in crops:
                for name, common, score in self._classifier(c, topk=topk):
                    e = pool.setdefault(name, {"common": common, "scores": []})
                    e["scores"].append(score)
            res = [{"name": k, "common": v["common"],
                    "score": float(np.mean(v["scores"])),
                    "views": len(v["scores"])}
                   for k, v in pool.items()]
            res.sort(key=lambda r: -r["score"])
            res = res[:topk]
        return [dict(r) for r in res]''',
    '''            pool = defaultdict(list)
            for c in crops:
                for name, common, score in self._classifier(c, topk=topk):
                    pool[name, common].append(score)
            res = sorted(({"name": n, "common": cm, "score": float(np.mean(v)),
                           "views": len(v)} for (n, cm), v in pool.items()),
                         key=lambda r: -r["score"])[:topk]
        return res'''))

subs.append((
    '''        if self._blob is None:
            parts = [self.points.tobytes(), self.tags.tobytes()]
            if self.has_prov:

                parts.append(np.clip(self.obs, 0, 65535).astype(np.uint16).tobytes())
                parts.append(np.clip(self.last_frame, 0, 65535)
                             .astype(np.uint16).tobytes())
                parts.append(np.clip(self.first_frame, 0, 65535)
                             .astype(np.uint16).tobytes())
            parts.append(self.colors.tobytes())
            self._blob = b"".join(parts)
        return self._blob''',
    '''        if self._blob is None:
            u16 = lambda a: np.clip(a, 0, 65535).astype(np.uint16).tobytes()
            parts = [self.points.tobytes(), self.tags.tobytes()]
            if self.has_prov:
                parts += [u16(self.obs), u16(self.last_frame),
                          u16(self.first_frame)]
            self._blob = b"".join(parts + [self.colors.tobytes()])
        return self._blob'''))

subs.append((
    '''        sub = self.points[::max(1, len(self.points) // 200000)]
        focus = np.median(sub, axis=0)
        radius = float(np.percentile(np.linalg.norm(sub - focus, axis=1), 90))
        fwd = np.zeros(3)
        for f in self.frames:
            fwd += f["extrinsic"][2, :3]
        if np.linalg.norm(fwd) > 1e-6:
            fwd /= np.linalg.norm(fwd)
        else:
            fwd = np.array([0.0, 0.0, 1.0])
        flat = fwd - self.up * float(fwd @ self.up)
        if np.linalg.norm(flat) > 1e-3:
            flat /= np.linalg.norm(flat)
        else:
            flat = fwd
        return {"focus": [float(x) for x in focus], "radius": max(radius, 0.2),
                "forward": [float(x) for x in flat]}''',
    '''        sub = self.points[::max(1, len(self.points) // 200000)]
        focus = np.median(sub, axis=0)
        radius = float(np.percentile(np.linalg.norm(sub - focus, axis=1), 90))
        fwd = np.sum([f["extrinsic"][2, :3] for f in self.frames] or [(0, 0, 1)],
                     axis=0)
        fwd = fwd / n if (n := np.linalg.norm(fwd)) > 1e-6 else np.array([0., 0., 1.])
        flat = fwd - self.up * float(fwd @ self.up)
        flat = flat / n if (n := np.linalg.norm(flat)) > 1e-3 else fwd
        return {"focus": focus.tolist(), "radius": max(radius, 0.2),
                "forward": [float(x) for x in flat]}'''))

subs.append((
    '''            "plants": [{
                "id": p["id"], "category": p["category"],
                "category_confidence": p.get("category_confidence", 0.0),
                "detection_score": p.get("detection_score", 0.0),
                "species": p.get("species"), "species_common": p.get("species_common"),
                "species_confidence": p.get("species_confidence", 0.0),
                "species_raw_score": p.get("species_raw_score", 0.0),
                "species_identified": p.get("species_identified", False),
                "species_candidates": p.get("species_candidates", []),
                "species_view_scores": p.get("species_view_scores", []),
                "species_view_top1": p.get("species_view_top1", []),
                "species_agreement": p.get("species_agreement"),
                "species_spread": p.get("species_spread"),
                "species_sharpening": p.get("species_sharpening"),
                "species_entropy": p.get("species_entropy"),
                "species_entropy_max": p.get("species_entropy_max"),
                "species_unsupported": p.get("species_unsupported", False),
                "views": p.get("views", 0), "frames": p.get("frames", []),
                "crop": p.get("crop", ""),
                "bbox_min": p["bbox_min"], "bbox_max": p["bbox_max"],
                "centroid": p["centroid"], "metrics": p.get("metrics", {}),
            } for p in self.plants],''',
    '''            "plants": [{k: p.get(k, d) for k, d in PLANT_FIELDS.items()}
                       for p in self.plants],'''))

subs.append((
    '''            "up": [float(x) for x in self.up],
            "ground": {"normal": [float(x) for x in self.ground_n],
                       "offset": float(self.ground_d), "min": float(gmin)},
            "bbox_min": [float(x) for x in self.points.min(0)],
            "bbox_max": [float(x) for x in self.points.max(0)],''',
    '''            "up": self.up.tolist(),
            "ground": {"normal": self.ground_n.tolist(),
                       "offset": float(self.ground_d), "min": float(gmin)},
            "bbox_min": self.points.min(0).tolist(),
            "bbox_max": self.points.max(0).tolist(),'''))

subs.append((
    '''                "position": [float(x) for x in
                             (-f["extrinsic"][:, :3].T @ f["extrinsic"][:, 3])],''',
    '''                "position": (-f["extrinsic"][:, :3].T
                             @ f["extrinsic"][:, 3]).tolist(),'''))

subs.append((
    '''            etag = '"%x-%x"' % (int(st.st_mtime), st.st_size)''',
    '''            etag = f'"{int(st.st_mtime):x}-{st.st_size:x}"' '''.rstrip() ))

subs.append((
    '''    def _q(self, key, default):
        from urllib.parse import parse_qs
        q = parse_qs(urlparse(self.path).query)
        return q.get(key, [default])[0]''',
    '''    def _q(self, key, default):
        return parse_qs(urlparse(self.path).query).get(key, [default])[0]'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:200])
    s = s.replace(old, new, 1)

s = s.replace('''        except Exception as exc:
            import traceback
            traceback.print_exc()''', '''        except Exception as exc:
            traceback.print_exc()''')

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("view_map3d refactored")
