"""The reconstructed map in a gravity-aligned frame, plus real plant prompts."""
import json
import os

import cv2
import numpy as np

from detect_plants import source_to_model_transform
from .views import plant_views


def ground_frame(n, d):
    """Rotation R and offset so that aligned = p @ R.T + t has z = height."""
    n = np.asarray(n, np.float64)
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    R = np.stack([e1, e2, n])
    t = np.array([0.0, 0.0, d])
    return R, t


class Yard:
    """Wraps a MapScene: aligned points, colours, and per-plant views."""

    def __init__(self, scene):
        self.s = scene
        self.R, self.t = ground_frame(scene.ground_n, scene.ground_d)
        self.xyz = (scene.points.astype(np.float64) @ self.R.T + self.t).astype(np.float32)
        self.rgb = scene.colors
        self.art = scene.art
        self.model_hw = tuple(scene.model_hw)

    def to_aligned(self, p):
        return (np.asarray(p, np.float64) @ self.R.T + self.t).astype(np.float32)

    def aligned_extrinsic(self, E):
        """cam-from-world for aligned coords: x_cam = Rc (R^T (a - t)) + tc."""
        Rc, tc = E[:3, :3], E[:3, 3]
        Ra = Rc @ self.R.T
        return np.concatenate([Ra, (tc - Ra @ self.t)[:, None]], 1)

    def seeds(self):
        path = os.path.join(self.art, "plant_points.npz")
        S = np.load(path)
        return {p["id"]: S[f"pts_{p['id']}"].astype(np.float32)
                for p in self.s.plants if f"pts_{p['id']}" in S}

    def views(self):
        """{plant id: [view dict]} with boxes converted to model pixels."""
        seeds = self.seeds()
        frames = {f["id"]: f for f in self.s.frames}
        pv = plant_views(self.art, self.s.plants, seeds, frames, self.model_hw)
        cache = {}
        out = {}
        for pid, vs in pv.items():
            out[pid] = []
            for v in vs:
                fid = v["frame"]
                if fid not in cache:
                    f = frames[fid]
                    depth = self.s._frame_arrays(fid)[0]
                    hs, ws = cv2.imread(f["jpg"], cv2.IMREAD_GRAYSCALE).shape
                    scale, y_off = source_to_model_transform((hs, ws), self.model_hw)
                    cache[fid] = (self.aligned_extrinsic(f["extrinsic"]),
                                  f["intrinsic"], depth, scale, y_off)
                E, K, depth, scale, y_off = cache[fid]
                x0, y0, x1, y1 = v["box"]
                out[pid].append({"E": E, "K": K, "depth": depth, "frame": fid,
                                 "box": (x0 * scale, y0 * scale - y_off,
                                         x1 * scale, y1 * scale - y_off)})
        return out, {pid: self.to_aligned(p) for pid, p in seeds.items()}


def write_bank(art, out):
    """Aligned yard points for synthetic backgrounds, plus each detected
    plant cut out of the map (radius-rule membership) so training can paste
    real reconstructed plants as well as PLANesT ones."""
    from view_map3d import MapScene
    s = MapScene(art, use_seg=False)
    y = Yard(s)
    cut_xyz, cut_rgb, cut_off, cut_center = [], [], [0], []
    for p in s.plants:
        idx = s.member.get(p["id"], np.zeros(0, np.int64))
        q = y.xyz[idx]
        if len(q) < 300:
            continue
        keep = q[:, 2] > 2 * s.voxel                  # drop the ground under it
        if len(q) > 60000:
            keep &= np.random.default_rng(p["id"]).random(len(q)) < 60000 / len(q)
        cut_xyz.append(q[keep])
        cut_rgb.append(y.rgb[idx][keep])
        cut_off.append(cut_off[-1] + int(keep.sum()))
        cut_center.append(np.median(q, 0))
    np.savez(out, xyz=y.xyz, rgb=y.rgb, tags=s.tags,
             cut_xyz=np.vstack(cut_xyz), cut_rgb=np.vstack(cut_rgb),
             cut_off=np.array(cut_off), cut_center=np.array(cut_center))
    print(f"bank: {len(y.xyz):,} points, {len(cut_center)} plant cut-outs")
    return out
