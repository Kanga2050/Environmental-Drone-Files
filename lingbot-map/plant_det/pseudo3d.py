"""3D-consistent pseudo-labels for the yard frames.

Each plant's points (from a finished candidate's plant_seg.npz) are projected
into every saved frame; the points that are in view and not hidden behind
the frame's depth map give that plant's box in that frame.  A plant then has
a box in every frame that sees it, not only where the detector fired, which
is what self-training on the detector's own boxes lacked.

  python -m plant_det.pseudo3d --cand ../cands/best_c7_t85 --out ../datasets/yard3d
"""
import argparse
import json
import os

import cv2
import numpy as np

from detect_plants import source_to_model_transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_vis", type=int, default=60)
    ap.add_argument("--min_frac", type=float, default=0.05,
                    help="visible share of the plant's points needed for a box")
    ap.add_argument("--max_area", type=float, default=0.6)
    ap.add_argument("--per_plant", type=int, default=30000)
    a = ap.parse_args()
    art = a.cand
    meta = json.load(open(os.path.join(art, "meta.json")))
    hm, wm = meta["model_hw"]
    plants = json.load(open(os.path.join(art, "plants.json"), encoding="utf-8"))["plants"]
    cls_of = {p["id"]: (1 if p["category"] == "tree" else 0) for p in plants}
    seg = np.load(os.path.join(art, "plant_seg.npz"))
    tags, ids = seg["tags"], seg["plant_ids"]
    xyz = np.load(os.path.join(art, "cloud.npz"))["points"].astype(np.float32)
    rng = np.random.default_rng(0)
    pts = {}
    for t, pid in enumerate(ids):
        idx = np.flatnonzero(tags == t)
        if len(idx) > a.per_plant:
            idx = rng.choice(idx, a.per_plant, replace=False)
        if len(idx):
            pts[int(pid)] = xyz[idx]
    img_dir, lab_dir = os.path.join(a.out, "images"), os.path.join(a.out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    n_box = n_img = 0
    for fid in meta["saved_frame_ids"]:
        jpg = os.path.join(art, "frames", f"{fid:06d}.jpg")
        npz = os.path.join(art, "frames", f"{fid:06d}.npz")
        if not (os.path.exists(jpg) and os.path.exists(npz)):
            continue
        f = np.load(npz)
        E, K, depth = f["extrinsic"], f["intrinsic"], f["depth"][..., 0].astype(np.float32)
        h, w = cv2.imread(jpg, cv2.IMREAD_GRAYSCALE).shape
        scale, y_off = source_to_model_transform((h, w), (hm, wm))
        lines = []
        for pid, P in pts.items():
            cam = P @ E[:3, :3].T + E[:3, 3]
            z = cam[:, 2]
            ok = z > 1e-3
            u = K[0, 0] * cam[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
            v = K[1, 1] * cam[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
            ok &= (u >= 0) & (u < wm) & (v >= 0) & (v < hm)
            if ok.sum() < a.min_vis:
                continue
            ui, vi = u[ok].astype(int), v[ok].astype(int)
            d = depth[vi, ui]
            vis = (d <= 0) | (z[ok] <= d * 1.05 + 0.02)      # not hidden behind the surface
            if vis.sum() < max(a.min_vis, a.min_frac * len(P)):
                continue
            uu, vv = u[ok][vis], v[ok][vis]
            x0, x1 = np.percentile(uu, [2, 98]) / scale
            y0, y1 = (np.percentile(vv, [2, 98]) + y_off) / scale
            x0, x1 = np.clip([x0, x1], 0, w)
            y0, y1 = np.clip([y0, y1], 0, h)
            bw, bh = (x1 - x0) / w, (y1 - y0) / h
            if bw * bh > a.max_area or bw < 0.02 or bh < 0.02:
                continue
            lines.append(f"{cls_of.get(pid, 0)} {(x0 + x1) / 2 / w:.5f} {(y0 + y1) / 2 / h:.5f} "
                         f"{bw:.5f} {bh:.5f}")
        dst = os.path.join(img_dir, f"y{fid:06d}.jpg")
        if not os.path.exists(dst):
            os.link(os.path.realpath(jpg), dst)
        with open(os.path.join(lab_dir, f"y{fid:06d}.txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        n_box += len(lines)
        n_img += 1
    print(f"{n_img} frames, {n_box} boxes ({n_box / max(n_img, 1):.1f} per frame)")


if __name__ == "__main__":
    main()
