"""Assign map points to detected plants with the trained Point Transformer.

For every plant from detect_plants.py, the 2D boxes it was detected in and
its lifted seed points become a prompt; the network predicts which points
in the surrounding region belong to it.  Where plants compete for a point,
the highest probability wins.  Writes <artifacts>/plant_seg.npz, which
view_map3d.py picks up automatically.

    python -m plant_seg.segment --artifacts ../maps/garden \
        --weights ../datasets/plant_seg_runs/pt1/best.pt
"""
import argparse
import os
import time

import numpy as np
import torch

from . import prompt as P
from .model import PointTransformerSeg
from .yard import Yard


@torch.no_grad()
def predict_plant(model, y, views, seeds, n_points, passes=2, bs=8, device="cuda", seed=0):
    lo, hi, diag = P.crop_region(seeds)
    pool = np.flatnonzero(np.all((y.xyz >= lo) & (y.xyz <= hi), 1))
    if len(pool) < 100:
        return pool, np.zeros(len(pool), np.float32), np.zeros(len(pool), np.float32)
    center = np.median(seeds, 0)
    feat_all = P.features(y.xyz[pool], y.rgb[pool], views, seeds, center, diag)
    rng = np.random.default_rng(seed)
    prob = np.zeros(len(pool), np.float32)
    veg = np.zeros(len(pool), np.float32)
    hits = np.zeros(len(pool), np.float32)
    for _ in range(passes):
        order = rng.permutation(len(pool))
        n_chunks = max(1, int(np.ceil(len(pool) / n_points)))
        chunks = []
        for c in range(n_chunks):
            ids = order[c * n_points:(c + 1) * n_points]
            if len(ids) < n_points:      # top up the last chunk with repeats
                ids = np.concatenate([ids, rng.choice(len(pool), n_points - len(ids))])
            chunks.append(ids)
        for b in range(0, len(chunks), bs):
            ids = np.stack(chunks[b:b + bs])
            f = torch.from_numpy(feat_all[ids]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m, v = model(f[..., :3], f)
            m = torch.sigmoid(m.float()).cpu().numpy()
            v = torch.sigmoid(v.float()).cpu().numpy()
            for k in range(len(ids)):
                np.add.at(prob, ids[k], m[k])
                np.add.at(veg, ids[k], v[k])
                np.add.at(hits, ids[k], 1)
    hits = np.maximum(hits, 1)
    return pool, prob / hits, veg / hits


def load_model(weights, device="cuda"):
    ck = torch.load(weights, map_location=device)
    model = PointTransformerSeg(ck["n_feat"], derived=ck.get("derived", False),
                                uniform_layers=ck.get("uniform_layers", ())).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, ck.get("n_points", 8192)


def segment(art, weights, threshold=0.5, passes=2, device="cuda", out_name="plant_seg.npz"):
    from view_map3d import MapScene
    scene = MapScene(art, use_seg=False)
    y = Yard(scene)
    views, seeds = y.views()
    model, n_points = load_model(weights, device)

    n = len(y.xyz)
    best_p = np.zeros(n, np.float32)
    best_i = np.full(n, -1, np.int16)
    veg = np.zeros(n, np.float32)
    per_plant = {}
    regions = {}
    rng = np.random.default_rng(0)
    t0 = time.time()
    for i, p in enumerate(scene.plants):
        pid = p["id"]
        vs, sd = views.get(pid, []), seeds.get(pid)
        if sd is None or len(sd) < 20 or not vs:
            print(f"  plant {pid}: no usable views/seeds, skipped")
            continue
        pool, prob, vg = predict_plant(model, y, vs, sd, n_points, passes, device=device)
        win = prob > best_p[pool]
        best_p[pool[win]] = prob[win]
        best_i[pool[win]] = i
        veg[pool] = np.maximum(veg[pool], vg)
        per_plant[pid] = int((prob > threshold).sum())
        # A sample of the region with this plant's probabilities, for
        # plant_seg.compare_view.
        keep = rng.choice(len(pool), min(len(pool), 40000), replace=False)
        regions[f"idx_{pid}"] = pool[keep]
        regions[f"prob_{pid}"] = prob[keep].astype(np.float16)
        print(f"  plant {pid}: {len(vs)} views, region {len(pool):,} pts, "
              f"mask {per_plant[pid]:,} (old {len(scene.member.get(pid, [])):,})", flush=True)
    tags = np.where(best_p > threshold, best_i, -1).astype(np.int16)
    print(f"segmented {len(per_plant)} plants in {time.time() - t0:.0f}s; "
          f"{(tags >= 0).sum():,} points assigned (old {(scene.tags >= 0).sum():,})")

    # Store in cloud.npz order so it is independent of MapScene's shuffle.
    inv = np.empty_like(scene.order)
    inv[scene.order] = np.arange(len(scene.order))
    out = os.path.join(art, out_name)
    np.savez_compressed(out, tags=tags[inv], prob=best_p[inv].astype(np.float16),
                        veg=veg[inv].astype(np.float16),
                        plant_ids=np.array([p["id"] for p in scene.plants]),
                        threshold=threshold, weights=os.path.abspath(weights))
    for k in [k for k in regions if k.startswith("idx_")]:
        regions[k] = scene.order[regions[k]].astype(np.int32)   # cloud.npz order
    np.savez_compressed(os.path.join(art, out_name.replace(".npz", "_regions.npz")), **regions)
    print(f"wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--out", default="plant_seg.npz",
                    help="file name inside --artifacts (view_map3d reads plant_seg.npz)")
    args = ap.parse_args()
    segment(args.artifacts, args.weights, args.threshold, args.passes, out_name=args.out)


if __name__ == "__main__":
    main()
