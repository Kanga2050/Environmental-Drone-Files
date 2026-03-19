
import os
import sys

import cv2
import numpy as np
import torch

REPO = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map"
sys.path.insert(0, REPO)
import detect_plants as dp
from lingbot_map.utils.geometry import unproject_depth_map_to_point_map

BASE = r"C:\Users\shaur\OneDrive\Desktop\Test"
SCENES = [
    ("yard-full  (all frames)", os.path.join(BASE, "artifacts_yard_final")),
    ("driveway   (must stay 1)", os.path.join(BASE, "artifacts")),
]
SEPS = [0.5, 0.7, 0.9, 1.2, 1.5]

det = dp.Owlv2Detector(r"C:\Users\shaur\AppData\Local\lingbot-map\models\owlv2", "cuda", 0.15)

def detect_scene(art):
    meta = dp.load_meta(art)
    model_hw = tuple(meta["model_hw"])
    conf_thr = meta.get("conf_threshold", 1.5)
    dets = []
    for fid in meta["saved_frame_ids"]:
        try:
            depth, conf, extr, intr, rgb = dp.load_frame(art, fid)
        except FileNotFoundError:
            continue
        if rgb is None:
            continue
        scale, y_off = dp.source_to_model_transform(rgb.shape[:2], model_hw)
        world = unproject_depth_map_to_point_map(depth[None], extr[None], intr[None])[0]
        rgb_model = cv2.resize(rgb, (model_hw[1], model_hw[0]), interpolation=cv2.INTER_AREA)
        boxes, scores, labels = det(rgb, dp.DEFAULT_PROMPTS)
        boxes, scores, labels = dp.nms_2d(boxes, scores, labels, 0.5)
        for box, score, label in zip(boxes, scores, labels):
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            crop = rgb[max(0, y0):y1, max(0, x0):x1]
            if crop.size == 0 or dp.vegetation_fraction(crop) < 0.06:
                continue
            geo = dp.lift_box_to_3d(box, world, conf, rgb_model, scale, y_off,
                                    conf_thr, 40, use_vegetation=True)
            if geo is None:
                continue
            dets.append({**geo, "frame": int(fid), "score": float(score),
                         "category": [(label, float(score))], "species": [],
                         "crop": "", "box_src": [float(v) for v in box]})
    return dets, meta

for name, art in SCENES:
    if not os.path.isdir(art):
        print(f"{name}: missing, skipped")
        continue
    dets, meta = detect_scene(art)
    allp = np.concatenate([d["points"] for d in dets]) if dets else np.zeros((1, 3))
    extent = float(np.linalg.norm(allp.max(0) - allp.min(0)))
    cluster_dist = 0.04 * extent
    row = []
    for sep in SEPS:
        inst = dp.cluster_detections(dets, cluster_dist)
        inst = dp.merge_overlapping(inst, 0.15, sep)
        kept = [i for i in inst if len({m["frame"] for m in i["members"]}) >= 2]
        sizes = sorted((len(i["members"]) for i in kept), reverse=True)[:3]
        row.append(f"sep {sep}: {len(kept):>2} inst (top members {sizes})")
    print(f"\n{name}  [{len(dets)} dets, extent {extent:.2f}]")
    for r in row:
        print("   ", r)
