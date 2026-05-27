"""Open Images plant/tree boxes -> YOLO dataset.

Classes: 0 plant (Plant, Houseplant, Lavender, Squash), 1 tree (Tree, Palm
tree, Maple, Willow, Christmas tree).  Only images with at least one such
box are used; flower-only images are skipped because a flower box is a part
of a plant, not a plant.
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import requests

PLANT = {"/m/05s2s", "/m/03fp41", "/m/04gth", "/m/0dv77"}
TREE = {"/m/07j7r", "/m/0cdl1", "/m/0cffdh", "/m/0mw_6", "/m/025nd"}
CLS = {**{m: 0 for m in PLANT}, **{m: 1 for m in TREE}}


def load(csv):
    parts = []
    for ch in pd.read_csv(csv, usecols=["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax",
                                        "IsGroupOf"], chunksize=2_000_000):
        parts.append(ch[ch.LabelName.isin(CLS)])
    return pd.concat(parts)


def fetch(split, iid, out_img, max_side, session):
    path = os.path.join(out_img, iid + ".jpg")
    if os.path.exists(path):
        return True
    for _ in range(3):
        try:
            r = session.get(f"https://s3.amazonaws.com/open-images-dataset/{split}/{iid}.jpg",
                            timeout=30)
            if r.status_code != 200:
                return False
            im = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                return False
            s = max_side / max(im.shape[:2])
            if s < 1:
                im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(path, im, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return True
        except Exception:
            pass
    return False


def build(df, split, name, root, max_side, workers, cap, seed=0):
    ids = df.ImageID.unique()
    rng = np.random.default_rng(seed)
    if cap and len(ids) > cap:
        ids = rng.choice(ids, cap, replace=False)
    out_img = os.path.join(root, "images", name)
    out_lab = os.path.join(root, "labels", name)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lab, exist_ok=True)
    groups = dict(tuple(df[df.ImageID.isin(set(ids))].groupby("ImageID")))
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=workers)
    session.mount("https://", adapter)
    ok = 0
    with ThreadPoolExecutor(workers) as ex:
        for i, (iid, good) in enumerate(zip(ids, ex.map(
                lambda x: fetch(split, x, out_img, max_side, session), ids))):
            if not good:
                continue
            ok += 1
            g = groups[iid]
            cx, cy = (g.XMin + g.XMax) / 2, (g.YMin + g.YMax) / 2
            w, h = g.XMax - g.XMin, g.YMax - g.YMin
            with open(os.path.join(out_lab, iid + ".txt"), "w") as f:
                for c, a, b, ww, hh in zip(g.LabelName.map(CLS), cx, cy, w, h):
                    f.write(f"{c} {a:.5f} {b:.5f} {ww:.5f} {hh:.5f}\n")
            if i % 5000 == 0:
                print(name, i, len(ids), ok, flush=True)
    print(name, "done", ok, "of", len(ids), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_side", type=int, default=960)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--train_cap", type=int, default=80000)
    a = ap.parse_args()
    val = pd.concat([load(os.path.join(a.raw, "oi_val_bbox.csv")),
                     load(os.path.join(a.raw, "oi_test_bbox.csv"))])
    print("val images", val.ImageID.nunique(), flush=True)
    build(val[val.ImageID.isin(load(os.path.join(a.raw, "oi_val_bbox.csv")).ImageID)],
          "validation", "val", a.out, a.max_side, a.workers, 0)
    test = load(os.path.join(a.raw, "oi_test_bbox.csv"))
    build(test, "test", "test", a.out, a.max_side, a.workers, 0)
    tr = load(os.path.join(a.raw, "oi_train_bbox.csv"))
    print("train images", tr.ImageID.nunique(), "boxes", len(tr),
          tr.LabelName.map(CLS).value_counts().to_dict(), flush=True)
    build(tr, "train", "train", a.out, a.max_side, a.workers, a.train_cap)
    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\n"
                f"test: images/test\nnames:\n  0: plant\n  1: tree\n")


if __name__ == "__main__":
    main()
