"""Backbone features for the health set: every photo as clean, mirrored and
two FPV-degraded views, through the plant classifier's fine-tuned BioCLIP-2.

    python feats.py OUT.npz [--ckpt models/classifier/best.pt] [--views 4]
"""
import argparse, csv, os, sys, zlib

import cv2
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "yardmap"))
from plant_det.hier import HierNet, prep_eval  # noqa: E402
from fpv import degrade  # noqa: E402

ROOT = os.environ.get("HEALTH_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
CLASSES = ["healthy", "stressed", "dead"]

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--ckpt", default=os.path.join(REPO, "models", "classifier", "best.pt"))
ap.add_argument("--views", type=int, default=4)
ap.add_argument("--bs", type=int, default=96)
a = ap.parse_args()

rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "meta.csv"), encoding="utf-8"))
        if os.path.exists(os.path.join(ROOT, r["cls"], f"{r['obs']}.jpg"))]
seen = set()
rows = [r for r in rows if not (r["obs"] in seen or seen.add(r["obs"]))]
print("images", len(rows), {c: sum(r["cls"] == c for r in rows) for c in CLASSES}, flush=True)

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
v = ck["vocab"]
net = HierNet(ck["backbone"], len(v["forms"]), len(v["families"]), len(v["genera"]), len(v["species"]),
              train_blocks=0)
net.load_state_dict(ck["model"], strict=False)
net = net.cuda().eval()


@torch.no_grad()
def embed(batch):
    x = torch.stack(batch).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = net.backbone((x - net.mean_t) / net.std_t)
    if f.dim() == 3:
        f = f[:, 0]
    return f.float().cpu().numpy().astype(np.float16)


def views(path, seed):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    s = 448 / max(bgr.shape[:2])
    if s < 1:
        bgr = cv2.resize(bgr, (int(bgr.shape[1] * s), int(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(seed)
    out = [bgr, bgr[:, ::-1]]
    while len(out) < a.views:
        d = degrade(bgr, rng)
        out.append(d if rng.random() < 0.5 else d[:, ::-1])
    return [prep_eval(np.ascontiguousarray(o[:, :, ::-1])) for o in out]


F, keep, buf = [], [], []
for i, r in enumerate(rows):
    vs = views(os.path.join(ROOT, r["cls"], f"{r['obs']}.jpg"), int(r["obs"]))
    if vs is None:
        continue
    keep.append(i)
    buf += vs
    if len(buf) >= a.bs * a.views:
        F.append(embed(buf))
        buf = []
    if i % 2000 == 0:
        print(i, flush=True)
if buf:
    F.append(embed(buf))
F = np.concatenate(F).reshape(len(keep), a.views, -1)
R = [rows[i] for i in keep]
y = np.array([CLASSES.index(r["cls"]) for r in R])
taxon = np.array([int(r["taxon"] or 0) for r in R])
# split by taxon so no species is in two splits
h = np.array([zlib.crc32(str(t).encode()) % 10 for t in taxon])
split = np.where(h < 8, 0, np.where(h < 9, 1, 2))
np.savez(a.out, feats=F, y=y, taxon=taxon, split=split, obs=np.array([int(r["obs"]) for r in R]),
         place=np.array([r["place"] for r in R]), name=np.array([r["name"] for r in R]))
print("saved", F.shape, "splits", np.bincount(split))
