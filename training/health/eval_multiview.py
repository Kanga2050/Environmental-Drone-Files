"""Score the plant-level decision on held-out iNaturalist observations that
have several photos, each FPV-degraded, the way a plant's views are scored.

    python eval_multiview.py MODEL.pt OUT.json
"""
import csv, json, os, sys
from collections import defaultdict

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "yardmap"))
from plant_det.health import HealthClassifier  # noqa: E402
from fpv import degrade  # noqa: E402

ROOT = os.environ.get("HEALTH_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
model, out = sys.argv[1], sys.argv[2]
hc = HealthClassifier(model, "cuda")
groups = defaultdict(list)
for r in csv.DictReader(open(os.path.join(ROOT, "multiview.csv"))):
    groups[(r["obs"], r["cls"])].append(int(r["k"]))
bad = set(open(os.path.join(ROOT, "conflict_obs.txt")).read().split())
res = []
for (obs, cls), ks in groups.items():
    if obs in bad or len(ks) < 2:
        continue
    ims = []
    for k in sorted(ks):
        im = cv2.imread(os.path.join(ROOT, "multiview", f"{obs}_{k}.jpg"))
        if im is None:
            continue
        s = 448 / max(im.shape[:2])
        if s < 1:
            im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
        ims.append(degrade(im, np.random.default_rng(int(obs) * 10 + k))[:, :, ::-1].copy())
    if len(ims) < 2:
        continue
    h = hc.classify_views(ims)
    res.append(dict(obs=obs, cls=cls, n=len(ims), label=h["label"], guess=h["guess"], conf=h["conf"], agree=h["agree"]))
n = len(res)
lab = [r for r in res if r["label"] != "unsure"]
summ = {"observations": n, "views_mean": float(np.mean([r["n"] for r in res])),
        "guess_acc": float(np.mean([r["guess"] == r["cls"] for r in res])),
        "labelled": len(lab) / n, "precision": float(np.mean([r["label"] == r["cls"] for r in lab])) if lab else None,
        "by_rule": {}}
for name, f in (("single-photo threshold", lambda r: r["conf"] >= hc.tau),
                ("agreement rule only", lambda r: r["conf"] < hc.tau and r["label"] != "unsure")):
    s = [r for r in res if f(r)]
    summ["by_rule"][name] = {"share": len(s) / n, "precision": float(np.mean([r["guess"] == r["cls"] for r in s])) if s else None}
json.dump({"summary": summ, "rows": res}, open(out, "w"), indent=1)
print(json.dumps(summ, indent=1))
