
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

REPO = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map"
sys.path.insert(0, REPO)
import detect_plants as dp

OUT = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--artifacts", default=r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_avg")
ap.add_argument("--views", type=int, default=5)
ap.add_argument("--models_dir", default=r"C:\Users\shaur\AppData\Local\lingbot-map\models")
args = ap.parse_args()

art = args.artifacts
plants = json.load(open(os.path.join(art, "plants.json"), encoding="utf-8"))["plants"]
clf = dp.BioClipClassifier(os.path.join(args.models_dir, "bioclip"), "cuda",
                           dp.DEFAULT_SPECIES)
names = [s[0] for s in dp.DEFAULT_SPECIES]
K = len(names)
MAXENT = np.log2(K)
print(f"{len(plants)} instances, {K} candidate species, "
      f"max entropy {MAXENT:.2f} bits\n")

by_frame = defaultdict(list)
for f in glob.glob(os.path.join(art, "crops", "*.jpg")):
    by_frame[int(os.path.basename(f).split("_")[0])].append(f)

def full_probs(crop):

    from PIL import Image
    import torch
    img = clf.preprocess(Image.fromarray(crop)).unsqueeze(0).to(clf.device)
    with torch.no_grad():
        f = clf.model.encode_image(img)
        f = (f / f.norm(dim=-1, keepdim=True)).float()
        return (100.0 * f @ clf.text_feats.T).softmax(dim=-1)[0].cpu().numpy()

def fused_probs(crops):
    from PIL import Image
    import torch
    batch = torch.stack([clf.preprocess(Image.fromarray(c)) for c in crops])
    with torch.no_grad():
        f = clf.model.encode_image(batch.to(clf.device))
        f = f / f.norm(dim=-1, keepdim=True)
        f = f.mean(dim=0, keepdim=True)
        f = (f / f.norm(dim=-1, keepdim=True)).float()
        return (100.0 * f @ clf.text_feats.T).softmax(dim=-1)[0].cpu().numpy()

def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

rows = []
for p in plants:
    pool = []
    for fid in p["frames"]:
        for f in sorted(by_frame.get(fid, [])):
            im = cv2.imread(f)
            if im is not None:
                pool.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    if not pool:
        continue
    pool.sort(key=lambda a: -dp.crop_quality(a))
    sel = pool[:args.views]

    per = np.stack([full_probs(c) for c in sel])
    fus = fused_probs(sel)
    top = int(np.argmax(fus))
    order = np.argsort(-fus)

    at_fused = per[:, top]
    view_top = [int(i) for i in per.argmax(1)]

    rows.append({
        "id": p["id"],
        "views_used": len(sel),
        "views_total": len(pool),
        "fused_name": names[top],
        "fused_score": float(fus[top]),
        "fused_margin": float(fus[order[0]] - fus[order[1]]),
        "fused_entropy": entropy(fus),
        "per_view_at_fused": [float(x) for x in at_fused],
        "view_min": float(at_fused.min()),
        "view_med": float(np.median(at_fused)),
        "view_max": float(at_fused.max()),
        "view_std": float(at_fused.std()),
        "view_top1_scores": [float(per[i].max()) for i in range(len(sel))],
        "view_entropy_mean": float(np.mean([entropy(v) for v in per])),
        "distinct_view_top1": len(set(view_top)),
        "view_top1_names": [names[i] for i in view_top],
        "agrees_with_fused": int(sum(1 for i in view_top if i == top)),
        "top5": [{"name": names[i], "score": float(fus[i])} for i in order[:5]],
        "reported_conf": p.get("species_confidence"),
        "reported_raw": p.get("species_raw_score"),
        "identified": p.get("species_identified"),
    })

print(f"{'id':>3s} {'v':>2s} {'fused name':<26s} {'raw':>6s} {'marg':>6s} "
      f"{'ent':>5s} | {'per-view score at that name':<28s} {'agree':>6s}")
for r in rows:
    print(f"{r['id']:>3d} {r['views_used']:>2d} {r['fused_name'][:26]:<26s} "
          f"{r['fused_score']:>6.3f} {r['fused_margin']:>6.3f} "
          f"{r['fused_entropy']:>5.2f} | "
          f"{r['view_min']:.3f}-{r['view_max']:.3f} "
          f"(med {r['view_med']:.3f}, sd {r['view_std']:.3f})  "
          f"{r['agrees_with_fused']:>2d}/{r['views_used']}")

fs = np.array([r["fused_score"] for r in rows])
sd = np.array([r["view_std"] for r in rows])
en = np.array([r["fused_entropy"] for r in rows])
ag = np.array([r["agrees_with_fused"] / r["views_used"] for r in rows])
print(f"\n{len(rows)} instances")
print(f"{'fused top-1 score':28s} min {fs.min():.3f}  med {np.median(fs):.3f}  "
      f"max {fs.max():.3f}")
print(f"{'per-view spread (sd)':28s} min {sd.min():.3f}  med {np.median(sd):.3f}  "
      f"max {sd.max():.3f}")
print(f"{'fused entropy (bits/5.64)':28s} min {en.min():.2f}   med {np.median(en):.2f}   "
      f"max {en.max():.2f}")
print(f"{'views agreeing with fused':28s} med {np.median(ag) * 100:.0f}%")

print("\nhow the single reported number relates to the spread behind it:")
for lo, hi, lab in ((0, .05, "tight (sd<0.05)"), (.05, .15, "moderate"),
                    (.15, 1, "wide (sd>0.15)")):
    m = (sd >= lo) & (sd < hi)
    if m.any():
        print(f"  {lab:<18s} {m.sum():>2d} instances, "
              f"median fused score {np.median(fs[m]):.3f}, "
              f"median agreement {np.median(ag[m]) * 100:.0f}%")

json.dump(rows, open(os.path.join(OUT, "species_dist.json"), "w"), indent=1)
print(f"\nwrote species_dist.json ({len(rows)} instances, full vectors summarised)")
