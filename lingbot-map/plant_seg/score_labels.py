"""Score plant point assignments against hand labels from plant_label.html.

    python -m plant_seg.score_labels --artifacts ../artifacts_yard_avg \
        --labels ../artifacts_yard_avg/plant_labels.json \
        --seg plant_seg.npz plant_seg_pt4.npz

Each labelled plant is matched to whichever predicted plant holds most of
its labelled points (so plant ids may differ, e.g. after re-running
detection).  Scores use the sampled region shown in the labelling page:
labelled points are positives, every other shown point is a negative.
"--seg radius" scores the old radius rule.
"""
import argparse
import json
import os

import numpy as np


def old_rule_tags(art):
    from view_map3d import MapScene
    s = MapScene(art, use_seg=False)
    tags = np.empty(s.n_cloud, np.int16)
    tags[s.order] = s.tags
    return tags


def score(tags, regions, labels):
    rows = []
    for pid, lab in labels.items():
        key = f"idx_{pid}"
        if key not in regions:
            continue
        region = regions[key]
        gt = np.isin(region, np.asarray(lab["points"], np.int64))
        if gt.sum() == 0:
            continue
        t = tags[region]
        hits = t[gt & (t >= 0)]
        if len(hits) == 0:
            rows.append({"id": int(pid), "iou": 0.0, "precision": 0.0, "recall": 0.0, "matched": None})
            continue
        best = np.bincount(hits).argmax()
        pred = t == best
        inter = (pred & gt).sum()
        rows.append({"id": int(pid), "matched": int(best),
                     "iou": float(inter / (pred | gt).sum()),
                     "precision": float(inter / max(pred.sum(), 1)),
                     "recall": float(inter / gt.sum())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--seg", nargs="+", default=["radius", "plant_seg.npz"])
    ap.add_argument("--all", action="store_true", help="include plants not marked done")
    ap.add_argument("--ids", type=int, nargs="*", default=None,
                    help="score only these labelled plant ids (e.g. the held-out ones)")
    args = ap.parse_args()
    with open(args.labels) as f:
        lab = json.load(f)["plants"]
    lab = {k: v for k, v in lab.items() if (args.all or v.get("done"))
           and (args.ids is None or int(k) in args.ids)}
    regions = np.load(os.path.join(args.artifacts, "plant_label_regions.npz"))
    print(f"{len(lab)} labelled plants")
    results = {}
    for name in args.seg:
        tags = (old_rule_tags(args.artifacts) if name == "radius"
                else np.load(os.path.join(args.artifacts, name))["tags"])
        rows = score(tags, regions, lab)
        results[name] = rows
        m = {k: np.mean([r[k] for r in rows]) for k in ("iou", "precision", "recall")}
        print(f"{name:<22} IoU {m['iou']:.3f}  precision {m['precision']:.3f}  recall {m['recall']:.3f}")
    print("\nper plant IoU:")
    ids = [r["id"] for r in next(iter(results.values()))]
    print("  plant " + "".join(f"{n[:14]:>16}" for n in results))
    for i, pid in enumerate(ids):
        print(f"  #{pid:<5}" + "".join(f"{results[n][i]['iou']:>16.3f}" for n in results))
    return results


if __name__ == "__main__":
    main()
