"""Score a hierarchical classifier (trained checkpoint or BioCLIP-2 zero-shot)
on the shared test sets: clean test, FPV-degraded test, novel species.

  python -m plant_det.eval_cls --ckpt ../datasets/cls_runs/NAME/best.pt
  python -m plant_det.eval_cls --zeroshot bioclip2 --tag _all
"""
import argparse
import json
import os

import numpy as np


def _stop(P, Y, hier, tau):
    from .train_cls import RANKS
    depth_of = {"plant": 0, "form": 0, "family": 1, "genus": 2, "species": 3}
    corr = depth = 0
    for i in range(len(Y)):
        d = hier.decide({k: P[k][i] for k in RANKS}, tau)
        lv = d["level"]
        if lv in ("plant", "form"):
            corr += lv == "plant" or Y[i, 0] < 0 or hier.v["forms"].index(d["form"][0]) == Y[i, 0]
            continue
        idx = {"family": hier.v["families"], "genus": hier.v["genera"],
               "species": hier.v["species"]}[lv].index(d[lv][0])
        corr += Y[i, RANKS.index(lv)] == idx
        depth += depth_of[lv]
    return {"correct": round(corr / len(Y), 4), "mean_depth": round(depth / len(Y), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--zeroshot", default=None, choices=[None, "bioclip2", "bioclip"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--n_test", type=int, default=3000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--tau_name", default="tau.json")
    ap.add_argument("--blend", default="mean", choices=["mean", "geo"])
    a = ap.parse_args()
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from .hier import Hierarchy, HierNet, ZeroShotNet
    from .train_cls import CLS, RANKS, ImageSet, evaluate, metrics
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        vocab = ck["vocab"]
        net = HierNet(ck["backbone"], len(vocab["forms"]), len(vocab["families"]),
                      len(vocab["genera"]), len(vocab["species"]), train_blocks=0)
        net.load_state_dict(ck["model"], strict=False)
        net = net.cuda().eval()
    else:
        vocab = json.load(open(os.path.join(CLS, f"vocab{a.tag}.json")))
        net = ZeroShotNet(vocab, a.zeroshot, "cuda")
    df = pd.read_csv(os.path.join(CLS, f"labels{a.tag}.csv"))
    ycols = [f"y_{k}" for k in RANKS]
    te = df[df.split == "test"].sample(frac=1, random_state=0).groupby("src").head(a.n_test)
    nv = df[df.split == "novel"].sample(frac=1, random_state=0).head(a.n_test)
    hier = Hierarchy(vocab, a.blend)
    res = {}
    keep = {}
    for name, d, fpv in (("test", te, False), ("test_fpv", te, True), ("novel", nv, False)):
        ds = ImageSet(CLS, d.path.to_numpy().astype(str), d[ycols].to_numpy().astype(np.int64),
                      False, fpv=fpv)
        P, Y = evaluate(net, hier, DataLoader(ds, batch_size=128, num_workers=8), "cuda")
        res[name] = metrics(P, Y, hier)
        keep[name] = (P, Y)
        # per source on the clean test
        if name == "test":
            src = d.src.to_numpy()
            for s in np.unique(src):
                m = src == s
                res[f"test_{s}"] = {f"acc_{k}": float((P[k][m].argmax(1).numpy()
                                                       == Y[m, j])[Y[m, j] >= 0].mean())
                                    for j, k in enumerate(RANKS)}
        print(name, json.dumps(res[name]), flush=True)
    # per-rank thresholds: the lowest tau at which that rank's answers are
    # >= target precise on the yard-like mix (FPV-degraded test + novel species)
    import torch
    P = {k: torch.cat([keep["test_fpv"][0][k], keep["novel"][0][k]]) for k in RANKS}
    Y = np.concatenate([keep["test_fpv"][1], keep["novel"][1]])
    tau = {}
    for r in RANKS:
        names = {"form": vocab["forms"], "family": vocab["families"], "genus": vocab["genera"],
                 "species": vocab["species"]}[r]
        conf, ok = [], []
        for i in range(len(Y)):
            d = hier.decide({k: P[k][i] for k in RANKS}, 0.0)
            nm, c = d[r]
            if nm is None:
                continue
            conf.append(c)
            # species of a novel plant is never right (it is not in the vocabulary)
            ok.append(Y[i, RANKS.index(r)] >= 0 and names.index(nm) == Y[i, RANKS.index(r)])
        conf, ok = np.array(conf), np.array(ok)
        tau[r] = 0.99
        for t in np.arange(0.2, 0.99, 0.01):
            m = conf >= t
            if m.sum() >= 30 and ok[m].mean() >= a.target:
                tau[r] = round(float(t), 2)
                break
    # deeper ranks never need less confidence than shallower ones
    tau["genus"] = max(tau["genus"], tau["family"])
    tau["species"] = max(tau["species"], tau["genus"])
    res["tau"] = tau
    tau_saved = dict(tau, blend=a.blend) if a.blend != "mean" else tau
    print("calibrated tau", tau, flush=True)
    if a.ckpt:
        json.dump(tau_saved, open(os.path.join(os.path.dirname(a.ckpt), a.tau_name), "w"))
    else:
        json.dump(tau, open(os.path.join(os.path.dirname(CLS), "cls_runs",
                                         f"tau_zeroshot_{a.zeroshot}.json"), "w"))
    for name, (Pn, Yn) in keep.items():
        res[name]["calibrated"] = metrics(Pn, Yn, hier, taus=())
        res[name]["calibrated"].update(_stop(Pn, Yn, hier, tau))
    out = a.out or (os.path.join(os.path.dirname(a.ckpt), a.tau_name.replace("tau", "eval"))
                    if a.ckpt else
                    os.path.join(os.path.dirname(CLS), "cls_runs", f"zeroshot_{a.zeroshot}.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
