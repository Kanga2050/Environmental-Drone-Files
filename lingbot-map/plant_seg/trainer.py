"""Train the promptable Point Transformer on synthetic yard scenes.

    python -m plant_seg.train --plants ../datasets/planest_npz \
        --bank ../datasets/yard_bank.npz --out ../datasets/plant_seg_runs/pt1
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import PointTransformerSeg
from .prompt import N_FEAT
from .synth import SynthScenes


def batches(pool, bs):
    """Stack samples streamed from the numpy-only worker pool."""
    from .synth import worker_sample
    it = pool.imap_unordered(worker_sample, iter(int, 1), chunksize=2)
    while True:
        items = [next(it) for _ in range(bs)]
        for attempt in range(30):
            try:
                yield tuple(torch.from_numpy(np.stack([x[k] for x in items]))
                            for k in range(4))
                break
            except MemoryError:
                time.sleep(2 + attempt)


def make_val(plants, bank, n_points, n):
    ds = SynthScenes(plants, bank, split="val", n_points=n_points, seed=1234)
    return [ds.sample() for _ in range(n)]


def losses(mask_logit, veg_logit, lab, veg):
    # lab -1 = unknown (hand-labelled plants only label what was shown)
    lk = (lab >= 0).float()
    lab = lab.clamp(min=0)
    bce = (F.binary_cross_entropy_with_logits(mask_logit, lab, reduction="none")
           * lk).sum() / lk.sum().clamp(min=1)
    prob = torch.sigmoid(mask_logit) * lk
    inter = (prob * lab).sum(1)
    dice = 1 - (2 * inter + 1) / (prob.sum(1) + lab.sum(1) + 1)
    known = (veg >= 0).float()            # -1 marks "might be vegetation"
    vbce = (F.binary_cross_entropy_with_logits(veg_logit, veg.clamp(min=0), reduction="none")
            * known).sum() / known.sum().clamp(min=1)
    return bce + dice.mean() + 0.3 * vbce


def iou(pred, lab):
    inter = (pred & lab).sum()
    union = (pred | lab).sum()
    return inter / max(union, 1)


def random_yaw(feat):
    B = feat.shape[0]
    a = torch.rand(B, device=feat.device) * 2 * math.pi
    c, s = torch.cos(a), torch.sin(a)
    x, y = feat[..., 0], feat[..., 1]
    feat = feat.clone()
    feat[..., 0] = c[:, None] * x - s[:, None] * y
    feat[..., 1] = s[:, None] * x + c[:, None] * y
    if torch.rand(()) < 0.5:
        feat[..., 0] = -feat[..., 0]
    return feat


@torch.no_grad()
def evaluate(model, val, device, bs=8):
    model.eval()
    ious, base_ious, src = [], [], []
    for i in range(0, len(val), bs):
        chunk = val[i:i + bs]
        feat = torch.from_numpy(np.stack([c[0] for c in chunk])).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m, _ = model(feat[..., :3], feat)
        pred = (m.float() > 0).cpu().numpy()
        for j, c in enumerate(chunk):
            lab = c[1] > 0.5
            ious.append(iou(pred[j], lab))
            base_ious.append(iou(c[3] > 0.5, lab))
            src.append(int(c[4]))
    model.train()
    ious, src = np.array(ious), np.array(src)
    by_src = {k: float(ious[src == v].mean()) if (src == v).any() else float("nan")
              for k, v in (("planest", 0), ("yard", 1))}
    return float(ious.mean()), float(np.mean(base_ious)), by_src


def run(args, pool):
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    real = None
    if args.real_labels:
        from .real_labels import RealLabelled
        real = RealLabelled(args.real_art, args.real_labels, args.n_points,
                            holdout=set(args.real_holdout))
        real_rng = np.random.default_rng(0)

    t0 = time.time()
    val = make_val(args.plants, args.bank, args.n_points, args.n_val)
    print(f"validation set: {len(val)} scenes in {time.time() - t0:.0f}s", flush=True)

    model = PointTransformerSeg(N_FEAT, derived=args.derived,
                                uniform_layers=args.uniform_layers).to(device)
    if args.init:
        model.load_compatible(torch.load(args.init, map_location=device)["model"])
        print(f"initialised from {args.init}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters,
                                                pct_start=0.05)
    loader = batches(pool, args.bs)
    log = []
    best = -1
    start = 0
    last_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
        else:
            for _ in range(ck["iter"]):
                sched.step()
        start = ck["iter"]
        best = ck.get("best", -1)
        lp = os.path.join(args.out, "log.json")
        if os.path.exists(lp):
            with open(lp) as f:
                log = json.load(f)
            best = max([best] + [r["val_iou"] for r in log])
        print(f"resumed from iteration {start} (best val IoU {best:.4f})", flush=True)
    t0 = time.time()
    run = 0.0
    oom = 0
    for it, (feat, lab, veg, _) in enumerate(loader, start + 1):
        if real is not None:
            # swap in hand-labelled real plants for part of the batch
            for b in range(args.real_per_batch):
                f, l, v, _, _ = real.sample(real_rng)
                feat[b], lab[b], veg[b] = (torch.from_numpy(f), torch.from_numpy(l),
                                           torch.from_numpy(v))
        feat, lab, veg = feat.to(device), lab.to(device), veg.to(device)
        feat = random_yaw(feat)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m, v = model(feat[..., :3], feat)
            loss = losses(m.float(), v.float(), lab, veg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            oom = 0
        except torch.OutOfMemoryError:
            # The GPU is shared with other programs; skip the batch and retry.
            oom += 1
            print(f"it {it}: CUDA OOM, skipping batch ({oom} in a row)", flush=True)
            opt.zero_grad(set_to_none=True)
            m = v = loss = None
            torch.cuda.empty_cache()
            time.sleep(2)
            if oom >= 20:
                raise
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it > 1 else loss.item()
        if it % 100 == 0:
            print(f"it {it} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / (it - start):.2f}s/it", flush=True)
        if it % 250 == 0 and it % args.eval_every and it != args.iters:
            torch.save({"model": model.state_dict(), "n_feat": N_FEAT, "iter": it,
                        "n_points": args.n_points, "derived": args.derived, "uniform_layers": args.uniform_layers, "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "best": best}, last_path)
        if it % args.eval_every == 0 or it == args.iters:
            v_iou, b_iou, by_src = evaluate(model, val, device)
            log.append({"iter": it, "loss": run, "val_iou": v_iou, "baseline_iou": b_iou,
                        "val_iou_by_target": by_src, "minutes": (time.time() - t0) / 60})
            print(f"== it {it}: val IoU {v_iou:.4f} (PLANesT targets {by_src['planest']:.4f}, "
                  f"yard targets {by_src['yard']:.4f}; seed-radius baseline {b_iou:.4f})",
                  flush=True)
            if v_iou > best:
                best = v_iou
                torch.save({"model": model.state_dict(), "n_feat": N_FEAT, "iter": it,
                            "n_points": args.n_points, "derived": args.derived, "uniform_layers": args.uniform_layers}, os.path.join(args.out, "best.pt"))
            torch.save({"model": model.state_dict(), "n_feat": N_FEAT, "iter": it,
                        "n_points": args.n_points, "derived": args.derived, "uniform_layers": args.uniform_layers, "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "best": best}, last_path)
            with open(os.path.join(args.out, "log.json"), "w") as f:
                json.dump(log, f, indent=1)
        if it >= args.iters:
            break

