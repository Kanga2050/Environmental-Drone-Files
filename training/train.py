"""Fine-tune the streaming GCT for foliage and the drone camera.

The released weights stay frozen; LoRA adapters go into the aggregator's
frame and global blocks (and optionally the camera head), and the DPT depth
head can be trained outright.  Each step is one TartanAir clip:

  * block mode   - S frames as one bidirectional block (the anchor pass)
  * stream mode  - a short anchor block then single frames through the KV
                   cache, gradients flowing through the cache (the streaming
                   pass every later video frame takes)

Losses, all invariant to the clip's unknown global scale (one median
gt/pred depth ratio per clip, detached):
  depth  confidence-weighted |log depth error| - a*log(conf)  (VGGT style)
  pose   rotation geodesic + translation L1 (in predicted units) relative to
         the first frame, + field-of-view L1
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import torch

import ckpt
import common
import lora
import tartan

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--envs", default="seasonsforest,gascola,neighborhood,japanesealley,seasidetown,oldtown")
ap.add_argument("--val_seqs", default="seasonsforest/Hard/P006,gascola/Hard/P009,neighborhood/Hard/P017,japanesealley/Hard/P005,oldtown/Hard/P007")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--head_lr", type=float, default=2e-5)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--lora_camera", action="store_true", help="LoRA in the camera head too")
ap.add_argument("--train_depth_head", action="store_true")
ap.add_argument("--S", type=int, default=8, help="frames per block-mode clip")
ap.add_argument("--stream_p", type=float, default=0.3, help="share of stream-mode steps")
ap.add_argument("--stream_anchor", type=int, default=4)
ap.add_argument("--stream_len", type=int, default=4)
ap.add_argument("--stream_grad_last", type=int, default=0,
                help="truncated backprop: only the last K streamed frames carry gradients; "
                     "the anchor and earlier frames run without, as context (0 = all)")
ap.add_argument("--stride", default="1,8")
ap.add_argument("--fpv_p", type=float, default=0.0)
ap.add_argument("--drone_p", type=float, default=0.0)
ap.add_argument("--crop_p", type=float, default=0.3)
ap.add_argument("--k1", type=float, default=0.0)
ap.add_argument("--w_pose", type=float, default=1.0)
ap.add_argument("--conf_alpha", type=float, default=0.2)
ap.add_argument("--eval_every", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no_val0", action="store_true")
args = ap.parse_args()

OUT = os.path.join(common.HERE, "runs", args.name)
os.makedirs(OUT, exist_ok=True)
json.dump(vars(args), open(os.path.join(OUT, "args.json"), "w"), indent=1)
torch.manual_seed(args.seed)
# Past ~15 GB Windows starts paging GPU memory to system RAM and a step takes
# several times longer; better to skip the rare oversized clip (OOM below).
torch.cuda.set_per_process_memory_fraction(0.90)
rng = np.random.default_rng(args.seed)

model = common.build()
for p in model.parameters():
    p.requires_grad = False
model.aggregator.to(torch.bfloat16)
ckpt.enable(model.aggregator)
ckpt.enable_depth_head(model)
wrapped = lora.add_lora(model.aggregator.frame_blocks, args.rank, args.rank)
wrapped += lora.add_lora(model.aggregator.global_blocks, args.rank, args.rank)
if args.lora_camera:
    wrapped += lora.add_lora(model.camera_head, args.rank, args.rank)
groups = [{"params": lora.lora_params(model), "lr": args.lr}]
if args.train_depth_head:
    for p in model.depth_head.parameters():
        p.requires_grad = True
    groups.append({"params": list(model.depth_head.parameters()), "lr": args.head_lr})
n_train = sum(p.numel() for g in groups for p in g["params"])
print(f"{len(wrapped)} LoRA layers, {n_train / 1e6:.1f}M trainable parameters", flush=True)
opt = torch.optim.AdamW(groups, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + np.cos(np.pi * min(s, args.steps) / args.steps)))

val_names = set(args.val_seqs.split(","))
all_seqs = tartan.sequences(args.envs.split(","))
key = lambda seq: "/".join(seq.replace("\\", "/").split("/")[-3:])
train_seqs = [s for s in all_seqs if key(s[1]) not in val_names]
val_seqs = [s for s in all_seqs if key(s[1]) in val_names]
print(f"{len(train_seqs)} train / {len(val_seqs)} val trajectories", flush=True)
lo, hi = (int(x) for x in args.stride.split(","))


def fov_gt(K, hw):
    H, W = hw
    return torch.stack([2 * torch.atan(H / 2 / K[:, 1, 1]), 2 * torch.atan(W / 2 / K[:, 0, 0])], -1)


def losses(out, clip, hw):
    dev = out["depth"].device
    depth, conf, c2w, K = common.decode(out, hw)
    depth, conf, c2w = depth[0], conf[0], c2w[0]
    gd, gv = clip["depth"].to(dev), clip["valid"].to(dev)
    s = common.depth_scale(depth.detach(), gd, gv)
    err = (torch.log(depth.clamp_min(1e-4) * s) - torch.log(gd.clamp_min(1e-4))).abs()
    l_depth = (conf * err - args.conf_alpha * torch.log(conf))[gv].mean()
    pc = torch.linalg.inv(c2w[:1].float()) @ c2w.float()
    gc = clip["c2w"].to(dev)
    R = pc[1:, :3, :3].transpose(-1, -2) @ gc[1:, :3, :3]
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    l_rot = torch.arccos(cos).mean()
    l_t = (pc[1:, :3, 3] - gc[1:, :3, 3] / s).abs().mean()
    l_fov = (out["pose_enc"][0, :, 7:9].float() - fov_gt(clip["K"].to(dev), hw)).abs().mean()
    total = l_depth + args.w_pose * (l_rot + l_t + l_fov)
    return total, {"depth": l_depth.item(), "rot": l_rot.item(), "t": l_t.item(),
                   "fov": l_fov.item()}


def step_block(clip):
    x = clip["images"][None].cuda()
    S = x.shape[1]
    common.reset(model)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, num_frame_for_scale=S, num_frame_per_block=S, causal_inference=True)
    return losses(out, clip, x.shape[-2:])


def step_stream(clip):
    """Anchor block, then single frames through the (differentiable) cache."""
    x = clip["images"][None].cuda()
    n = args.stream_anchor
    common.reset(model)
    outs = []
    S = x.shape[1]
    first_grad = S - args.stream_grad_last if args.stream_grad_last else 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.set_grad_enabled(first_grad == 0):
            outs.append(model(x[:, :n], num_frame_for_scale=n, num_frame_per_block=n,
                              causal_inference=True))
        for i in range(n, S):
            with torch.set_grad_enabled(i >= first_grad):
                outs.append(model(x[:, i:i + 1], num_frame_for_scale=n, num_frame_per_block=1,
                                  causal_inference=True))
    out = {k: torch.cat([o[k] for o in outs], 1) for k in ("depth", "depth_conf", "pose_enc")}
    return losses(out, clip, x.shape[-2:])


@torch.no_grad()
def validate(tag):
    """Streaming metrics on held-out trajectories, clean and with the drone look."""
    model.eval()
    res = {}
    for mode in ("clean", "drone"):
        rows = []
        for k, (env, seq, n) in enumerate(val_seqs):
            for j, start in enumerate((0, n // 2)):
                r2 = np.random.default_rng(1000 + 10 * k + j)
                ids = list(range(start, min(n, start + 24 * 4), 4))
                if len(ids) < 12:
                    continue
                clip = tartan.load_clip(seq, ids, r2, fpv=(mode == "drone"),
                                        drone=(mode == "drone"))
                d, c, c2w, K = common.run_stream(model, clip["images"], n_anchor=8)
                rows.append(common.metrics(d, c2w, clip))
        res[mode] = {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}
    model.train()
    print(f"[val {tag}] " + " | ".join(
        f"{mode}: " + " ".join(f"{k}={v:.3f}" for k, v in r.items()) for mode, r in res.items()),
        flush=True)
    return res


import queue
import threading

clips = queue.Queue(maxsize=6)


def producer(seed):
    r = np.random.default_rng(seed)
    while True:
        stream = r.random() < args.stream_p
        S = args.stream_anchor + args.stream_len if stream else args.S
        try:
            c = tartan.random_clip(train_seqs, r, S, (lo, hi), fpv_p=args.fpv_p,
                                   crop_p=args.crop_p, drone_p=args.drone_p, k1_max=args.k1)
        except Exception as e:                  # a damaged file: draw another clip
            print("data:", e, flush=True)
            continue
        clips.put((stream, c))


for w in range(3):
    threading.Thread(target=producer, args=(args.seed * 100 + w,), daemon=True).start()

hist = {"val": {}, "train": []}
if not args.no_val0:
    hist["val"]["0"] = validate(0)
model.train()
t0 = time.time()
run = {}
fails = 0
for step in range(1, args.steps + 1):
    stream, clip = clips.get()
    try:
        loss, parts = (step_stream if stream else step_block)(clip)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
    except (torch.OutOfMemoryError, FloatingPointError) as e:
        print(f"step {step}: skipped ({type(e).__name__}, {tuple(clip['images'].shape)})",
              flush=True)
        e = loss = parts = None
        opt.zero_grad(set_to_none=True)
        common.reset(model)
        gc.collect()
        torch.cuda.empty_cache()
        fails += 1
        if fails >= 20:
            raise RuntimeError("20 failed steps in a row; stopping")
        continue
    fails = 0
    sched.step()
    common.reset(model)
    if step % 50 == 0:
        torch.cuda.empty_cache()
    for k, v in parts.items():
        run[k] = 0.98 * run.get(k, v) + 0.02 * v
    if step % 25 == 0:
        el = time.time() - t0
        print(f"step {step} {el / step:.2f}s/it  " + " ".join(f"{k}={v:.4f}" for k, v in run.items())
              + f"  peak {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
        hist["train"].append({"step": step, **run})
    if step % args.eval_every == 0 or step == args.steps:
        hist["val"][str(step)] = validate(step)
        torch.save({"lora": lora.lora_state(model),
                    "depth_head": model.depth_head.state_dict() if args.train_depth_head else None,
                    "args": vars(args), "step": step}, os.path.join(OUT, "adapter.pt"))
        json.dump(hist, open(os.path.join(OUT, "hist.json"), "w"), indent=1)

# a plain checkpoint the pipeline loads exactly like models/gct.pt
model.eval()
sd = lora.merged_state_dict(model)
torch.save({"model": sd}, os.path.join(OUT, "gct_merged.pt"))
print("saved", os.path.join(OUT, "gct_merged.pt"), flush=True)
