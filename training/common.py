"""Shared pieces for GCT fine-tuning: model loading, running a clip the way
the yard pipeline does (anchor block, then one frame at a time through the KV
cache), and depth / pose metrics up to the unknown global scale."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "yardmap"))

from gct.models.gct_stream import GCTStream                       # noqa: E402
from gct.utils.pose_enc import pose_encoding_to_extri_intri       # noqa: E402

MODELS = os.path.join(ROOT, "models")


def build(weights=os.path.join(MODELS, "gct.pt"), device="cuda", kv_window=16,
          scale_frames=12, max_frame_num=4096, grad_ckpt=True):
    m = GCTStream(img_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=max_frame_num,
                  kv_cache_sliding_window=kv_window, kv_cache_scale_frames=scale_frames,
                  kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
                  kv_cache_max_context_frames=256, use_sdpa=True, camera_num_iterations=4,
                  use_gradient_checkpoint=grad_ckpt)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if "lora_" not in k]
    if real_missing:
        print(f"missing {len(real_missing)} keys, e.g. {real_missing[:3]}")
    del ck, sd
    return m.to(device)


def reset(model):
    model.clean_kv_cache()
    model.aggregator.total_frames_processed = 0


def decode(out, hw):
    """-> depth (B,S,H,W), conf (B,S,H,W), c2w (B,S,4,4), K (B,S,3,3)."""
    E, K = pose_encoding_to_extri_intri(out["pose_enc"].float(), hw)
    B, S = E.shape[:2]
    c2w = torch.zeros(B, S, 4, 4, device=E.device, dtype=E.dtype)
    c2w[..., :3, :4] = E                     # this model's encoding is camera-to-world
    c2w[..., 3, 3] = 1
    return out["depth"][..., 0].float(), out["depth_conf"].float(), c2w, K


def run_stream(model, images, n_anchor, keyframe_interval=1, grad=False):
    """images (S,3,H,W).  First n_anchor frames as one bidirectional block, the
    rest one by one through the cache, exactly as reconstruct.py runs a video.
    Returns depth (S,H,W), conf, c2w (S,4,4), K."""
    reset(model)
    hw = images.shape[-2:]
    dev = next(model.parameters()).device
    outs = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
        x = images[:n_anchor][None].to(dev)
        outs.append(decode(model(x, num_frame_for_scale=n_anchor, num_frame_per_block=n_anchor,
                                 causal_inference=True), hw))
        for i in range(n_anchor, images.shape[0]):
            key = keyframe_interval <= 1 or (i - n_anchor) % keyframe_interval == 0
            if not key:
                model._set_skip_append(True)
            o = model(images[i:i + 1][None].to(dev), num_frame_for_scale=n_anchor,
                      num_frame_per_block=1, causal_inference=True)
            if not key:
                model._set_skip_append(False)
            outs.append(decode(o, hw))
    reset(model)
    return [torch.cat([o[k] for o in outs], 1)[0] for k in range(4)]


def rel_to_first(c2w):
    return torch.linalg.inv(c2w[:1]) @ c2w


def depth_scale(pred, gt, valid):
    """Median ratio gt/pred over valid pixels of the whole clip (one scale)."""
    r = gt[valid] / pred[valid].clamp_min(1e-6)
    return r.median() if r.numel() else torch.tensor(1.0, device=pred.device)


def rot_err_deg(Ra, Rb):
    c = ((Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.arccos(c.clamp(-1, 1)))


@torch.no_grad()
def metrics(depth, c2w, gt):
    """Clip metrics.  Depth is aligned with one median scale per clip; poses
    are compared relative to the first frame, translations after a
    least-squares scale (ATE) and also with the depth scale (consistency)."""
    dev = depth.device
    gd, gv = gt["depth"].to(dev), gt["valid"].to(dev)
    s = depth_scale(depth, gd, gv)
    p, g = depth[gv] * s, gd[gv]
    absrel = ((p - g).abs() / g).mean().item()
    d1 = (torch.maximum(p / g, g / p) < 1.25).float().mean().item()
    pc = rel_to_first(c2w.double())
    gc = gt["c2w"].to(dev).double()
    rerr = rot_err_deg(pc[1:, :3, :3], gc[1:, :3, :3]).mean().item()
    # relative rotation between consecutive frames
    dpr = torch.linalg.inv(pc[:-1]) @ pc[1:]
    dgt = torch.linalg.inv(gc[:-1]) @ gc[1:]
    rpe_r = rot_err_deg(dpr[:, :3, :3], dgt[:, :3, :3]).mean().item()
    tp, tg = pc[:, :3, 3], gc[:, :3, 3]
    span = (tg - tg.mean(0)).norm(dim=1).mean().clamp_min(1e-6)
    ls = (tp * tg).sum() / (tp * tp).sum().clamp_min(1e-12)
    ate = ((tp * ls - tg).norm(dim=1).mean() / span).item()
    ate_d = ((tp * s.double() - tg).norm(dim=1).mean() / span).item()
    return {"absrel": absrel, "d1": d1, "rot": rerr, "rpe_rot": rpe_r, "ate": ate,
            "ate_dscale": ate_d}
