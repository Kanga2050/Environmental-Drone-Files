"""TartanAir clips (MonST3R-style preprocessed copy: per frame RGB png, metric
depth npy, camera-to-world pose and intrinsics npz) as GCT training input.

A clip is S frames of one trajectory taken every `stride` frames.  Images are
conditioned the way the yard pipeline conditions its video (optional FPV look,
then CLAHE), resized to the model grid (width 518, height a multiple of 14),
optionally after a random crop that changes aspect and field of view."""
import glob
import os

import cv2
import numpy as np
import torch

from fpv import FPVLook

ROOT = os.environ.get("TARTAN_ROOT", r"C:\Users\shaur\gct_data\tartanair")
MAX_DEPTH = 80.0            # sky and far terrain are not supervised


def sequences(envs=None, root=ROOT):
    out = []
    for env in sorted(os.listdir(root)):
        if envs and env not in envs:
            continue
        for seq in sorted(glob.glob(os.path.join(root, env, "*", "P*"))):
            n = len(glob.glob(os.path.join(seq, "*_rgb.png")))
            if n >= 40:
                out.append((env, seq, n))
    return out


def clahe(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# The drone camera as GLOMAP calibrated it on the garden video, after the
# pipeline's edge crop: 57 x 60 degree field of view shown on a 1076 x 874
# frame, i.e. a 4:3 analog picture stretched 1.31x sideways.  On the model grid
# that is 518 x 420 with fx ~ 481, fy ~ 366.
DRONE_HFOV, DRONE_VFOV, DRONE_HW = 56.6, 59.7, (420, 518)


def drone_crop(K, h, w, rng, jitter=0.12):
    """Crop (fractions) of a pinhole frame matching the drone's field of view."""
    fx, fy = K[0, 0], K[1, 1]
    cw = 2 * fx * np.tan(np.radians(DRONE_HFOV / 2)) * rng.uniform(1 - jitter, 1 + jitter) / w
    ch = 2 * fy * np.tan(np.radians(DRONE_VFOV / 2)) * rng.uniform(1 - jitter, 1 + jitter) / h
    cw, ch = min(cw, 1.0), min(ch, 1.0)
    x0, y0 = rng.uniform(0, 1 - cw), rng.uniform(0, 1 - ch)
    return (y0, y0 + ch, x0, x0 + cw)


def radial(img, K, k1, interp):
    """Warp a pinhole image as seen through a lens with radial term k1."""
    h, w = img.shape[:2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x, y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
    f = 1 + k1 * (x * x + y * y)
    return cv2.remap(img, (x * f * K[0, 0] + K[0, 2]).astype(np.float32),
                     (y * f * K[1, 1] + K[1, 2]).astype(np.float32), interp,
                     borderMode=cv2.BORDER_REFLECT)


def load_clip(seq, ids, rng=None, fpv=False, crop=None, width=518, patch=14, out_hw=None,
              drone=False, k1=0.0):
    """-> dict of tensors: images (S,3,H,W) in [0,1], depth (S,H,W), valid
    (S,H,W) bool, c2w (S,4,4) relative to the first frame, K (S,3,3)."""
    imgs, deps, poses, Ks = [], [], [], []
    look = FPVLook(rng) if fpv else None
    for i in ids:
        rgb = cv2.cvtColor(cv2.imread(os.path.join(seq, f"{i:06d}_rgb.png")), cv2.COLOR_BGR2RGB)
        d = np.load(os.path.join(seq, f"{i:06d}_depth.npy")).astype(np.float32)
        cam = np.load(os.path.join(seq, f"{i:06d}_cam.npz"))
        K = cam["camera_intrinsics"].astype(np.float64).copy()
        h, w = d.shape
        if drone and crop is None:
            crop = drone_crop(K, h, w, rng)
            out_hw = DRONE_HW
        if crop is not None:                      # (y0, y1, x0, x1) fractions
            y0, y1, x0, x1 = (int(crop[0] * h), int(crop[1] * h), int(crop[2] * w),
                              int(crop[3] * w))
            rgb, d = rgb[y0:y1, x0:x1], d[y0:y1, x0:x1]
            K[0, 2] -= x0
            K[1, 2] -= y0
            h, w = d.shape
        if k1:
            rgb = radial(rgb, K, k1, cv2.INTER_LINEAR)
            d = radial(d, K, k1, cv2.INTER_NEAREST)
        if look is not None:
            rgb = look(rgb)
        rgb = clahe(rgb)
        H = out_hw[0] if out_hw else round(h * width / w / patch) * patch
        width = out_hw[1] if out_hw else width
        sx, sy = width / w, H / h
        rgb = cv2.resize(rgb, (width, H), interpolation=cv2.INTER_AREA)
        d = cv2.resize(d, (width, H), interpolation=cv2.INTER_NEAREST)
        K[0] *= sx
        K[1] *= sy
        imgs.append(rgb)
        deps.append(d)
        poses.append(cam["camera_pose"].astype(np.float64))
        Ks.append(K)
    c2w = np.stack(poses)
    c2w = np.linalg.inv(c2w[0])[None] @ c2w
    deps = np.stack(deps)
    valid = np.isfinite(deps) & (deps > 0.05) & (deps < MAX_DEPTH)
    return {"images": torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0,
            "depth": torch.from_numpy(np.nan_to_num(deps)),
            "valid": torch.from_numpy(valid),
            "c2w": torch.from_numpy(c2w).float(),
            "K": torch.from_numpy(np.stack(Ks)).float()}


def random_clip(seqs, rng, S, stride_range=(1, 6), fpv_p=0.5, crop_p=0.5, drone_p=0.0,
                k1_max=0.0):
    env, seq, n = seqs[rng.integers(len(seqs))]
    stride = int(rng.integers(stride_range[0], stride_range[1] + 1))
    span = (S - 1) * stride
    if span >= n:
        stride = max(1, (n - 1) // (S - 1))
        span = (S - 1) * stride
    start = int(rng.integers(0, n - span))
    ids = list(range(start, start + span + 1, stride))
    if rng.random() < 0.5:                      # trajectories run both ways
        ids = ids[::-1]
    crop = None
    drone = rng.random() < drone_p
    if not drone and rng.random() < crop_p:                   # narrower field of view / other aspect
        fw = rng.uniform(0.6, 1.0)
        # never taller than the yard frames (420 on a 518-wide grid): taller
        # inputs cost up to 65% more tokens and run out of memory
        fh = min(rng.uniform(0.6, 1.0), fw * 640 / 480 * 420 / 518)
        y0, x0 = rng.uniform(0, 1 - fh), rng.uniform(0, 1 - fw)
        crop = (y0, y0 + fh, x0, x0 + fw)
    k1 = rng.uniform(-k1_max, k1_max) if k1_max else 0.0
    clip = load_clip(seq, ids, rng, fpv=rng.random() < fpv_p, crop=crop, drone=drone, k1=k1)
    clip["env"], clip["seq"], clip["ids"] = env, seq, ids
    return clip
