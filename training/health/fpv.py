"""Make ordinary photos look like the yard frames: analog FPV video (Foxeer
camera), cropped and CLAHE-equalised by stream_video.py.

What the frames show, and what is simulated here:
  * low effective resolution, worse vertically (analog lines, then upscaled)
  * colour smeared sideways and shifted (narrow-band chroma)
  * cross-colour: fine luma texture (foliage!) decoded as rainbow chroma,
    which turns leaves into purple/green blobs
  * luma noise and per-line jitter, oversaturation, colour cast
  * CLAHE on lightness, JPEG
"""
import cv2
import numpy as np


def degrade(bgr, rng, strength=1.0, res=(170, 380)):
    h, w = bgr.shape[:2]
    # 1. analog resolution: small frame, fewer lines than columns
    lw = int(rng.uniform(*res))
    lh = int(lw * h / w * rng.uniform(0.45, 0.75))
    lo = cv2.resize(bgr, (lw, max(lh, 32)), interpolation=cv2.INTER_AREA)
    ycc = cv2.cvtColor(lo, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycc[:, :, 0]
    # 2. cross-colour from high-frequency luma on a horizontal carrier
    hp = cv2.GaussianBlur(y, (0, 0), 0.8) - cv2.GaussianBlur(y, (0, 0), rng.uniform(1.8, 3.0))
    xs = np.arange(lo.shape[1], dtype=np.float32)
    ys = np.arange(lo.shape[0], dtype=np.float32)[:, None]
    phase = xs[None, :] * rng.uniform(0.3, 0.9) + ys * rng.uniform(0.0, 0.6) + rng.uniform(0, 6.28)
    amp = rng.uniform(1.5, 4.0) * strength * rng.uniform(0.4, 1.0)
    ycc[:, :, 1] += amp * hp * np.cos(phase)
    ycc[:, :, 2] += amp * hp * np.sin(phase)
    if rng.random() < 0.7:                  # luma bandwidth: horizontal smear
        ycc[:, :, 0] = cv2.GaussianBlur(ycc[:, :, 0], (0, 0), sigmaX=rng.uniform(0.5, 1.3),
                                        sigmaY=0.01)
    # 3. narrow chroma: horizontal smear and shift
    k = int(rng.integers(3, 9)) | 1
    for c in (1, 2):
        ycc[:, :, c] = cv2.blur(ycc[:, :, c], (k, 1))
        ycc[:, :, c] = np.roll(ycc[:, :, c], int(rng.integers(-2, 3)), axis=1)
    # 4. noise: luma grain and per-line offsets
    ycc[:, :, 0] += rng.normal(0, rng.uniform(1, 6), y.shape)
    ycc[:, :, 0] += rng.normal(0, rng.uniform(0, 3), (y.shape[0], 1))
    ycc[:, :, 1:] += rng.normal(0, rng.uniform(1, 4), ycc[:, :, 1:].shape)
    # 5. saturation boost / colour cast around neutral chroma
    sat = rng.uniform(0.9, 1.3)
    ycc[:, :, 1:] = 128 + (ycc[:, :, 1:] - 128) * sat + rng.uniform(-8, 8, 2)
    lo = cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    out = cv2.resize(lo, (w, h), interpolation=cv2.INTER_LINEAR)
    # 6. CLAHE on lightness (stream_video.py --clahe)
    if rng.random() < 0.8:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=rng.uniform(1.5, 3.0),
                                       tileGridSize=(8, 8)).apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    q = int(rng.uniform(45, 90))
    out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])[1],
                       cv2.IMREAD_COLOR)
    return out
