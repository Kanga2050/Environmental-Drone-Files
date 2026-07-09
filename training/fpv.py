"""The yard video's analog FPV look (Foxeer camera over an analog link),
applied consistently to every frame of a clip: one camera, one link, so the
resolution, chroma smear and colour cast are drawn once per clip and only the
noise changes frame to frame.  Same recipe as the detector's training look."""
import cv2
import numpy as np


class FPVLook:

    def __init__(self, rng, strength=1.0):
        self.rng = rng
        self.lw = int(rng.uniform(170, 380))
        self.vfrac = rng.uniform(0.45, 0.75)
        self.hp_sigma = rng.uniform(1.8, 3.0)
        self.carrier = (rng.uniform(0.3, 0.9), rng.uniform(0.0, 0.6))
        self.amp = rng.uniform(1.5, 4.0) * strength * rng.uniform(0.4, 1.0)
        self.luma_smear = rng.uniform(0.5, 1.3) if rng.random() < 0.7 else 0.0
        self.k = int(rng.integers(3, 9)) | 1
        self.shift = int(rng.integers(-2, 3))
        self.grain = rng.uniform(1, 6)
        self.line = rng.uniform(0, 3)
        self.cnoise = rng.uniform(1, 4)
        self.sat = rng.uniform(0.9, 1.3)
        self.cast = rng.uniform(-8, 8, 2)
        self.q = int(rng.uniform(45, 90))

    def __call__(self, rgb):
        rng = self.rng
        h, w = rgb.shape[:2]
        lh = max(int(self.lw * h / w * self.vfrac), 32)
        lo = cv2.resize(rgb, (self.lw, lh), interpolation=cv2.INTER_AREA)
        ycc = cv2.cvtColor(lo, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        y = ycc[:, :, 0]
        hp = cv2.GaussianBlur(y, (0, 0), 0.8) - cv2.GaussianBlur(y, (0, 0), self.hp_sigma)
        xs = np.arange(lo.shape[1], dtype=np.float32)
        ys = np.arange(lo.shape[0], dtype=np.float32)[:, None]
        phase = xs[None, :] * self.carrier[0] + ys * self.carrier[1] + rng.uniform(0, 6.28)
        ycc[:, :, 1] += self.amp * hp * np.cos(phase)
        ycc[:, :, 2] += self.amp * hp * np.sin(phase)
        if self.luma_smear:
            ycc[:, :, 0] = cv2.GaussianBlur(ycc[:, :, 0], (0, 0), sigmaX=self.luma_smear,
                                            sigmaY=0.01)
        for c in (1, 2):
            ycc[:, :, c] = np.roll(cv2.blur(ycc[:, :, c], (self.k, 1)), self.shift, axis=1)
        ycc[:, :, 0] += rng.normal(0, self.grain, y.shape)
        ycc[:, :, 0] += rng.normal(0, self.line + 1e-6, (y.shape[0], 1))
        ycc[:, :, 1:] += rng.normal(0, self.cnoise, ycc[:, :, 1:].shape)
        ycc[:, :, 1:] = 128 + (ycc[:, :, 1:] - 128) * self.sat + self.cast
        lo = cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        out = cv2.resize(lo, (w, h), interpolation=cv2.INTER_LINEAR)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, self.q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)   # channel order is kept (RGB in, RGB out)
