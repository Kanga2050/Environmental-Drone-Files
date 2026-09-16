from collections import deque
import cv2
import numpy as np

K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def blur(g):
    out = []
    for ax in (0, 1):
        k = np.ones((9, 1) if ax == 0 else (1, 9), np.float32) / 9
        b = cv2.filter2D(g, -1, k)
        di = np.abs(np.diff(g, axis=ax))
        db = np.abs(np.diff(b, axis=ax))
        out.append(1 - np.maximum(di - db, 0).sum() / max(di.sum(), 1e-6))
    return float(max(out))


def raw(rgb, w=320):
    g = cv2.cvtColor(cv2.resize(rgb, (w, max(8, int(rgb.shape[0] * w / rgb.shape[1]))), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu, sd = float(g.mean()), float(g.std())
    H, W = rgb.shape[:2]
    c = cv2.cvtColor(rgb[H // 2 - min(H, 240) // 2:H // 2 + min(H, 240) // 2, W // 2 - min(W, 320) // 2:W // 2 + min(W, 320) // 2],
                     cv2.COLOR_RGB2GRAY).astype(np.float32)
    noise = float(np.median(np.abs(cv2.filter2D(c, -1, K))[1:-1, 1:-1]) * np.sqrt(np.pi / 2) / 6)
    feat = len(cv2.FastFeatureDetector_create(20).detect(g.astype(np.uint8), None))
    clip = float(((g < 8) | (g > 247)).mean())
    return {"luma": round(mu, 1), "std": round(sd, 1), "noise": round(noise, 2), "blur": round(blur(g), 3),
            "feat": feat, "clip": round(clip, 3)}


class Score:
    def __init__(self, n=3000, warm=40, luma=14.0, std=18.0):
        self.h = {k: deque(maxlen=n) for k in ("blur", "noise", "feat")}
        self.warm, self.luma, self.std = warm, luma, std

    def __call__(self, m):
        if m["luma"] < self.luma or m["std"] < self.std:
            return 0.0
        for k in self.h:
            self.h[k].append(m[k])
        if len(self.h["blur"]) < self.warm:
            return None
        md = {k: float(np.median(v)) for k, v in self.h.items()}
        s = float(np.clip(1.15 * (1 - m["blur"]) / max(1 - md["blur"], 1e-6), 0, 1)) ** 2
        n = float(np.clip(1.25 * md["noise"] / max(m["noise"], 1e-6), 0, 1))
        f = min(m["feat"] / max(md["feat"], 1.0), 1.0) ** 0.5
        e = float(np.clip(1 - 2 * m["clip"], 0, 1))
        return round(s * n * f * e, 3)
