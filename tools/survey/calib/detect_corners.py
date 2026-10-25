import json, sys
import cv2
import numpy as np

cap = cv2.VideoCapture("checker.mp4")
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W, H = int(cap.get(3)), int(cap.get(4))
fl = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_LARGER
import os, pickle
obj, img, used = [], [], []
if os.path.exists("det.pkl"):
    obj, img, used = pickle.load(open("det.pkl", "rb"))
for fi in ([] if used else range(0, n, 12)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, im = cap.read()
    if not ok:
        continue
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g[int(H * 0.84):, :] = 128
    g[:int(H * 0.06), :int(W * 0.15)] = 128
    found, c, meta = cv2.findChessboardCornersSBWithMeta(g, (9, 6), fl)
    if not found:
        continue
    rows, cols = meta.shape
    c = c.reshape(rows, cols, 2)
    yy, xx = np.mgrid[0:rows, 0:cols]
    o = np.stack([xx, yy, np.zeros_like(xx)], -1).reshape(-1, 1, 3).astype(np.float64)
    p = c.reshape(-1, 1, 2).astype(np.float64)
    spread = (p[:, 0, 0].max() - p[:, 0, 0].min()) / W
    if len(p) < 60:
        continue
    obj.append(o)
    img.append(p)
    used.append((fi, rows, cols, round(float(spread), 2)))
pickle.dump((obj, img, used), open("det.pkl", "wb"))
print(f"{len(used)} frames with a grid; corners per frame median {int(np.median([r * c for _, r, c, _ in used]))}")
allp = np.concatenate([p[:, 0] for p in img])
print(f"coverage x {allp[:, 0].min():.0f}-{allp[:, 0].max():.0f} of {W}, y {allp[:, 1].min():.0f}-{allp[:, 1].max():.0f} of {H}")
K = np.array([[W / 3, 0, W / 2], [0, W / 3, H / 2], [0, 0, 1]], np.float64)
D = np.zeros((4, 1))
flags = 1 | 2 | 8
keep = list(range(len(obj)))
for it in range(4):
    try:
        rms, K, D, rv, tv = cv2.fisheye.calibrate([obj[i] for i in keep], [img[i] for i in keep], (W, H), K, D,
                                                  flags=flags, criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-7))
    except cv2.error as e:
        bad = str(e)
        print("calibration error, dropping a frame:", bad[:120])
        keep = keep[:-1]
        continue
    errs = []
    for j, i in enumerate(keep):
        pr, _ = cv2.fisheye.projectPoints(obj[i], rv[j], tv[j], K, D)
        errs.append(float(np.sqrt(np.mean(np.sum((pr - img[i]) ** 2, -1)))))
    errs = np.array(errs)
    print(f"pass {it}: {len(keep)} frames, rms {rms:.3f} px, worst {errs.max():.2f}")
    lim = max(1.0, np.median(errs) * 3)
    if errs.max() <= lim:
        break
    keep = [i for i, e in zip(keep, errs) if e <= lim]
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
print(f"fx {fx:.1f} fy {fy:.1f} (ratio fy/fx {fy / fx:.3f}) cx {cx:.1f} cy {cy:.1f}")
print("D", D.ravel().round(4).tolist())
print(f"horizontal FOV at image edge ~{2 * np.degrees(np.arctan2(W / 2, fx)):.0f} deg (pinhole equiv), fisheye theta at edge {np.degrees((W / 2) / fx):.0f} deg")
json.dump({"model": "fisheye", "width": W, "height": H, "K": K.tolist(), "D": D.ravel().tolist(), "rms_px": rms,
           "frames": len(keep)}, open("foxeer_1080.json", "w"), indent=1)
