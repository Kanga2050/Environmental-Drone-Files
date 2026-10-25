import json, pickle
import cv2
import numpy as np
obj, img, used = pickle.load(open("det.pkl", "rb"))
W, H = 1920, 1080
o = [x.reshape(-1, 3).astype(np.float32) for x in obj]
p = [x.reshape(-1, 2).astype(np.float32) for x in img]
res = {}
for name, fl in (("fisheye", None), ("rational", cv2.CALIB_RATIONAL_MODEL)):
    keep = list(range(len(o)))
    for it in range(4):
        try:
            if name == "fisheye":
                K0 = np.array([[600, 0, W / 2], [0, 600, H / 2], [0, 0, 1]], np.float64)
                rms, K, D, rv, tv = cv2.fisheye.calibrate([o[i].reshape(-1, 1, 3).astype(np.float64) for i in keep],
                                                          [p[i].reshape(-1, 1, 2).astype(np.float64) for i in keep], (W, H),
                                                          K0, np.zeros(4), flags=1 | 2 | 8,
                                                          criteria=(3, 200, 1e-7))
            else:
                rms, K, D, rv, tv = cv2.calibrateCamera([o[i] for i in keep], [p[i] for i in keep], (W, H), None, None, flags=fl)
        except cv2.error as e:
            print(name, "error", str(e).splitlines()[-1][:150]); break
        errs = []
        for j, i in enumerate(keep):
            if name == "fisheye":
                pr, _ = cv2.fisheye.projectPoints(o[i].reshape(-1, 1, 3).astype(np.float64), rv[j], tv[j], K, D)
            else:
                pr, _ = cv2.projectPoints(o[i], rv[j], tv[j], K, D)
            errs.append(float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - p[i]) ** 2, -1)))))
        errs = np.array(errs)
        print(f"{name} pass {it}: {len(keep)} frames rms {rms:.3f} worst {errs.max():.2f}")
        lim = max(1.0, np.median(errs) * 3)
        res[name] = dict(K=K.tolist(), D=np.ravel(D).tolist(), rms=float(rms), frames=len(keep))
        if errs.max() <= lim:
            break
        keep = [i for i, e in zip(keep, errs) if e <= lim]
for k, v in res.items():
    K = np.array(v["K"]); print(k, "fx %.1f fy %.1f fy/fx %.3f cx %.1f cy %.1f" % (K[0, 0], K[1, 1], K[1, 1] / K[0, 0], K[0, 2], K[1, 2]), "D", np.round(v["D"], 4).tolist())
json.dump(dict(width=W, height=H, **{k: v for k, v in res.items()}), open("foxeer_1080.json", "w"), indent=1)
