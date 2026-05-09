
import json
import os

import numpy as np

ART = r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_avg"
meta = json.load(open(os.path.join(ART, "meta.json"), encoding="utf-8"))
UP = None

def unit(v):
    return v / np.linalg.norm(v)

def gravity_basis(up):
    ref = np.array([0, 0, 1.0]) if abs(up[1]) > 0.9 else np.array([0, 1.0, 0])
    e0 = unit(np.cross(ref, up))
    e1 = unit(np.cross(up, e0))
    return e0, e1

def adopt(basis, up):
    right = basis[:, 0]
    fwd = -basis[:, 2]
    e0, e1 = gravity_basis(up)
    h = fwd - up * float(fwd @ up)
    yaw = np.arctan2(h @ e1, h @ e0)
    pitch = np.arcsin(np.clip(fwd @ up, -1, 1))
    right0 = unit(np.cross(fwd, up))
    up0 = unit(np.cross(right0, fwd))
    roll = np.arctan2(right @ up0, right @ right0)
    return yaw, pitch, roll

def apply_cam(yaw, pitch, roll, up):
    e0, e1 = gravity_basis(up)
    fwd = unit(e0 * (np.cos(yaw) * np.cos(pitch))
               + e1 * (np.sin(yaw) * np.cos(pitch)) + up * np.sin(pitch))
    right = unit(np.cross(fwd, up))
    if roll:
        up0 = unit(np.cross(right, fwd))
        right = unit(right * np.cos(roll) + up0 * np.sin(roll))
    upv = unit(np.cross(right, fwd))
    return np.stack([right, upv, -fwd], axis=1)

ups, poses = [], []
for fid in meta["saved_frame_ids"]:
    f = os.path.join(ART, "frames", f"{fid:06d}.npz")
    if not os.path.exists(f):
        continue
    E = np.load(f)["extrinsic"].astype(np.float64)
    ups.append(E[1, :3])
    poses.append((fid, E))
UP = unit(-np.mean(ups, axis=0))
print(f"world up {np.round(UP, 4)}   over {len(poses)} recorded frames\n")

errs, rolls, flips = [], [], []
for fid, E in poses:
    r0, r1, r2 = E[0, :3], E[1, :3], E[2, :3]
    basis = np.stack([r0, -r1, -r2], axis=1)
    yaw, pitch, roll = adopt(basis, UP)
    back = apply_cam(yaw, pitch, roll, UP)
    errs.append(np.abs(back - basis).max())
    rolls.append(np.degrees(roll))

    old_fwd = basis[:, 2]
    flips.append(np.degrees(np.arccos(np.clip(old_fwd @ unit(r2), -1, 1))))

errs, rolls, flips = np.array(errs), np.array(rolls), np.array(flips)
print(f"round trip basis error : max {errs.max():.2e}  "
      f"(machine zero is ~1e-15)")
print(f"  exact for all {len(errs)} poses: {bool(errs.max() < 1e-9)}")
print()
print(f"roll the old camera could not hold: "
      f"median |{np.median(np.abs(rolls)):.1f}|deg, "
      f"p90 {np.percentile(np.abs(rolls), 90):.1f}deg, "
      f"max {np.abs(rolls).max():.1f}deg")
print(f"  frames rolled more than 5deg : "
      f"{100 * (np.abs(rolls) > 5).mean():.0f}%")
print()
print(f"old enterFrame forward vs true view direction: "
      f"min {flips.min():.2f}deg, max {flips.max():.2f}deg")
print(f"  -> the fly camera was aimed "
      f"{'exactly backwards' if flips.min() > 179.9 else 'inconsistently'}")
