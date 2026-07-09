"""Score a reconstruction of the 7-minute garden video without ground truth.

  pose     GCT camera centres vs the GLOMAP (global SfM) solution of the same
           video, after a similarity alignment: ATE as a share of the
           trajectory's extent, rotation error, and relative rotation error
           over 1 s.
  revisit  the 42 GLOMAP-verified revisits (same spot seen minutes apart):
           how far apart GCT puts the two cameras compared with GLOMAP, as
           a share of the scene radius.  Drift shows up here first.
  depth    multi-view consistency of saved depth maps: frame i's depth
           warped into frame j (about 1/3 s later) vs frame j's own depth.
  focal    predicted fx, fy vs GLOMAP's calibration on the model grid.

    python eval_yard.py MAP_DIR [MAP_DIR ...]
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = np.load(os.path.join(HERE, "ref", "glomap_poses.npz"))
REVISITS = json.load(open(os.path.join(HERE, "ref", "revisit_gt.json")))


def load_run(art):
    meta = json.load(open(os.path.join(art, "meta.json")))
    out = {"meta": meta}
    if os.path.exists(os.path.join(art, "poses.npy")) and "frame_source_ids" in meta:
        c2w = np.load(os.path.join(art, "poses.npy")).astype(np.float64)
        out["src"] = np.array(meta["frame_source_ids"])
        out["c2w"] = c2w
    else:                                   # older runs: saved frames only
        n, stride = meta["num_scale_frames"], meta["anchor_stride"]
        emitted = np.array(meta["source_frame_ids"])
        consumed = (n - 1) * stride + 1
        ids, c2w = [], []
        for fid in meta["saved_frame_ids"]:
            e = fid * stride if fid < n else consumed + fid - n
            if e >= len(emitted):
                continue
            w2c = np.eye(4)
            w2c[:3] = np.load(os.path.join(art, "frames", f"{fid:06d}.npz"))["extrinsic"]
            ids.append(emitted[e])
            c2w.append(np.linalg.inv(w2c))
        out["src"], out["c2w"] = np.array(ids), np.stack(c2w)
    return out


def umeyama(src, dst):
    """Similarity (s, R, t) minimising |s R src + t - dst|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def rot_deg(Ra, Rb):
    c = (np.trace(Ra.transpose(0, 2, 1) @ Rb, axis1=1, axis2=2) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def pose_scores(run):
    ref_c2w = np.linalg.inv(np.concatenate([REF["w2c"], np.tile([[[0, 0, 0, 1]]], (len(REF["w2c"]), 1, 1))], 1))
    ref = dict(zip(REF["src_ids"].tolist(), range(len(ref_c2w))))
    pairs = [(i, ref[s]) for i, s in enumerate(run["src"]) if s in ref]
    gi = np.array([p[0] for p in pairs])
    ri = np.array([p[1] for p in pairs])
    P, G = run["c2w"][gi], ref_c2w[ri]
    s, R, t = umeyama(P[:, :3, 3], G[:, :3, 3])
    Pc = (s * (R @ P[:, :3, 3].T)).T + t
    extent = np.linalg.norm(G[:, :3, 3] - G[:, :3, 3].mean(0), axis=1)
    radius = np.percentile(extent, 90)
    ate = np.linalg.norm(Pc - G[:, :3, 3], axis=1)
    Rp = R @ P[:, :3, :3]
    rot = rot_deg(Rp, G[:, :3, :3])
    # orientation drift on its own: the one rotation that best aligns all
    # predicted camera orientations with GLOMAP's
    U, _, Vt = np.linalg.svd(np.einsum("nij,nkj->ik", G[:, :3, :3], P[:, :3, :3]))
    Ro = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    rot_o = rot_deg(Ro @ P[:, :3, :3], G[:, :3, :3])
    # relative rotation over ~1 s (30 source frames)
    src = run["src"][gi]
    pos = {sid: k for k, sid in enumerate(src)}
    a = [k for k, sid in enumerate(src) if sid + 30 in pos]
    b = [pos[src[k] + 30] for k in a]
    rpe = rot_deg(P[a, :3, :3].transpose(0, 2, 1) @ P[b, :3, :3],
                  G[a, :3, :3].transpose(0, 2, 1) @ G[b, :3, :3]) if a else np.array([np.nan])
    # revisits
    cen = {sid: Pc[k] for k, sid in enumerate(src)}
    gcen = {sid: G[k, :3, 3] for k, sid in enumerate(src)}
    rv = []
    for p in REVISITS:
        a_, b_ = p["a"], p["b"]
        if a_ in cen and b_ in cen:
            rv.append(np.linalg.norm((cen[a_] - cen[b_]) - (gcen[a_] - gcen[b_])) / radius)
    return {"frames": int(len(pairs)), "ate": float(np.median(ate) / radius),
            "ate_mean": float(ate.mean() / radius), "rot_deg": float(np.median(rot)),
            "rot_deg_orient": float(np.median(rot_o)),
            "rpe_rot_1s": float(np.nanmedian(rpe)), "revisits": len(rv),
            "revisit_err": float(np.median(rv)) if rv else None,
            "revisit_under_10pct": float(np.mean(np.array(rv) < 0.10)) if rv else None}


def depth_scores(art, gap_max=12, max_pairs=200, conf_gate=True):
    fs = sorted(glob.glob(os.path.join(art, "frames", "*.npz")))
    fr = {int(os.path.basename(f)[:6]): f for f in fs}
    keys = sorted(fr)
    pairs = [(a, b) for a, b in zip(keys, keys[1:]) if 0 < b - a <= gap_max]
    if not pairs:
        return {}
    step = max(1, len(pairs) // max_pairs)
    rels, goods, fxs, fys = [], [], [], []
    for a, b in pairs[::step]:
        A, B = np.load(fr[a]), np.load(fr[b])
        da, db = A["depth"].astype(np.float32).squeeze(), B["depth"].astype(np.float32).squeeze()
        ca = A["conf"].astype(np.float32).squeeze()
        Ka, Kb = A["intrinsic"], B["intrinsic"]
        fxs.append(Ka[0, 0])
        fys.append(Ka[1, 1])
        Ta, Tb = np.eye(4), np.eye(4)
        Ta[:3], Tb[:3] = A["extrinsic"], B["extrinsic"]          # world-to-camera
        h, w = da.shape
        v, u = np.mgrid[0:h:4, 0:w:4]
        z = da[v, u]
        m = ((ca[v, u] > 1.5) if conf_gate else np.ones_like(z, bool)) & (z > 0)
        pts = np.stack([(u - Ka[0, 2]) / Ka[0, 0] * z, (v - Ka[1, 2]) / Ka[1, 1] * z, z, np.ones_like(z)], -1)[m]
        q = (Tb @ np.linalg.inv(Ta) @ pts.T).T
        zb = q[:, 2]
        ub = q[:, 0] / zb * Kb[0, 0] + Kb[0, 2]
        vb = q[:, 1] / zb * Kb[1, 1] + Kb[1, 2]
        ok = (zb > 0) & (ub >= 0) & (ub < w - 1) & (vb >= 0) & (vb < h - 1)
        if ok.sum() < 100:
            continue
        dz = db[vb[ok].round().astype(int), ub[ok].round().astype(int)]
        rel = np.abs(dz - zb[ok]) / np.maximum(dz, 1e-6)
        rels.append(np.median(rel))
        goods.append(np.mean(rel < 0.05))
    gk = REF["K"][0]
    ghw = REF["hw"]
    return {"depth_pairs": len(rels), "depth_relerr": float(np.median(rels)),
            "depth_within5pct": float(np.mean(goods)),
            "fx": float(np.median(fxs)), "fy": float(np.median(fys)),
            "fx_glomap": float(gk[0, 0] * 518 / ghw[1]), "fy_glomap": float(gk[1, 1] * 420 / ghw[0])}


def score(art):
    run = load_run(art)
    s = {"map": os.path.basename(os.path.normpath(art)), "frames_run": run["meta"]["num_frames"]}
    s.update(pose_scores(run))
    s.update(depth_scores(art))
    a = depth_scores(art, conf_gate=False)     # every pixel: not flattered by a stricter confidence
    s["depth_relerr_all"], s["depth_within5pct_all"] = a.get("depth_relerr"), a.get("depth_within5pct")
    cloud = np.load(os.path.join(art, "cloud.npz"))
    s["points"] = int(len(cloud["points"]))
    return s


if __name__ == "__main__":
    rows = [score(a) for a in sys.argv[1:]]
    for r in rows:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
