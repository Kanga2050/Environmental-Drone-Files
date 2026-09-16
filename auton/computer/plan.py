import json
import math
import numpy as np
from scipy import ndimage
from link import M, off


def rank(v, m):
    out = np.zeros_like(v, dtype=float)
    if m.sum() > 1:
        o = np.argsort(np.argsort(v[m]))
        out[m] = o / (m.sum() - 1)
    return out


def priority(cl, w, minn):
    n = cl["n"]
    m = n >= minn
    bad = rank(cl["bad"], m) ** 2
    p = w["bad"] * bad + w["plant"] * (cl["plant"] + 0.5 * cl["lowveg"]) + w["thin"] * cl["thin"]
    p = np.where(m, np.clip(p / (w["bad"] + w["plant"]), 0, 1), w["unseen"])
    return p


def regions(cl, pri, thr, minc=3, tile=6):
    m = ndimage.binary_closing(pri > thr, iterations=1) & np.isfinite(pri)
    lab, k = ndimage.label(m)
    out = []
    for r in range(1, k + 1):
        ij = np.argwhere(lab == r)
        t = ij // tile
        for key in np.unique(t, axis=0):
            s = ij[(t == key).all(1)]
            if len(s) >= minc:
                out.append(s)
    return out


def why(cl, ij):
    f = lambda k: float(np.nanmean(cl[k][ij[:, 0], ij[:, 1]]))
    n = cl["n"][ij[:, 0], ij[:, 1]]
    parts = []
    for k, lab in (("bad", "poor images"), ("thin", "thin coverage"), ("plant", "plants"), ("lowveg", "low veg"), ("struct", "structure")):
        v = f(k)
        if v > (0.5 if k == "bad" else 0.25):
            parts.append(f"{lab} {v:.0%}")
    u = float((n < 5).mean())
    if u > 0.2:
        parts.append(f"unseen {u:.0%}")
    return ", ".join(parts) or "low quality"


def mow(pts, gap, pad):
    c = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - c) if len(pts) > 1 else (None, None, np.eye(2))
    u, v = Vt[0], np.r_[-Vt[0][1], Vt[0][0]]
    a, b = (pts - c) @ u, (pts - c) @ v
    a0, a1, b0, b1 = a.min() - pad, a.max() + pad, b.min() - pad, b.max() + pad
    k = max(1, math.ceil((b1 - b0) / gap))
    out = []
    for i in range(k + 1):
        bb = b0 + (b1 - b0) * i / k
        ends = (a0, a1) if i % 2 == 0 else (a1, a0)
        out += [c + e * u + bb * v for e in ends]
    return np.array(out), math.degrees(math.atan2(u[1], u[0]))


def build(cl, o, start, g):
    w = {"bad": g.wbad, "plant": g.wplant, "thin": g.wthin, "unseen": g.wunseen}
    pri = priority(cl, w, g.minn)
    lo, cs = np.asarray(cl["lo"]), cl["size"]
    regs = []
    for ij in regions(cl, pri, g.thr):
        xy = lo + (ij + 0.5) * cs
        top = np.nanmax(cl["top"][ij[:, 0], ij[:, 1]]) if np.isfinite(cl["top"][ij[:, 0], ij[:, 1]]).any() else 0.0
        alt = float(np.clip(max(g.amin, top + g.clear), g.amin, g.amax))
        hag = max(alt - max(top, 0.0), 2.0)
        gap = 2 * hag * math.tan(math.radians(g.fov / 2)) * (1 - g.side)
        wp, yaw = mow(xy, max(gap, 1.0), cs / 2)
        score = float(np.maximum(pri[ij[:, 0], ij[:, 1]] - g.thr, 0).sum() * cs * cs)
        L = float(np.linalg.norm(np.diff(wp, axis=0), axis=1).sum())
        regs.append({"xy": xy, "wp": wp, "alt": round(alt, 1), "gap": round(gap, 1), "yaw": round(yaw), "score": score,
                     "area": round(len(ij) * cs * cs), "why": why(cl, ij), "len": L, "top": round(float(top), 1)})
    regs.sort(key=lambda r: -r["score"])
    keep, t = [], 0.0
    for r in regs:
        dt = (r["len"] + 20) / g.speed + 3 * len(r["wp"])
        if t + dt > g.budget:
            continue
        keep.append(r)
        t += dt
    order, cur, left = [], np.asarray(start[:2], float), keep[:]
    while left:
        j = min(range(len(left)), key=lambda i: min(np.linalg.norm(left[i]["wp"][0] - cur), np.linalg.norm(left[i]["wp"][-1] - cur)))
        r = left.pop(j)
        if np.linalg.norm(r["wp"][-1] - cur) < np.linalg.norm(r["wp"][0] - cur):
            r["wp"] = r["wp"][::-1]
        order.append(r)
        cur = r["wp"][-1]
    hi = max([r["alt"] for r in order] + [g.amin])
    transit = float(min(g.amax, max(hi, float(np.nanmax(cl["top"])) + g.clear if np.isfinite(cl["top"]).any() else hi)))
    it = [(M.MAV_CMD_NAV_WAYPOINT, *o, 0, (0, 0, 0, 0)),
          (M.MAV_CMD_DO_CHANGE_SPEED, 0, 0, 0, (1, g.speed, -1, 0))]
    ins = []
    for i, r in enumerate(order):
        lat, lon = off(*o, r["wp"][0][1], r["wp"][0][0])
        it.append((M.MAV_CMD_NAV_WAYPOINT, lat, lon, transit, (0, 0, 0, 0)))
        for x, y in r["wp"]:
            lat, lon = off(*o, y, x)
            it.append((M.MAV_CMD_NAV_WAYPOINT, lat, lon, r["alt"], (0, 0, 0, 0)))
        lat, lon = off(*o, r["wp"][-1][1], r["wp"][-1][0])
        it.append((M.MAV_CMD_NAV_WAYPOINT, lat, lon, transit, (0, 0, 0, 0)))
        ins.append({"id": i + 1, "why": r["why"], "alt": r["alt"], "gap": r["gap"], "yaw": r["yaw"], "area": r["area"],
                    "top": r["top"], "wps": len(r["wp"]), "center": [round(float(v), 1) for v in r["xy"].mean(0)]})
    it.append((M.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0, 0, 0, (0, 0, 0, 0)))
    return ins, it, pri, order, transit, t


class Cfg:
    def __init__(self, **k):
        d = dict(wbad=1.0, wplant=0.5, wthin=0.6, wunseen=0.5, minn=20, thr=0.45, amin=6.0, amax=40.0, clear=4.0,
                 fov=90.0, side=0.7, speed=4.0, budget=240.0)
        d.update(k)
        self.__dict__.update(d)


def show(ins, transit, t):
    print(f"{len(ins)} regions, transit {transit:.0f} m, ~{t:.0f} s")
    for r in ins:
        print(f"  {r['id']}: {r['area']:>4} m2 at {r['center']}  alt {r['alt']} m (canopy {r['top']} m)  gap {r['gap']} m  {r['wps']} wps  [{r['why']}]")


def main():
    import argparse
    from scan import save
    a = argparse.ArgumentParser()
    a.add_argument("cells")
    a.add_argument("--origin", default="37.8700,-122.2700")
    a.add_argument("--out", default="survey.json")
    a.add_argument("--url", default=None)
    for k, v in Cfg().__dict__.items():
        a.add_argument("--" + k, type=type(v), default=v)
    g = a.parse_args()
    z = np.load(g.cells)
    cl = {k: z[k] for k in z.files}
    cl["lo"], cl["size"] = cl["lo"].tolist(), float(cl["size"])
    o = tuple(float(x) for x in g.origin.split(","))
    ins, it, pri, order, transit, t = build(cl, o, (0, 0), g)
    show(ins, transit, t)
    json.dump({"origin": o, "instructions": ins, "items": it}, open(g.out, "w"), indent=1)
    np.save(g.out.replace(".json", "_pri.npy"), pri)
    if g.url:
        from link import conn, upload
        m = conn(g.url)
        print("uploaded" if upload(m, it) else "upload failed")


if __name__ == "__main__":
    main()
