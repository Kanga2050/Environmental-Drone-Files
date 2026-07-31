import numpy as np, json, os, sys
import fe, fos
from scipy.spatial import cKDTree

xx = fe.X[fe.EC[:, :4]]
Lm = np.array([np.linalg.norm(xx[:, a] - xx[:, b], axis=1) for a, b in fe.EDGE]).max(0)
V = fe.geom(np.arange(fe.NE))[2]
GOOD = V / Lm ** 3 * 6 * np.sqrt(2) > 0.02
near = np.zeros(fe.NN, bool)
for q in cKDTree(fe.X).query_ball_point(fe.X[np.r_[fe.TIE["a"], fe.TIE["b"].ravel()]], 2.5): near[q] = True
BULK = GOOD & ~near[fe.EC].any(1)
CEN = xx.mean(1)


def region(c):
    x, y, z = c
    if abs(z) < 8: return "rear arm" if (x < 40 or x > 160) else "rear arm root"
    if abs(z - 248) < 8: return "front arm" if (x < 40 or x > 160) else "front arm root"
    return "body shell"


REGA = None


def stressfile(case):
    if os.path.exists(f"{case}/S.npy"): return np.load(f"{case}/S.npy"), f"{case}/{case}.vtu"
    if os.path.exists(f"dyn/t_{case}_S.npy"): return np.load(f"dyn/t_{case}_S.npy"), f"dyn/t_{case}.vtu"
    if os.path.exists(f"dyn/t_{case}.vtu"):
        import meshio
        U = meshio.read(f"dyn/t_{case}.vtu").point_data["U"].ravel()
        S = fe.stress(U.astype(float)).mean(1)
        np.save(f"dyn/t_{case}_S.npy", S.astype(np.float32))
        return S, f"dyn/t_{case}.vtu"
    return None, None


def run(case):
    S, vtu = stressfile(case)
    if S is None: return None
    r = fos.evaluate(S, ".", BULK)
    vmv = fos.fi(S.astype(float), np.tile([0, 0, 1.0], (len(S), 1)))[2]
    out = dict(case=case, vm_max=float(vmv[BULK].max()), vm_loc=CEN[np.flatnonzero(BULK)[vmv[BULK].argmax()]].tolist())
    out["vm_region"] = region(out["vm_loc"])
    for k in ("planar_z", "planar_y", "planar_belly", "conformal"):
        e = r[k]["elem"]
        out[k] = dict(fos=r[k]["fos_min"], mode=r[k]["mode"], loc=CEN[e].round(1).tolist(), region=region(CEN[e]))
    out["iso_fos"] = r["iso"]["fos_min"]
    global REGA
    if REGA is None: REGA = np.array([region(c) for c in CEN])
    byreg = {}
    for name in ("rear arm", "rear arm root", "front arm", "front arm root", "body shell"):
        m = BULK & (REGA == name)
        if m.any(): byreg[name] = dict(vm=float(vmv[m].max()), fos_planar_z=float(r["planar_z_map"][m].min()), fos_conformal=float(r["conformal_map"][m].min()))
    out["regions"] = byreg
    np.save(f"post_{case}_fos.npy", np.stack([r["planar_z_map"], r["conformal_map"], vmv]).astype(np.float32))
    json.dump(out, open(f"post_{case}.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    REG = None
    for c in sys.argv[1:]:
        o = run(c)
        if o: print(c, json.dumps({k: o[k] for k in ("vm_max", "vm_region", "iso_fos")}), {k: (round(o[k]["fos"], 2), o[k]["mode"], o[k]["region"]) for k in ("planar_z", "planar_y", "planar_belly", "conformal")}, json.dumps(o["regions"]))
