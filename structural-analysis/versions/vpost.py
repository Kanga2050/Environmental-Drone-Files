import sys, os, json, numpy as np
D = os.path.dirname(os.path.abspath(__file__))
v = sys.argv[1]
os.chdir(os.path.join(D, v))
sys.path.insert(0, os.path.join(D, "../fea"))
import fe, deck, fos
from scipy.spatial import cKDTree
X = fe.X
xx = X[fe.EC[:, :4]]
Lm = np.array([np.linalg.norm(xx[:, a] - xx[:, b], axis=1) for a, b in fe.EDGE]).max(0)
V = fe.geom(np.arange(fe.NE))[2]
good = V / Lm ** 3 * 6 * np.sqrt(2) > 0.02
ex = np.r_[fe.TIE["a"], fe.TIE["b"].ravel(), np.concatenate([deck.cs[k] for k in deck.cs])]
near = np.zeros(fe.NN, bool)
for q in cKDTree(X).query_ball_point(X[ex], 4.0): near[q] = True
z0 = X[:, 2].min()
for q in cKDTree(X).query_ball_point(X[np.flatnonzero(X[:, 2] < z0 + 3)], 6.0): near[q] = True
bulk = good & ~near[fe.EC].any(1)
cen = xx.mean(1)
zf = deck.MOT[2][2]
def region(c):
    x, y, z = c
    if abs(z) < 8: return "rear arm" if (x < 40 or x > 160) else "rear arm root"
    if abs(z - zf) < 8: return "front arm" if (x < 40 or x > 160) else "front arm root"
    return "body shell"
reg = np.array([region(c) for c in cen])
out = dict(version=v, info=json.load(open("info.json")))
for case, path in (("hover", "hover/S.npy"), ("vto_max", "vto_max/S.npy"), ("drop100", "dyn/t_drop100_S.npy"), ("pole3", "dyn/t_pole3_S.npy")):
    if not os.path.exists(path): continue
    S = np.load(path).astype(float)
    r = fos.evaluate(S, ".", bulk)
    vm = fos.fi(S, np.tile([0, 0, 1.0], (len(S), 1)))[2]
    o = {}
    for name in ("rear arm", "rear arm root", "front arm", "front arm root", "body shell"):
        m = bulk & (reg == name)
        if m.sum() < 50: continue
        o[name] = dict(vm999=float(np.percentile(vm[m], 99.9)), vm_max=float(vm[m].max()),
                       fos_planar=float(np.percentile(r["planar_z_map"][m], 0.1)), fos_conformal=float(np.percentile(r["conformal_map"][m], 0.1)))
    o["all"] = dict(vm999=float(np.percentile(vm[bulk], 99.9)), fos_planar=float(np.percentile(r["planar_z_map"][bulk], 0.1)), fos_conformal=float(np.percentile(r["conformal_map"][bulk], 0.1)))
    out[case] = o
if os.path.exists("vto_max/U.npy") or os.path.exists("vto_max/vto_max.vtu"):
    pass
if not os.path.exists("dyn/freqs.json") and os.path.exists("dyn/modes.npz"):
    json.dump((np.load("dyn/modes.npz")["w"] / 2 / np.pi).tolist(), open("dyn/freqs.json", "w"))
f = json.load(open("dyn/freqs.json")) if os.path.exists("dyn/freqs.json") else None
out["freqs"] = [x for x in f if x > 1][:10] if f else None
for c in ("drop100", "pole3"):
    p = f"dyn/t_{c}.json"
    if os.path.exists(p):
        d = json.load(open(p)); out[c + "_F"] = d["F_peak"]
json.dump(out, open("vpost.json", "w"), indent=1, default=float)
print(v, json.dumps({k: out[k]["all"] for k in ("hover", "vto_max", "drop100", "pole3") if k in out}))
