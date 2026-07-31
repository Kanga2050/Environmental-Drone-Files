import numpy as np, json, os, sys, math, run
W, rho = run.W, run.rho
A = [math.pi * R * R for _, _, R in run.D]


def forces(c, lo=None, hi=None):
    h = np.genfromtxt(f"{c}/history.csv", delimiter=",", names=True)
    it = h["Inner_Iter"]
    hi = hi or it.max(); lo = lo if lo is not None else max(0, hi - 300)
    sel = (it >= lo) & (it <= hi)
    ref = h["RefForce"][-1]
    F = np.array([h[k][sel].mean() * ref for k in ("CFx", "CFy", "CFz")])
    M = np.array([h[k][sel].mean() * ref for k in ("CMx", "CMy", "CMz")])
    Fs = np.array([h[k][sel].std() * ref for k in ("CFx", "CFy", "CFz")])
    return F, M, Fs, int(it.max()), float(h["rmsP"][-1]) if "rmsP" in h.dtype.names else None


def power(meta):
    P = 0.0
    for i, T in enumerate(meta["T"]):
        if T <= 0: continue
        P += T * meta["mdot"][i] / (rho * A[i])
    return P


def one(c, lo=None, hi=None):
    m = json.load(open(f"{c}/meta.json"))
    F, M, Fs, n, r = forces(c, lo, hi)
    u = np.array(m["u"]); V = np.linalg.norm(u)
    T = sum(m["T"])
    out = dict(case=c, V=V, T=T, F=F.tolist(), Fstd=Fs.tolist(), M=M.tolist(), iters=n, P_ideal=power(m), theta=m.get("theta"))
    if V > 1:
        ev = -u / V; up = np.array([0, -ev[2], ev[1]]); up /= np.linalg.norm(up)
        out.update(L=float(F @ up), D=float(-F @ ev), Tlift=float(T * up[2]), Tprop=float(T * ev[2]))
        out["net_up"] = out["L"] + out["Tlift"] - W
        out["net_fwd"] = out["Tprop"] - out["D"]
    else:
        out.update(download=float(-F[2]), download_frac=float(-F[2] / T) if T > 0 else None)
    return out


if __name__ == "__main__":
    res = {}
    for c in sys.argv[1:]:
        if not os.path.exists(f"{c}/history.csv"): continue
        lo = 400 if c.startswith(("ff", "sweep")) else None
        hi = 900 if c.startswith(("ff", "sweep")) else None
        res[c] = one(c, lo, hi)
        print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[c].items() if k not in ("case",)})
    old = json.load(open("summary.json")) if os.path.exists("summary.json") else {}
    old.update(res)
    json.dump(old, open("summary.json", "w"), indent=1)
