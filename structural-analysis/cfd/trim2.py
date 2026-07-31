import numpy as np, json, sys, run, ff
W = run.W
tr = json.load(open("trim.json")); sm = json.load(open("summary.json"))
th = np.radians(tr["theta"]); L = np.array(tr["L"]); D = np.array(tr["D"])
pL, pD = np.polyfit(th, L, 2), np.polyfit(th, D, 2)
out = {}
for V in map(int, sys.argv[1:]):
    c = f"ff{V}"; r = sm[c]; s = (V / 15) ** 2
    t0 = np.radians(r["theta"]); T0 = r["T"]
    dL = r["L"] - np.polyval(pL, t0) * s; dD = r["D"] - np.polyval(pD, t0) * s
    tt = np.radians(np.linspace(3, 80, 1541)); best = None
    for t in tt:
        Lb = np.polyval(pL, t) * s; Db = np.polyval(pD, t) * s
        T = T0
        for _ in range(30):
            k = T / T0
            T = (W - Lb - dL * k) / np.cos(t)
        f = T * np.sin(t) - (Db + dD * T / T0)
        if best is not None and np.sign(f) != np.sign(best[2]):
            out[V] = dict(theta=float(np.degrees(t)), T=float(T), dL=float(dL), dD=float(dD), theta0=float(np.degrees(t0)), T0=T0)
            break
        best = (t, T, f)
    print(V, out.get(V))
    if V in out and "--make" in sys.argv[0:1] + sys.argv:
        pass
json.dump(out, open("trim2.json", "w"), indent=1)
