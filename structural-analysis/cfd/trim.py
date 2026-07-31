import numpy as np, json, glob, run
W = run.W
def forces(c):
    h = np.genfromtxt(f"{c}/history.csv", delimiter=",", names=True)
    it = h["Inner_Iter"]; sel = (it >= 400) & (it <= 900)
    ref = h["RefForce"][-1]
    return np.array([h[k][sel].mean() * ref for k in ("CFx", "CFy", "CFz")]), np.array([h[k][sel].mean() * ref for k in ("CMx", "CMy", "CMz")])
th, L, D, Mx = [], [], [], []
for c in sorted(glob.glob("sweep15_p*")):
    t = np.radians(int(c.split("p")[-1]))
    F, M = forces(c)
    ev = np.array([0, np.cos(t), np.sin(t)]); up = np.array([0, -np.sin(t), np.cos(t)])
    th.append(t); D.append(-F @ ev); L.append(F @ up); Mx.append(M[0])
th, L, D = map(np.array, (th, L, D))
for a, l, dd in zip(np.degrees(th), L, D): print(f"theta {a:.0f}  L {l:.3f} N  D {dd:.3f} N  L/D {l/dd:.2f}")
out = {}
for V in (10, 15, 20):
    s = (V / 15) ** 2
    tt = np.radians(np.linspace(5, 80, 751))
    Li = np.interp(tt, th, L, left=None) * s; Di = np.interp(tt, th, D) * s
    Li = np.polyval(np.polyfit(th, L, 2), tt) * s; Di = np.polyval(np.polyfit(th, D, 2), tt) * s
    T = (W - Li) / np.cos(tt)
    f = T * np.sin(tt) - Di
    i = np.flatnonzero(np.sign(f[:-1]) != np.sign(f[1:]))
    if len(i):
        k = i[0]; a = tt[k]; Tk = T[k]
        out[V] = dict(theta=float(np.degrees(a)), T=float(Tk), L=float(Li[k]), D=float(Di[k]), T_over_W=float(Tk / W))
        print(V, out[V])
json.dump(dict(theta=np.degrees(th).tolist(), L=L.tolist(), D=D.tolist(), W=W, trim=out), open("trim.json", "w"), indent=1)
