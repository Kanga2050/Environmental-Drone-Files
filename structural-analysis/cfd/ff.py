import numpy as np, json, glob, sys, run, swirl
W, cg = run.W, run.cg
def moments():
    th, Mx = [], []
    for c in sorted(glob.glob("sweep15_p*")):
        h = np.genfromtxt(f"{c}/history.csv", delimiter=",", names=True)
        it = h["Inner_Iter"]; sel = (it >= 400) & (it <= 900)
        th.append(int(c.split("p")[-1])); Mx.append(h["CMx"][sel].mean() * h["RefForce"][-1])
    return np.radians(th), np.array(Mx)
th, Mx = moments()
def splitM(T, Ma):
    Tf = (T * cg[1] / 1000 - Ma) / 0.2
    Tf = min(max(Tf, 0.05 * T), 0.95 * T)
    return [(T - Tf) / 2] * 2 + [Tf / 2] * 2
def make(name, V, theta, T, mesh="../m45.su2"):
    t = np.radians(theta)
    Ma = np.polyval(np.polyfit(th, Mx, 2), t) * (V / 15) ** 2
    Ts = splitM(T, Ma)
    u = [0, -V * np.cos(t), -V * np.sin(t)]
    swirl.case(name, u, Ts)
    s = open(f"{name}/case.cfg").read().replace("ITER= 2200", "ITER= 1000").replace("MESH_FILENAME= ../m45.su2", f"MESH_FILENAME= {mesh}")
    open(f"{name}/case.cfg", "w").write(s)
    m = json.load(open(f"{name}/meta.json")); m.update(theta=theta, V=V, Ma=float(Ma)); json.dump(m, open(f"{name}/meta.json", "w"), indent=1)
    print(name, theta, T, [round(x, 2) for x in Ts], round(float(Ma), 3))
if __name__ == "__main__":
    tr = json.load(open("trim.json"))["trim"]
    for V in (10, 15, 20):
        d = tr[str(V)]
        make(f"ff{V}", V, d["theta"], d["T"])
