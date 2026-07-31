import numpy as np, json, math, sys, os, run
SPIN = [1, -1, -1, 1]
CQ = 0.011
TPL = "tpl/example_inlet.dat"
def blocks():
    L = open(TPL).read().splitlines()
    out, i = [], 1
    while i < len(L):
        tag = L[i].split("=")[1].strip(); n = int(L[i + 1].split("=")[1]); hdr = L[i + 3]
        a = np.array([[float(x) for x in l.split()] for l in L[i + 4:i + 4 + n]])
        out.append((tag, hdr, a)); i += 4 + n
    return out
def g(rn):
    return rn / (rn ** 2 + 0.3 ** 2)
def case(name, u, Ts, iters=2200):
    run.cfg(name, u, Ts, mesh="../m45.su2", iters=iters)
    m = json.load(open(f"{name}/meta.json"))
    u = np.array(u, float)
    L = [f"NMARK= {len(run.D)}"]
    sw = []
    for k, (tag, hdr, a) in enumerate(blocks()):
        x0, y0, R = run.D[k]
        T = Ts[k]; mag = m["jet"][k]
        un = math.sqrt(max(mag ** 2 - u[0] ** 2 - u[1] ** 2, 0))
        Q = CQ * T
        rr = np.linspace(0, R, 2001); I = np.trapz(g(rr / R) * rr * 2 * np.pi * rr, rr)
        C = Q / (run.rho * un * I) if un > 0 and T > 0 else 0.0
        dx, dy = a[:, 0] - x0 / 1000, a[:, 1] - y0 / 1000
        r = np.hypot(dx, dy) + 1e-9
        vt = C * g(r / R) * SPIN[k]
        vx = u[0] + vt * (-dy / r); vy = u[1] + vt * (dx / r); vz = np.full(len(a), -un)
        V = np.c_[vx, vy, vz]; n = np.linalg.norm(V, axis=1)
        a = a.copy(); a[:, 4] = n; a[:, 5:8] = V / n[:, None]
        sw.append(dict(Q=Q, vt_peak=float(np.abs(vt).max()), swirl_deg=float(np.degrees(np.arctan2(np.abs(vt).max(), un)))))
        L += [f"MARKER_TAG= {tag}", f"NROW={len(a)}", "NCOL=10", hdr] + ["\t".join(f"{v:.15e}" for v in row) for row in a]
    open(f"{name}/inlet.dat", "w").write("\n".join(L) + "\n")
    s = open(f"{name}/case.cfg").read()
    s = s.replace("CFL_NUMBER= 10", "CFL_NUMBER= 5").replace("CFL_ADAPT_PARAM= ( 0.5, 1.2, 5.0, 100.0 )", "CFL_ADAPT_PARAM= ( 0.5, 1.1, 2.0, 30.0 )")
    s = s.replace("MARKER_WALL_FUNCTIONS= ( body, STANDARD_WALL_FUNCTION )\n", "")
    s += "SPECIFIED_INLET_PROFILE= YES\nINLET_FILENAME= inlet.dat\n"
    open(f"{name}/case.cfg", "w").write(s)
    m["swirl"] = sw; json.dump(m, open(f"{name}/meta.json", "w"), indent=1)
    print(name, [round(x["vt_peak"], 2) for x in sw], [round(x["swirl_deg"], 1) for x in sw])
if __name__ == "__main__":
    W, sp = run.W, run.split
    k = 1.62
    case("hover_sw", [0, 0, -0.05], sp(k * W))
    case("vto_max_sw", [0, 0, -8.0], sp(4 * 13.73))
    case("gust_rear10", [0, 10.0, -0.05], sp(k * W))
    case("gust_side10", [-10.0, 0, -0.05], sp(k * W))
