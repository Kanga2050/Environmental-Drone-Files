import deck, json, sys, numpy as np, subprocess
from run_fea import pmap
X = deck.X
z0 = X[:, 2].min()
CASES = {
 "hover": dict(cfd="hover_sw"),
 "vto_max": dict(cfd="vto_max_sw"),
 "gust_rear10": dict(cfd="gust_rear10"),
 "gust_side10": dict(cfd="gust_side10"),
 "ff10": dict(cfd="ff10"),
 "ff15": dict(cfd="ff15"),
 "ff20": dict(cfd="ff20"),
 "modal": dict(modal=True),
}
def prep(name, spec):
    k = spec.get("scale", 1.0)
    pf = pmap(spec["cfd"], k) if spec.get("cfd") else None
    T = spec.get("T")
    if spec.get("cfd") and T is None:
        T = [t * k for t in json.load(open(f"../cfd/{spec['cfd']}/meta.json"))["T"]]
    m = deck.write(name, T=T, pfun=pf, fixed=spec.get("fixed"), accel=spec.get("accel"), extra=spec.get("extra"), modal=spec.get("modal", False))
    print(name, json.dumps({q: m.get(q) for q in ("T", "F_aero", "a")}, default=float), flush=True)
if __name__ == "__main__":
    for n in sys.argv[1:]:
        prep(n, CASES[n])
        subprocess.run([sys.executable, "fe.py", n, "modal" if n == "modal" else "static"])
