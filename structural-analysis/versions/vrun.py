import sys, os, json, subprocess, time
F = os.path.abspath(os.path.join(os.path.dirname(__file__), "../fea"))
v = sys.argv[1]
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), v))
sys.path.insert(0, F)
import numpy as np, deck
PY = sys.executable
def fe(name, mode="static"):
    t = time.time()
    r = subprocess.run([PY, os.path.join(F, "fe.py"), name, mode], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=F))
    open(f"{name}.log", "w").write(r.stdout + r.stderr)
    print(name, "rc", r.returncode, int(time.time() - t), flush=True)
W = deck.MT * 1e-3 * 9.81
cg = deck.CG
ym = [m[1] for m in deck.MOT]
yr, yf = ym[0], ym[2]
def split(T):
    f = (cg[1] - yr) / (yf - yr)
    return [T * (1 - f) / 2] * 2 + [T * f / 2] * 2
info = dict(mass_g=deck.MT, struct_g=deck.mstr, cg=cg.tolist(), W=W, mot=deck.MOT)
json.dump(info, open("info.json", "w"), indent=1, default=float)
print(v, info, flush=True)
if os.path.exists("hover/res.json") and os.path.exists("vto_max/res.json") and os.path.exists("dyn/harmonic.json"):
    SKIP = True
else:
    SKIP = False
deck.write("hover", T=split(1.62 * W))
json.dump({"T": split(1.62 * W)}, open("hover_meta.json", "w"))
deck.write("vto_max", T=split(4 * 13.73))
deck.write("modal", modal=True)
if not os.path.exists("hover/res.json"): fe("hover")
if not os.path.exists("vto_max/res.json"): fe("vto_max")
env = dict(os.environ, PYTHONPATH=F, HOVER_META=os.path.abspath("hover_meta.json"), NSPEED=os.environ.get("NSPEED", "91"), NM="40")
t = time.time()
if not os.path.exists("dyn/harmonic.json"):
    r = subprocess.run([PY, os.path.join(F, "dyn.py"), "harmonic"], capture_output=True, text=True, env=env)
    open("dyn.log", "w").write(r.stdout + r.stderr)
print("dyn rc", r.returncode, int(time.time() - t), flush=True)
for f in ("dyn/sig.npy", "dyn/K.npz", "dyn/M.npz"):
    if os.path.exists(f): os.remove(f)
for c, te in (("drop100", "0.012"), ("pole3", "0.005")):
    if os.path.exists(f"dyn/t_{c}.json"): continue
    t = time.time()
    r = subprocess.run([PY, os.path.join(F, "tdyn.py"), c], capture_output=True, text=True, env=dict(env, DT="2e-5", TEND=te))
    open(f"tdyn_{c}.log", "w").write(r.stdout + r.stderr)
    print("tdyn", c, r.returncode, int(time.time() - t), flush=True)
d = json.load(open("dyn/t_drop100.json"))
n = d["F_peak"] / W
for f in ("dyn/modes.npz", "dyn/sig.npy", "dyn/K.npz", "dyn/M.npz"):
    if os.path.exists(f) and f != "dyn/modes.npz": os.remove(f)
import glob
for f in glob.glob("*/U.npy") + glob.glob("dyn/*.vtu"):
    os.remove(f)
print("done", v, "n", n, flush=True)  # CLEANUP
