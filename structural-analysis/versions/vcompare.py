import json, os, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25})
D = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in glob.glob(f"{D}/v*/vpost.json"):
    d = json.load(open(f))
    if "hover" not in d or "drop100" not in d: continue
    rows.append(d)
rows.sort(key=lambda d: int(d["version"][1:]))
V = [d["version"] for d in rows]
x = np.arange(len(V))
g = lambda case, key, reg="all": [d[case].get(reg, {}).get(key, np.nan) if case in d else np.nan for d in rows]
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
ax[0, 0].bar(x, [d["info"]["struct_g"]["fus"] + d["info"]["struct_g"]["can"] for d in rows], color="#5d7ea8")
ax[0, 0].bar(x, [d["info"]["struct_g"]["can"] for d in rows], color="#b8c4d1")
ax[0, 0].set_title("Printed structure mass (g)  [light = canopy]")
ax[0, 1].plot(x, g("vto_max", "fos_conformal"), "o-", lw=2, color="#3b6ea5", label="full-throttle takeoff")
ax[0, 1].plot(x, g("hover", "fos_conformal"), "s--", lw=1.5, color="#7f8c8d", label="hover")
ax[0, 1].set_yscale("log"); ax[0, 1].set_title("Flight factor of safety (conformal print)"); ax[0, 1].legend(frameon=False)
ax[1, 0].plot(x, g("drop100", "fos_planar"), "o--", lw=1.5, color="#c0392b", label="planar layers")
ax[1, 0].plot(x, g("drop100", "fos_conformal"), "o-", lw=2, color="#27ae60", label="conformal layers")
ax[1, 0].plot(x, g("pole3", "fos_conformal"), "^-", lw=1.5, color="#8e44ad", label="3 m/s arm strike (conformal)")
ax[1, 0].axhline(1, color="k", lw=0.8); ax[1, 0].set_title("Crash factor of safety (1 m drop, arm strike)"); ax[1, 0].legend(frameon=False, fontsize=9)
f1 = [d["freqs"][0] if d.get("freqs") else np.nan for d in rows]
f2 = [np.mean(d["freqs"][1:4]) if d.get("freqs") else np.nan for d in rows]
ax[1, 1].plot(x, f1, "o-", lw=2, color="#e67e22", label="first mode"); ax[1, 1].plot(x, f2, "s--", lw=1.5, color="#16a085", label="arm bending modes (2nd–4th, mean)")
ax[1, 1].set_title("Natural frequencies (Hz)"); ax[1, 1].legend(frameon=False, fontsize=9)
for a in ax.ravel():
    a.set_xticks(x); a.set_xticklabels([v.upper() for v in V])
fig.suptitle("Manta V3 design history: the same analysis on every saved milestone", fontsize=12)
fig.tight_layout(); fig.savefig(f"{D}/../img/manta_versions.png"); plt.close(fig)
for d in rows:
    print(d["version"], round(d["info"]["mass_g"]), {c: (round(d[c]["all"]["fos_planar"], 2), round(d[c]["all"]["fos_conformal"], 2)) for c in ("hover", "vto_max", "drop100", "pole3") if c in d}, [round(f, 1) for f in (d.get("freqs") or [])[:5]])
