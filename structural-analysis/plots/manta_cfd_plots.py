import numpy as np, json, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25})
os.chdir(os.path.dirname(os.path.abspath(__file__)))
tr = json.load(open("cfd/trim.json"))
sm = json.load(open("cfd/summary.json"))
W = tr["W"]
th = np.array(tr["theta"]); L = np.array(tr["L"]); D = np.array(tr["D"])
aoa = 51.3 - th
fig, ax = plt.subplots(figsize=(8, 4.3))
ax.plot(aoa, L, "o-", lw=2, color="#3b6ea5", label="Body lift")
ax.plot(aoa, D, "s-", lw=2, color="#c0392b", label="Body drag")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Body angle of attack (deg), props off"); ax.set_ylabel("Force at 15 m/s (N)")
ax.set_title(f"Manta body forces at 15 m/s (weight {W:.1f} N)")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("img/manta_body_polar.png"); plt.close(fig)
EFF = 0.52
BAT = 22.2 * 1.3 * 0.8
rows = [("Hover (trimmed)", 0, sm["hover_sw"]["T"], sm["hover_sw"]["P_ideal"])]
for V in (10, 15, 20):
    c = f"ff{V}b" if f"ff{V}b" in sm else f"ff{V}"
    if c in sm:
        rows.append((f"{V} m/s cruise", V, sm[c]["T"], sm[c]["P_ideal"]))
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
names = [r[0] for r in rows]; Tw = [r[2] / W for r in rows]; P = [r[3] / EFF for r in rows]
cols = ["#7f8c8d"] + ["#3b6ea5"] * (len(rows) - 1)
ax[0].bar(names, Tw, color=cols); ax[0].axhline(1, color="k", lw=0.8, ls=":"); ax[0].set_ylabel("Total thrust / weight"); ax[0].set_title("Thrust needed")
for i, v in enumerate(Tw): ax[0].text(i, v + 0.03, f"{v:.2f}", ha="center")
ax[1].bar(names, [BAT / p * 60 for p in P], color=cols); ax[1].set_ylabel("Flight time on 1300 mAh 6S (min)"); ax[1].set_title("Estimated endurance")
for i, p in enumerate(P): ax[1].text(i, BAT / p * 60 + 0.2, f"{BAT/p*60:.1f} min\n{p:.0f} W", ha="center", fontsize=9)
for a in ax: a.tick_params(axis="x", rotation=15)
fig.tight_layout(); fig.savefig("img/manta_power_endurance.png"); plt.close(fig)
print([(r[0], round(r[2] / W, 2), round(r[3] / EFF), round(BAT / (r[3] / EFF) * 60, 1)) for r in rows])
