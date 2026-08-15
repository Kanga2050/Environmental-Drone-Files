import numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25})
h = json.load(open("fea/dyn/harmonic.json"))
z = np.load("fea/dyn/modes.npz")
f = z["w"] / 2 / np.pi
fe_ = f[f > 1]
rpm = np.array(h["rps"]) * 60
hov = np.mean(h["hover_rps"]) * 60; mx = h["max_rps"] * 60
fig, ax = plt.subplots(figsize=(8.5, 5))
for fi in fe_[fe_ < 1700]:
    ax.axhline(fi, color="#9aa5b1", lw=0.7, zorder=1)
r = np.linspace(0, 34000, 50)
ax.plot(r, r / 60, color="#3b6ea5", lw=2.2, label="1× rotor speed (imbalance)")
ax.plot(r, 3 * r / 60, color="#c0392b", lw=2.2, label="3× rotor speed (blade pass, 3 blades)")
ax.axvspan(hov * 0.85, hov * 1.15, color="#27ae60", alpha=0.15, label=f"hover band ({hov/1000:.1f}k rpm ±15%)")
ax.axvline(mx, color="#555", ls="--", lw=1); ax.text(mx - 300, 60, "full\nthrottle", ha="right", fontsize=9)
ax.set_xlim(0, 34000); ax.set_ylim(0, 1700)
ax.set_xlabel("Motor speed (rpm)"); ax.set_ylabel("Frequency (Hz)")
ax.set_title("Campbell diagram: frame natural frequencies (grey) vs. prop excitation")
ax.legend(loc="upper left", frameon=True, fontsize=9)
fig.tight_layout(); fig.savefig("img/manta_campbell.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.semilogy(rpm, h["fc_g_1P"], color="#3b6ea5", lw=2, label="1× (prop imbalance, ISO G6.3)")
ax.semilogy(rpm, h["fc_g_3P"], color="#c0392b", lw=2, label="3× (blade-pass thrust ripple, 5%)")
ax.axvspan(hov * 0.85, hov * 1.15, color="#27ae60", alpha=0.15, label="hover band")
ax.axhline(0.3, color="#555", ls=":", lw=1); ax.text(800, 0.33, "0.3 g: typical comfortable IMU vibration level", fontsize=9, color="#555")
ax.set_xlabel("Motor speed (rpm)"); ax.set_ylabel("Vibration at flight controller (g)")
ax.set_title("Forced response at the flight controller, all four motors at the same speed")
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig("img/manta_frf_fc.png"); plt.close(fig)
print("elastic modes", np.round(fe_[:12], 1).tolist(), "hover rpm", hov, "max", mx)
i = int(np.argmax(h["fc_g_1P"])); print("1P peak", rpm[i], h["fc_g_1P"][i], "vm max", max(h["vm_1P"]), max(h["vm_3P"]))
j = np.argmin(abs(rpm - hov)); print("at hover 1P", h["fc_g_1P"][j], "3P", h["fc_g_3P"][j])
