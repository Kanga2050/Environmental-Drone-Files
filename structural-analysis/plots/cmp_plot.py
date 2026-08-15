import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25})
s = json.load(open("cfd/summary.json"))
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
lab = ["Manta body", "Flat plate,\nsame outline"]; col = ["#3b6ea5", "#95a5a6"]
dl = [s["hover_sw"]["download_frac"] * 100, s["hover_flat"]["download_frac"] * 100]
ax[0].bar(lab, dl, color=col); ax[0].set_ylabel("% of thrust lost"); ax[0].set_title("Hover: prop wash pushing\nthe body down")
for i, v in enumerate(dl): ax[0].text(i, v + 0.8, f"{v:.0f}%", ha="center")
w = 0.35; x = np.arange(2)
for j, (k, kk) in enumerate((("ff15_off", "ff15"), ("ff15_flat_off", "ff15_flat"))):
    ax[1].bar(j - w / 2, s[k]["L"], w, color=col[j], alpha=0.45, label="props off" if j == 0 else None)
    ax[1].bar(j + w / 2, s[kk]["L"], w, color=col[j], label="props on" if j == 0 else None)
    ax[2].bar(j - w / 2, s[k]["D"], w, color=col[j], alpha=0.45)
    ax[2].bar(j + w / 2, s[kk]["D"], w, color=col[j])
    ax[1].text(j - w / 2, s[k]["L"] + 0.1, f"{s[k]['L']:.1f}", ha="center", fontsize=9); ax[1].text(j + w / 2, s[kk]["L"] + 0.1, f"{s[kk]['L']:.1f}", ha="center", fontsize=9)
    ax[2].text(j - w / 2, s[k]["D"] + 0.1, f"{s[k]['D']:.1f}", ha="center", fontsize=9); ax[2].text(j + w / 2, s[kk]["D"] + 0.1, f"{s[kk]['D']:.1f}", ha="center", fontsize=9)
for a, t in ((ax[1], "Body lift at 15 m/s (N)"), (ax[2], "Body drag at 15 m/s (N)")):
    a.set_xticks(x); a.set_xticklabels(lab); a.set_title(t)
ax[1].legend(frameon=False, fontsize=9)
fig.suptitle("Manta body vs. a thin flat plate of the same outline (pale = props off, solid = props on)", fontsize=11)
fig.tight_layout(); fig.savefig("img/manta_vs_flat.png"); plt.close(fig)
print("ok")
