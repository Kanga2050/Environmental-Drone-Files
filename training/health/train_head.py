"""Train the plant-health head on cached backbone features.

    python train_head.py FEATS.npz OUT.pt [--hidden 512] [--epochs 40] [--views clean|fpv|all]

Reports accuracy and macro-F1 on the held-out taxa for clean photos and for
FPV-degraded views separately, the confusion matrix, and the confidence at
which predictions reach 85% precision.
"""
import argparse, json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["healthy", "stressed", "dead"]
ap = argparse.ArgumentParser()
ap.add_argument("feats")
ap.add_argument("out")
ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--drop", type=float, default=0.3)
ap.add_argument("--views", default="all", choices=["clean", "fpv", "all"])
ap.add_argument("--ls", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--drop_obs", default=None, help="file of observation ids to leave out (conflicting labels)")
a = ap.parse_args()
torch.manual_seed(a.seed)

d = np.load(a.feats)
X = torch.from_numpy(d["feats"].astype(np.float32))      # N, V, D
y = torch.from_numpy(d["y"]).long()
sp = d["split"].copy()
if a.drop_obs:
    bad = {int(x) for x in open(a.drop_obs).read().split()}
    sp[np.isin(d["obs"], list(bad))] = 3          # in no split
    print("dropped", int((sp == 3).sum()), "conflicting photos")
V = X.shape[1]
tv = {"clean": [0, 1], "fpv": list(range(2, V)), "all": list(range(V))}[a.views]
mu = X[sp == 0][:, tv].reshape(-1, X.shape[2]).mean(0)
sd = X[sp == 0][:, tv].reshape(-1, X.shape[2]).std(0) + 1e-4
Xn = (X - mu) / sd


def head(dim):
    if a.hidden:
        return nn.Sequential(nn.Dropout(a.drop), nn.Linear(dim, a.hidden), nn.GELU(), nn.Dropout(a.drop),
                             nn.Linear(a.hidden, len(CLASSES)))
    return nn.Sequential(nn.Dropout(a.drop), nn.Linear(dim, len(CLASSES)))


net = head(X.shape[2]).cuda()
tr = np.where(sp == 0)[0]
cnt = np.bincount(d["y"][tr], minlength=3)
w = torch.tensor(cnt.sum() / (3 * cnt), dtype=torch.float32).cuda()
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
steps = a.epochs * (len(tr) // 256 + 1)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
Xg, yg = Xn.cuda(), y.cuda()
tvt = torch.tensor(tv).cuda()


@torch.no_grad()
def predict(idx, vs):
    net.eval()
    lg = torch.stack([net(Xg[idx][:, v]) for v in vs]).mean(0)
    net.train()
    return lg


def metrics(lg, yy, T=1.0):
    p = F.softmax(lg / T, -1)
    pr = p.argmax(1)
    acc = float((pr == yy).float().mean())
    f1s = []
    cm = torch.zeros(3, 3, dtype=torch.long)
    for t, q in zip(yy.cpu(), pr.cpu()):
        cm[t, q] += 1
    for c in range(3):
        tp = cm[c, c].item(); fp = cm[:, c].sum().item() - tp; fn = cm[c].sum().item() - tp
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    return dict(acc=round(acc, 4), macro_f1=round(float(np.mean(f1s)), 4),
                f1={c: round(f, 4) for c, f in zip(CLASSES, f1s)}, cm=cm.tolist())


va, te = np.where(sp == 1)[0], np.where(sp == 2)[0]
fpv = list(range(2, V))
best, best_state = -1, None
for ep in range(a.epochs):
    perm = np.random.default_rng(a.seed + ep).permutation(tr)
    for i in range(0, len(perm), 256):
        b = torch.from_numpy(perm[i:i + 256]).cuda()
        v = tvt[torch.randint(len(tv), (len(b),), device="cuda")]
        xb = Xg[b, v]
        loss = F.cross_entropy(net(xb), yg[b], weight=w, label_smoothing=a.ls)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    m = metrics(predict(va, fpv), yg[va])
    if m["macro_f1"] > best:
        best, best_state = m["macro_f1"], {k: t.clone() for k, t in net.state_dict().items()}
    if ep % 5 == 4 or ep == a.epochs - 1:
        print(f"ep {ep + 1} loss {loss.item():.3f} val-fpv {m['acc']:.3f} f1 {m['macro_f1']:.3f}", flush=True)
net.load_state_dict(best_state)

# temperature on validation (FPV views), then the 85%-precision confidence
lv = predict(va, fpv)
Ts = np.linspace(0.5, 3, 26)
nll = [F.cross_entropy(lv / T, yg[va]).item() for T in Ts]
T = float(Ts[int(np.argmin(nll))])
pv = F.softmax(lv / T, -1)
conf, pr = pv.max(1)
ok = (pr == yg[va]).float()
tau = 1.0
for t in np.linspace(0.34, 0.99, 66):
    s = conf >= t
    if s.sum() >= 30 and ok[s].mean() >= 0.85:
        tau = float(t)
        break
res = {"val_fpv": metrics(lv, yg[va], T), "test_clean": metrics(predict(te, [0, 1]), yg[te], T),
       "test_fpv": metrics(predict(te, fpv), yg[te], T), "temp": T, "tau85": tau}
lt = F.softmax(predict(te, fpv) / T, -1)
c, p = lt.max(1)
s = c >= tau
res["test_fpv_at_tau"] = dict(coverage=round(float(s.float().mean()), 4),
                              precision=round(float((p[s] == yg[te][s]).float().mean()), 4) if s.any() else None)
res["args"] = vars(a)
print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=1))
torch.save({"head": net.cpu().state_dict(), "hidden": a.hidden, "drop": a.drop, "mu": mu, "sd": sd,
            "classes": CLASSES, "temp": T, "tau": tau, "metrics": res}, a.out)
