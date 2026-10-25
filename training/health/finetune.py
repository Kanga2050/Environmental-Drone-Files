"""Fine-tune the last blocks of the classifier backbone together with a
health head, with the FPV degradation applied on the fly.

    python finetune.py HEAD.pt OUT.pt [--blocks 4] [--epochs 3] [--drop_obs FILE]

Starts from a head trained by train_head.py (and its feature normalisation),
uses the same species-level split, and reports the same metrics.
"""
import argparse, csv, json, os, sys, zlib

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "yardmap"))
from plant_det.hier import HierNet, prep_eval  # noqa: E402
from plant_det.health import make_head  # noqa: E402
from fpv import degrade  # noqa: E402

ROOT = os.environ.get("HEALTH_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
CLASSES = ["healthy", "stressed", "dead"]
ap = argparse.ArgumentParser()
ap.add_argument("head")
ap.add_argument("out")
ap.add_argument("--ckpt", default=os.path.join(REPO, "models", "classifier", "best.pt"))
ap.add_argument("--blocks", type=int, default=4)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--head_lr", type=float, default=3e-4)
ap.add_argument("--p_fpv", type=float, default=0.6)
ap.add_argument("--drop_obs", default=None)
ap.add_argument("--workers", type=int, default=6)
a = ap.parse_args()

rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "meta.csv"), encoding="utf-8"))
        if os.path.exists(os.path.join(ROOT, r["cls"], f"{r['obs']}.jpg"))]
seen = set()
rows = [r for r in rows if not (r["obs"] in seen or seen.add(r["obs"]))]
bad = set(open(a.drop_obs).read().split()) if a.drop_obs else set()
rows = [r for r in rows if r["obs"] not in bad]
split = lambda r: (lambda h: 0 if h < 8 else 1 if h < 9 else 2)(zlib.crc32(str(int(r["taxon"] or 0)).encode()) % 10)
parts = {s: [r for r in rows if split(r) == s] for s in (0, 1, 2)}
if __name__ == "__main__":
    print({s: len(v) for s, v in parts.items()}, flush=True)


class DS(torch.utils.data.Dataset):
    def __init__(self, rows, train, fpv_seed=None):
        self.rows, self.train, self.fpv_seed = rows, train, fpv_seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        bgr = cv2.imread(os.path.join(ROOT, r["cls"], f"{r['obs']}.jpg"))
        if bgr is None:
            bgr = np.full((224, 224, 3), 124, np.uint8)
        s = 448 / max(bgr.shape[:2])
        if s < 1:
            bgr = cv2.resize(bgr, (int(bgr.shape[1] * s), int(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)
        if self.train:
            rng = np.random.default_rng()
            h, w = bgr.shape[:2]
            f = rng.uniform(0.6, 1.0)
            ch, cw = int(h * f), int(w * f)
            y0, x0 = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
            bgr = bgr[y0:y0 + ch, x0:x0 + cw]
            if rng.random() < 0.5:
                bgr = bgr[:, ::-1]
            if rng.random() < a.p_fpv:
                bgr = degrade(np.ascontiguousarray(bgr), rng)
        elif self.fpv_seed is not None:
            bgr = degrade(bgr, np.random.default_rng(int(r["obs"]) + self.fpv_seed))
        return prep_eval(np.ascontiguousarray(bgr[:, :, ::-1])), CLASSES.index(r["cls"])


def main():
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    v = ck["vocab"]
    net = HierNet(ck["backbone"], len(v["forms"]), len(v["families"]), len(v["genera"]), len(v["species"]),
                  train_blocks=a.blocks, grad_ckpt=True)
    net.load_state_dict(ck["model"], strict=False)
    net = net.cuda()
    hk = torch.load(a.head, map_location="cpu", weights_only=False)
    head = make_head(len(hk["mu"]), hk["hidden"], hk["drop"], 3)
    head.load_state_dict(hk["head"])
    head = head.cuda()
    mu, sd = hk["mu"].cuda(), hk["sd"].cuda()


    def logits(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = net.backbone((x - net.mean_t) / net.std_t)
        if f.dim() == 3:
            f = f[:, 0]
        return head((f.float() - mu) / sd)


    bb = [p for p in net.backbone.parameters() if p.requires_grad]
    print("trainable backbone params", sum(p.numel() for p in bb), flush=True)
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr}, {"params": head.parameters(), "lr": a.head_lr}], weight_decay=0.05)
    cnt = np.bincount([CLASSES.index(r["cls"]) for r in parts[0]], minlength=3)
    w = torch.tensor(cnt.sum() / (3 * cnt), dtype=torch.float32).cuda()
    dl = torch.utils.data.DataLoader(DS(parts[0], True), a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                                     persistent_workers=True)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, a.head_lr], total_steps=steps, pct_start=0.1)


    @torch.no_grad()
    def predict(rows, fpv_seed):
        net.eval(); head.eval()
        out, ys = [], []
        for x, y in torch.utils.data.DataLoader(DS(rows, False, fpv_seed), 64, num_workers=a.workers):
            x = x.cuda()
            out.append(0.5 * (logits(x) + logits(torch.flip(x, dims=[3]))).float().cpu())
            ys.append(y)
        net.train(); head.train()
        return torch.cat(out), torch.cat(ys)


    def metrics(lg, yy, T=1.0):
        pr = F.softmax(lg / T, -1).argmax(1)
        cm = torch.zeros(3, 3, dtype=torch.long)
        for t, q in zip(yy, pr):
            cm[t, q] += 1
        f1 = []
        for c in range(3):
            tp = cm[c, c].item(); fp = cm[:, c].sum().item() - tp; fn = cm[c].sum().item() - tp
            f1.append(2 * tp / max(1, 2 * tp + fp + fn))
        return dict(acc=round(float((pr == yy).float().mean()), 4), macro_f1=round(float(np.mean(f1)), 4),
                    f1={c: round(x, 4) for c, x in zip(CLASSES, f1)}, cm=cm.tolist())


    net.train()
    best, best_state, step = -1, None, 0
    for ep in range(a.epochs):
        run = []
        for x, y in dl:
            loss = F.cross_entropy(logits(x.cuda()), y.cuda(), weight=w, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run.append(loss.item())
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{steps} loss {np.mean(run[-100:]):.3f}", flush=True)
        lv, yv = predict(parts[1], 1000)
        m = metrics(lv, yv)
        print(f"epoch {ep + 1} val-fpv acc {m['acc']:.3f} f1 {m['macro_f1']:.3f}", flush=True)
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            best_state = ({k: t.detach().cpu().clone() for k, t in net.backbone.state_dict().items()},
                          {k: t.detach().cpu().clone() for k, t in head.state_dict().items()})
    net.backbone.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])

    lv, yv = predict(parts[1], 1000)
    Ts = np.linspace(0.5, 3, 26)
    T = float(Ts[int(np.argmin([F.cross_entropy(lv / t, yv).item() for t in Ts]))])
    pv = F.softmax(lv / T, -1)
    conf, pr = pv.max(1)
    ok = (pr == yv).float()
    tau = 1.0
    for t in np.linspace(0.34, 0.99, 66):
        s = conf >= t
        if s.sum() >= 30 and ok[s].mean() >= 0.85:
            tau = float(t)
            break
    lc, yc = predict(parts[2], None)
    lf, yf = predict(parts[2], 2000)
    pt = F.softmax(lf / T, -1)
    c, p = pt.max(1)
    s = c >= tau
    res = {"val_fpv": metrics(lv, yv, T), "test_clean": metrics(lc, yc, T), "test_fpv": metrics(lf, yf, T), "temp": T,
           "tau85": tau, "test_fpv_at_tau": dict(coverage=round(float(s.float().mean()), 4),
                                                 precision=round(float((p[s] == yf[s]).float().mean()), 4) if s.any() else None),
           "args": vars(a)}
    print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=1))
    torch.save({"head": head.cpu().state_dict(), "hidden": hk["hidden"], "drop": hk["drop"], "mu": hk["mu"], "sd": hk["sd"],
                "classes": CLASSES, "temp": T, "tau": tau, "metrics": res,
                "backbone": {k: t.half() for k, t in net.backbone.state_dict().items()}}, a.out)


if __name__ == "__main__":
    main()
