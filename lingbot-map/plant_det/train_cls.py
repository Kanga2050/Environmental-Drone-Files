"""Train the hierarchical plant classifier on iNat 2021 + PlantNet-300K +
iNat open-data (California) images.

  python -m plant_det.train_cls labels                      # build labels.csv / vocab.json
  python -m plant_det.train_cls train --backbone bioclip2 --train_blocks 4 --name c_bioclip2
"""
import argparse
import json
import os
import time

import numpy as np

CLS = r"C:\Users\shaur\Desktop\Test\datasets\cls"
RUNS = r"C:\Users\shaur\Desktop\Test\datasets\cls_runs"


# ----------------------------------------------------------------- labels

def build_labels(a):
    import pandas as pd
    from .taxonomy import FORMS, Taxonomy
    tax = Taxonomy(os.path.join(a.open, "taxa.csv.gz"), a.gift)
    dfs = []
    import glob
    for src in a.label_sources.split(","):
        for p in sorted(glob.glob(os.path.join(a.cls, src + "*.csv"))):
            if "select" in p:
                continue
            d = pd.read_csv(p)
            d["src"] = src
            dfs.append(d)
            print(p, len(d), flush=True)
    df = pd.concat(dfs, ignore_index=True)
    # earlier selections truncated hybrids to "Genus ×"; those rows are dropped
    df = df[~df.name.str.strip().str.endswith("×")]
    names = df.name.unique()
    tid = {n: tax.find_species(n) for n in names}
    df["tid"] = df.name.map(tid)
    print("unmatched names", int(df.tid.isna().sum()), "images,",
          sum(v is None for v in tid.values()), "names", flush=True)
    df = df[df.tid.notna()].copy()
    df["tid"] = df.tid.astype(np.int64)
    lin = {t: tax.lineage(t) for t in df.tid.unique()}
    df["family"] = df.tid.map(lambda t: lin[t].get("family"))
    df["genus"] = df.tid.map(lambda t: lin[t].get("genus"))
    df["species"] = df.tid.map(lambda t: lin[t].get("species"))
    df = df[df.family.notna() & df.genus.notna()]
    sp_tid = {}
    for t, L in lin.items():
        if L.get("species"):
            sp_tid.setdefault(L["species"], L.get("species_id", t))
    forms = tax.build_forms(list(set(sp_tid.values())))
    df["form"] = df.species.map(lambda s: forms.get(sp_tid.get(s)) if isinstance(s, str) else None)
    # genus-only images get the genus' majority form
    gform = df[df.form.notna()].groupby("genus").form.agg(lambda s: s.mode().iloc[0])
    df.loc[df.form.isna(), "form"] = df.loc[df.form.isna(), "genus"].map(gform)
    print("form known", round(df.form.notna().mean(), 3),
          df.form.value_counts().to_dict(), flush=True)

    rng = np.random.default_rng(0)
    # novel species: all images held out to test naming above species level
    tr_counts = df[df.split == "train"].species.value_counts()
    cand = tr_counts[tr_counts >= 20].index.to_numpy()
    novel = set(rng.choice(cand, int(0.03 * len(cand)), replace=False))
    df.loc[df.species.isin(novel), "split"] = "novel"
    df.loc[df.split == "val", "split"] = "test"
    tr = df[df.split == "train"]
    sp_counts = tr.species.value_counts()
    species = sorted(sp_counts[sp_counts >= 3].index)
    genera = sorted(df.genus.unique())
    families = sorted(df.family.unique())
    gi = {g: i for i, g in enumerate(genera)}
    fi = {f: i for i, f in enumerate(families)}
    si = {s: i for i, s in enumerate(species)}
    g_fam = df.drop_duplicates("genus").set_index("genus").family
    s_gen = df.drop_duplicates("species").set_index("species").genus
    vocab = {"forms": FORMS, "families": families, "genera": genera, "species": species,
             "genus_family": [fi[g_fam[g]] for g in genera],
             "species_genus": [gi[s_gen[s]] for s in species],
             "novel_species": sorted(novel)}
    # full taxonomic strings (BioCLIP's text format) for zero-shot baselines
    rep = {}
    for t, L in lin.items():
        for r in ("family", "genus", "species"):
            if L.get(r):
                rep.setdefault((r, L[r]), L)
    def taxstr(r, n):
        L = rep.get((r, n), {})
        parts = ["Plantae"] + [L.get(k, "") for k in ("phylum", "class", "order", "family")]
        if r in ("genus", "species"):
            parts.append(L.get("genus", ""))
        if r == "species":
            parts.append(n.split()[-1])
        return " ".join(x for x in parts if x)
    rank_of = {"families": "family", "genera": "genus", "species": "species"}
    vocab["tax"] = {r: [taxstr(rank_of[r], n) for n in vocab[r]]
                    for r in ("families", "genera", "species")}
    df["y_form"] = df.form.map({f: i for i, f in enumerate(FORMS)}).fillna(-1).astype(int)
    df["y_family"] = df.family.map(fi).astype(int)
    df["y_genus"] = df.genus.map(gi).astype(int)
    df["y_species"] = df.species.map(si).fillna(-1).astype(int)
    df[["path", "src", "split", "species", "y_form", "y_family", "y_genus", "y_species"]].to_csv(
        os.path.join(a.cls, f"labels{a.tag}.csv"), index=False)
    json.dump(vocab, open(os.path.join(a.cls, f"vocab{a.tag}.json"), "w"))
    print("splits", df.split.value_counts().to_dict(), flush=True)
    print(f"vocab: {len(families)} families, {len(genera)} genera, {len(species)} species",
          flush=True)


# ----------------------------------------------------------------- data

class ImageSet:
    """Numpy-only dataset (safe to pickle into spawn workers)."""

    def __init__(self, root, paths, labels, train, size=224, fpv=False):
        self.root, self.paths, self.labels = root, paths, labels
        self.train, self.size, self.fpv = train, size, fpv

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import cv2
        im = cv2.imread(os.path.join(self.root, self.paths[i]), cv2.IMREAD_COLOR)
        if im is None:
            im = np.zeros((self.size, self.size, 3), np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        rng = np.random.default_rng()
        if self.train:
            im = augment(im, rng, self.size)
        else:
            if self.fpv:                    # fixed yard-like degradation per image
                from .fpv import degrade
                im = degrade(np.ascontiguousarray(im[:, :, ::-1]), np.random.default_rng(i),
                             res=(70, 220))[:, :, ::-1]
            im = center(im, self.size)
        return np.ascontiguousarray(im.transpose(2, 0, 1)), self.labels[i]


def center(im, size):
    import cv2
    h, w = im.shape[:2]
    s = size / min(h, w)
    im = cv2.resize(im, (max(size, round(w * s)), max(size, round(h * s))),
                    interpolation=cv2.INTER_AREA)
    h, w = im.shape[:2]
    y, x = (h - size) // 2, (w - size) // 2
    return im[y:y + size, x:x + size]


def augment(im, rng, size):
    """Random-resized crop, flip, and video-like degradations (blur, colour
    casts, low contrast, padding to square like the yard crops)."""
    import cv2
    from .fpv import degrade
    if rng.random() < 0.5:                  # analog FPV look of the yard frames
        im = degrade(np.ascontiguousarray(im[:, :, ::-1]), rng,
                     res=(70, 220))[:, :, ::-1]
    h, w = im.shape[:2]
    if rng.random() < 0.25:                 # whole image padded, as yard crops are
        s = max(h, w)
        c = np.full((s, s, 3), 124, np.uint8)
        c[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = im
        im = cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
    else:
        area = h * w * rng.uniform(0.25, 1.0)
        ar = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
        cw = int(min(w, round(np.sqrt(area * ar))))
        ch = int(min(h, round(np.sqrt(area / ar))))
        x = rng.integers(0, w - cw + 1)
        y = rng.integers(0, h - ch + 1)
        im = cv2.resize(im[y:y + ch, x:x + cw], (size, size), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        im = im[:, ::-1]
    im = im.astype(np.float32)
    if rng.random() < 0.3:                  # low-res video frame
        f = rng.uniform(0.3, 0.7)
        small = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        im = cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.2:
        im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.8:                  # brightness / contrast / colour cast
        im = (im - 128) * rng.uniform(0.7, 1.3) + 128 + rng.uniform(-25, 25)
        im = im * rng.uniform(0.85, 1.15, 3)
    if rng.random() < 0.2:
        g = im.mean(2, keepdims=True)
        im = g + (im - g) * rng.uniform(0.3, 1.0)
    return np.clip(im, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- train / eval

RANKS = ["form", "family", "genus", "species"]
W = {"form": 0.5, "family": 0.5, "genus": 0.5, "species": 1.0}


def evaluate(net, hier, loader, device, temp=None, max_batches=None):
    import torch
    net.eval()
    P = {k: [] for k in RANKS}
    Y = []
    with torch.no_grad():
        for b, (x, y) in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            x = x.to(device, non_blocking=True).float() / 255
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = net(x)
            p = hier.probs({k: v.float() for k, v in lg.items()}, temp)
            for k in RANKS:
                P[k].append(p[k].cpu())
            Y.append(y)
    P = {k: torch.cat(v) for k, v in P.items()}
    Y = torch.cat(Y).numpy()
    return P, Y


def metrics(P, Y, hier, taus=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    out = {}
    for j, k in enumerate(RANKS):
        m = Y[:, j] >= 0
        if m.any():
            out[f"acc_{k}"] = float((P[k][m].argmax(1).numpy() == Y[m, j]).mean())
    # hierarchical stopping: correctness of the deepest confident answer
    depth_of = {"plant": 0, "form": 0, "family": 1, "genus": 2, "species": 3}
    n = len(Y)
    for tau in taus:
        corr, depth = 0, 0
        for i in range(n):
            d = hier.decide({k: P[k][i] for k in RANKS}, tau)
            lv = d["level"]
            if lv in ("plant", "form"):
                corr += 1
                continue
            idx = {"family": hier.v["families"], "genus": hier.v["genera"],
                   "species": hier.v["species"]}[lv].index(d[lv][0])
            ok = Y[i, RANKS.index(lv)] == idx
            corr += ok
            depth += depth_of[lv]
        out[f"tau{tau}"] = {"correct": round(corr / n, 4), "mean_depth": round(depth / n, 3)}
    return out


def train(a):
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from .hier import HierNet, Hierarchy

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = os.path.join(RUNS, a.name)
    os.makedirs(out, exist_ok=True)
    vocab = json.load(open(os.path.join(a.cls, f"vocab{a.tag}.json")))
    df = pd.read_csv(os.path.join(a.cls, f"labels{a.tag}.csv"))
    if a.sources:
        df = df[df.src.isin(a.sources.split(","))]
    ycols = [f"y_{k}" for k in RANKS]
    tr = df[df.split == "train"]
    te = df[df.split == "test"].sample(frac=1, random_state=0).groupby("src").head(a.n_test)
    nv = df[df.split == "novel"].sample(frac=1, random_state=0).head(a.n_test)
    mk = lambda d, t, fpv=False: ImageSet(a.cls, d.path.to_numpy().astype(str),
                                          d[ycols].to_numpy().astype(np.int64), t, fpv=fpv)
    # species-balanced-ish sampling: weight 1/sqrt(images of that species)
    key = tr.species.fillna(tr.y_genus.astype(str))
    cnt = key.map(key.value_counts())
    w = torch.tensor((1.0 / np.sqrt(cnt.to_numpy())), dtype=torch.double)
    per_epoch = a.epoch_size or len(tr)
    sampler = WeightedRandomSampler(w, per_epoch, replacement=True)
    dl = DataLoader(mk(tr, True), batch_size=a.bs, sampler=sampler, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)
    dl_te = DataLoader(mk(te, False), batch_size=128, num_workers=6)
    dl_nv = DataLoader(mk(nv, False), batch_size=128, num_workers=6)
    dl_fp = DataLoader(mk(te, False, True), batch_size=128, num_workers=6)
    print(f"train {len(tr)}  test {len(te)}  novel {len(nv)}  epoch {per_epoch}", flush=True)

    device = "cuda"
    net = HierNet(a.backbone, len(vocab["forms"]), len(vocab["families"]),
                  len(vocab["genera"]), len(vocab["species"]), a.train_blocks,
                  grad_ckpt=a.grad_ckpt).to(device)
    hier = Hierarchy(vocab)
    if a.init and not os.path.exists(os.path.join(out, "last.pt")):
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        sd = net.state_dict()
        src = ck["model"]
        old_v = ck["vocab"]
        # head rows are copied by name, so a checkpoint with a different
        # vocabulary (e.g. after adding classes) still starts from what it knew
        key_of = {"form": "forms", "family": "families", "genus": "genera", "species": "species"}
        copied = {}
        for k, v in src.items():
            if k.startswith("heads.") and v.shape != sd[k].shape:
                rank = k.split(".")[1]
                old_idx = {n: i for i, n in enumerate(old_v[key_of[rank]])}
                new_names = vocab[key_of[rank]]
                tgt = sd[k].clone()
                n = 0
                for j, nm in enumerate(new_names):
                    if nm in old_idx:
                        tgt[j] = v[old_idx[nm]]
                        n += 1
                sd[k] = tgt
                copied[k] = f"{n}/{len(new_names)}"
            elif k in sd and v.shape == sd[k].shape:
                sd[k] = v
        net.load_state_dict(sd)
        print("init from", a.init, "remapped head rows:", copied, flush=True)
    head_p = [p for n, p in net.named_parameters() if n.startswith("heads") and p.requires_grad]
    bb_p = [p for n, p in net.named_parameters() if not n.startswith("heads") and p.requires_grad]
    print(f"trainable backbone params {sum(p.numel() for p in bb_p)/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW([{"params": head_p, "lr": a.lr},
                             {"params": bb_p, "lr": a.lr * a.bb_lr_mult}], weight_decay=0.05)
    steps = a.epochs * (per_epoch // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * a.bb_lr_mult],
                                                total_steps=steps, pct_start=0.05)
    start_ep = 0
    if os.path.exists(os.path.join(out, "last.pt")):
        ck = torch.load(os.path.join(out, "last.pt"), map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep = ck["epoch"] + 1
        print("resumed at epoch", start_ep, flush=True)
    log = open(os.path.join(out, "log.txt"), "a")
    best = -1
    for ep in range(start_ep, a.epochs):
        net.train()
        t0 = time.time()
        run = {k: 0.0 for k in RANKS}
        for it, (x, y) in enumerate(dl):
            x = x.to(device, non_blocking=True).float() / 255
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = net(x)
            loss = 0
            for j, k in enumerate(RANKS):
                lk = F.cross_entropy(lg[k].float(), y[:, j], ignore_index=-1,
                                     label_smoothing=0.1)
                if torch.isfinite(lk):
                    loss = loss + W[k] * lk
                    run[k] += lk.item() if it % 50 == 0 else 0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            if it % 200 == 0:
                msg = (f"ep {ep} it {it}/{per_epoch // a.bs} loss {loss.item():.3f} "
                       f"{(it + 1) * a.bs / (time.time() - t0):.0f} img/s")
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
        P, Y = evaluate(net, hier, dl_te, device)
        m = metrics(P, Y, hier)
        Pn, Yn = evaluate(net, hier, dl_nv, device)
        mn = metrics(Pn, Yn, hier)
        Pf, Yf = evaluate(net, hier, dl_fp, device)
        mf = metrics(Pf, Yf, hier)
        rec = {"epoch": ep, "test": m, "test_fpv": mf, "novel": mn, "time": time.time() - t0}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        state = {"model": net.state_dict(), "vocab": vocab, "backbone": a.backbone,
                 "epoch": ep, "metrics": rec}
        torch.save({**state, "opt": opt.state_dict(), "sched": sched.state_dict()},
                   os.path.join(out, "last.pt"))
        score = (m.get("acc_species", 0) + m.get("acc_genus", 0)
                 + mf.get("acc_species", 0) + mf.get("acc_genus", 0))
        if score > best:
            best = score
            torch.save(state, os.path.join(out, "best.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["labels", "train"])
    ap.add_argument("--cls", default=CLS)
    ap.add_argument("--open", default=r"C:\Users\shaur\Desktop\Test\datasets\inat_open")
    ap.add_argument("--gift", default=r"C:\Users\shaur\Desktop\Test\datasets\gift")
    ap.add_argument("--name", default="c0")
    ap.add_argument("--backbone", default="dinov2_b")
    ap.add_argument("--train_blocks", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--epoch_size", type=int, default=0)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bb_lr_mult", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--n_test", type=int, default=3000)
    ap.add_argument("--sources", default="")
    ap.add_argument("--tag", default="", help="labels{tag}.csv / vocab{tag}.json")
    ap.add_argument("--init", default="", help="start from this checkpoint's weights")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--label_sources", default="inat21,plantnet,open")
    a = ap.parse_args()
    {"labels": build_labels, "train": train}[a.what](a)


if __name__ == "__main__":
    main()
