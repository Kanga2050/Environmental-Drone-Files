"""Classifier images from iNat 2021, PlantNet-300K and iNat open data.

Each source is decoded, resized to at most --max_side px and written to
<out>/img/<source>/, with an index CSV <out>/<source>.csv (path, name, split).
Names are raw scientific names; taxonomy.py maps them to one tree later.

  python -m plant_det.prep_cls inat21 --raw datasets/raw --out datasets/cls
  python -m plant_det.prep_cls plantnet --raw datasets/raw --out datasets/cls
  python -m plant_det.prep_cls open_select --open datasets/inat_open --out datasets/cls
  python -m plant_det.prep_cls open_fetch --out datasets/cls
"""
import argparse
import csv
import io
import json
import os
import tarfile
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

MAX_SIDE = 384


def save(job):
    data, path = job
    if os.path.exists(path):
        return True
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return False
    s = MAX_SIDE / max(im.shape[:2])
    if s < 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.imwrite(path, im, [cv2.IMWRITE_JPEG_QUALITY, 90])


def bounded_map(pool, fn, it, batch=2000):
    """pool.imap reads its whole input ahead (MemoryError on a 40 GB tar);
    feed it in fixed-size batches instead."""
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == batch:
            yield from pool.map(fn, buf, chunksize=32)
            buf = []
    if buf:
        yield from pool.map(fn, buf, chunksize=32)


def write_index(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "name", "split"])
        w.writerows(rows)


def inat21(a):
    rows = []
    d_img = os.path.join(a.out, "img", "inat21")
    os.makedirs(d_img, exist_ok=True)
    with Pool(a.workers) as pool:
        for tar_name, js, split in (("val.tar.gz", "val.json", "test"),
                                    ("train_mini.tar.gz", "train_mini.json", "train")):
            if split not in a.parts.split(","):
                continue
            meta = json.load(open(os.path.join(a.raw, js)))
            cats = {c["image_dir_name"]: c["name"] for c in meta["categories"]
                    if c["kingdom"] == "Plantae" and c["phylum"] in
                    ("Tracheophyta", "Bryophyta", "Marchantiophyta")}

            def jobs():
                with tarfile.open(os.path.join(a.raw, tar_name), "r|gz") as tf:
                    for m in tf:
                        if not m.isfile():
                            continue
                        parts = m.name.split("/")
                        if len(parts) < 3 or parts[-2] not in cats:
                            continue
                        out = os.path.join(d_img, f"{split}_{parts[-1]}")
                        rows.append((os.path.relpath(out, a.out), cats[parts[-2]], split))
                        yield tf.extractfile(m).read(), out

            n = 0
            for ok in bounded_map(pool, save, jobs()):
                n += 1
                if n % 20000 == 0:
                    print(tar_name, n, flush=True)
            print(tar_name, "done", n, flush=True)
    write_index(os.path.join(a.out, f"inat21_{a.parts.replace(',', '_')}.csv"), rows)


def plantnet(a):
    z = zipfile.ZipFile(os.path.join(a.raw, "plantnet_300K.zip"))
    names = [n for n in z.namelist() if n.endswith(".json") and "name" in n.lower()]
    print("name files", names, flush=True)
    sp = json.loads(z.read(names[0]))
    d_img = os.path.join(a.out, "img", "plantnet")
    os.makedirs(d_img, exist_ok=True)
    rows = []

    def jobs():
        for info in z.infolist():
            p = info.filename.split("/")
            if not info.filename.lower().endswith(".jpg") or len(p) < 4:
                continue
            split, sid = p[-3], p[-2]
            split = {"val": "val", "test": "test"}.get(split, "train")
            out = os.path.join(d_img, p[-1])
            rows.append((os.path.relpath(out, a.out), sp[sid], split))
            yield z.read(info), out

    n = 0
    with Pool(a.workers) as pool:
        for ok in bounded_map(pool, save, jobs()):
            n += 1
            if n % 20000 == 0:
                print("plantnet", n, flush=True)
    write_index(os.path.join(a.out, "plantnet.csv"), rows)
    print("plantnet done", n, flush=True)


def open_select(a):
    """California plant observations (any quality grade, so cultivated garden
    plants are included), capped per taxon, first photo per observation."""
    import pandas as pd
    taxa = pd.read_csv(os.path.join(a.open, "taxa.csv.gz"), sep="\t",
                       usecols=["taxon_id", "ancestry", "rank_level", "rank", "name"])
    plant = taxa[taxa.ancestry.fillna("").str.startswith("48460/47126/211194")]   # vascular
    keep = plant[plant.rank_level <= 10]                    # species and below
    tid_name = dict(zip(keep.taxon_id, keep.name))
    print("vascular species-level taxa", len(tid_name), flush=True)
    sel = []
    for ch in pd.read_csv(os.path.join(a.open, "observations.csv.gz"), sep="\t",
                          usecols=["observation_uuid", "latitude", "longitude", "taxon_id",
                                   "quality_grade", "observer_id"],
                          chunksize=5_000_000):
        ch = ch[(ch.latitude.between(32.5, 42.0)) & (ch.longitude.between(-124.5, -114.1))
                & ch.taxon_id.isin(tid_name)]
        sel.append(ch)
        print("obs chunk", sum(len(s) for s in sel), flush=True)
    obs = pd.concat(sel)
    from .taxonomy import clean_name
    obs["name"] = obs.taxon_id.map(tid_name).map(clean_name)
    if a.only_hybrids:
        obs = obs[obs.name.str.contains(" × ", regex=False)]
    cnt = obs.name.value_counts()
    names = cnt[cnt >= a.min_per_taxon].index[:a.max_taxa]
    obs = obs[obs.name.isin(set(names))]
    # cultivated (casual) first so garden plants are not drowned by wild ones
    obs = obs.sample(frac=1, random_state=0)
    obs["casual"] = obs.quality_grade.eq("casual")
    obs = obs.sort_values("casual", ascending=False, kind="stable")
    obs = obs.groupby("name").head(a.per_taxon)
    print("selected obs", len(obs), "taxa", obs.name.nunique(), flush=True)
    want = set(obs.observation_uuid)
    ph = []
    for ch in pd.read_csv(os.path.join(a.open, "photos.csv.gz"), sep="\t",
                          usecols=["photo_id", "observation_uuid", "extension", "position"],
                          chunksize=5_000_000):
        ch = ch[ch.observation_uuid.isin(want) & (ch.position == 0)]
        ph.append(ch)
    ph = pd.concat(ph).drop_duplicates("observation_uuid")
    m = obs.merge(ph, on="observation_uuid")
    # split by observer so test photos come from people never seen in training
    h = m.observer_id.astype(np.int64) * 2654435761 % 100
    m["split"] = np.where(h < 5, "test", "train")
    m[["photo_id", "extension", "name", "split", "quality_grade"]].to_csv(
        os.path.join(a.out, a.select), index=False)
    print("photos", len(m), flush=True)


def open_fetch(a):
    import pandas as pd
    import requests
    from concurrent.futures import ThreadPoolExecutor
    m = pd.read_csv(os.path.join(a.out, a.select))
    m = m.groupby("name").head(a.fetch_cap)
    d_img = os.path.join(a.out, "img", "open")
    os.makedirs(d_img, exist_ok=True)
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=64))

    def get(r):
        out = os.path.join(d_img, f"{r.photo_id}.jpg")
        if os.path.exists(out):
            return out
        for _ in range(3):
            try:
                x = s.get(f"https://inaturalist-open-data.s3.amazonaws.com/photos/"
                          f"{r.photo_id}/medium.{r.extension}", timeout=30)
                if x.status_code == 200 and save((x.content, out)):
                    return out
                if x.status_code in (403, 404):
                    return None
            except Exception:
                pass
        return None

    rows = []
    with ThreadPoolExecutor(64) as ex:
        for i, (r, out) in enumerate(zip(m.itertuples(), ex.map(get, m.itertuples()))):
            if out:
                rows.append((os.path.relpath(out, a.out), r.name, r.split))
            if i % 20000 == 0:
                print("open", i, len(m), len(rows), flush=True)
    write_index(os.path.join(a.out, a.select.replace("_select", "").replace("select", "open")), rows)
    print("open done", len(rows), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["inat21", "plantnet", "open_select", "open_fetch"])
    ap.add_argument("--raw", default=r"C:\Users\shaur\Desktop\Test\datasets\raw")
    ap.add_argument("--open", default=r"C:\Users\shaur\Desktop\Test\datasets\inat_open")
    ap.add_argument("--out", default=r"C:\Users\shaur\Desktop\Test\datasets\cls")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--parts", default="test,train")
    ap.add_argument("--per_taxon", type=int, default=100)
    ap.add_argument("--min_per_taxon", type=int, default=15)
    ap.add_argument("--max_taxa", type=int, default=6000)
    ap.add_argument("--fetch_cap", type=int, default=60)
    ap.add_argument("--only_hybrids", action="store_true")
    ap.add_argument("--select", default="open_select.csv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    globals()[a.what](a)


if __name__ == "__main__":
    main()
