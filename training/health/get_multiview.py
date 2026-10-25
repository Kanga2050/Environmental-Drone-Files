"""Extra photos of the held-out test observations, to score the multi-view rule.

    python get_multiview.py   (writes data/multiview/{obs}_{k}.jpg and multiview.csv)
"""
import concurrent.futures as cf
import csv, json, os, time, urllib.request, zlib

ROOT = os.environ.get("HEALTH_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
OUT = os.path.join(ROOT, "multiview")
os.makedirs(OUT, exist_ok=True)


def get(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "manta-health-trainer/1.0 (personal research)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            time.sleep(3 * (k + 1))
    return None


rows = list(csv.DictReader(open(os.path.join(ROOT, "meta.csv"), encoding="utf-8")))
test = {}
for r in rows:
    if zlib.crc32(str(int(r["taxon"] or 0)).encode()) % 10 == 9:
        test.setdefault(r["obs"], r["cls"])
ids = sorted(test)
jobs = []
for i in range(0, len(ids), 100):
    b = get(f"https://api.inaturalist.org/v1/observations?id={','.join(ids[i:i + 100])}&per_page=100")
    time.sleep(1.1)
    for o in json.loads(b)["results"] if b else []:
        for k, ph in enumerate(o.get("photos", [])[:4]):
            if ph.get("license_code") in ("cc0", "cc-by", "cc-by-nc"):
                jobs.append((str(o["id"]), k, ph["url"].replace("square", "medium")))
print("photos", len(jobs), flush=True)


def fetch(j):
    path = os.path.join(OUT, f"{j[0]}_{j[1]}.jpg")
    if not os.path.exists(path):
        b = get(j[2])
        if not b:
            return None
        open(path, "wb").write(b)
    return j


with cf.ThreadPoolExecutor(8) as ex:
    got = [j for j in ex.map(fetch, jobs) if j]
with open(os.path.join(ROOT, "multiview.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["obs", "k", "cls"])
    for o, k, _ in got:
        w.writerow([o, k, test[o]])
print("saved", len(got))
