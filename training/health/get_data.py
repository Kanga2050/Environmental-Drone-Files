import concurrent.futures as cf
import csv, json, os, sys, time, urllib.request

ROOT = os.environ.get("HEALTH_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
API = "https://api.inaturalist.org/v1/observations"
CLASSES = {"healthy": 38, "stressed": 39, "dead": 40}
PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 7000
PER_TAXON = 25
PLACES = [("ca", "&place_id=14"), ("world", "")]


def get(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "manta-health-trainer/1.0 (personal research)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            time.sleep(3 * (k + 1))
    return None


def listing(cls, val):
    out, per_taxon = [], {}
    for tag, place in PLACES:
        want = PER_CLASS // 2 if tag == "ca" else PER_CLASS - len(out)
        got = 0
        id_below = None
        while got < want:
            url = (f"{API}?taxon_id=47126&term_id=36&term_value_id={val}&photos=true&quality_grade=research"
                   f"&photo_license=cc0,cc-by,cc-by-nc&per_page=200&order_by=id&order=desc{place}"
                   + (f"&id_below={id_below}" if id_below else ""))
            b = get(url)
            time.sleep(1.1)
            if b is None:
                break
            res = json.loads(b)["results"]
            if not res:
                break
            for o in res:
                id_below = o["id"]
                tx = (o.get("taxon") or {}).get("id")
                if per_taxon.get(tx, 0) >= PER_TAXON or not o.get("photos"):
                    continue
                per_taxon[tx] = per_taxon.get(tx, 0) + 1
                ph = o["photos"][0]
                out.append(dict(obs=o["id"], cls=cls, taxon=tx, name=(o.get("taxon") or {}).get("name", ""),
                                place=tag, lat=(o.get("geojson") or {}).get("coordinates", [None, None])[1],
                                url=ph["url"].replace("square", "medium"), license=ph.get("license_code")))
                got += 1
                if got >= want:
                    break
        print(cls, tag, got, flush=True)
    return out


def fetch(r):
    path = os.path.join(ROOT, r["cls"], f"{r['obs']}.jpg")
    if os.path.exists(path):
        return True
    b = get(r["url"])
    if b:
        open(path, "wb").write(b)
        return True
    return False


rows = []
for cls, val in CLASSES.items():
    os.makedirs(os.path.join(ROOT, cls), exist_ok=True)
    rows += listing(cls, val)
with open(os.path.join(ROOT, "meta.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
ok = 0
with cf.ThreadPoolExecutor(8) as ex:
    for i, done in enumerate(ex.map(fetch, rows)):
        ok += done
        if i % 1000 == 0:
            print(f"downloaded {i}/{len(rows)}", flush=True)
print("images", ok, "of", len(rows))
