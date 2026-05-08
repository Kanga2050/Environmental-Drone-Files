
import json
import time
import urllib.request

import numpy as np

NEW, OLD = "http://127.0.0.1:8090", "http://127.0.0.1:8095"

def get(base, path, raw=False):
    with urllib.request.urlopen(base + path, timeout=120) as r:
        b = r.read()
    return b if raw else json.loads(b)

def post(base, path, payload):
    req = urllib.request.Request(base + path, data=payload,
                                 headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())

for base in (NEW, OLD):
    for _ in range(40):
        try:
            get(base, "/api/scene.json")
            break
        except Exception:
            time.sleep(10)

a, b = get(NEW, "/api/scene.json"), get(OLD, "/api/scene.json")

a.pop("live_classifier", None)
b.pop("live_classifier", None)

def walk(x, y, path=""):
    bad = []
    if type(x) is not type(y):
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            return [f"{path}: type {type(x).__name__} vs {type(y).__name__}"]
    if isinstance(x, dict):
        for k in set(x) | set(y):
            if k not in x or k not in y:
                bad.append(f"{path}.{k}: present in only one")
            else:
                bad += walk(x[k], y[k], f"{path}.{k}")
    elif isinstance(x, list):
        if len(x) != len(y):
            return [f"{path}: len {len(x)} vs {len(y)}"]
        for i, (p, q) in enumerate(zip(x, y)):
            bad += walk(p, q, f"{path}[{i}]")
    elif isinstance(x, float) or isinstance(y, float):
        if not (abs(float(x) - float(y)) <= 1e-9 * max(1, abs(float(y)))):
            bad.append(f"{path}: {x} != {y}")
    elif x != y:
        bad.append(f"{path}: {x!r} != {y!r}")
    return bad

d = walk(a, b, "scene")
print(f"scene.json: {len(d)} differences")
for x in d[:12]:
    print("   ", x)

pa, pb = get(NEW, "/api/points.bin", True), get(OLD, "/api/points.bin", True)
print(f"points.bin: {len(pa):,} vs {len(pb):,} bytes, identical={pa == pb}")

fid = a["frames"][60]["id"]
for ep in (f"/api/frame/{fid}/seen.bin", f"/api/frame/{fid}/created.bin",
           f"/api/frame/{fid}/own.bin", "/api/plant/0/points.bin"):
    x, y = get(NEW, ep, True), get(OLD, ep, True)
    print(f"{ep}: identical={x == y} ({len(x):,} bytes)")

rng = np.random.default_rng(0)
sel = np.sort(rng.choice(a["n_points"], 40000, replace=False)).astype(np.uint32)
ra, rb = post(NEW, "/api/select", sel.tobytes()), post(OLD, "/api/select", sel.tobytes())
d = walk(ra, rb, "select")
print(f"\nselect (40k points): {len(d)} differences")
for x in d[:12]:
    print("   ", x)
print(f"  n={ra['n']} height={ra.get('height_m'):.6f} vol={ra.get('volume_m3'):.6g}")
print(f"  overlap={[(o['id'], o['n']) for o in ra['overlap']]}")
print(f"  views={[v['frame'] for v in ra['views']]}")
