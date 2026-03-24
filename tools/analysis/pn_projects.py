
import glob
import os
import sys

import cv2
import numpy as np
import requests

KEY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PLANTNET_KEY", "")
BASE = "https://my-api.plantnet.org/v2/identify"
PROJECTS = ["k-southwestern-u-s-a", "k-world-flora", "useful", "all"]

crops = sorted(glob.glob(
    r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_avg\crops\*.jpg"))
rng = np.random.default_rng(0)
pick = [crops[i] for i in rng.choice(len(crops), 12, replace=False)]
print(f"{len(crops)} crops available, testing {len(pick)}\n")

def ask(project, paths):
    files = []
    for p in paths:
        im = cv2.imread(p)
        ok, buf = cv2.imencode(".jpg", im)
        if ok:
            files.append(("images", (os.path.basename(p), buf.tobytes(),
                                     "image/jpeg")))
    if not files:
        return None, "no image"
    try:
        r = requests.post(f"{BASE}/{project}",
                          params={"api-key": KEY,
                                  "include-related-images": "false"},
                          files=files,
                          data={"organs": ["auto"] * len(files)}, timeout=60)
    except Exception as e:
        return None, f"net {e}"
    if r.status_code == 404:
        return None, "no match"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    res = r.json().get("results", [])
    if not res:
        return None, "empty"
    top = res[0]
    return (top["species"]["scientificNameWithoutAuthor"],
            float(top["score"])), ""

print(f"{'crop':<18s} " + "".join(f"{p[:20]:<30s}" for p in PROJECTS))
tally = {p: [] for p in PROJECTS}
for c in pick:
    row = f"{os.path.basename(c):<18s} "
    for p in PROJECTS:
        hit, err = ask(p, [c])
        if hit:
            tally[p].append(hit[1])
            row += f"{hit[0][:18]:<19s}{hit[1]:<11.3f}"
        else:
            row += f"{err:<30s}"
    print(row)

print()
for p in PROJECTS:
    v = np.array(tally[p]) if tally[p] else np.zeros(0)
    print(f"{p:<24s} matched {len(v):>2d}/{len(pick)}  "
          f"mean top-1 {v.mean() if len(v) else 0:.3f}  "
          f"max {v.max() if len(v) else 0:.3f}  "
          f">=0.10: {(v >= .10).sum() if len(v) else 0}")
