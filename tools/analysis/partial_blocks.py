import argparse, glob, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", default=r"C:/Users/shaur/AppData/Local/sfm-work/out_glomap")
ap.add_argument("--images", default=r"C:/Users/shaur/AppData/Local/sfm-work/dense_glomap/images")
ap.add_argument("--block", type=int, default=250)
ap.add_argument("--overlap", type=int, default=40)
ap.add_argument("--json", required=True)
ap.add_argument("--seed", default=None)
a = ap.parse_args()

names = sorted(os.listdir(a.images))
step = a.block - a.overlap
CODES = {3221225477: "0xC0000005 ACCESS_VIOLATION", 3221226505: "0xC0000409 STACK_BUFFER_OVERRUN"}

def vcount(p):
    with open(p, "rb") as f:
        for _ in range(20):
            l = f.readline().decode("ascii", "ignore")
            if l.startswith("element vertex"): return int(l.split()[-1])
    return 0

pat = re.compile(r"block (\d+): patch_match_stereo failed \((\d+)\), (\d+)/(\d+) geometric maps present")
seen = {}
for line in open(os.path.join(a.out_dir, "run.out"), errors="ignore"):
    m = pat.search(line)
    if m:
        bi, rc, got, want = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        seen[bi] = {"exit_code": rc, "fault": CODES.get(rc, hex(rc)),
                    "geometric_maps": got, "expected_maps": want,
                    "map_coverage": round(got / want, 3)}

if a.seed and os.path.exists(a.seed):
    sd = json.load(open(a.seed))["records"]
    for k, v in sd.items():
        bi = int(k)
        if bi not in seen:
            seen[bi] = {"exit_code": v["exit_code"],
                        "fault": CODES.get(v["exit_code"], hex(v["exit_code"])),
                        "geometric_maps": v["geometric_maps"],
                        "expected_maps": v["expected_maps"],
                        "map_coverage": round(v["geometric_maps"] / v["expected_maps"], 3),
                        "source": "observed"}

rows = []
for bi, d in sorted(seen.items()):
    lo = bi * step
    blk = names[lo:lo + a.block]
    ply = os.path.join(a.out_dir, f"block_{bi:04d}.ply")
    pts = vcount(ply) if os.path.exists(ply) else 0
    missing = [n for n in blk[d["geometric_maps"]:]]
    rows.append({"block": bi, "frames": [blk[0], blk[-1]] if blk else [],
                 "frame_index_range": [lo, lo + len(blk) - 1],
                 "points_recovered": pts,
                 "points_per_map": round(pts / max(d["geometric_maps"], 1)),
                 "uncovered_frames": len(missing),
                 "est_points_if_complete": round(pts / max(d["map_coverage"], 1e-9)),
                 "est_points_lost": round(pts / max(d["map_coverage"], 1e-9)) - pts, **d})

total_lost = sum(r["est_points_lost"] for r in rows)
out = {"note": "blocks where patch_match_stereo crashed partway; fusion used only the geometric "
                "maps written before the fault. Rerun these to recover the remainder.",
       "rerun_hint": "delete the listed block_XXXX.ply and relaunch; the script recomputes only missing blocks",
       "partial_blocks": len(rows), "est_total_points_lost": total_lost, "blocks": rows}
os.makedirs(os.path.dirname(a.json), exist_ok=True)
json.dump(out, open(a.json, "w"), indent=1)
print(f"{len(rows)} partial blocks, est {total_lost:,} points recoverable -> {a.json}")
for r in rows:
    print(f"  block {r['block']:>2}  {r['frames'][0]}-{r['frames'][1]}  "
          f"{r['geometric_maps']:>3}/{r['expected_maps']} maps  "
          f"{r['points_recovered']:>9,} pts  est +{r['est_points_lost']:>9,}  {r['fault']}")
