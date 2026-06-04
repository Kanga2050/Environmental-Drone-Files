"""Run one detector/classifier candidate end to end on the yard map, in its own
artifacts folder (cloud and frames are linked from the original, never copied
or modified):

  detect_plants -> plant_seg.segment (Point Transformer) -> eval_yard -> export_viewer

  python -m plant_det.run_candidate --name yolo11m_bioclip2 \
      --detect "--detector yolo --det_weights W.pt --classifier hier --cls_weights C.pt"
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = os.path.dirname(HERE)
SRC = os.path.join(TEST, "artifacts_yard_avg")
SEG_W = os.path.join(TEST, "datasets", "plant_seg_runs", "pt7_slim", "best.pt")


def link_artifacts(dst):
    os.makedirs(dst, exist_ok=True)
    for f in ("cloud.npz", "meta.json"):
        if not os.path.exists(os.path.join(dst, f)):
            os.link(os.path.join(SRC, f), os.path.join(dst, f))
    if not os.path.exists(os.path.join(dst, "frames")):
        subprocess.run(["cmd", "/c", "mklink", "/J", os.path.join(dst, "frames"),
                        os.path.join(SRC, "frames")], check=True, capture_output=True)


def run(cmd, log):
    print(">", " ".join(cmd), flush=True)
    log.write("> " + " ".join(cmd) + "\n")
    log.flush()
    r = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--detect", default="", help="extra detect_plants.py arguments")
    ap.add_argument("--root", default=os.path.join(TEST, "cands"))
    ap.add_argument("--skip_detect", action="store_true")
    ap.add_argument("--desc_det", default="")
    ap.add_argument("--desc_cls", default="")
    ap.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()
    art = os.path.join(a.root, a.name)
    link_artifacts(art)
    if a.desc_det or a.desc_cls:
        with open(os.path.join(art, "cand.json"), "w", encoding="utf-8") as f:
            json.dump({"detector": a.desc_det, "classifier": a.desc_cls, "detect_args": a.detect}, f)
    py = sys.executable
    with open(os.path.join(art, "run.log"), "a", encoding="utf-8") as log:
        if not a.skip_detect:
            for f in ("plant_views.json",):
                if os.path.exists(os.path.join(art, f)):
                    os.remove(os.path.join(art, f))
            run([py, "detect_plants.py", "--artifacts", art] + shlex.split(a.detect), log)
        run([py, "-m", "plant_seg.segment", "--artifacts", art, "--weights", SEG_W,
             "--threshold", "0.4"], log)
        run([py, "-m", "plant_seg.eval_yard", "--artifacts", art], log)
        if a.viewer:
            run([py, "export_viewer.py", "--artifacts", art,
                 "--out", os.path.join(TEST, f"yard_viewer_{a.name}.html"),
                 "--min_conf", "0.0"], log)
    with open(os.path.join(art, "plants.json"), encoding="utf-8") as f:
        plants = json.load(f)["plants"]
    print(f"{a.name}: {len(plants)} plants")
    for p in plants:
        h = p.get("hierarchy") or {}
        print(f"  #{p['id']:<3} {p['category']:<10} views {p['views']:<3} "
              f"{h.get('level', '-'):<8} {p['species']} ({p['species_common']}) "
              f"{p['species_confidence']:.2f}")


if __name__ == "__main__":
    main()
