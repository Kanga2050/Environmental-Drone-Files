"""Live mapping from a camera (or any stream OpenCV can open).

    python yardmap/live.py --source 0                 # first camera / capture card
    python yardmap/live.py --source rtsp://host/stream
    python yardmap/live.py --source video/clip.mp4 --realtime   # replay a file

Opens a browser viewer (http://localhost:8080) that grows the coloured point
cloud as frames arrive, shows the camera and its path, and marks plants as
they are found: the detector ensemble runs on the newest frame every few
seconds, each box is lifted to 3D with that frame's depth and pose, nearby
sightings are merged into one plant, and every plant is named (growth form ->
family -> genus -> species, stopping where unsure) from its best views.

Reconstruction always works on the newest frame; frames that arrive while the
model is busy are dropped, so the map keeps up with a live feed.  Press
"Finish" in the viewer (or Ctrl-C) to stop: the map is saved to maps/NAME and,
unless --no_export, the full offline pipeline (detection over all saved frames,
Point Transformer point assignment, viewer export) writes viewers/NAME.html.
"""
import argparse
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from reconstruct import StreamMapper, LiveViewer          # noqa: E402

M = os.path.join(ROOT, "models")


class LatestFrame:
    """Reads a source on its own thread and keeps only the newest frame."""

    def __init__(self, source, realtime=False):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.realtime = realtime
        self.frame, self.idx, self.done = None, -1, False
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        t0, n = time.time(), 0
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                self.done = True
                return
            n += 1
            if self.realtime:                       # a file: pace it like a live feed
                wait = t0 + n / self.fps - time.time()
                if wait > 0:
                    time.sleep(wait)
            with self.lock:
                self.frame, self.idx = bgr, n - 1

    def latest(self, after=-1):
        with self.lock:
            if self.frame is None or self.idx <= after:
                return None, after
            return cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB), self.idx


class PlantTracker:
    """Detections lifted to 3D and merged into plant instances."""

    def __init__(self, args, device):
        from detect_plants import YoloDetector
        from plant_det.hier import HierClassifier
        import json
        self.det = YoloDetector(f"{M}/detector_fpv.pt,{M}/detector_yard.pt", device,
                                threshold=args.det_threshold, tta=False)
        tau = json.load(open(os.path.join(M, "classifier", "tau85.json")))
        self.cls = HierClassifier(os.path.join(M, "classifier", "best.pt"), device, tau=tau)
        self.plants = []          # dicts: center, n, crops [(score, crop)], label, kind
        self.merge_frac = args.merge_frac

    def update(self, rgb, r, hw, extent):
        boxes, scores, _ = self.det(rgb)
        if not len(boxes):
            return []
        H, W = rgb.shape[:2]
        depth, conf = np.squeeze(r["depth"]), np.squeeze(r["conf"])
        sy, sx = depth.shape[0] / H, depth.shape[1] / W
        K, c2w = r["K"], r["c2w"]
        dist = max(self.merge_frac * extent, 1e-3)
        seen = []
        for (x0, y0, x1, y1), s in zip(boxes, scores):
            if (x1 - x0) * (y1 - y0) > 0.85 * H * W:
                continue
            u0, u1 = int(x0 * sx), max(int(x1 * sx), int(x0 * sx) + 1)
            v0, v1 = int(y0 * sy), max(int(y1 * sy), int(y0 * sy) + 1)
            d, c = depth[v0:v1, u0:u1], conf[v0:v1, u0:u1]
            ok = (c > 1.5) & (d > 0)
            if ok.sum() < 20:
                continue
            vv, uu = np.nonzero(ok)
            z = np.median(d[ok])
            u, v = np.median(uu) + u0, np.median(vv) + v0
            p = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z, 1.0])
            center = (c2w @ p)[:3]
            crop = rgb[int(y0):int(y1), int(x0):int(x1)]
            best = min(self.plants, key=lambda q: np.linalg.norm(q["center"] - center),
                       default=None)
            if best is not None and np.linalg.norm(best["center"] - center) < dist:
                best["center"] = (best["center"] * best["n"] + center) / (best["n"] + 1)
                best["n"] += 1
                best["crops"] = sorted(best["crops"] + [(float(s), crop)],
                                       key=lambda t: -t[0])[:8]
                best["dirty"] = True
                q = best
            else:
                q = {"id": len(self.plants), "center": center, "n": 1,
                     "crops": [(float(s), crop)], "label": "plant", "dirty": True}
                self.plants.append(q)
            seen.append((q, (x0, y0, x1, y1)))
        for q in {id(q): q for q, _ in seen}.values():
            if q["dirty"] and (q["n"] in (1, 3) or q["n"] % 5 == 0):
                q["label"] = self.name(q)
                q["dirty"] = False
        return seen

    def name(self, q):
        d = self.cls.classify_views([c for _, c in q["crops"]])
        lvl = d["level"]
        if lvl == "plant":
            return "plant"
        if lvl == "form":
            return d["form"][0]
        return f"{d[lvl][0]} ({d['form'][0]})"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default="0", help="camera index, stream URL or video file")
    ap.add_argument("--name", default=time.strftime("live_%Y%m%d_%H%M"))
    ap.add_argument("--gct", default=os.path.join(M, "gct.pt"))
    ap.add_argument("--realtime", action="store_true", help="pace a video file like a live feed")
    ap.add_argument("--anchor_frames", type=int, default=12)
    ap.add_argument("--anchor_sec", type=float, default=0.5,
                    help="seconds between anchor frames (the start needs real motion)")
    ap.add_argument("--keyframe_interval", type=int, default=6,
                    help="cache every Nth processed frame (live runs at a few fps)")
    ap.add_argument("--crop_margins", default="0.05,0.14,0.22,0.22",
                    help="TOP,BOTTOM,LEFT,RIGHT trimmed off (drone OSD); 0,0,0,0 for a clean camera")
    ap.add_argument("--no_clahe", action="store_true")
    ap.add_argument("--detect_every", type=float, default=3.0, help="seconds between detections")
    ap.add_argument("--det_threshold", type=float, default=0.25)
    ap.add_argument("--merge_frac", type=float, default=0.05,
                    help="sightings closer than this share of the scene extent are one plant")
    ap.add_argument("--no_plants", action="store_true")
    ap.add_argument("--artifact_every", type=int, default=5)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--point_size", type=float, default=0.001)
    ap.add_argument("--vis_every", type=int, default=5, help="send the cloud to the viewer every N frames")
    ap.add_argument("--no_export", action="store_true")
    ap.add_argument("--max_seconds", type=float, default=0, help="stop by itself after this long")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    src = int(args.source) if args.source.isdigit() else args.source
    art = os.path.join(ROOT, "maps", args.name)
    mapper = StreamMapper(
        args.gct, num_scale_frames=args.anchor_frames, anchor_stride=1,
        keyframe_interval=args.keyframe_interval, kv_cache_sliding_window=16,
        crop_margins=tuple(float(v) for v in args.crop_margins.split(",")),
        clahe=not args.no_clahe, min_frame_luma=14, min_frame_std=18,
        sky_model=os.path.join(M, "skyseg.onnx"), save_artifacts=art,
        artifact_every=args.artifact_every, source=str(args.source), fps=10.0)
    tracker = None if args.no_plants else PlantTracker(args, device)
    viewer = LiveViewer(args.port, args.point_size)
    finish = {"flag": False}
    btn = viewer.server.gui.add_button("Finish and export")
    btn.on_click(lambda _: finish.update(flag=True))
    status = viewer.server.gui.add_text("Status", "collecting anchor frames", disabled=True)
    plants_txt = viewer.server.gui.add_text("Plants", "-", disabled=True)
    print(f"\n  Live viewer: http://localhost:{args.port}\n")

    feed = LatestFrame(src, realtime=args.realtime or not isinstance(src, int) and
                       os.path.exists(str(src)))
    last_idx, last_anchor, last_det = -1, 0.0, 0.0
    extent = 1.0
    try:
        t_start = time.time()
        while not finish["flag"]:
            if args.max_seconds and time.time() - t_start > args.max_seconds:
                break
            rgb, idx = feed.latest(last_idx)
            if rgb is None:
                if feed.done:
                    break
                time.sleep(0.005)
                continue
            if not mapper.started and time.time() - last_anchor < args.anchor_sec:
                time.sleep(0.01)            # spread the anchor over real motion
                continue
            last_idx = idx
            if not mapper.started:
                last_anchor = time.time()
            r = mapper.push(rgb, idx)
            if r is None:
                status.value = f"anchor {len(mapper.anchor)}/{args.anchor_frames}"
                continue
            if mapper.frame_idx % args.vis_every and not r["anchor"]:
                continue
            p, c = mapper.snapshot()
            if len(p) > 1000:
                lo, hi = np.percentile(p[::50], [5, 95], axis=0)
                extent = float(np.linalg.norm(hi - lo))
            H, W = mapper.hw
            el = time.time() - mapper.t0
            status.value = (f"{mapper.frame_idx} frames, {len(p):,} points, "
                            f"{mapper.frame_idx / max(el, 1e-6):.1f} fps")
            viewer.update(p, c, r["c2w"][:3, :3], r["c2w"][:3, 3],
                          2 * float(np.arctan(r["K"][1, 2] / r["K"][1, 1])), W / H,
                          frames=mapper.frame_idx)
            if tracker and time.time() - last_det > args.detect_every:
                last_det = time.time()
                full = r.get("rgb")
                seen = tracker.update(full, r, (H, W), extent) if full is not None else []
                for q in tracker.plants:
                    viewer.server.scene.add_label(f"/plants/{q['id']}", q["label"],
                                                  position=q["center"].astype(np.float32))
                    viewer.server.scene.add_icosphere(
                        f"/plants/{q['id']}/dot", radius=0.01 * extent, color=(60, 220, 90),
                        position=q["center"].astype(np.float32))
                plants_txt.value = ", ".join(f"{q['label']}" for q in tracker.plants[-8:]) or "-"
                if full is not None:
                    img = full.copy()
                    for q, (x0, y0, x1, y1) in seen:
                        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (60, 220, 90), 2)
                        cv2.putText(img, q["label"][:28], (int(x0) + 2, int(y0) + 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    img = cv2.resize(img, (W, H))
                    viewer.server.scene.add_image("/camera/latest", img, render_width=0.16,
                                                  render_height=0.16 * H / W,
                                                  position=r["c2w"][:3, 3],
                                                  wxyz=_wxyz(r["c2w"][:3, :3]))
    except KeyboardInterrupt:
        pass
    mapper.flush()
    print(f"\nStopped after {mapper.frame_idx} frames; saving {art}")
    mapper.save()
    if not args.no_export and mapper.started:
        status.value = "exporting the offline viewer..."
        sh = os.path.join(ROOT, "make_viewer.sh")
        subprocess.run(["bash", sh, "--from-map", args.name], cwd=ROOT, check=False)
        print(f"Viewer: {os.path.join(ROOT, 'viewers', args.name + '.html')}")


def _wxyz(R):
    import viser.transforms as vtf
    return vtf.SO3.from_matrix(R).wxyz


if __name__ == "__main__":
    main()
