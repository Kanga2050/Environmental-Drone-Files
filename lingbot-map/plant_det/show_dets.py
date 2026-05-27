"""Draw detector boxes on yard frames, side by side with the boxes OWLv2
produced (the ones kept in artifacts_yard_avg/plants.json).

  python -m plant_det.show_dets --weights W.pt --out grid.jpg [--frames 8]
"""
import argparse
import json
import os

import cv2
import numpy as np

TEST = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(TEST, "artifacts_yard_avg")


def owl_boxes(art=ART):
    out = {}
    for p in json.load(open(os.path.join(art, "plants.json"), encoding="utf-8"))["plants"]:
        for b in p.get("boxes", []):
            out.setdefault(b["frame"], []).append((b["box"], p["id"]))
    return out


def draw(im, boxes, color, labels=None):
    for k, b in enumerate(boxes):
        x0, y0, x1, y1 = [int(v) for v in b]
        cv2.rectangle(im, (x0, y0), (x1, y1), color, 3)
        if labels:
            cv2.putText(im, labels[k], (x0 + 4, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color, 2)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--fids", default="")
    a = ap.parse_args()
    from ultralytics import YOLO, RTDETR
    m = (RTDETR if "rtdetr" in a.weights.lower() else YOLO)(a.weights)
    meta = json.load(open(os.path.join(ART, "meta.json")))
    fids = [int(x) for x in a.fids.split(",")] if a.fids else \
        list(np.array(meta["saved_frame_ids"])[np.linspace(0, len(meta["saved_frame_ids"]) - 1,
                                                            a.frames).astype(int)])
    owl = owl_boxes()
    rows = []
    for fid in fids:
        im = cv2.imread(os.path.join(ART, "frames", f"{fid:06d}.jpg"))
        r = m.predict(im, imgsz=a.imgsz, conf=a.conf, agnostic_nms=True, verbose=False)[0]
        lab = [f"{r.names[int(c)]} {s:.2f}" for c, s in zip(r.boxes.cls, r.boxes.conf)]
        left = draw(im.copy(), [b for b, _ in owl.get(fid, [])], (0, 200, 255),
                    [f"owl #{i}" for _, i in owl.get(fid, [])])
        right = draw(im.copy(), r.boxes.xyxy.cpu().numpy(), (60, 255, 60), lab)
        cv2.putText(left, f"frame {fid}  OWLv2 (kept)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)
        cv2.putText(right, f"trained detector", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)
        row = np.hstack([left, right])
        rows.append(cv2.resize(row, (1200, int(1200 * row.shape[0] / row.shape[1]))))
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
