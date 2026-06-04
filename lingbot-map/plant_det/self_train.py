"""Self-training on the yard video: the FPV-trained detector labels the yard
frames (test-time augmentation, confident boxes only), then a short fine-tune
on those frames mixed with FPV-style Open Images keeps it from drifting.

  python -m plant_det.self_train --weights ../datasets/det_runs/y11m_fpv/weights/best.pt
"""
import argparse
import os
import random

DS = r"C:\Users\shaur\Desktop\Test\datasets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--frames", default=os.path.join(DS, "yard_frames", "images"))
    ap.add_argument("--out", default=os.path.join(DS, "yard_st"))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--max_frac", type=float, default=0.6)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--oi", type=int, default=15000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--name", default="y11m_fpv_st")
    a = ap.parse_args()
    from ultralytics import YOLO
    m = YOLO(a.weights)
    lab_dir = os.path.join(a.out, "labels")
    os.makedirs(lab_dir, exist_ok=True)
    frames = sorted(f for f in os.listdir(a.frames) if f.endswith(".jpg"))
    n_box = 0
    for i in range(0, len(frames), 16):
        batch = [os.path.join(a.frames, f) for f in frames[i:i + 16]]
        for f, r in zip(frames[i:i + 16], m.predict(batch, imgsz=640, conf=a.conf, augment=True,
                                                    agnostic_nms=True, verbose=False)):
            lines = []
            for (cx, cy, w, h), c in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist()):
                if w * h <= a.max_frac:
                    lines.append(f"{int(c)} {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}")
            n_box += len(lines)
            with open(os.path.join(lab_dir, f[:-4] + ".txt"), "w") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"pseudo-labelled {len(frames)} frames, {n_box} boxes", flush=True)

    # Ultralytics finds labels by swapping /images/ for /labels/ in the path.
    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    for f in frames:
        dst = os.path.join(img_dir, f)
        if not os.path.exists(dst):
            os.link(os.path.join(a.frames, f), dst)
    rng = random.Random(0)
    oi_img = os.path.join(DS, "oi_plants_fpv", "images", "train")
    oi = rng.sample(sorted(os.listdir(oi_img)), a.oi)
    lines = [os.path.join(img_dir, f) for f in frames] * a.repeat + \
        [os.path.join(oi_img, f) for f in oi]
    rng.shuffle(lines)
    with open(os.path.join(a.out, "train.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    yaml = os.path.join(a.out, "data.yaml")
    with open(yaml, "w") as fh:
        fh.write(f"path: {a.out}\ntrain: {os.path.join(a.out, 'train.txt')}\n"
                 f"val: {os.path.join(DS, 'oi_plants_fpv', 'images', 'val')}\n"
                 f"names:\n  0: plant\n  1: tree\n")
    YOLO(a.weights).train(data=yaml, epochs=a.epochs, imgsz=640, batch=16, workers=6,
                          project=os.path.join(DS, "det_runs"), name=a.name, exist_ok=True,
                          lr0=0.002, lrf=0.2, warmup_epochs=0.5, cos_lr=True, close_mosaic=1,
                          hsv_h=0.03, hsv_s=0.6, hsv_v=0.4, degrees=5.0, scale=0.6, mixup=0.05,
                          plots=True, amp=True)


if __name__ == "__main__":
    main()
