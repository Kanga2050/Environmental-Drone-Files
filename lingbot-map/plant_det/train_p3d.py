import os
import random

from ultralytics import YOLO



def main():
    DS = "C:/Users/shaur/Desktop/Test/datasets"
    y = DS + "/yard3d/images"
    oi = DS + "/oi_plants_fpv/images/train"
    rng = random.Random(0)
    lines = [f"{y}/{f}" for f in sorted(os.listdir(y))] * 6 + \
            [f"{oi}/{f}" for f in rng.sample(sorted(os.listdir(oi)), 12000)]
    rng.shuffle(lines)
    with open(DS + "/yard3d/train.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(DS + "/yard3d/data.yaml", "w") as fh:
        fh.write(f"path: {DS}/yard3d\n"
                 f"train: {DS}/yard3d/train.txt\n"
                 f"val: {DS}/oi_plants_fpv/images/val\n"
                 "names:\n  0: plant\n  1: tree\n")
    YOLO(DS + "/det_runs/y11m_fpv/weights/best.pt").train(
        data=DS + "/yard3d/data.yaml", epochs=5, imgsz=640, batch=16, workers=6,
        project=DS + "/det_runs", name="y11m_p3d", exist_ok=True, lr0=0.002, lrf=0.2,
        warmup_epochs=0.5, cos_lr=True, close_mosaic=1, hsv_h=0.03, hsv_s=0.6, hsv_v=0.4,
        degrees=5.0, scale=0.6, mixup=0.05, amp=True, plots=False)


if __name__ == "__main__":
    main()
