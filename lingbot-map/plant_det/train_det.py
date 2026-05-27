"""Fine-tune an Ultralytics detector (COCO-pretrained) on Open Images plant/tree boxes."""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default=r"C:\Users\shaur\Desktop\Test\datasets\det_runs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    from ultralytics import YOLO, RTDETR
    if a.resume:
        YOLO(f"{a.project}/{a.name}/weights/last.pt").train(resume=True)
        return
    m = (RTDETR if "rtdetr" in a.model else YOLO)(a.model)
    m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers,
            project=a.project, name=a.name, exist_ok=True, cos_lr=True, close_mosaic=5,
            fraction=a.fraction, patience=100, plots=True, amp=True, cache=False,
            # video frames: blur, colour casts, perspective; no vertical flips (gravity)
            hsv_h=0.03, hsv_s=0.6, hsv_v=0.4, degrees=5.0, perspective=0.0005,
            scale=0.6, mixup=0.05)


if __name__ == "__main__":
    main()
