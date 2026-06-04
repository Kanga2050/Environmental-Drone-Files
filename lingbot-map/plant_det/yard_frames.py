"""Extract yard video frames prepared exactly like stream_video.py does
(margins crop + CLAHE), for detector self-training."""
import argparse
import os

import cv2

from stream_video import make_frame_prep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"C:\Users\shaur\Desktop\Test\video\VID_20260913_192111.mp4")
    ap.add_argument("--out", default=r"C:\Users\shaur\Desktop\Test\datasets\yard_frames\images")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--margins", default="0.05,0.14,0.22,0.22")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prep = make_frame_prep(clahe=True, margins=tuple(float(v) for v in a.margins.split(",")))
    cap = cv2.VideoCapture(a.video)
    i = n = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % a.stride == 0:
            ok, bgr = cap.retrieve()
            if ok:
                rgb = prep(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                if g.mean() >= 14 and g.std() >= 18:          # stream_video's usable test
                    cv2.imwrite(os.path.join(a.out, f"v{i:05d}.jpg"),
                                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                    n += 1
        i += 1
    print("frames", n, "of", i)


if __name__ == "__main__":
    main()
