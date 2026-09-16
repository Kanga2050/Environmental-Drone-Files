import argparse
import json
import socket
import threading
import time
import cv2
import numpy as np
from iq import raw, Score


class Cam:
    def __init__(self, src, fps):
        self.file = not str(src).isdigit()
        self.cap = cv2.VideoCapture(src) if self.file else cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {src}")
        self.fps = max(self.cap.get(cv2.CAP_PROP_FPS), 0) or fps
        self.frame, self.t, self.n, self.done = None, 0.0, 0, False
        self.lock = threading.Lock()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            ok, f = self.cap.read()
            if not ok:
                self.done = True
                return
            with self.lock:
                self.frame, self.t = f, time.time()
                self.n += 1
            if self.file:
                time.sleep(1 / self.fps)

    def get(self):
        with self.lock:
            return self.frame, self.t, self.n


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--src", default="0")
    a.add_argument("--fps", type=float, default=30)
    a.add_argument("--tap", default="127.0.0.1:14600")
    a.add_argument("--rec", default=None)
    a.add_argument("--bench", type=float, default=0)
    a.add_argument("--every", type=int, default=1)
    g = a.parse_args()
    cam = Cam(g.src, g.fps)
    host, port = g.tap.rsplit(":", 1)
    sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sc = Score()
    rec = None
    last, done, ms, qs = 0, 0, [], []
    t0 = time.time()
    while not cam.done:
        f, t, n = cam.get()
        if f is None or n == last:
            time.sleep(0.002)
            continue
        if rec is None and g.rec:
            rec = cv2.VideoWriter(g.rec, cv2.VideoWriter_fourcc(*"mp4v"), cam.fps, (f.shape[1], f.shape[0]))
        if rec is not None:
            rec.write(f)
        skipped = n - last - 1
        last = n
        if n % g.every:
            continue
        s = time.perf_counter()
        m = raw(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        q = sc(m)
        ms.append((time.perf_counter() - s) * 1000)
        done += 1
        if q is not None:
            qs.append(q)
        sk.sendto(json.dumps({"t": t, "f": n, "iq": m}).encode(), (host, int(port)))
        if g.bench and time.time() - t0 > g.bench:
            break
        if done % 150 == 0:
            el = time.time() - t0
            print(f"{n} frames {n / el:.1f} fps in, {done / el:.1f} scored, {np.mean(ms[-150:]):.1f} ms, lag {(time.time() - t) * 1000:.0f} ms, skipped {skipped}, q {np.median(qs[-150:]) if qs else 0:.2f}", flush=True)
    if rec is not None:
        rec.release()
    el = time.time() - t0
    f, t, n = cam.get()
    print(json.dumps({"seconds": round(el, 1), "camera_fps": round(n / el, 1), "scored_fps": round(done / el, 1),
                      "ms_mean": round(float(np.mean(ms)), 1) if ms else None, "ms_p99": round(float(np.percentile(ms, 99)), 1) if ms else None,
                      "frames_missed": int(n - done * g.every), "frame": list(f.shape) if f is not None else None,
                      "q_median": round(float(np.median(qs)), 2) if qs else None, "keeps_up": bool(ms and np.percentile(ms, 99) < 1000 / max(n / el, 1))}))


if __name__ == "__main__":
    main()
