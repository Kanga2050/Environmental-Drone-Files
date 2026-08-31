import argparse
import bisect
import json
import math
import queue
import socket
import threading
import time
import numpy as np
import geo
import plan
from link import M, conn, beat, ne, off, upload, cmd, mode, goto


class Ctl:
    def __init__(self, m, pl, g):
        self.m, self.pl, self.g = m, pl, g
        self.it = pl["items"]
        self.loit = len(self.it) - 2
        self.alt = pl["alt"]
        (la, lo), (lb, lob) = pl["a"], pl["b"]
        self.o = ((la + lb) / 2, (lo + lob) / 2)
        n0, e0 = ne(*self.o, la, lo)
        n1, e1 = ne(*self.o, lb, lob)
        self.box = (min(n0, n1), max(n0, n1), min(e0, e1), max(e0, e1))
        self.ts, self.pos = [], []
        self.cells = {}
        self.win = []
        self.cur = 0
        self.bat = 100
        self.state = "watch"
        self.done = False
        self.adj = time.time()
        self.todo, self.tgt, self.arr, self.sent, self.t0 = [], None, None, 0, 0
        self.pairs, self.rc, self.pending, self.busy = [], 0, None, False
        self.sc = None
        if g.model:
            from scan import Scanner
            self.sc = Scanner(g.model)

    def mav(self, r):
        k = r.get_type()
        if k == "GLOBAL_POSITION_INT":
            self.ts.append(time.time())
            self.pos.append((r.lat / 1e7, r.lon / 1e7, r.relative_alt / 1e3))
            if len(self.ts) > 4000:
                del self.ts[:1000], self.pos[:1000]
        elif k == "MISSION_CURRENT":
            self.cur = r.seq
        elif k == "SYS_STATUS" and r.battery_remaining >= 0:
            self.bat = r.battery_remaining
        elif k == "RC_CHANNELS":
            v = r.chan8_raw
            if self.g.approve == "rc" and v > 1700 >= self.rc and self.pending:
                print("ch8 approve")
                self.fly()
            self.rc = v
        elif k == "STATUSTEXT":
            print("fc:", r.text)

    def at(self, t):
        i = bisect.bisect_left(self.ts, t)
        best = min((j for j in (i - 1, i) if 0 <= j < len(self.ts)),
                   key=lambda j: abs(self.ts[j] - t), default=None)
        if best is None or abs(self.ts[best] - t) > 1:
            return None
        return self.pos[best]

    def key(self, lat, lon):
        n, e = ne(*self.o, lat, lon)
        c = self.g.cell
        return (math.floor(n / c), math.floor(e / c))

    def tap(self, d):
        if self.m.flightmode not in ("AUTO", "GUIDED"):
            return
        p = self.at(d["t"] - self.g.lag)
        if p is None or p[2] < 3:
            return
        if "p" in d:
            n, e = ne(*self.o, p[0], p[1])
            self.pairs.append((d["p"], (e, n, p[2])))
            if len(self.pairs) > 20000:
                del self.pairs[:5000]
        s = self.cells.setdefault(self.key(p[0], p[1]), [0.0, 0])
        s[0] += d["q"]
        s[1] += 1
        self.win.append((d["t"], d["q"]))

    def bad(self):
        c = self.g.cell
        n0, n1, e0, e1 = self.box
        out = []
        for i in range(math.floor(n0 / c), math.ceil(n1 / c)):
            for j in range(math.floor(e0 / c), math.ceil(e1 / c)):
                s, k = self.cells.get((i, j), (0.0, 0))
                if k < self.g.minn or s / k < self.g.lo:
                    out.append(off(*self.o, (i + 0.5) * c, (j + 0.5) * c))
        return out

    def retarget(self, alt):
        for i, it in enumerate(self.it):
            if 3 <= i <= self.loit:
                it[3] = alt
        cur = self.cur
        if upload(self.m, self.it):
            cmd(self.m, M.MAV_CMD_DO_SET_MISSION_CURRENT, cur)
            self.alt = alt
            print(f"alt {alt:.0f} m")
        else:
            print("alt change rejected")

    def redo(self):
        p = self.pos[-1] if self.pos else None
        left = self.bad()
        if p is None or not left or self.bat < self.g.bat:
            print(f"no redo ({len(left)} cells, bat {self.bat}%)")
            return False
        path = []
        while left and len(path) < self.g.maxre:
            q = min(left, key=lambda x: (x[0] - p[0]) ** 2 + (x[1] - p[1]) ** 2)
            left.remove(q)
            path.append(q)
            p = q
        print(f"redo {len(path)} cells")
        self.todo, self.tgt, self.state, self.t0 = path, None, "redo", time.time()
        mode(self.m, "GUIDED")
        return True

    def scan(self):
        if self.busy or not self.sc:
            return False
        self.busy = True
        threading.Thread(target=self.work, daemon=True).start()
        return True

    def work(self):
        try:
            g = self.g
            if len(self.pairs) < 30:
                raise RuntimeError(f"only {len(self.pairs)} pose pairs")
            a, b = np.array([x[0] for x in self.pairs]), np.array([x[1] for x in self.pairs])
            if np.ptp(b[:, :2], 0).max() < 5:
                raise RuntimeError("flight too small to align")
            T, err = geo.robust(a, b)
            d = np.load(g.snap)
            p = geo.apply(T, d["points"].astype(np.float64))
            c = d["colors"].astype(np.float32) / 255.0
            n0, n1, e0, e1 = self.box
            pad = g.pad
            m = (p[:, 0] > e0 - pad) & (p[:, 0] < e1 + pad) & (p[:, 1] > n0 - pad) & (p[:, 1] < n1 + pad)
            print(f"scan: {m.sum():,} pts, scale {T[0]:.2f}, fit {err:.2f} m")
            t = time.time()
            from scan import cells, save
            ps, pq, nj = self.sc(p[m], c[m])
            cl = cells(p[m], ps, pq, g.scell, (e0, n0), (e1, n1))
            save("scan_cells.npz", cl)
            q = self.pos[-1]
            n, e = ne(*self.o, q[0], q[1])
            cfg = plan.Cfg(amin=g.samin, amax=g.amax, budget=g.budget, speed=g.sspeed, thr=g.thr, clear=g.clear)
            ins, it, pri, order, transit, dur = plan.build(cl, self.o, (e, n), cfg)
            print(f"scan took {time.time() - t:.1f} s ({nj} patches)")
            plan.show(ins, transit, dur)
            json.dump({"origin": self.o, "instructions": ins, "items": it}, open("survey.json", "w"), indent=1)
            if not ins:
                print("nothing worth resurveying")
                self.redo()
                return
            self.pending = it
            if g.approve == "auto":
                self.fly()
            else:
                print("flip CH8 or type 'fly' to send" if g.approve == "rc" else "type 'fly' to send")
        except Exception as ex:
            print(f"scan failed: {ex}")
            if self.state == "watch" and self.cur == self.loit:
                self.redo()
        finally:
            self.busy = False

    def fly(self):
        if not self.pending:
            print("no survey pending")
            return
        it, self.pending = self.pending, None
        if not upload(self.m, it):
            print("survey upload failed")
            return
        self.it, self.loit, self.state = [list(x) for x in it], -1, "watch"
        cmd(self.m, M.MAV_CMD_DO_SET_MISSION_CURRENT, 1)
        mode(self.m, "AUTO")
        print(f"survey sent: {len(it)} items")

    def step(self):
        now = time.time()
        fm = self.m.flightmode
        self.win = [w for w in self.win if now - w[0] < self.g.every]
        if self.state == "watch" and fm == "AUTO":
            if 3 <= self.cur < self.loit and now - self.adj > self.g.every and len(self.win) >= 10:
                q = sum(w[1] for w in self.win) / len(self.win)
                na = self.alt
                if q < self.g.lo:
                    na = max(self.g.amin, self.alt - self.g.step)
                elif q > self.g.hi:
                    na = min(self.g.amax, self.alt + self.g.step)
                self.adj = now
                if na != self.alt:
                    print(f"q {q:.2f}")
                    self.retarget(na)
            if self.cur == self.loit and not self.done:
                self.done = True
                if not self.scan():
                    self.redo()
        elif self.state == "redo":
            if fm != "GUIDED" and now - self.t0 > 3:
                print(f"pilot took over ({fm})")
                self.state = "watch"
                return
            if self.tgt is None:
                if not self.todo or self.bat < self.g.bat:
                    print("redo finished")
                    self.state = "watch"
                    mode(self.m, "RTL")
                    return
                self.tgt, self.arr, self.t0, self.sent = self.todo.pop(0), None, now, 0
                print(f"cell {self.tgt[0]:.6f},{self.tgt[1]:.6f} ({len(self.todo)} left)")
            if now - self.sent > 2:
                goto(self.m, *self.tgt, self.g.ralt)
                self.sent = now
            p = self.pos[-1]
            n, e = ne(p[0], p[1], *self.tgt)
            if math.hypot(n, e) < 2.5 and abs(p[2] - self.g.ralt) < 2:
                self.arr = self.arr or now
            if (self.arr and now - self.arr > self.g.dwell) or now - self.t0 > 60:
                self.tgt = None

    def say(self, s):
        w = s.split()
        if not w:
            return
        if w[0] == "alt" and len(w) == 2:
            self.retarget(float(w[1]))
        elif w[0] == "go" and len(w) >= 3:
            self.state = "hold"
            mode(self.m, "GUIDED")
            time.sleep(0.5)
            goto(self.m, float(w[1]), float(w[2]), float(w[3]) if len(w) > 3 else self.alt)
        elif w[0] == "auto":
            self.state = "watch"
            mode(self.m, "AUTO")
        elif w[0] == "redo":
            self.redo()
        elif w[0] in ("rtl", "land", "loiter"):
            self.state = "watch"
            mode(self.m, w[0].upper())
        elif w[0] == "scan":
            print("scanning" if self.scan() else "scan unavailable")
        elif w[0] == "fly":
            self.fly()
        elif w[0] == "cells":
            print(f"{len(self.bad())} cells to redo, {len(self.cells)} seen")
        else:
            print("alt N | go LAT LON [ALT] | auto | redo | scan | fly | rtl | land | loiter | cells")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="udpin:0.0.0.0:14550")
    p.add_argument("--plan", default="plan.json")
    p.add_argument("--tap", type=int, default=14600)
    p.add_argument("--lag", type=float, default=0.2)
    p.add_argument("--cell", type=float, default=8)
    p.add_argument("--minn", type=int, default=3)
    p.add_argument("--lo", type=float, default=0.25)
    p.add_argument("--hi", type=float, default=0.6)
    p.add_argument("--every", type=float, default=15)
    p.add_argument("--step", type=float, default=5)
    p.add_argument("--amin", type=float, default=12)
    p.add_argument("--amax", type=float, default=40)
    p.add_argument("--ralt", type=float, default=15)
    p.add_argument("--dwell", type=float, default=4)
    p.add_argument("--maxre", type=int, default=12)
    p.add_argument("--bat", type=int, default=35)
    p.add_argument("--model", default=None)
    p.add_argument("--snap", default="live.npz")
    p.add_argument("--approve", choices=["rc", "key", "auto"], default="rc")
    p.add_argument("--scell", type=float, default=2.0)
    p.add_argument("--pad", type=float, default=4.0)
    p.add_argument("--samin", type=float, default=6.0)
    p.add_argument("--sspeed", type=float, default=4.0)
    p.add_argument("--thr", type=float, default=0.45)
    p.add_argument("--clear", type=float, default=4.0)
    p.add_argument("--budget", type=float, default=240.0)
    p.add_argument("--up", action="store_true")
    g = p.parse_args()
    pl = json.load(open(g.plan))
    m = conn(g.url)
    print(f"link sys {m.target_system}")
    if g.up:
        print("uploaded" if upload(m, pl["items"]) else "upload failed")
    c = Ctl(m, pl, g)
    sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sk.bind(("0.0.0.0", g.tap))
    sk.setblocking(False)
    q = queue.Queue()
    threading.Thread(target=lambda: [q.put(l) for l in iter(input, None)], daemon=True).start()
    hb = 0
    try:
        while True:
            if time.time() - hb > 1:
                beat(m)
                hb = time.time()
            while (r := m.recv_match(blocking=False)) is not None:
                c.mav(r)
            while True:
                try:
                    c.tap(json.loads(sk.recv(2048)))
                except (BlockingIOError, ValueError):
                    break
            while not q.empty():
                c.say(q.get())
            c.step()
            time.sleep(0.02)
    except (KeyboardInterrupt, EOFError):
        pass
    json.dump({f"{i},{j}": v for (i, j), v in c.cells.items()}, open("cells.json", "w"))


if __name__ == "__main__":
    main()
