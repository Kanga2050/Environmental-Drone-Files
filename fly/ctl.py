import argparse
import bisect
import json
import math
import queue
import socket
import threading
import time
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
        elif w[0] == "cells":
            print(f"{len(self.bad())} cells to redo, {len(self.cells)} seen")
        else:
            print("alt N | go LAT LON [ALT] | auto | redo | rtl | land | loiter | cells")


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
                    c.tap(json.loads(sk.recv(512)))
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
