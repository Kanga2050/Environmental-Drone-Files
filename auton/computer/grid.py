import argparse
import json
import math
from link import M, conn, off, ne, upload


def lanes(a, b, gap, yaw, ext=0.0):
    lat0, lon0 = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    pts = [ne(lat0, lon0, *a), ne(lat0, lon0, *b), ne(lat0, lon0, a[0], b[1]), ne(lat0, lon0, b[0], a[1])]
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    rot = [(n * c + e * s, -n * s + e * c) for n, e in pts]
    u0, u1 = min(p[0] for p in rot) - ext, max(p[0] for p in rot) + ext
    v0, v1 = min(p[1] for p in rot), max(p[1] for p in rot)
    k = max(1, math.ceil((v1 - v0) / gap))
    out = []
    for i in range(k + 1):
        v = v0 + (v1 - v0) * i / k
        ends = [(u0, v), (u1, v)] if i % 2 == 0 else [(u1, v), (u0, v)]
        for u, w in ends:
            out.append(off(lat0, lon0, u * c - w * s, u * s + w * c))
    return out


def plan(a, b, alt, gap, yaw, speed, hold, ext=0.0):
    wps = lanes(a, b, gap, yaw, ext)
    it = [(M.MAV_CMD_NAV_WAYPOINT, *wps[0], 0, (0, 0, 0, 0)),
          (M.MAV_CMD_NAV_TAKEOFF, 0, 0, alt, (0, 0, 0, 0)),
          (M.MAV_CMD_DO_CHANGE_SPEED, 0, 0, 0, (1, speed, -1, 0))]
    it += [(M.MAV_CMD_NAV_WAYPOINT, lat, lon, alt, (0, 0, 0, 0)) for lat, lon in wps]
    it.append((M.MAV_CMD_NAV_LOITER_TIME, *wps[-1], alt, (hold, 0, 0, 0)))
    it.append((M.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0, 0, 0, (0, 0, 0, 0)))
    return {"a": a, "b": b, "alt": alt, "gap": gap, "yaw": yaw, "speed": speed, "hold": hold, "items": it}


def ll(s):
    return tuple(float(x) for x in s.split(","))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a", type=ll)
    p.add_argument("b", type=ll)
    p.add_argument("--alt", type=float, default=25)
    p.add_argument("--gap", type=float, default=0)
    p.add_argument("--fov", type=float, default=90)
    p.add_argument("--side", type=float, default=0.6)
    p.add_argument("--yaw", type=float, default=0)
    p.add_argument("--speed", type=float, default=5)
    p.add_argument("--hold", type=float, default=20)
    p.add_argument("--tilt", type=float, default=0)
    p.add_argument("--out", default="plan.json")
    p.add_argument("--url", default=None)
    g = p.parse_args()
    gap = g.gap or 2 * g.alt * math.tan(math.radians(g.fov / 2)) * (1 - g.side)
    pl = plan(g.a, g.b, g.alt, gap, g.yaw, g.speed, g.hold, g.alt * math.tan(math.radians(g.tilt)))
    json.dump(pl, open(g.out, "w"), indent=1)
    print(f"{len(pl['items']) - 5} wps, gap {gap:.1f} m -> {g.out}")
    if g.url:
        m = conn(g.url)
        print("uploaded" if upload(m, pl["items"]) else "upload failed")


if __name__ == "__main__":
    main()
