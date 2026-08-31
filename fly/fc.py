import argparse
import re
import time
from link import M, conn, cmd


def parm(path):
    out = []
    for l in open(path):
        w = l.replace(",", " ").split()
        if len(w) >= 2 and not w[0].startswith("#"):
            out.append((w[0], float(w[1])))
    return out


def alias(name, v):
    yield name, v
    if name == "RTL_ALT":
        yield "RTL_ALT_M", v / 100
    elif name == "WPNAV_SPEED":
        yield "WP_SPD", v / 100
    k = re.match(r"SR(\d)_(.+)", name)
    if k:
        yield f"MAV{int(k[1]) + 1}_{k[2]}", v


def resolve(m, name, v):
    for n, x in alias(name, v):
        cur = get(m, n)
        if cur is not None:
            return n, x, cur
    return name, v, None


def get(m, name, t=2.0):
    m.mav.param_request_read_send(m.target_system, m.target_component, name.encode(), -1)
    end = time.time() + t
    while time.time() < end:
        r = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if r is not None and r.param_id == name:
            return r.param_value
    return None


def put(m, name, v, tries=3):
    for _ in range(tries):
        m.mav.param_set_send(m.target_system, m.target_component, name.encode(), v, M.MAV_PARAM_TYPE_REAL32)
        end = time.time() + 1.5
        while time.time() < end:
            r = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
            if r is not None and r.param_id == name:
                if abs(r.param_value - v) < 1e-3 * max(1, abs(v)):
                    return True
                break
    return False


def main():
    a = argparse.ArgumentParser()
    a.add_argument("parm", nargs="?", default="manta.parm")
    a.add_argument("--url", default="COM3")
    a.add_argument("--baud", type=int, default=115200)
    a.add_argument("--check", action="store_true")
    a.add_argument("--reboot", action="store_true")
    g = a.parse_args()
    from pymavlink import mavutil
    m = mavutil.mavlink_connection(g.url, baud=g.baud, source_system=255)
    m.wait_heartbeat(timeout=30)
    ok, diff, miss = 0, [], []
    for k in range(3):
        todo = parm(g.parm)
        for name, v in todo:
            name, v, cur = resolve(m, name, v)
            if cur is None:
                if k == 2 or g.check:
                    miss.append(name)
                continue
            if abs(cur - v) < 1e-3 * max(1, abs(v)):
                ok += 1 if k == 0 else 0
                continue
            if g.check:
                diff.append((name, cur, v))
            elif not put(m, name, v):
                diff.append((name, cur, v))
        if g.check or not diff and not any(resolve(m, n, v)[2] is None for n, v in todo):
            break
        diff = []
    print(f"{len(parm(g.parm))} params: {ok} already set")
    for n, c, v in diff:
        print(f"  {n}: {c:g} -> {v:g}" if g.check else f"  failed {n} ({c:g} -> {v:g})")
    if miss:
        print("  not on this firmware yet (enable the feature, reboot, run again): " + " ".join(sorted(set(miss))))
    if g.reboot and not g.check:
        cmd(m, M.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN, 1)
        print("rebooting")


if __name__ == "__main__":
    main()
