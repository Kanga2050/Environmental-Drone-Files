import math
import time
from pymavlink import mavutil

M = mavutil.mavlink
R = 6378137.0
REL = M.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT


def conn(url, wait=60):
    m = mavutil.mavlink_connection(url, source_system=255, source_component=M.MAV_COMP_ID_MISSIONPLANNER)
    if not m.wait_heartbeat(timeout=wait):
        raise SystemExit(f"no heartbeat on {url}")
    for i, hz in ((33, 4), (42, 1), (1, 1)):
        cmd(m, M.MAV_CMD_SET_MESSAGE_INTERVAL, i, 1e6 / hz)
    return m


def beat(m):
    m.mav.heartbeat_send(M.MAV_TYPE_GCS, M.MAV_AUTOPILOT_INVALID, 0, 0, 0)


def off(lat, lon, n, e):
    return (lat + math.degrees(n / R),
            lon + math.degrees(e / (R * math.cos(math.radians(lat)))))


def ne(lat0, lon0, lat, lon):
    return (math.radians(lat - lat0) * R,
            math.radians(lon - lon0) * R * math.cos(math.radians(lat0)))


def send(m, it, seq):
    cmd, lat, lon, alt, p = it
    m.mav.mission_item_int_send(m.target_system, m.target_component, seq, REL, cmd,
                                0, 1, *p, int(lat * 1e7), int(lon * 1e7), alt, 0)


def upload(m, items, tries=5):
    for _ in range(tries):
        m.mav.mission_count_send(m.target_system, m.target_component, len(items), 0)
        t = time.time()
        while time.time() - t < 5:
            r = m.recv_match(type=["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"],
                             blocking=True, timeout=1)
            if r is None:
                continue
            if r.get_type() == "MISSION_ACK":
                if r.type == M.MAV_MISSION_ACCEPTED:
                    return True
                time.sleep(2)
                break
            if r.seq < len(items):
                send(m, items[r.seq], r.seq)
                t = time.time()
    return False


def cmd(m, c, *p):
    p = list(p) + [0] * (7 - len(p))
    m.mav.command_long_send(m.target_system, m.target_component, c, 0, *p)


def mode(m, name):
    m.set_mode(m.mode_mapping()[name])


def goto(m, lat, lon, alt):
    m.mav.set_position_target_global_int_send(
        0, m.target_system, m.target_component, REL, 0b110111111000,
        int(lat * 1e7), int(lon * 1e7), alt, 0, 0, 0, 0, 0, 0, 0, 0)
