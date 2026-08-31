import argparse
import os
import shutil
import subprocess
import urllib.request

URL = "https://firmware.ardupilot.org/Copter/{rel}/speedybeef4v5/arducopter_with_bl.hex"


def hex2bin(src, dst):
    mem, base = {}, 0
    for l in open(src):
        l = l.strip()
        if not l.startswith(":"):
            continue
        b = bytes.fromhex(l[1:])
        if sum(b) & 0xFF:
            raise ValueError("bad checksum")
        n, addr, t, d = b[0], (b[1] << 8) | b[2], b[3], b[4:4 + b[0]]
        if t == 0:
            for i, x in enumerate(d):
                mem[base + addr + i] = x
        elif t == 2:
            base = int.from_bytes(d, "big") << 4
        elif t == 4:
            base = int.from_bytes(d, "big") << 16
        elif t == 1:
            break
    lo, hi = min(mem), max(mem)
    out = bytearray(b"\xff" * (hi - lo + 1))
    for a, x in mem.items():
        out[a - lo] = x
    open(dst, "wb").write(out)
    return lo, len(out)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--rel", default="stable")
    a.add_argument("--hex", default=None)
    a.add_argument("--dfu", action="store_true")
    g = a.parse_args()
    h = g.hex or "arducopter_with_bl.hex"
    if not g.hex:
        print("downloading", URL.format(rel=g.rel))
        urllib.request.urlretrieve(URL.format(rel=g.rel), h)
    lo, n = hex2bin(h, "arducopter_with_bl.bin")
    print(f"{h}: {n / 1024:.0f} KiB at 0x{lo:08x}")
    if not g.dfu:
        print("hold BOOT, plug USB, then flash the .hex with Betaflight Configurator (Load Firmware [Local], Full chip erase) or rerun with --dfu")
        return
    dfu = shutil.which("dfu-util")
    if not dfu:
        raise SystemExit("dfu-util not found")
    subprocess.run([dfu, "-a", "0", "-d", "0483:df11", "-s", f"0x{lo:08x}:mass-erase:force:leave", "-D", "arducopter_with_bl.bin"], check=True)
    print("flashed; it reboots into ArduPilot. next: python fc.py manta.parm --url COMx --reboot")


if __name__ == "__main__":
    main()
