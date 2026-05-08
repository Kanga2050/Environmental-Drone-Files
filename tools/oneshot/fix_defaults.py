import io
p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\detect_plants.py"
s = io.open(p, encoding="utf-8").read()
old = '''    ap.add_argument("--crop_from_3d", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Re-cut each classifier view around where the fused "
                         "instance's own 3D points land in that frame, instead of "
                         "using the per-frame detection box. The box is whatever "
                         "the detector drew, which here is routinely a plant plus "
                         "the fence behind it plus a band of sky")'''
new = '''    ap.add_argument("--crop_from_3d", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Re-cut each classifier view around where the fused "
                         "instance's own 3D points land in that frame, instead of "
                         "using the per-frame detection box. Off by default "
                         "because it was measured and did not help: the instance's "
                         "pooled points carry the wall and ground the plant stands "
                         "on, so the projected box drifts onto them, and it fails "
                         "outright on sparse instances. Split-half consistency "
                         "17/38 against 19/42 without it")'''
assert old in s
io.open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("crop_from_3d now defaults off")
