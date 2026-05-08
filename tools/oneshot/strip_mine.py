
import ast
import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = [r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\stream_video.py",
           r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\detect_plants.py"]

def literals(path):
    try:
        tree = ast.parse(io.open(path, encoding="utf-8").read())
    except SyntaxError:
        return []
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]

def comment_lines(text):
    out = set()
    for ln in text.splitlines():
        t = ln.strip()
        if t.startswith("#") and len(t) > 2:
            out.add(t)
    return out

added = set()
for f in sorted(glob.glob(os.path.join(HERE, "patch_*.py"))):
    lits = literals(f)

    olds, news = set(), set()
    for i, s in enumerate(lits):
        (olds if i % 2 == 0 else news).update(comment_lines(s))
    for s in lits:
        news.update(comment_lines(s))
    added |= news
print(f"{len(added)} distinct comment lines seen across patch scripts")

for path in TARGETS:
    src = io.open(path, encoding="utf-8").read()
    lines = src.splitlines(True)
    keep, dropped = [], 0
    for ln in lines:
        if ln.strip() in added:
            dropped += 1
            continue
        keep.append(ln)
    out = "".join(keep)
    out = re.sub(r"\n{3,}", "\n\n", out)
    try:
        ast.parse(out)
    except SyntaxError as e:
        sys.exit(f"{path}: refusing to write, would not parse: {e}")
    io.open(path, "w", encoding="utf-8", newline="\n").write(out)
    print(f"{os.path.basename(path)}: dropped {dropped} comment lines, "
          f"{len(lines)} -> {len(out.splitlines())}")
