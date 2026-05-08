import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\view_map3d.py"
s = io.open(p, encoding="utf-8").read()

lines = s.splitlines(True)
start = next(i for i, l in enumerate(lines) if l.startswith("        pairs = ("))
end = next(i for i, l in enumerate(lines)
           if i > start and "large enough to crop" in l)
new = '''        picks = [(v, c) for v in self.best_views(idx, topk=max_views * 2)
                 if (c := self.crop_for_view(v)) is not None][:max_views]
        if not picks:
            return {"error": "no view of this selection is large enough to crop"}
        used, crops = zip(*picks)
'''
lines[start:end + 1] = [new]
s = "".join(lines)

s = s.replace("res = self._classify(crops, topk=topk)",
              "res = self._classify(list(crops), topk=topk)", 1)
s = s.replace('''        self.frame_by_id = {f["id"]: f for f in self.frames}
        self.plants_by_id = {p["id"]: p for p in self.plants}''',
              '''        self.frame_by_id = {f["id"]: f for f in self.frames}''', 1)

import ast
ast.parse(s)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("identify fixed, dead dict removed,", len(s.splitlines()), "lines")
