import io
import re
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\viewer\app.js"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''const $ = (id) => document.getElementById(id);''',
    '''const $ = (id) => document.getElementById(id);
const v3 = (a) => new THREE.Vector3().fromArray(a);'''))

subs.append(('''  setMode("frame " + f.id);''', '''  setMode(`frame ${f.id}`);'''))
subs.append(('''    tx.textContent = "#" + p.id;''', '''    tx.textContent = `#${p.id}`;'''))
subs.append(('''  $("frameVal").textContent = "#" + f.id;''',
             '''  $("frameVal").textContent = `#${f.id}`;'''))
subs.append((
    '''        tag >= 0 ? "#" + S.scene.plants[tag].id + " " + S.scene.plants[tag].category''',
    '''        tag >= 0 ? `#${S.scene.plants[tag].id} ${S.scene.plants[tag].category}`'''))
subs.append((
    '''  return `<i>${p.species}</i>${p.species_common ? " — " + p.species_common : ""}`;''',
    '''  return `<i>${p.species}</i>${p.species_common ? ` — ${p.species_common}` : ""}`;'''))
subs.append((
    '''        <span class="bn">${c.name === top ? "<b>" + c.name + "</b>" : c.name}</span>''',
    '''        <span class="bn">${c.name === top ? `<b>${c.name}</b>` : c.name}</span>'''))
subs.append((
    '''        S.calibrated ? "1 u = " + S.scale.toFixed(4) + " m" : "uncalibrated"}</div>''',
    '''        S.calibrated ? `1 u = ${S.scale.toFixed(4)} m` : "uncalibrated"}</div>'''))
subs.append((
    '''    (t) => t.classList.toggle("on", t.id === "tab-" + name));''',
    '''    (t) => t.classList.toggle("on", t.id === `tab-${name}`));'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:160])
    s = s.replace(old, new, 1)

out, i, n, hits = [], 0, len(s), 0
NEEDLE = "new THREE.Vector3().fromArray("
while i < n:
    j = s.find(NEEDLE, i)
    if j < 0:
        out.append(s[i:])
        break
    out.append(s[i:j])
    k = j + len(NEEDLE)
    depth = 1
    while k < n and depth:
        depth += (s[k] == "(") - (s[k] == ")")
        k += 1
    out.append("v3(" + s[j + len(NEEDLE):k - 1] + ")")
    hits += 1
    i = k
s = "".join(out)

s = s.replace("const v3 = (a) => v3(a);",
              "const v3 = (a) => new THREE.Vector3().fromArray(a);", 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print(f"app.js: {hits} fromArray -> v3, {len(subs)} template literals, "
      f"{len(s.splitlines())} lines")
