
import io
import sys

base = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\viewer"

def patch(fname, subs):
    path = base + "\\" + fname
    s = io.open(path, encoding="utf-8").read()
    for old, new in subs:
        if old not in s:
            sys.exit(f"[{fname}] NOT FOUND:\n" + old[:240])
        s = s.replace(old, new, 1)
    io.open(path, "w", encoding="utf-8").write(s)
    print("patched", fname)

patch("index.html", [
    ('''    <div id="plantList"></div>''',
     '''    <div class="ctl row tight" style="margin-bottom:8px">
      <button class="act" id="exportCsv">Export table (CSV)</button>
      <span id="plantCount" class="t2" style="color:var(--dim);font-size:11px"></span>
    </div>
    <div id="plantList"></div>'''),
])

patch("app.js", [

    ('''  // Nearest first, and cap it: forty labels stacked on one shrub row is noise.
  cands.sort((a, b) => a.w - b.w);
  for (const c of cands.slice(0, 28)) {''',
     '''  // Nearest first, and then drop any label that would sit on one already
  // placed. A cap alone does not help when the plants are a hedge row: the
  // survivors still pile into the same hundred pixels and none is readable.
  cands.sort((a, b) => a.w - b.w);
  const placed = [];
  const fits = (c) => {
    const w = 130, h = 17;
    for (const q of placed) {
      if (Math.abs(q.px - c.px) < w && Math.abs(q.py - c.py) < h) return false;
    }
    placed.push(c);
    return true;
  };
  for (const c of cands.filter(fits).slice(0, 34)) {'''),

    ('''  $("scaleReset").onclick = resetScale;''',
     '''  $("scaleReset").onclick = resetScale;

  $("exportCsv").onclick = () => {
    const u = S.calibrated ? "m" : "u";
    const k = S.calibrated ? S.scale : 1;
    const head = ["id", "category", "category_conf", "detection", "species",
                  "common", "species_raw", "species_vote", "identified", "views",
                  "cloud_points", `height_${u}`, `crown_w_${u}`, `crown_d_${u}`,
                  `volume_${u}3`, "mean_r", "mean_g", "mean_b",
                  "x", "y", "z"];
    const rows = S.scene.plants
      .filter((p) => p.detection_score >= S.conf)
      .map((p) => {
        const m = p.metrics || {};
        const c = m.mean_rgb || [0, 0, 0];
        return [p.id, p.category, p.category_confidence, p.detection_score,
                p.species || "", p.species_common || "", p.species_raw_score,
                p.species_confidence, p.species_identified ? 1 : 0, p.views,
                m.n_cloud_points || 0,
                ((m.height_m || 0) * k).toFixed(4),
                ((m.crown_w_m || 0) * k).toFixed(4),
                ((m.crown_d_m || 0) * k).toFixed(4),
                ((m.volume_m3 || 0) * k ** 3).toFixed(6),
                ...c.map((v) => Math.round(v)),
                ...p.centroid.map((v) => v.toFixed(4))];
      });
    const esc = (v) => (/[",\\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v);
    download([head, ...rows].map((r) => r.map(esc).join(",")).join("\\n"),
             `plants_${S.scene.artifacts}.csv`, "text/csv");
  };'''),

    ('''      <button class="act" id="selClear">Clear</button>
    </div>''',
     '''      <button class="act" id="selClear">Clear</button>
      <button class="act" id="selPly">Save .ply</button>
    </div>'''),

    ('''  $("selClear").onclick = clearSelection;''',
     '''  $("selClear").onclick = clearSelection;
  $("selPly").onclick = () => exportPly(S.selection.idx);'''),

    ('''function toggleIsolate() {''',
     '''/** Hand a string or buffer to the browser as a file. */
function download(data, name, mime) {
  const url = URL.createObjectURL(new Blob([data], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

/** Write a point selection as binary PLY, so it opens in anything.
 *
 * Positions go out in scene units unless a scale has been set, in which case
 * they are converted to metres -- a file whose units depend on a slider would
 * be worse than useless in CloudCompare six months from now, so the chosen
 * convention is recorded in a comment in the header.
 */
function exportPly(idx) {
  if (!idx || !idx.length) return;
  const k = S.calibrated ? S.scale : 1;
  const header =
    "ply\\nformat binary_little_endian 1.0\\n" +
    `comment exported from ${S.scene.artifacts}\\n` +
    `comment units ${S.calibrated ? "metres (calibrated)" : "arbitrary (uncalibrated)"}\\n` +
    `element vertex ${idx.length}\\n` +
    "property float x\\nproperty float y\\nproperty float z\\n" +
    "property uchar red\\nproperty uchar green\\nproperty uchar blue\\n" +
    "end_header\\n";
  const head = new TextEncoder().encode(header);
  const body = new ArrayBuffer(idx.length * 15);
  const dv = new DataView(body);
  for (let j = 0; j < idx.length; j++) {
    const i = idx[j], o = j * 15;
    dv.setFloat32(o, S.positions[i * 3] * k, true);
    dv.setFloat32(o + 4, S.positions[i * 3 + 1] * k, true);
    dv.setFloat32(o + 8, S.positions[i * 3 + 2] * k, true);
    dv.setUint8(o + 12, S.colors[i * 3]);
    dv.setUint8(o + 13, S.colors[i * 3 + 1]);
    dv.setUint8(o + 14, S.colors[i * 3 + 2]);
  }
  const out = new Uint8Array(head.length + body.byteLength);
  out.set(head, 0);
  out.set(new Uint8Array(body), head.length);
  download(out, `selection_${idx.length}pts.ply`, "application/octet-stream");
  hudInfo(`saved ${idx.length.toLocaleString()} points as PLY`);
}

function toggleIsolate() {'''),

    ('''  dom.plantList.innerHTML = rows.map(({ p, i }) => {''',
     '''  $("plantCount").textContent =
    `${rows.length} of ${S.scene.plants.length} shown`;
  dom.plantList.innerHTML = rows.map(({ p, i }) => {'''),
])
print("done")
