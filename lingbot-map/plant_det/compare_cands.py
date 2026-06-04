"""One offline HTML page comparing every candidate pipeline on the yard.

  python -m plant_det.compare_cands --out ../plant_candidates.html

Reads Test/cands/<name>/ (plants.json, plant_seg.npz, plant_seg_eval.json,
cand.json) plus the current artifacts_yard_avg as the OWLv2 + BioCLIP
reference, classifier scores from datasets/cls_runs/*/eval.json and
detector scores from datasets/det_runs/*/results.csv.
"""
import argparse
import base64
import csv
import glob
import html
import json
import os

import cv2
import numpy as np

TEST = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DS = os.path.join(TEST, "datasets")


def b64jpg(im, side=None, q=82):
    if side and max(im.shape[:2]) > side:
        s = side / max(im.shape[:2])
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return "data:image/jpeg;base64," + base64.b64encode(
        cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()


def colour(i):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1, .85, 1)
    return int(b * 255), int(g * 255), int(r * 255)


def green_stats(art):
    """Label-free point checks: how many cloud points are assigned, and how
    green (excess-green) the assigned points are compared with the cloud."""
    p = os.path.join(art, "plant_seg.npz")
    if not os.path.exists(p):
        return {}
    seg = np.load(p)
    tags = seg["tags"]
    rgb = np.load(os.path.join(art, "cloud.npz"))["colors"].astype(np.float32)
    if rgb.max() <= 1.0:
        rgb *= 255
    exg = 2 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]
    green = exg > 8
    on = tags >= 0
    return {"assigned": int(on.sum()),
            "green_purity": float(green[on].mean()) if on.any() else 0.0,
            "green_coverage": float(on[green].mean())}


def load_cand(name, art, meta):
    pj = os.path.join(art, "plants.json")
    if not os.path.exists(pj):
        return None
    plants = json.load(open(pj, encoding="utf-8"))["plants"]
    ev = os.path.join(art, "plant_seg_eval.json")
    ev = json.load(open(ev))["summary"] if os.path.exists(ev) else {}
    levels = {}
    for p in plants:
        lv = (p.get("hierarchy") or {}).get("level") or (
            "species" if p.get("species_identified") else "plant")
        levels[lv] = levels.get(lv, 0) + 1
    return {"name": name, "art": art, "meta": meta, "plants": plants, "eval": ev,
            "levels": levels, "points": green_stats(art)}


def frame_panel(c, fid, width=520):
    im = cv2.imread(os.path.join(c["art"], "frames", f"{fid:06d}.jpg"))
    if im is None:
        return None
    for p in c["plants"]:
        for b in p.get("boxes", []):
            if b["frame"] != fid:
                continue
            x0, y0, x1, y1 = [int(v) for v in b["box"]]
            col = colour(p["id"])
            cv2.rectangle(im, (x0, y0), (x1, y1), col, 3)
            name = p.get("species") or p.get("category") or ""
            cv2.putText(im, f"#{p['id']} {name}"[:28], (x0 + 4, y0 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
    return b64jpg(im, width)


def plant_card(c, p):
    h = p.get("hierarchy")
    crop = os.path.join(c["art"], "crops", p.get("crop") or "")
    img = cv2.imread(crop) if p.get("crop") else None
    thumb = f'<img src="{b64jpg(img, 200)}" alt="">' if img is not None else ""
    rows = ""
    if h:
        depth = {"plant": -1, "form": 0, "family": 1, "genus": 2, "species": 3}[h["level"]]
        for k, r in enumerate(("form", "family", "genus", "species")):
            x = h.get(r) or {}
            on = k <= depth or (r == "form" and depth >= 0)
            nm = html.escape(str(x.get("name") or "-"))
            rows += (f'<div class="rk {"on" if on else ""}"><span>{r}</span><b>{nm}</b>'
                     f'<i style="--w:{100 * (x.get("conf") or 0):.0f}%"></i>'
                     f'<em>{(x.get("conf") or 0):.2f}</em></div>')
    else:
        nm = html.escape(p.get("species") or "-")
        rows = (f'<div class="rk on"><span>species</span><b>{nm}</b>'
                f'<i style="--w:{100 * (p.get("species_confidence") or 0):.0f}%"></i>'
                f'<em>{(p.get("species_confidence") or 0):.2f}</em></div>')
    title = html.escape(p.get("species") or p.get("category") or "plant")
    cat = html.escape(str(p.get("category"))) + " · " if p.get("species") else ""
    return (f'<div class="card"><div class="ct">{thumb}</div><div class="cb">'
            f'<div class="ch">#{p["id"]} <b>{title}</b> <span class="dim">'
            f'{cat}{p["views"]} views · det '
            f'{p["detection_score"]:.2f}</span></div>{rows}</div></div>')


def pct(x):
    return "—" if x is None or x != x else f"{100 * x:.1f}"


def cls_table():
    rows = []
    for f in sorted(glob.glob(os.path.join(DS, "cls_runs", "*", "eval.json"))) + \
            sorted(glob.glob(os.path.join(DS, "cls_runs", "zeroshot_*.json"))):
        r = json.load(open(f))
        name = os.path.basename(os.path.dirname(f)) if f.endswith("eval.json") else \
            os.path.basename(f)[:-5]
        rows.append((name, r))
    if not rows:
        return "<p class='dim'>No classifier scores yet.</p>"
    taus = "".join(f"<li><b>{html.escape(n)}</b>: " + ", ".join(
        f"{k} {v:.2f}" for k, v in r.get("tau", {}).items()) + "</li>" for n, r in rows)
    out = ["<table><tr><th>classifier</th><th>set</th><th>form</th><th>family</th>"
           "<th>genus</th><th>species</th><th>τ=0.5 correct</th><th>τ=0.5 depth</th>"
           "<th>calibrated τ correct</th><th>calibrated depth</th></tr>"]
    for name, r in rows:
        for s in ("test", "test_fpv", "novel"):
            m = r.get(s, {})
            t = m.get("tau0.5", {})
            cal = m.get("calibrated", {})
            cells = "".join(f"<td>{pct(m.get('acc_' + k))}</td>"
                            for k in ("form", "family", "genus", "species"))
            out.append(f"<tr><td>{html.escape(name)}</td><td>{s}</td>{cells}"
                       f"<td>{pct(t.get('correct'))}</td>"
                       f"<td>{t.get('mean_depth', float('nan')):.2f}</td>"
                       f"<td>{pct(cal.get('correct'))}</td>"
                       f"<td>{cal.get('mean_depth', float('nan')):.2f}</td></tr>")
    out.append("</table><p class='dim'>Calibrated per-rank thresholds (lowest τ at which "
               "that rank is ≥90% right on FPV-degraded + novel-species images):</p><ul>"
               + taus + "</ul>")
    return "".join(out)


def det_table():
    out = ["<table><tr><th>detector run</th><th>epochs</th><th>val mAP50</th>"
           "<th>val mAP50-95</th><th>precision</th><th>recall</th></tr>"]
    n = 0
    for f in sorted(glob.glob(os.path.join(DS, "det_runs", "*", "results.csv"))):
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        rows = [{k.strip(): v for k, v in r.items()} for r in rows]
        best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0)))
        out.append(f"<tr><td>{os.path.basename(os.path.dirname(f))}</td><td>{len(rows)}</td>"
                   f"<td>{float(best['metrics/mAP50(B)']):.3f}</td>"
                   f"<td>{float(best['metrics/mAP50-95(B)']):.3f}</td>"
                   f"<td>{float(best['metrics/precision(B)']):.3f}</td>"
                   f"<td>{float(best['metrics/recall(B)']):.3f}</td></tr>")
        n += 1
    out.append("</table>")
    return "".join(out) if n else "<p class='dim'>No detector scores yet.</p>"


CSS = """
:root{--bg:#f7f7f4;--fg:#1d1f1c;--dim:#6b6f68;--card:#fff;--line:#deded8;--acc:#2f7d4f;
--bar:#cfe3d5;--off:#e9e9e4}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151714;--fg:#e8e9e4;
--dim:#9a9e96;--card:#1e211d;--line:#30342e;--acc:#6cc28d;--bar:#2c4a36;--off:#2a2d28}}
:root[data-theme="dark"]{--bg:#151714;--fg:#e8e9e4;--dim:#9a9e96;--card:#1e211d;--line:#30342e;
--acc:#6cc28d;--bar:#2c4a36;--off:#2a2d28}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1280px;margin:0 auto;padding:24px 16px 64px}main>*{min-width:0}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 10px}h3{font-size:15px;margin:22px 0 8px}
.dim{color:var(--dim)}.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--card)}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap}
th{color:var(--dim);font-weight:600}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,360px),1fr));gap:10px}
.card{display:flex;gap:10px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}
.ct img{width:110px;height:110px;object-fit:cover;border-radius:6px;display:block}
.cb{flex:1;min-width:0}.ch{margin-bottom:4px;overflow:hidden;text-overflow:ellipsis}
.rk{display:grid;grid-template-columns:54px 1fr 60px 34px;gap:6px;align-items:center;font-size:12px;
color:var(--dim)}.rk b{font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rk.on{color:var(--fg)}.rk.on b{font-weight:650}
.rk i{height:6px;background:var(--off);border-radius:3px;position:relative}
.rk i:after{content:"";position:absolute;inset:0;width:var(--w);background:var(--bar);border-radius:3px}
.rk.on i:after{background:var(--acc)}.rk em{font-style:normal;text-align:right}
.frames{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,380px),1fr));gap:10px}
.frames figure{margin:0}.frames img{width:100%;border-radius:6px;display:block}
figcaption{font-size:12px;color:var(--dim);margin-top:2px}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:10px 0}
summary{cursor:pointer;font-weight:600}
a{color:var(--acc)}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(TEST, "plant_candidates.html"))
    ap.add_argument("--frames", default="")
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--best", default="", help="candidate name to mark as the pick")
    a = ap.parse_args()
    cands = []
    ref = load_cand("owlv2_bioclip (current)", os.path.join(TEST, "artifacts_yard_avg"),
                    {"detector": "OWLv2, 7 text prompts", "classifier": "BioCLIP zero-shot, 50 species",
                     "viewer": "yard_viewer.html"})
    if ref:
        cands.append(ref)
    for d in sorted(glob.glob(os.path.join(TEST, "cands", "*"))):
        mp = os.path.join(d, "cand.json")
        meta = json.load(open(mp)) if os.path.exists(mp) else {}
        c = load_cand(os.path.basename(d), d, meta)
        if c:
            cands.append(c)
    if a.frames:
        fids = [int(x) for x in a.frames.split(",")]
    else:
        # frames where most candidates kept boxes, spread through the video
        score = {}
        for c in cands:
            seen = {}
            for p in c["plants"]:
                for b in p.get("boxes", []):
                    seen[b["frame"]] = seen.get(b["frame"], 0) + 1
            for f, n in seen.items():
                score[f] = score.get(f, 0) + 10 + min(n, 5)
        fids = []
        for f in sorted(score, key=lambda f: -score[f]):
            if all(abs(f - g) >= 400 for g in fids):
                fids.append(f)
            if len(fids) == a.n_frames:
                break
        fids.sort()

    summ = ["<table><tr><th>candidate</th><th>detector</th><th>classifier</th><th>plants</th>"
            "<th>named to</th><th>points</th><th>green purity</th><th>green coverage</th>"
            "<th>box precision</th><th>seed recall</th><th>viewer</th></tr>"]
    for c in cands:
        lv = " ".join(f"{k} {v}" for k, v in sorted(c["levels"].items()))
        pts = c["points"]
        ev = c["eval"]
        star = " ★" if c["name"] == a.best else ""
        v = c["meta"].get("viewer", f"yard_viewer_{c['name']}.html")
        link = f'<a href="{v}">open</a>' if os.path.exists(os.path.join(TEST, v)) else "-"
        summ.append(
            f"<tr><td><b>{html.escape(c['name'])}{star}</b></td>"
            f"<td>{html.escape(c['meta'].get('detector', ''))}</td>"
            f"<td>{html.escape(c['meta'].get('classifier', ''))}</td>"
            f"<td>{len(c['plants'])}</td><td>{lv}</td>"
            f"<td>{pts.get('assigned', 0):,}</td>"
            f"<td>{100 * pts.get('green_purity', 0):.0f}%</td>"
            f"<td>{100 * pts.get('green_coverage', 0):.0f}%</td>"
            f"<td>{ev.get('new_box_precision', float('nan')):.3f}</td>"
            f"<td>{ev.get('new_seed_recall', float('nan')):.3f}</td>"
            f"<td>{link}</td></tr>")
    summ.append("</table>")

    body = [f"<h1>Plant detector &amp; classifier candidates</h1>",
            "<p class='dim'>Each candidate: its own box detector → lift to 3D → "
            "Point Transformer point assignment (unchanged, pt7_slim) → hierarchical naming "
            "(growth form → family → genus → species, stopping at the first rank below the "
            "confidence threshold). The reference row is the pipeline behind yard_viewer.html. "
            "Green purity = share of assigned points that are vegetation-coloured; green coverage "
            "= share of vegetation-coloured map points assigned to some plant; box precision / "
            "seed recall as in plant_seg.eval_yard (each relative to its own boxes).</p>",
            "<h2>Summary</h2><div class='wrap'>", "".join(summ), "</div>",
            "<h2>Classifier benchmarks</h2><p class='dim'>Held-out images: test = iNat 2021 val + "
            "PlantNet test + California observers never seen in training; test_fpv = the same images "
            "degraded to look like the yard's analog FPV frames; novel = species held out entirely, "
            "so the right answer is to stop at genus or family. Accuracies in %.</p>"
            "<div class='wrap'>", cls_table(), "</div>",
            "<h2>Detector benchmarks</h2><p class='dim'>Open Images plant/tree boxes, validation "
            "split (the _fpv runs validate on the FPV-degraded copy).</p><div class='wrap'>",
            det_table(), "</div>"]
    body.append("<h2>Same frames, each candidate's boxes</h2>")
    for fid in fids:
        figs = []
        for c in cands:
            src = frame_panel(c, fid)
            if src:
                figs.append(f"<figure><img src='{src}' alt=''><figcaption>"
                            f"{html.escape(c['name'])}</figcaption></figure>")
        body.append(f"<h3>frame {fid}</h3><div class='frames'>{''.join(figs)}</div>")
    body.append("<h2>Plants found by each candidate</h2>")
    for c in cands:
        cards = "".join(plant_card(c, p) for p in c["plants"])
        body.append(f"<details {'open' if c['name'] == a.best else ''}><summary>"
                    f"{html.escape(c['name'])} — {len(c['plants'])} plants</summary>"
                    f"<div class='cards' style='margin-top:8px'>{cards}</div></details>")
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Plant Candidates</title><style>{CSS}</style></head><body><main>"
            + "".join(body) + "</main></body></html>")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {a.out} ({len(page) / 1e6:.1f} MB, {len(cands)} candidates)")


if __name__ == "__main__":
    main()
