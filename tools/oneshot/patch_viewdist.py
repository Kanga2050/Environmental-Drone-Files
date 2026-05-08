
import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\view_map3d.py"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''                "species_identified": p.get("species_identified", False),''',
    '''                "species_identified": p.get("species_identified", False),
                "species_candidates": p.get("species_candidates", []),
                "species_view_scores": p.get("species_view_scores", []),
                "species_view_top1": p.get("species_view_top1", []),
                "species_agreement": p.get("species_agreement"),
                "species_spread": p.get("species_spread"),
                "species_sharpening": p.get("species_sharpening"),
                "species_entropy": p.get("species_entropy"),
                "species_entropy_max": p.get("species_entropy_max"),
                "species_unsupported": p.get("species_unsupported", False),'''))

for old, new in subs:
    if old not in s:
        sys.exit("view_map3d NOT FOUND:\n" + old[:240])
    s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("view_map3d: species distribution served")

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\viewer\app.js"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''function renderPlantDetail(i) {''',
    '''// Two different spreads, and conflating them is how a shaky name looks solid.
// `candidates` is how the classifier split its belief across species for the
// fused answer; `views` is how much each individual look supported the name that
// won. A tall first bar with scattered view bars means the mean embedding made
// the call, not the evidence.
function distBars(p) {
  const cand = p.species_candidates || [];
  const views = p.species_view_scores || [];
  if (!cand.length && !views.length) return "";
  const top = p.species || "";
  let html = "";
  if (cand.length) {
    html += `<div class="dist"><div class="distlab">across species</div>` +
      cand.map((c) => `<div class="bar" title="${c.name} ${c.score.toFixed(3)}">
        <span class="bn">${c.name === top ? "<b>" + c.name + "</b>" : c.name}</span>
        <span class="bt"><i style="width:${Math.max(1, c.score * 100).toFixed(1)}%"></i></span>
        <span class="bv">${c.score.toFixed(3)}</span></div>`).join("") + `</div>`;
  }
  if (views.length) {
    const tops = p.species_view_top1 || [];
    html += `<div class="dist"><div class="distlab">across views &mdash; each
      look's own score for ${top || "the reported name"}</div>` +
      views.map((v, k) => {
        const agree = tops[k] === top;
        return `<div class="bar" title="${tops[k] || ""}">
          <span class="bn">view ${k + 1}${agree ? " &#10003;" : ""}</span>
          <span class="bt"><i class="${agree ? "ok" : "no"}"
            style="width:${Math.max(1, v * 100).toFixed(1)}%"></i></span>
          <span class="bv">${v.toFixed(3)}</span></div>`;
      }).join("") + `</div>`;
  }
  return html;
}

function renderPlantDetail(i) {'''))

subs.append((
    '''      <div class="k">vote share</div><div class="v">${
        (p.species_confidence || 0).toFixed(2)}</div>
      <div class="k">views</div><div class="v">${p.views}</div>''',
    '''      <div class="k">vote share</div><div class="v">${
        (p.species_confidence || 0).toFixed(2)}</div>
      ${p.species_agreement == null ? "" : `<div class="k">views agreeing</div>
        <div class="v">${(p.species_agreement * 100).toFixed(0)}%</div>`}
      ${p.species_spread == null ? "" : `<div class="k">per-view spread</div>
        <div class="v">&plusmn;${p.species_spread.toFixed(3)}</div>`}
      ${p.species_sharpening == null ? "" : `<div class="k" title="fused score
        divided by the best single view's score for the same name">from fusing</div>
        <div class="v">${p.species_sharpening.toFixed(2)}&times;</div>`}
      ${p.species_entropy == null ? "" : `<div class="k" title="0 = one species,
        ${(p.species_entropy_max || 0).toFixed(2)} = no preference at all">entropy</div>
        <div class="v">${p.species_entropy.toFixed(2)} / ${
          (p.species_entropy_max || 0).toFixed(2)} bits</div>`}
      <div class="k">views</div><div class="v">${p.views}</div>'''))

subs.append((
    '''    <div class="btnrow">
      <button class="act" id="pdFly">Fly to (F)</button>''',
    '''    ${distBars(p)}
    ${p.species_unsupported ? `<div class="warn">No individual view picked this
      species &mdash; the name comes only from the averaged embedding.</div>` : ""}
    <div class="btnrow">
      <button class="act" id="pdFly">Fly to (F)</button>'''))

subs.append((
    '''                  "common", "species_raw", "species_vote", "identified", "views",''',
    '''                  "common", "species_raw", "species_vote", "agreement",
                  "view_spread", "sharpening", "entropy", "identified", "views",'''))

subs.append((
    '''                p.species || "", p.species_common || "", p.species_raw_score,
                p.species_confidence, p.species_identified ? 1 : 0, p.views,''',
    '''                p.species || "", p.species_common || "", p.species_raw_score,
                p.species_confidence, p.species_agreement, p.species_spread,
                p.species_sharpening, p.species_entropy,
                p.species_identified ? 1 : 0, p.views,'''))

for old, new in subs:
    if old not in s:
        sys.exit("app.js NOT FOUND:\n" + old[:240])
    s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("app.js: distribution bars + spread fields + CSV columns")

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\viewer\style.css"
s = io.open(p, encoding="utf-8").read()
s += '''
/* Species confidence distributions. Bars are deliberately plain: the point is
   comparing lengths, and anything more decorative competes with the 3D view. */
.dist { margin-top: 10px; }
.distlab { color: #8b949e; font-size: 11px; margin-bottom: 4px; }
.bar { display: grid; grid-template-columns: 96px 1fr 40px; gap: 6px;
       align-items: center; margin: 2px 0; font-size: 11px; }
.bar .bn { color: #c9d1d9; overflow: hidden; text-overflow: ellipsis;
           white-space: nowrap; }
.bar .bt { background: #21262d; border-radius: 2px; height: 9px; overflow: hidden; }
.bar .bt i { display: block; height: 100%; background: #58a6ff; }
.bar .bt i.ok { background: #3fb950; }
.bar .bt i.no { background: #6e7681; }
.bar .bv { color: #8b949e; text-align: right; font-variant-numeric: tabular-nums; }
.warn { margin-top: 8px; padding: 6px 8px; border-radius: 4px; font-size: 11px;
        background: #3d2a14; color: #e3b341; border: 1px solid #5c3f1a; }
'''
io.open(p, "w", encoding="utf-8").write(s)
print("style.css: bar styles")
