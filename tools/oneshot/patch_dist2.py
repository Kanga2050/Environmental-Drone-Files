
import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\detect_plants.py"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''        # The fused name beating every view it was averaged from means no single
        # view actually saw this species; the peak is an artefact of the mean.
        unsupported = bool(at_fused and sp_raw > max(at_fused))''',
    '''        # How much of the reported confidence the views actually carried.
        # Fusion sharpening a call the views already agree on is the point of
        # fusing; a large ratio means the peak came from the mean embedding
        # instead, and on this footage that is where the wrong names sit.
        sharpening = (round(float(sp_raw / max(max(at_fused), 1e-6)), 3)
                      if at_fused else None)
        # No view's own top-1 is the reported name, so nothing independently saw
        # it. min_species_agreement already gates on this; it is recorded here
        # so the record says why, rather than only that the plant failed.
        unsupported = bool(view_top1) and not any(t == sci_top for t in view_top1)'''))

subs.append((
    '''            "species_spread": spread,''',
    '''            "species_spread": spread,
            "species_sharpening": sharpening,'''))

subs.append((
    '''                                       and len(members) >= args.min_species_views
                                       and not unsupported),''',
    '''                                       and len(members) >= args.min_species_views),'''))

subs.append((
    '''        # Pull the distribution apart once, so the record can carry it.
        dist = inst.get("species_dist") or {}''',
    '''        # Pull the distribution apart once, so the record can carry it.
        sci_top = (sp.split("|", 1)[0] if sp and "|" in sp else (sp or ""))
        dist = inst.get("species_dist") or {}'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:240])
    s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("detect_plants: gate corrected, sharpening ratio reported")
