import io, sys
p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\detect_plants.py"
s = io.open(p, encoding="utf-8").read()
old = '''        if fused:
            method = "fused"
            fsci, fcom, fraw = fused[0]
            total = sum(x[2] for x in fused) or 1.0
            if sp and sp.split("|")[0] == fsci:
                pass                       # views already agreed with the fusion
            else:
                agreement = sp_conf if sp else 0.0
            sp = f"{fsci}|{fcom}"
            sp_raw = fraw
            sp_conf = fraw / total
            if not any(m.get("species") for m in members):
                agreement = sp_conf        # nothing to cross-check against'''
new = '''        if fused:
            method = "fused"
            fsci, fcom, fraw = fused[0]
            total = sum(x[2] for x in fused) or 1.0
            # Agreement has to be measured against the name actually being
            # reported, which is the fused one -- reusing the vote share would
            # quietly certify the fused name with a different species' support.
            tops = [m["species"][0][0].split("|")[0]
                    for m in members if m.get("species")]
            agreement = (sum(t == fsci for t in tops) / len(tops)) if tops else None
            sp = f"{fsci}|{fcom}"
            sp_raw = fraw
            sp_conf = fraw / total
            if agreement is None:
                # Nothing to cross-check against (a metered API classified once),
                # so fall back to the fused margin and say so in the output.
                agreement = sp_conf
                method = "fused-unchecked"'''
assert old in s, "not found"
io.open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("agreement fixed")
