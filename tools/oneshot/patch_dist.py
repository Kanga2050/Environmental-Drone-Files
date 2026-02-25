
import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\detect_plants.py"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''    def classify_many(self, crops, topk=3):
        """One answer from several views of the same individual.''',
    '''    def probs(self, crops):
        """The full candidate distribution for each crop, in one forward pass."""
        from PIL import Image
        batch = torch.stack([self.preprocess(Image.fromarray(c)) for c in crops])
        with torch.no_grad():
            f = self.model.encode_image(batch.to(self.device))
            f = f / f.norm(dim=-1, keepdim=True)
            mean = f.mean(dim=0, keepdim=True)
            mean = mean / mean.norm(dim=-1, keepdim=True)
            per = (100.0 * f.float() @ self.text_feats.T).softmax(dim=-1)
            fused = (100.0 * mean.float() @ self.text_feats.T).softmax(dim=-1)[0]
        return per.cpu().numpy(), fused.cpu().numpy()

    def classify_views(self, crops, topk=5):
        """Fused answer plus the per-view distribution it was averaged from.

        The fused score alone cannot distinguish "every view says 0.4" from
        "one view says 0.95 and four say 0.02" -- the mean embedding is what
        gets softmaxed, so a name no individual view favours can still come out
        peaked. Returning the spread makes that visible instead of hiding it
        behind a confident-looking number.
        """
        if not crops:
            return None
        per, fused = self.probs(crops)
        order = np.argsort(-fused)[:topk]
        top = int(order[0])

        def ent(v):
            v = v[v > 0]
            return float(-(v * np.log2(v)).sum())

        return {
            "fused": [(self.species[i][0], self.species[i][1], float(fused[i]))
                      for i in order],
            "per_view": [[(self.species[i][0], self.species[i][1], float(v[i]))
                          for i in np.argsort(-v)[:topk]] for v in per],
            "at_fused": [float(v[top]) for v in per],
            "entropy": ent(fused),
            "max_entropy": float(np.log2(len(self.species))),
        }

    def classify_many(self, crops, topk=3):
        """One answer from several views of the same individual.'''))

subs.append((
    '''            per_view = (not args.fuse_views) or args.classifier == "bioclip"
            if per_view:
                for (q, m, _), v in zip(chosen, views):
                    m["species"] = [(f"{a}|{b}", c) for a, b, c in classifier(v)]
                    n_calls += 1

            if args.fuse_views and hasattr(classifier, "classify_many"):
                fused = classifier.classify_many(views, topk=5)
                if fused:
                    inst["species_fused"] = fused
                n_calls += 1''',
    '''            # One batched pass covers both questions when the classifier can
            # do it, which is cheaper than N per-view calls plus a fused call.
            if args.fuse_views and hasattr(classifier, "classify_views"):
                dist = classifier.classify_views(views, topk=5)
                if dist:
                    inst["species_fused"] = dist["fused"]
                    inst["species_dist"] = dist
                    for (q, m, _), pv in zip(chosen, dist["per_view"]):
                        m["species"] = [(f"{a}|{b}", c) for a, b, c in pv]
                n_calls += 1
            else:
                per_view = (not args.fuse_views) or args.classifier == "bioclip"
                if per_view:
                    for (q, m, _), v in zip(chosen, views):
                        m["species"] = [(f"{a}|{b}", c)
                                        for a, b, c in classifier(v)]
                        n_calls += 1

                if args.fuse_views and hasattr(classifier, "classify_many"):
                    fused = classifier.classify_many(views, topk=5)
                    if fused:
                        inst["species_fused"] = fused
                    n_calls += 1'''))

subs.append((
    '''            "species_candidates": [
                {"name": a, "common": b, "score": round(float(c), 4)}
                for a, b, c in (fused or [])],''',
    '''            "species_candidates": [
                {"name": a, "common": b, "score": round(float(c), 4)}
                for a, b, c in (fused or [])],
            # What the single score above is standing in for. Views that
            # disagree, or a fused name that outscores every view it came from,
            # both mean the number is less settled than it looks.
            "species_view_scores": view_scores,
            "species_view_top1": view_top1,
            "species_spread": spread,
            "species_entropy": ent_bits,
            "species_entropy_max": ent_max,
            "species_unsupported": unsupported,''',))

subs.append((
    '''        best = max(members, key=lambda m: m["score"])''',
    '''        # Pull the distribution apart once, so the record can carry it.
        dist = inst.get("species_dist") or {}
        at_fused = dist.get("at_fused") or []
        view_scores = [round(float(x), 4) for x in at_fused]
        view_top1 = [m["species"][0][0].split("|")[0]
                     for m in members if m.get("species")]
        spread = round(float(np.std(at_fused)), 4) if len(at_fused) > 1 else None
        ent_bits = (round(float(dist["entropy"]), 3)
                    if "entropy" in dist else None)
        ent_max = (round(float(dist["max_entropy"]), 3)
                   if "max_entropy" in dist else None)
        # The fused name beating every view it was averaged from means no single
        # view actually saw this species; the peak is an artefact of the mean.
        unsupported = bool(at_fused and sp_raw > max(at_fused))

        best = max(members, key=lambda m: m["score"])'''))

subs.append((
    '''            "species_identified": bool(sp_raw >= args.min_species_score
                                       and agreement >= args.min_species_agreement
                                       and len(members) >= args.min_species_views),''',
    '''            "species_identified": bool(sp_raw >= args.min_species_score
                                       and agreement >= args.min_species_agreement
                                       and len(members) >= args.min_species_views
                                       and not unsupported),'''))

for old, new in subs:
    if isinstance(new, tuple):
        new = new[0]
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:240])
    s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("detect_plants: per-view distribution recorded, unsupported names gated")
