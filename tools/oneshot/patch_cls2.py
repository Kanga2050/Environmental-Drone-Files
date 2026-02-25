
import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\detect_plants.py"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''    ap.add_argument("--classify_top_k", type=int, default=5,''',
    '''    ap.add_argument("--fuse_views", action=argparse.BooleanOptionalAction, default=True,
                    help="Send several views of one instance to the classifier as a "
                         "single identification, instead of classifying each crop "
                         "and voting. Both classifiers support it and it is what "
                         "they are built for; voting was this pipeline's own idea")
    ap.add_argument("--stack_views", type=int, default=0,
                    help="Build the first view by registering and averaging this "
                         "many of the best crops. Cuts the analog chroma noise, at "
                         "the risk of smearing if the views do not register. 0 off")
    ap.add_argument("--min_crop_side", type=int, default=224,
                    help="Upscale crops shorter than this with Lanczos before "
                         "classifying, so the classifier's own resize is not the "
                         "limiting filter")
    ap.add_argument("--classify_top_k", type=int, default=5,'''))

subs.append((
    '''            scored.sort(key=lambda t: -t[0])
            for q, m, rgb_crop in scored[:args.classify_top_k]:
                crop = rgb_crop
                if args.foliage_crop:
                    crop = foliage_subcrop(crop)
                m["crop_quality"] = round(float(q), 4)
                sp = classifier(crop)
                m["species"] = [(f"{a}|{b}", c) for a, b, c in sp]
                n_calls += 1
        print(f"Classified {n_calls} crops "
              f"(top {args.classify_top_k} per instance, {len(instances)} instances)")''',
    '''            scored.sort(key=lambda t: -t[0])
            chosen = scored[:args.classify_top_k]
            views = []
            for q, m, rgb_crop in chosen:
                crop = rgb_crop
                if args.foliage_crop:
                    crop = foliage_subcrop(crop)
                m["crop_quality"] = round(float(q), 4)
                views.append(upscale_crop(crop, args.min_crop_side))

            if args.stack_views > 1 and len(views) > 1:
                comp = stack_crops(views[:args.stack_views])
                if comp is not None:
                    views = [comp] + views[1:]

            # Per-view answers are kept even when fusing, because they are the
            # only way to see whether the views agree -- and a fused score is
            # exactly as confident-looking when every view was a coin flip.
            # Local classifiers get this for free; a metered API does not, so
            # there it happens only when fusing is off.
            per_view = (not args.fuse_views) or args.classifier == "bioclip"
            if per_view:
                for (q, m, _), v in zip(chosen, views):
                    m["species"] = [(f"{a}|{b}", c) for a, b, c in classifier(v)]
                    n_calls += 1

            if args.fuse_views and hasattr(classifier, "classify_many"):
                fused = classifier.classify_many(views, topk=5)
                if fused:
                    inst["species_fused"] = fused
                n_calls += 1
        how = ("fused per instance" if args.fuse_views else "per crop")
        print(f"Classified {n_calls} calls, {how} "
              f"(top {args.classify_top_k} views per instance, "
              f"{len(instances)} instances)")'''))

subs.append((
    '''        sp, sp_conf, sp_raw = vote_label(members, "species")''',
    '''        sp, sp_conf, sp_raw = vote_label(members, "species")
        # Cross-view agreement stays the honesty check whichever way the name was
        # produced: it is the share of views that independently landed on the
        # same species, and a fused request cannot report it about itself.
        agreement = sp_conf
        method = "vote"
        fused = inst.get("species_fused")
        if fused:
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
                agreement = sp_conf        # nothing to cross-check against'''))

subs.append((
    '''            "species_identified": bool(sp_raw >= args.min_species_score''',
    '''            "species_method": method,
            "species_agreement": round(float(agreement), 3),
            "species_candidates": [
                {"name": a, "common": b, "score": round(float(c), 4)}
                for a, b, c in (fused or [])],
            "species_identified": bool(sp_raw >= args.min_species_score'''))

subs.append((
    '''                                       and sp_conf >= args.min_species_agreement
                                       and len(members) >= args.min_species_views),''',
    '''                                       and agreement >= args.min_species_agreement
                                       and len(members) >= args.min_species_views),'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:260])
    s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8").write(s)
print("multi-view classification wired")
