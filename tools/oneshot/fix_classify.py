import io
p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\view_map3d.py"
s = io.open(p, encoding="utf-8").read()
old = '''        if self.classifier_kind == "plantnet" and hasattr(self._classifier, "classify_many"):
            res = self._classifier.classify_many(crops, topk=topk)
        else:
            # Average the per-view distributions rather than taking a majority:
            # a species that is second in every view is a better answer than one
            # that wins a single blurry frame.
            pool = {}
            for c in crops:
                for name, common, score in self._classifier(c, topk=topk):
                    e = pool.setdefault(name, {"common": common, "scores": []})
                    e["scores"].append(score)
            res = [{"name": k, "common": v["common"],
                    "score": float(np.mean(v["scores"])),
                    "views": len(v["scores"])}
                   for k, v in pool.items()]
            res.sort(key=lambda r: -r["score"])
            res = res[:topk]
        return [dict(r) for r in res]'''
new = '''        if hasattr(self._classifier, "classify_many"):
            # One identification from all the views together. Both classifiers
            # support this and it is what they are built for; classifying each
            # crop and pooling the answers is a worse approximation of it.
            fused = self._classifier.classify_many(crops, topk=topk)
            res = [{"name": a, "common": b, "score": float(c),
                    "views": len(crops)} for a, b, c in fused]
        else:
            pool = {}
            for c in crops:
                for name, common, score in self._classifier(c, topk=topk):
                    e = pool.setdefault(name, {"common": common, "scores": []})
                    e["scores"].append(score)
            res = [{"name": k, "common": v["common"],
                    "score": float(np.mean(v["scores"])),
                    "views": len(v["scores"])}
                   for k, v in pool.items()]
            res.sort(key=lambda r: -r["score"])
            res = res[:topk]
        return [dict(r) for r in res]'''
assert old in s
io.open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("server now fuses views for every classifier")
