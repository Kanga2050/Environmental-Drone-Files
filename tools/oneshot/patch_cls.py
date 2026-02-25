
import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\detect_plants.py"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''        idx = np.argsort(-probs)[:topk]
        return [(self.species[i][0], self.species[i][1], float(probs[i])) for i in idx]''',
    '''        idx = np.argsort(-probs)[:topk]
        return [(self.species[i][0], self.species[i][1], float(probs[i])) for i in idx]

    def classify_many(self, crops, topk=3):
        """One answer from several views of the same individual.

        Averages the L2-normalised image embeddings and classifies that, rather
        than averaging the per-view answers. For a contrastive model the mean
        embedding is the usual ensembling point: it cancels view-specific noise
        before the softmax sharpens it, where averaging probabilities lets one
        confident bad view dominate.
        """
        from PIL import Image
        if not crops:
            return []
        batch = torch.stack([self.preprocess(Image.fromarray(c)) for c in crops])
        with torch.no_grad():
            f = self.model.encode_image(batch.to(self.device))
            f = f / f.norm(dim=-1, keepdim=True)
            f = f.mean(dim=0, keepdim=True)
            f = (f / f.norm(dim=-1, keepdim=True)).float()
            probs = (100.0 * f @ self.text_feats.T).softmax(dim=-1)[0].cpu().numpy()
        idx = np.argsort(-probs)[:topk]
        return [(self.species[i][0], self.species[i][1], float(probs[i])) for i in idx]'''))

subs.append((
    '''        out = []
        for item in results:
            sp = item.get("species", {})
            sci = sp.get("scientificNameWithoutAuthor", "unknown")
            commons = sp.get("commonNames") or [""]
            out.append((sci, commons[0], float(item.get("score", 0.0))))
        return out''',
    '''        return self._parse(results)

    @staticmethod
    def _parse(results):
        out = []
        for item in results:
            sp = item.get("species", {})
            sci = sp.get("scientificNameWithoutAuthor", "unknown")
            commons = sp.get("commonNames") or [""]
            out.append((sci, commons[0], float(item.get("score", 0.0))))
        return out

    def classify_many(self, crops, topk=3, organs=None):
        """Identify one plant from up to five images in a single request.

        This is what the API is actually built for -- the service weighs the
        views jointly against one individual -- and it is a different thing from
        classifying each crop and voting, which is what this pipeline did
        before. On footage where no single frame carries enough detail, the
        difference is the whole result. It also costs one quota unit instead of
        five.
        """
        import requests
        crops = [c for c in crops if c is not None and c.size][:5]
        if not crops:
            return []
        files, data = [], []
        for i, c in enumerate(crops):
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(c, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                continue
            files.append(("images", (f"view{i}.jpg", buf.tobytes(), "image/jpeg")))
            data.append(("organs", organs[i] if organs else "auto"))
        if not files:
            return []
        try:
            r = requests.post(
                self.ENDPOINT,
                params={"api-key": self.api_key, "include-related-images": "false"},
                files=files, data=data, timeout=self.timeout)
            r.raise_for_status()
            return self._parse(r.json().get("results", [])[:topk])
        except Exception as exc:
            print(f"    Pl@ntNet multi-view request failed: {exc}")
            return []'''))

subs.append((
    '''def foliage_subcrop(crop_rgb, min_frac=0.04, min_keep_frac=0.35):''',
    '''def upscale_crop(crop_rgb, min_side=224, max_factor=4.0):
    """Enlarge a small crop with a good filter before the classifier shrinks it.

    Both classifiers resize to roughly 224 internally with whatever filter they
    were built with, and a 60-pixel crop blown up by a bilinear resize is mush.
    Doing it here with Lanczos adds no information but costs much less of what
    was there. Capped, because a 20-pixel crop upscaled eightfold is still a
    20-pixel crop and only wastes the classifier's time.
    """
    if crop_rgb is None or crop_rgb.size == 0:
        return crop_rgb
    h, w = crop_rgb.shape[:2]
    short = min(h, w)
    if short >= min_side:
        return crop_rgb
    f = min(max_factor, min_side / max(short, 1))
    if f <= 1.01:
        return crop_rgb
    return cv2.resize(crop_rgb, (int(round(w * f)), int(round(h * f))),
                      interpolation=cv2.INTER_LANCZOS4)

def stack_crops(crops, max_shift=0.25, min_corr=0.6):
    """Average several views of one plant into a lower-noise composite.

    The chroma noise on analog video is close to independent between frames
    while the plant is not, so registering a few crops and averaging trades
    nothing real for a clear drop in noise. Registration is affine by ECC
    against the best crop; any view that will not converge, or that converges
    to a poor correlation, is dropped rather than smeared in -- a misaligned
    frame in an average is worse than no frame at all.

    Returns the reference crop unchanged if nothing else registers, so the
    caller never has to handle a failure case.
    """
    if not crops:
        return None
    ref = crops[0]
    if len(crops) == 1:
        return ref
    h, w = ref.shape[:2]
    ref_g = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    acc = ref.astype(np.float32).copy()
    used = 1
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4)
    for c in crops[1:]:
        try:
            cr = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(cr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            warp = np.eye(2, 3, dtype=np.float32)
            corr, warp = cv2.findTransformECC(ref_g, g, warp, cv2.MOTION_AFFINE,
                                              crit, None, 5)
            if corr < min_corr:
                continue
            if (abs(warp[0, 2]) > max_shift * w) or (abs(warp[1, 2]) > max_shift * h):
                continue
            aligned = cv2.warpAffine(cr, warp, (w, h),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REFLECT)
            acc += aligned.astype(np.float32)
            used += 1
        except cv2.error:
            continue
    return np.clip(acc / used, 0, 255).astype(np.uint8)

def foliage_subcrop(crop_rgb, min_frac=0.04, min_keep_frac=0.35):'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:240])
    s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8").write(s)
print("classifier multi-view + crop conditioning added")
