"""Hierarchical plant classifier: one backbone, four heads
(growth form, family, genus, species), and top-down inference that stops at
the deepest rank whose confidence clears a threshold.

Family/genus confidence blends the rank's own head with the species head
marginalised up the tree, so a plant whose exact species was never seen can
still be named confidently at genus or family.  Taxonomic picks are made
top-down (genus only among the chosen family's genera, species only among
the chosen genus's species), so the answer is always one consistent path.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def make_backbone(name):
    """-> (module mapping images to features, feature dim, mean, std, blocks list)."""
    if name in ("bioclip2", "bioclip"):
        import open_clip
        if name == "bioclip2":
            m, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
        else:
            m, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained=r"C:\Users\shaur\AppData\Local\lingbot-map\models"
                                      r"\bioclip\open_clip_pytorch_model.bin")
        v = m.visual
        dim = v.output_dim
        return v, dim, CLIP_MEAN, CLIP_STD, list(v.transformer.resblocks)
    import timm
    timm_name = {"dinov2_b": "vit_base_patch14_reg4_dinov2.lvd142m",
                 "dinov2_l": "vit_large_patch14_reg4_dinov2.lvd142m",
                 "convnext_s": "convnext_small.fb_in22k_ft_in1k"}[name]
    kw = {"img_size": 224} if "dinov2" in name else {}
    m = timm.create_model(timm_name, pretrained=True, num_classes=0, **kw)
    blocks = list(m.blocks) if hasattr(m, "blocks") else list(m.stages)
    return m, m.num_features, IMNET_MEAN, IMNET_STD, blocks


class HierNet(nn.Module):

    def __init__(self, backbone, n_form, n_family, n_genus, n_species, train_blocks=-1,
                 grad_ckpt=False):
        super().__init__()
        self.backbone_name = backbone
        self.backbone, dim, self.mean, self.std, blocks = make_backbone(backbone)
        if grad_ckpt:
            self.backbone.set_grad_checkpointing(True)
        if train_blocks >= 0:               # -1: train everything
            for p in self.backbone.parameters():
                p.requires_grad = False
            for b in blocks[len(blocks) - train_blocks:] if train_blocks else []:
                for p in b.parameters():
                    p.requires_grad = True
            for n, p in self.backbone.named_parameters():
                if n.startswith(("ln_post", "proj", "norm", "head", "fc_norm")) and train_blocks:
                    p.requires_grad = True
        self.drop = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({
            "form": nn.Linear(dim, n_form), "family": nn.Linear(dim, n_family),
            "genus": nn.Linear(dim, n_genus), "species": nn.Linear(dim, n_species)})
        self.register_buffer("mean_t", torch.tensor(self.mean).view(1, 3, 1, 1))
        self.register_buffer("std_t", torch.tensor(self.std).view(1, 3, 1, 1))

    def forward(self, x):
        """x: float images in [0,1], (B,3,224,224)."""
        f = self.backbone((x - self.mean_t) / self.std_t)
        if f.dim() == 3:
            f = f[:, 0]
        f = self.drop(f)
        return {k: h(f) for k, h in self.heads.items()}


class Hierarchy:
    """Vocabulary and parent links: species -> genus -> family."""

    def __init__(self, vocab, blend="mean"):
        self.v = vocab
        self.sp_gen = torch.tensor(vocab["species_genus"])
        self.gen_fam = torch.tensor(vocab["genus_family"])
        self.blend = blend

    def probs(self, logits, temp=None):
        """Per-rank probabilities, blended with the species marginals.

        blend="mean" averages a rank's own head with the species head summed
        up the tree; blend="geo" takes their geometric mean, so a rank is only
        confident when both agree (one confident head alone is not enough)."""
        temp = temp or {}
        p = {k: F.softmax(v.float() / temp.get(k, 1.0), -1) for k, v in logits.items()}
        sp = p["species"]
        g_marg = torch.zeros_like(p["genus"]).index_add_(1, self.sp_gen.to(sp.device), sp)
        if self.blend == "geo":
            f_marg = torch.zeros_like(p["family"]).index_add_(1, self.gen_fam.to(sp.device),
                                                              0.5 * (p["genus"] + g_marg))
            g = torch.sqrt(p["genus"] * g_marg)
            f = torch.sqrt(p["family"] * f_marg)
        else:
            g = 0.5 * (p["genus"] + g_marg)
            f_marg = torch.zeros_like(p["family"]).index_add_(1, self.gen_fam.to(sp.device), g)
            f = 0.5 * (p["family"] + f_marg)
        return {"form": p["form"], "family": f, "genus": g, "species": sp}

    def decide(self, p, tau=0.5, tau_form=0.5):
        """One image (or fused views): p = {rank: (C,) tensor}.  Returns a dict
        with the chosen name per rank, its confidence, and the deepest rank
        that cleared tau."""
        v = self.v
        out = {"path": []}
        fi = int(p["form"].argmax())
        out["form"] = (v["forms"][fi], float(p["form"][fi]))
        fam = int(p["family"].argmax())
        gens = (self.gen_fam == fam).nonzero().flatten()
        gen = int(gens[p["genus"][gens].argmax()]) if len(gens) else int(p["genus"].argmax())
        sps = (self.sp_gen == gen).nonzero().flatten()
        spi = int(sps[p["species"][sps].argmax()]) if len(sps) else -1
        # confidence of the deeper pick is the joint, so it can never exceed its parent
        cf = float(p["family"][fam])
        cg = min(cf, float(p["genus"][gen]))
        cs = min(cg, float(p["species"][spi])) if spi >= 0 else 0.0
        out["family"] = (v["families"][fam], cf)
        out["genus"] = (v["genera"][gen], cg)
        out["species"] = (v["species"][spi], cs) if spi >= 0 else (None, 0.0)
        t = tau if isinstance(tau, dict) else {"form": tau_form, "family": tau,
                                                "genus": tau, "species": tau}
        level = "plant"
        if out["form"][1] >= t["form"]:
            level = "form"
        for r, c in (("family", cf), ("genus", cg), ("species", cs)):
            if c >= t[r]:
                level = r
            else:
                break
        out["level"] = level
        return out


def fuse(prob_list):
    """Average per-view probabilities (views of one plant)."""
    return {k: torch.stack([p[k] for p in prob_list]).mean(0) for k in prob_list[0]}


class HierClassifier:
    """Inference wrapper used by detect_plants.py."""

    def __init__(self, ckpt_path, device="cuda", tau=None):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        v = ck["vocab"]
        self.net = HierNet(ck["backbone"], len(v["forms"]), len(v["families"]),
                           len(v["genera"]), len(v["species"]), train_blocks=0)
        self.net.load_state_dict(ck["model"], strict=False)
        self.net = self.net.to(device).eval()
        self.temp = ck.get("temp", {})
        import json
        import os
        tj = os.path.join(os.path.dirname(ckpt_path), "tau.json")
        self.tau = tau if tau is not None else (
            json.load(open(tj)) if os.path.exists(tj) else 0.5)
        # a threshold file may say which blend it was calibrated for
        blend = self.tau.pop("blend", "mean") if isinstance(self.tau, dict) else "mean"
        self.h = Hierarchy(v, blend)
        self.device = device
        self.common = v.get("common", {})

    @torch.no_grad()
    def view_probs(self, crops):
        """crops: list of HxWx3 uint8 RGB.  -> list of per-view prob dicts."""
        x = torch.stack([prep_eval(c) for c in crops]).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = self.net(x)
            lf = self.net(torch.flip(x, dims=[3]))          # mirrored view, averaged in
        p = self.h.probs({k: t.float() for k, t in lg.items()}, self.temp)
        pf = self.h.probs({k: t.float() for k, t in lf.items()}, self.temp)
        p = {k: 0.5 * (p[k] + pf[k]) for k in p}
        return [{k: t[i].cpu() for k, t in p.items()} for i in range(len(crops))]

    def classify_views(self, crops):
        per = self.view_probs(crops)
        fused = fuse(per)
        d = self.h.decide(fused, self.tau)
        d["per_view"] = [self.h.decide(q, self.tau) for q in per]
        d["top_species"] = [(self.h.v["species"][i], float(fused["species"][i]))
                            for i in fused["species"].topk(5).indices.tolist()]
        return d


def prep_eval(rgb, size=224):
    """Pad to square (keep the whole plant), resize, to float tensor."""
    import cv2
    h, w = rgb.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s, 3), 124, np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = rgb
    im = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(im).permute(2, 0, 1).float() / 255.0


FORM_PROMPTS = {"tree": "a tree", "shrub": "a shrub", "subshrub": "a small woody subshrub",
                "forb": "a non-woody flowering herb", "graminoid": "a grass or sedge",
                "climber": "a climbing vine", "palm": "a palm tree", "fern": "a fern",
                "succulent": "a succulent plant", "moss": "a moss"}


class ZeroShotNet(torch.nn.Module):
    """BioCLIP(-2) zero-shot heads over the same vocabulary: each rank's
    logits are similarities to that rank's taxonomic text ("a photo of
    Plantae Tracheophyta ... Rosaceae Rosa chinensis")."""

    def __init__(self, vocab, model="bioclip2", device="cuda"):
        super().__init__()
        import open_clip
        if model == "bioclip2":
            m, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
            tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
        else:
            m, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained=r"C:\Users\shaur\AppData\Local\lingbot-map\models"
                                      r"\bioclip\open_clip_pytorch_model.bin")
            tok = open_clip.get_tokenizer("ViT-B-16")
        self.m = m.to(device).eval()
        self.scale = float(m.logit_scale.exp())
        texts = {"form": [f"a photo of {FORM_PROMPTS[f]}." for f in vocab["forms"]],
                 "family": [f"a photo of {t}." for t in vocab["tax"]["families"]],
                 "genus": [f"a photo of {t}." for t in vocab["tax"]["genera"]],
                 "species": [f"a photo of {t}." for t in vocab["tax"]["species"]]}
        self.T = {}
        with torch.no_grad():
            for k, ts in texts.items():
                fs = []
                for i in range(0, len(ts), 512):
                    f = m.encode_text(tok(ts[i:i + 512]).to(device))
                    fs.append(F.normalize(f.float(), dim=-1))
                self.T[k] = torch.cat(fs)
        self.register_buffer("mean_t", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1).to(device))
        self.register_buffer("std_t", torch.tensor(CLIP_STD).view(1, 3, 1, 1).to(device))

    def forward(self, x):
        f = F.normalize(self.m.encode_image((x - self.mean_t) / self.std_t).float(), dim=-1)
        return {k: self.scale * f @ t.T for k, t in self.T.items()}


class ZeroShotClassifier(HierClassifier):

    def __init__(self, vocab_path, model="bioclip2", device="cuda", tau=0.5):
        import json
        v = json.load(open(vocab_path))
        self.net = ZeroShotNet(v, model, device)
        self.h = Hierarchy(v)
        self.temp, self.tau, self.device = {}, tau, device
