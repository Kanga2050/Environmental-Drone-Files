"""One taxonomy for every image source: iNaturalist taxa.csv lineages plus
GIFT growth forms.

Every species is keyed by its iNaturalist taxon id; family and genus come
from its ancestry.  Growth form is not a taxonomic rank, so it is a separate
label: GIFT's species-level value where there is one, else the majority of
the species' genus, else of its family (only when that majority is clear),
with fixed rules for groups GIFT does not separate (ferns, mosses, palms,
succulents, climbers).
"""
import collections
import json
import os
import re

import pandas as pd

FORMS = ["tree", "shrub", "subshrub", "forb", "graminoid", "climber", "palm", "fern",
         "succulent", "moss"]
RANKS = ["form", "family", "genus", "species"]

SUCCULENT_FAMILIES = {"Cactaceae", "Crassulaceae", "Aizoaceae", "Didiereaceae",
                      "Portulacaceae", "Talinaceae", "Anacampserotaceae"}
SUCCULENT_GENERA = {"Aloe", "Agave", "Haworthia", "Haworthiopsis", "Gasteria", "Yucca",
                    "Hesperoyucca", "Dasylirion", "Furcraea", "Kleinia", "Senecio_succ",
                    "Curio", "Sansevieria", "Beschorneria", "Manfreda", "Bulbine",
                    "Kalanchoe", "Pachypodium", "Adenium", "Hoodia", "Stapelia", "Huernia"}
MOSS_PHYLA = {"Bryophyta", "Marchantiophyta", "Anthocerotophyta"}
FERN_CLASSES = {"Polypodiopsida", "Lycopodiopsida"}
GRAMINOID_FAMILIES = {"Poaceae", "Cyperaceae", "Juncaceae", "Restionaceae", "Typhaceae"}


def clean_name(name):
    """'Lactuca virosa L.' -> 'Lactuca virosa'; hybrids keep their cross sign:
    'Photinia × fraseri Dress' -> 'Photinia × fraseri'."""
    parts = name.replace("_", " ").split()
    if len(parts) >= 3 and parts[1] in ("×", "x", "X"):
        return f"{parts[0]} × {parts[2]}"
    return " ".join(parts[:2]) if len(parts) >= 2 else name


class Taxonomy:

    def __init__(self, taxa_csv, gift_dir=None):
        t = pd.read_csv(taxa_csv, sep="\t", dtype={"ancestry": str})
        t = t[t.ancestry.fillna("").str.startswith("48460/47126")]       # Plantae
        self.t = t.set_index("taxon_id")
        self.rank = self.t["rank"].to_dict()
        self.name = self.t["name"].to_dict()
        self.anc = self.t["ancestry"].to_dict()
        act = t[t.active.astype(str).str.lower() == "true"]
        self.by_name = {}
        for tid, n, r in zip(t.taxon_id, t.name, t["rank"]):
            self.by_name.setdefault((n, r), tid)
        # active taxa win over inactive synonyms with the same name
        for tid, n, r in zip(act.taxon_id, act.name, act["rank"]):
            self.by_name[(n, r)] = tid
        self.gift = self._gift(gift_dir) if gift_dir else {}
        self._form_cache = {}

    def find_species(self, name):
        n = clean_name(name)
        tid = self.by_name.get((n, "species"))
        if tid is None and " × " in n:
            tid = self.by_name.get((n, "hybrid"))
        if tid is None:
            # hybrids / infraspecific names fall back to the genus
            tid = self.by_name.get((n.split()[0], "genus"))
        return tid

    def lineage(self, tid):
        """{rank: name} for phylum/class/order/family/genus/species and the ids."""
        out = {}
        chain = [int(x) for x in self.anc.get(tid, "").split("/") if x] + [tid]
        for a in chain:
            r = self.rank.get(a)
            if r in ("phylum", "class", "order", "family", "genus", "species"):
                out[r] = self.name[a]
                out[r + "_id"] = a
        # varieties / subspecies / hybrids roll up to their species
        if self.rank.get(tid) == "hybrid":
            out.setdefault("species", clean_name(self.name[tid]))     # 'Photinia × fraseri'
        elif self.rank.get(tid) in ("subspecies", "variety", "form", "infrahybrid"):
            out.setdefault("species", " ".join(self.name[tid].split()[:2]))
        return out

    @staticmethod
    def _gift(gift_dir):
        sp = json.load(open(os.path.join(gift_dir, "species.json")))
        names = {r["work_ID"]: r["work_species"] for r in sp}
        val = {}
        for key in ("gf2", "gf1", "climber", "climber1"):
            p = os.path.join(gift_dir, key + ".json")
            if os.path.exists(p):
                for r in json.load(open(p)):
                    n = names.get(r["work_ID"])
                    if n and r["trait_value"]:
                        val.setdefault(n, {})[key] = r["trait_value"]
        return val

    def _gift_form(self, species):
        g = self.gift.get(species)
        if not g:
            return None
        if g.get("climber") in ("liana", "vine") or g.get("climber1") == "obligatory":
            return "climber"
        v = g.get("gf2") or g.get("gf1")
        if not v:
            return None
        v = v.split("/")[0]
        return {"herb": "forb", "forb": "forb", "graminoid": "graminoid", "subshrub": "subshrub",
                "shrub": "shrub", "tree": "tree", "palm": "palm"}.get(v)

    def build_forms(self, species_ids):
        """Growth form for each species id (None when unknown)."""
        lin = {s: self.lineage(s) for s in species_ids}
        direct = {}
        for s, L in lin.items():
            f = self._rule_form(L)
            if f is None and "species" in L:
                f = self._gift_form(L["species"])
            direct[s] = f
        # genus/family majorities over every GIFT species we know (not just ours)
        votes = {"genus": collections.defaultdict(collections.Counter),
                 "family": collections.defaultdict(collections.Counter)}
        for n in self.gift:
            f = self._gift_form(n)
            tid = self.by_name.get((n, "species"))
            if f is None or tid is None:
                continue
            L = self.lineage(tid)
            for r in votes:
                if r in L:
                    votes[r][L[r]][f] += 1
        out = {}
        for s, L in lin.items():
            f = direct[s]
            for r, need in (("genus", 0.7), ("family", 0.8)):
                if f is None and L.get(r) in votes[r]:
                    c = votes[r][L[r]]
                    top, n = c.most_common(1)[0]
                    if n / sum(c.values()) >= need and sum(c.values()) >= 3:
                        f = top
            if f == "forb" and L.get("family") in GRAMINOID_FAMILIES:
                f = "graminoid"
            out[s] = f
        return out

    @staticmethod
    def _rule_form(L):
        if L.get("phylum") in MOSS_PHYLA:
            return "moss"
        if L.get("class") in FERN_CLASSES:
            return "fern"
        if L.get("family") == "Arecaceae":
            return "palm"
        if L.get("family") in SUCCULENT_FAMILIES or L.get("genus") in SUCCULENT_GENERA:
            return "succulent"
        return None
