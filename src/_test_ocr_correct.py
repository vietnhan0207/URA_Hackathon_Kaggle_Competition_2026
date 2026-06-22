"""Measure the upside of two OCR-text correction ideas on TRAIN (CER vs GT):
  (1) brand-splice: when a product rule fires, replace the fuzzy-matched brand
      span in our OCR with the canonical UPPERCASE brand surface form.
  (2) re-inference proxy: how much CER would drop if dominant-family images had
      their brand tokens perfect (upper bound on rule-correction).
Also: does cleaner OCR let MORE product rules fire (product F1 knock-on)?
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from data import load_train_labels
from product_calibrated import CalibratedRuleHead, fold
from run_ocr import cache_path
from scoring import cer, token_f1

labels = load_train_labels()
ocr = pd.read_parquet(cache_path("vietocr_ft", "all"))[["image_id", "ocr_text"]]
df = labels.merge(ocr, on="image_id", suffixes=("_gt", "_ocr")).reset_index(drop=True)
df["ocr_text_ocr"] = df["ocr_text_ocr"].fillna("")

# canonical UPPERCASE brand strings keyed by a folded trigger
BRAND_SURFACE = [
    (r"canfoco|halong can", "HALONG CANFOCO"),
    (r"do hop ha long|ha long", "ĐỒ HỘP HẠ LONG"),
    (r"pate.*cot den|cot den.*pate|cot den", "PATÊ CỘT ĐÈN HẢI PHÒNG"),
]


def brand_splice(ocr_text):
    """Replace the best fuzzy-matching span with the canonical brand surface."""
    t = ocr_text
    f = fold(t)
    out = t
    for trig, surf in BRAND_SURFACE:
        if re.search(trig, f):
            # find a window in t whose fold best matches the surface fold
            sf = fold(surf)
            toks = t.split()
            best_i, best_j, best = -1, -1, 70
            for i in range(len(toks)):
                for L in range(1, min(7, len(toks) - i) + 1):
                    span = " ".join(toks[i:i + L])
                    sc = fuzz.ratio(fold(span), sf)
                    if sc > best:
                        best, best_i, best_j = sc, i, i + L
            if best_i >= 0:
                out = " ".join(toks[:best_i] + [surf] + toks[best_j:])
                t, f = out, fold(out)
    return out


# dominant-family subset with GT text
dom = df[df["ocr_text_gt"].map(fold).str.contains(r"pate|cot den|canfoco|do hop ha long", regex=True)]
dom = dom[dom["ocr_text_gt"].str.strip() != ""].copy()

cer_base = dom.apply(lambda r: cer(r["ocr_text_gt"], r["ocr_text_ocr"]), axis=1).mean()
dom["spliced"] = dom["ocr_text_ocr"].map(brand_splice)
cer_splice = dom.apply(lambda r: cer(r["ocr_text_gt"], r["spliced"]), axis=1).mean()

print(f"dominant-family train images w/ GT text: {len(dom)}")
print(f"  CER baseline (ft OCR)      : {cer_base:.4f}")
print(f"  CER after brand-splice     : {cer_splice:.4f}   (delta {cer_splice-cer_base:+.4f})")
print(f"  -> ocr_term delta on these : {(cer_base-cer_splice):+.4f}")
print(f"  -> fraction of full train  : {len(dom)/len(df):.1%}  "
      f"(composite impact ~ 0.4 * {len(dom)/len(df):.2f} * {(cer_base-cer_splice):+.4f} = "
      f"{0.4*len(dom)/len(df)*(cer_base-cer_splice):+.4f})")

# upper bound: if brand tokens were PERFECT, what's the floor CER?
print("\nCER distribution on dominant-family (where is error concentrated?):")
c = dom.apply(lambda r: cer(r["ocr_text_gt"], r["ocr_text_ocr"]), axis=1)
for lo, hi in [(0, .1), (.1, .3), (.3, .6), (.6, 1.01)]:
    m = (c >= lo) & (c < hi)
    print(f"  CER [{lo:.1f},{hi:.1f}): {m.sum():>4} imgs ({m.mean():.0%})  mean CER {c[m].mean() if m.any() else 0:.3f}")
print("  (CER>0.6 band = total OCR failures; rules can't reconstruct full news sentences)")
