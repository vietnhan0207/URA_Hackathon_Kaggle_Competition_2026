"""For the dominant families where the brand IS in our OCR but F1<1, WHAT does the
head emit? trigger-gap (emits '') vs family-collision (emits wrong family).
Decides whether the fix is 'add a trigger' or 'fix disambiguation priority'.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import pandas as pd

from data import load_train_labels
from product_calibrated import CalibratedRuleHead
from run_ocr import cache_path
from scoring import token_f1


def _fold(s):
    s = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


labels = load_train_labels()
ocr = pd.read_parquet(cache_path("vietocr_ft", "all"))[["image_id", "ocr_text"]]
df = labels.merge(ocr, on="image_id", suffixes=("_gt", "_ocr")).reset_index(drop=True)
df["ocr_text_ocr"] = df["ocr_text_ocr"].fillna("")

head = CalibratedRuleHead(use_classifier_fallback=True, min_pprod=0.55,
                          gate_threshold=0.75).fit(
    labels[["image_id", "ocr_text", "product_name"]])
df["pred"] = head.predict_batch(df["ocr_text_ocr"])
df["f1"] = [token_f1(g, p) for g, p in zip(df["product_name"], df["pred"])]
df["focr"] = df["ocr_text_ocr"].map(_fold)
df["gt_head"] = df["product_name"].map(lambda s: " ".join(_fold(s).split()[:2]))
df["brand_in_ocr"] = [bool(h) and all(w in f for w in h.split())
                      for h, f in zip(df["gt_head"], df["focr"])]

FAM = {
    "Ha Long / Canfoco": r"ha long|canfoco|cafoco|do hop",
    "Pate Cot Den":      r"cot den|pate|pate",
    "NAN":               r"\bnan\b",
    "Nestle":            r"nestle",
    "Highlands":         r"highland",
}
for fam, pat in FAM.items():
    sub = df[(df["product_name"].str.strip() != "") &
             (df["product_name"].map(_fold).str.contains(pat, regex=True)) &
             (df["f1"] < 0.999) & (df["brand_in_ocr"])]
    if not len(sub):
        continue
    emits = Counter("<EMPTY>" if not str(p).strip() else p for p in sub["pred"])
    print(f"\n=== {fam}: {len(sub)} brand-present misses (F1<1) — what we EMIT ===")
    for val, c in emits.most_common(6):
        print(f"  {c:>3}  -> {val[:55]}")
    # what GT wants on these
    gts = Counter(sub["product_name"])
    print(f"   GT labels on these rows (top): "
          f"{', '.join(f'{g[:30]}({c})' for g, c in gts.most_common(4))}")
