"""Where is the calibrated head LOSING product F1 on real OCR, and is any of it
recoverable with a new/expanded brand rule (no better OCR needed)?

For each train image: predict on REAL vietocr_ft OCR, score token_f1 vs GT product.
Rank the biggest losses by GT product family -> shows exactly which families to
add rules for, and whether the brand token is even present in our OCR (fixable)
or absent (OCR-limited, not fixable by rules).
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from data import load_train_labels
from product_calibrated import CalibratedRuleHead, fold
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

print(f"train images: {len(df)}")
print(f"mean product token-F1 (real OCR): {df['f1'].mean():.4f}")
print(f"  rows GT non-empty: {(df['product_name'].str.strip()!='').sum()}")
print(f"  rows pred non-empty: {(df['pred'].str.strip()!='').sum()}\n")

# ---- biggest losses by GT product family ----
loss = df[(df["product_name"].str.strip() != "") & (df["f1"] < 0.999)].copy()
loss["lost"] = 1.0 - loss["f1"]
# does our OCR contain the GT brand head-token? (is the loss recoverable?)
loss["gt_head"] = loss["product_name"].map(lambda s: " ".join(_fold(s).split()[:2]))
loss["focr"] = loss["ocr_text_ocr"].map(_fold)
loss["brand_in_ocr"] = [
    bool(h) and all(w in f for w in h.split())
    for h, f in zip(loss["gt_head"], loss["focr"])
]

print("=== Top GT product families by TOTAL lost F1 (real OCR) ===")
print(f"{'lostF1':>8} {'n':>4} {'meanF1':>7} {'brand_in_ocr%':>13}  product_name")
g = loss.groupby("product_name").agg(
    lost=("lost", "sum"), n=("lost", "size"), meanf1=("f1", "mean"),
    binocr=("brand_in_ocr", "mean")).sort_values("lost", ascending=False)
for name, r in g.head(25).iterrows():
    print(f"{r['lost']:>8.2f} {int(r['n']):>4} {r['meanf1']:>7.3f} {r['binocr']*100:>12.0f}%  {name[:60]}")

# ---- the recoverable subset: brand IS in our OCR but we still lose F1 ----
rec = loss[loss["brand_in_ocr"]]
print(f"\nRecoverable (brand token present in OCR but F1<1): {len(rec)} rows, "
      f"total lost F1 {rec['lost'].sum():.1f}")
print("  -> these are addressable by a rule/canonical-string fix (no better OCR needed)")
print(f"Unrecoverable (brand token ABSENT from OCR): {len(loss)-len(rec)} rows, "
      f"total lost F1 {loss['lost'].sum()-rec['lost'].sum():.1f}")
print("  -> OCR-limited; rules can't help (the brand isn't in the text we read)")

# composite ceiling if we recovered ALL the 'brand_in_ocr' losses to F1=1
gain_f1 = rec["lost"].sum() / len(df)
print(f"\nMax product-F1 lift if ALL recoverable rows hit F1=1: +{gain_f1:.4f} "
      f"-> composite +{0.6*gain_f1:.4f} (upper bound, optimistic)")
