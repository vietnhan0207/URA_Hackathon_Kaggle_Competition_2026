"""Investigate the OCR-correction lever: for dominant-family images, how much of
the CER comes from the brand span, and is the GT OCR brand surface form
consistent enough to correct toward?

CER is case- AND diacritic-sensitive (scoring.cer does no normalization), so the
correction must match GT casing/diacritics to help.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from data import load_train_labels
from run_ocr import cache_path
from scoring import cer


def fold(s):
    s = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


labels = load_train_labels()
ocr = pd.read_parquet(cache_path("vietocr_ft", "all"))[["image_id", "ocr_text"]]
df = labels.merge(ocr, on="image_id", suffixes=("_gt", "_ocr"))
df["fgt"] = df["ocr_text_gt"].map(fold)

# pate / canfoco family images
fam = df[df["fgt"].str.contains(r"pate|cot den|canfoco|do hop ha long", regex=True, na=False)]
fam = fam[fam["ocr_text_gt"].str.strip() != ""]
print(f"dominant-family train images with GT text: {len(fam)}")
print(f"mean CER (our ft OCR vs GT) on these: "
      f"{fam.apply(lambda r: cer(r['ocr_text_gt'], r['ocr_text_ocr']), axis=1).mean():.4f}")
print(f"mean GT ocr length: {fam['ocr_text_gt'].str.len().mean():.0f} chars\n")

print("=== 12 examples: GT ocr_text  ||  our OCR  (product) ===")
for _, r in fam.head(12).iterrows():
    print(f"GT : {r['ocr_text_gt'][:95]}")
    print(f"OCR: {r['ocr_text_ocr'][:95]}")
    print(f"     product={r['product_name']!r}  CER={cer(r['ocr_text_gt'], r['ocr_text_ocr']):.3f}")
    print()

# how often does the GT ocr_text literally contain a canonical brand string?
print("=== GT ocr_text brand-surface-form frequency (exact substring counts) ===")
for canon in ["Đồ hộp Hạ Long", "ĐỒ HỘP HẠ LONG", "Pate Cột Đèn Hải Phòng",
              "PATÊ CỘT ĐÈN HẢI PHÒNG", "Halong Canfoco", "HALONG CANFOCO",
              "Cột Đèn", "CỘT ĐÈN", "Hạ Long", "HẠ LONG"]:
    n = df["ocr_text_gt"].str.contains(re.escape(canon)).sum()
    print(f"  {n:>4}  '{canon}'")
