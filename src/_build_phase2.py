"""Build the PHASE-2 submission: our calibrated head (fit on unchanged train) +
empty-gate, applied to the phase-2 OCR parquet. Reports distribution so we see
whether phase-2 content matches train (Halong/Pate/NAN) or is more general.
"""
import re
import unicodedata

import pandas as pd

import config
from data import load_train_labels
from empty_gate import EmptyGate
from product_calibrated import CalibratedRuleHead
from run_ocr import cache_path

import config as _cfg
P2 = str(_cfg.CACHE_DIR / "ocr_vietocr_ft_phase2test.parquet")


def fold(s):
    s = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return re.sub(r"[^a-z0-9 ]", " ", s)


labels = load_train_labels()
head = CalibratedRuleHead(use_classifier_fallback=True, min_pprod=0.55,
                          gate_threshold=0.75).fit(
    labels[["image_id", "ocr_text", "product_name"]])

# empty-gate fit on train OCR
feats = labels.merge(pd.read_parquet(cache_path("vietocr_ft", "all")), on="image_id")
feats["gt_empty"] = (feats["ocr_text_x"].fillna("").str.strip() == "").astype(int)
gate = EmptyGate(threshold=0.6).fit(
    feats.rename(columns={"ocr_text_y": "ocr_text"}), feats["gt_empty"])

ocr = pd.read_parquet(P2)
sub = ocr[["image_id", "ocr_text"]].copy()
sub["ocr_text"] = sub["ocr_text"].fillna("")
if "mean_conf" not in sub: sub["mean_conf"], sub["n_boxes"] = 0.0, 0
sub = sub.merge(ocr[["image_id", "mean_conf", "n_boxes"]], on="image_id", how="left", suffixes=("", "_y"))
mask = gate.is_empty(sub)
sub.loc[mask, "ocr_text"] = ""
print(f"empty-gate zeroed {int(mask.sum())} of {len(sub)} OCR rows")

sub["product_name"] = head.predict_batch(sub["ocr_text"])

# distribution diagnostics
f = sub["ocr_text"].map(fold)
dom = f.str.contains(r"canfoco|cot den|\bnan\b|nestle|highland|pate|ha long", regex=True)
print(f"\nphase-2 rows whose OCR contains a DOMINANT-family marker: {int(dom.sum())} / {len(sub)} ({dom.mean():.0%})")
print(f"product fill: {(sub.product_name.str.strip()!='').mean():.1%}")
print("\ntop emitted products:")
vc = sub[sub.product_name.str.strip() != ""].product_name.value_counts().head(15)
for n, c in vc.items(): print(f"  {c:>4}  {n}")

import csv as _csv


def to_brand(product):
    """Best-effort brand from our product string (NO brand labels in train yet)."""
    f = fold(product)
    if not f.strip():
        return ""
    if re.search(r"nestle|\bnan\b|optipro|infinipro|milo|beba", f):
        return "Nestlé"
    if re.search(r"ha long|halong|canfoco|cot den|pate|do hop", f):
        return "Ha Long Canfoco"   # host-documented brand name
    if "highland" in f:
        return "Highlands Coffee"
    if "aptamil" in f:
        return "Aptamil"
    return product  # long tail: brand == product (best guess without labels)


out = sub[["image_id", "ocr_text", "product_name"]].copy()
out["brand_name"] = out["product_name"].map(to_brand)
# Phase-2 column order: image_id, ocr_text, brand_name, product_name
out = out[["image_id", "ocr_text", "brand_name", "product_name"]]
# Kaggle validator rejects truly-empty cells: use ' ' (space) like the accepted
# 0.6685 file, and QUOTE_ALL. ' ' scores identically to '' (token_f1/CER treat
# it as empty) but passes validation.
for col in ["ocr_text", "brand_name", "product_name"]:
    out[col] = out[col].fillna("").astype(str)
    out.loc[out[col].str.strip() == "", col] = " "
path = config.SUBMISSIONS_DIR / "submission_phase2_calibrated.csv"
out.to_csv(path, index=False, encoding="utf-8", quoting=_csv.QUOTE_ALL)
print(f"\nwrote {path} | rows {len(out)} | cols {list(out.columns)}")
print(f"brand fill={(out.brand_name != ' ').mean():.1%} | product fill={(out.product_name != ' ').mean():.1%}")
