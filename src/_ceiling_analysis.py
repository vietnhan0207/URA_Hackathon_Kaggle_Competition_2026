"""Quantify what it takes to reach >0.7: decompose the score and show the value of
each lever (perfect OCR vs perfect product) on the cached val. Also re-verifies the
metric (oracle=1.0, all-empty reference)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from data import load_train_labels, load_split_ids
from scoring import composite_score, token_f1, cer
from product_extract import ProductExtractor
from run_ocr import cache_path

labels = load_train_labels()
allc = pd.read_parquet(cache_path("paddleocr", "all"))
allc["n_chars"] = allc["ocr_text"].fillna("").str.len()
cached = set(allc.image_id)
tr_ids, va_ids = load_split_ids("train"), load_split_ids("val")
tr = labels[labels.image_id.isin(tr_ids)]
va = labels[labels.image_id.isin(va_ids & cached)].merge(
    allc[["image_id", "ocr_text", "mean_conf", "n_boxes", "n_chars"]], on="image_id",
    suffixes=("_gt", "_ocr"))

# --- metric re-verification ---
print("METRIC CHECK:")
print("  oracle (GT==pred):", composite_score(
    va.rename(columns={"ocr_text_gt": "ocr_text"})[["image_id", "ocr_text", "product_name"]],
    va.rename(columns={"ocr_text_gt": "ocr_text"})[["image_id", "ocr_text", "product_name"]]))
emp = va[["image_id"]].copy(); emp["ocr_text"] = ""; emp["product_name"] = ""
gtv = va.rename(columns={"ocr_text_gt": "ocr_text"})[["image_id", "ocr_text", "product_name"]]
print("  all-empty:", composite_score(gtv, emp))

# --- best current pipeline ---
ext = ProductExtractor(min_class_count=12, gate_threshold=0.55).fit(tr)
preds = ext.predict_batch(va.ocr_text_ocr)
sub = pd.DataFrame({"image_id": va.image_id, "ocr_text": va.ocr_text_ocr, "product_name": preds})
cur = composite_score(gtv, sub, return_components=True)
print(f"\nCURRENT (PaddleOCR + product head): {cur}")

# --- lever values ---
# perfect OCR (CER=0), keep our product preds:
perfect_ocr = sub.copy(); perfect_ocr["ocr_text"] = va.ocr_text_gt.values
print("  if OCR were PERFECT (keep our product):", composite_score(gtv, perfect_ocr, return_components=True))
# perfect product, keep our OCR:
perfect_prod = sub.copy(); perfect_prod["product_name"] = va.product_name.values
print("  if PRODUCT were PERFECT (keep our OCR):", composite_score(gtv, perfect_prod, return_components=True))

# --- empty vs non-empty breakdown ---
ne = va[va.ocr_text_gt.str.strip() != ""]
ne_cer = float(np.mean([cer(g, p) for g, p in zip(ne.ocr_text_gt, ne.ocr_text_ocr)]))
nep = va[va.product_name.str.strip() != ""]
nep_f1 = float(np.mean([token_f1(g, p) for g, p in
                        zip(nep.product_name, ext.predict_batch(nep.ocr_text_ocr))]))
print(f"\nNON-EMPTY rows: OCR CER={ne_cer:.3f} (n={len(ne)}) | product F1={nep_f1:.3f} (n={len(nep)})")
print(f"empty-OCR GT rate={ (va.ocr_text_gt.str.strip()=='').mean():.1%} | "
      f"empty-product GT rate={ (va.product_name.str.strip()=='').mean():.1%}")

# --- target: what CER/F1 hit 0.70? ---
print("\nTo reach composite 0.70 (illustrative):")
for f1 in (0.65, 0.70, 0.75):
    need_cer = 1 - (0.70 - 0.6 * f1) / 0.4
    print(f"  if overall F1={f1:.2f} -> need overall CER <= {need_cer:.3f}")
