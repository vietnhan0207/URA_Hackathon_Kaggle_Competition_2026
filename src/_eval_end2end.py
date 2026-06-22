"""First real end-to-end val composite: PaddleOCR text (CER) + product preds on it.
Trains product head on GT train text (train-OCR not ready yet) and predicts on the
REAL OCR'd val text. Compares product F1 on oracle vs real OCR.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from data import load_train_labels, load_split_ids
from scoring import composite_score, token_f1
from product_extract import ProductExtractor
from run_ocr import cache_path

labels = load_train_labels()
tr = labels[labels.image_id.isin(load_split_ids("train"))].reset_index(drop=True)
va = labels[labels.image_id.isin(load_split_ids("val"))].reset_index(drop=True)

ocr_val = pd.read_parquet(cache_path("paddleocr", "val"))[["image_id", "ocr_text"]]
ocr_val = ocr_val.rename(columns={"ocr_text": "ocr_pred"})
va = va.merge(ocr_val, on="image_id", how="left")
va["ocr_pred"] = va["ocr_pred"].fillna("")

ext = ProductExtractor(min_class_count=5, gate_threshold=0.45).fit(tr)


def pf1(preds, gts):
    return float(pd.Series([token_f1(g, p) for g, p in zip(gts, preds)]).mean())


# product F1: oracle OCR vs real OCR text as classifier input
f1_oracle = pf1(ext.predict_batch(va.ocr_text), va.product_name)
f1_real = pf1(ext.predict_batch(va.ocr_pred), va.product_name)
print(f"product F1  | oracle-OCR input: {f1_oracle:.4f} | real-OCR input: {f1_real:.4f}")

# full end-to-end composite on val (real OCR text + product from real OCR text)
sub = pd.DataFrame({
    "image_id": va.image_id,
    "ocr_text": va.ocr_pred,
    "product_name": ext.predict_batch(va.ocr_pred),
})
gt = va[["image_id", "ocr_text", "product_name"]]
print("\nEND-TO-END val composite (PaddleOCR + product head):")
print(" ", composite_score(gt, sub, return_components=True))

# reference points on val
empty = gt.copy(); empty["ocr_text"] = ""; empty["product_name"] = ""
print("  all-empty:", composite_score(gt, empty, return_components=True))
ocr_only = sub.copy(); ocr_only["product_name"] = ""
print("  OCR-only (empty product):", composite_score(gt, ocr_only, return_components=True))
