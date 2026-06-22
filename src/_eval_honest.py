"""Honest end-to-end val composite on the FIXED split + cached PaddleOCR.
Restricts to val images that have OCR cached. Compares product head trained on
clean GT text vs OCR'd train text, and the effect of the empty-gate.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from data import load_train_labels, load_split_ids
from scoring import composite_score
from product_extract import ProductExtractor
from run_ocr import cache_path

labels = load_train_labels()
allc = pd.read_parquet(cache_path("paddleocr", "all"))
allc["n_chars"] = allc["ocr_text"].fillna("").str.len()

tr_ids, va_ids = load_split_ids("train"), load_split_ids("val")
ocr = allc.set_index("image_id")
cached = set(allc.image_id)

# OCR'd train rows (with GT product) and val rows (cached only)
tr = labels[labels.image_id.isin(tr_ids & cached)].merge(
    allc[["image_id", "ocr_text"]], on="image_id", suffixes=("_gt", "_ocr"))
va = labels[labels.image_id.isin(va_ids & cached)].merge(
    allc[["image_id", "ocr_text", "mean_conf", "n_boxes", "n_chars"]],
    on="image_id", suffixes=("_gt", "_ocr"))
print(f"cached train {len(tr)} | cached val {len(va)}")

# empty-gate classifier (predict GT-empty-ocr from features), threshold 0.6
gf = ["log_boxes", "mean_conf", "log_chars"]
m = labels.merge(allc, on="image_id")
m["log_boxes"] = np.log1p(m["n_boxes"]); m["log_chars"] = np.log1p(m["n_chars"])
m["gt_empty"] = (m["ocr_text_x"].fillna("").str.strip() == "").astype(int)
mt = m[m.image_id.isin(tr_ids & cached)]
gate = LogisticRegression(max_iter=1000, class_weight="balanced").fit(mt[gf], mt["gt_empty"])

va2 = m[m.image_id.isin(va_ids & cached)].copy()
va2["p_empty"] = gate.predict_proba(va2[gf])[:, 1]
gated = va2.set_index("image_id")["p_empty"] >= 0.6


def run(train_on_ocr: bool, use_gate: bool):
    train_df = (tr[["image_id", "ocr_text_ocr", "product_name"]]
                .rename(columns={"ocr_text_ocr": "ocr_text"}) if train_on_ocr
                else labels[labels.image_id.isin(tr_ids)][["image_id", "ocr_text", "product_name"]])
    ext = ProductExtractor(min_class_count=5, gate_threshold=0.45).fit(train_df)
    ocr_text = va["ocr_text_ocr"].copy()
    if use_gate:
        g = va["image_id"].map(lambda i: bool(gated.get(i, False))).values
        ocr_text = ocr_text.where(~g, "")
    sub = pd.DataFrame({"image_id": va.image_id, "ocr_text": ocr_text,
                        "product_name": ext.predict_batch(ocr_text)})
    gt = va[["image_id", "ocr_text_gt", "product_name"]].rename(columns={"ocr_text_gt": "ocr_text"})
    return composite_score(gt, sub, return_components=True)


print("\nproduct head on GT text,  no gate :", run(False, False))
print("product head on GT text,  +gate   :", run(False, True))
print("product head on OCR text, no gate :", run(True, False))
print("product head on OCR text, +gate   :", run(True, True))
