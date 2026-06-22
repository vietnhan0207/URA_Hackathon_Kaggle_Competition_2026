"""Head-to-head CER: rapidOCR vs PaddleOCR on the same val images.
PaddleOCR text comes from the 'all' cache (the per-split val cache is pre-split-fix
and stale). Compares on the intersection of images both engines have OCR'd.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from data import load_train_labels
from scoring import cer
from product_extract import fold
from run_ocr import cache_path

gt = load_train_labels().set_index("image_id")["ocr_text"]
rapid = pd.read_parquet(cache_path("rapidocr", "val"))[["image_id", "ocr_text"]]
paddle = pd.read_parquet(cache_path("paddleocr", "all"))[["image_id", "ocr_text"]]

ids = sorted(set(rapid.image_id) & set(paddle.image_id))
print(f"comparing on {len(ids)} shared val images\n")
r = rapid.set_index("image_id")["ocr_text"]
p = paddle.set_index("image_id")["ocr_text"]


def metrics(textmap):
    raw = np.mean([cer(gt[i], textmap.get(i, "")) for i in ids])
    fld = np.mean([cer(fold(gt[i]), fold(textmap.get(i, ""))) for i in ids])
    fill = np.mean([1 if str(textmap.get(i, "")).strip() else 0 for i in ids])
    return raw, fld, fill


for name, tm in [("PaddleOCR", p.to_dict()), ("rapidOCR", r.to_dict())]:
    raw, fld, fill = metrics(tm)
    print(f"{name:>10}: CER {raw:.4f} (ocr_term {1-raw:.4f}) | "
          f"diacritic-folded CER {fld:.4f} | fill {fill:.1%}")
