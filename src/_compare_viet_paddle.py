"""Definitive CER comparison: VietOCR vs PaddleOCR on the val split (now that
ocr_vietocr_all.parquet is available). Raw + diacritic-folded CER on shared val images."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from data import load_train_labels, load_split_ids
from scoring import cer
from product_extract import fold
from run_ocr import cache_path

gt = load_train_labels().set_index("image_id")["ocr_text"]
va_ids = load_split_ids("val")
viet = pd.read_parquet(cache_path("vietocr", "all")).set_index("image_id")["ocr_text"].fillna("")
padd = pd.read_parquet(cache_path("paddleocr", "all")).set_index("image_id")["ocr_text"].fillna("")

ids = sorted(va_ids & set(viet.index) & set(padd.index))
print(f"VietOCR all rows: {len(viet)} | comparing on {len(ids)} shared val images\n")


def metrics(s):
    raw = np.mean([cer(gt[i], s.get(i, "")) for i in ids])
    fld = np.mean([cer(fold(gt[i]), fold(s.get(i, ""))) for i in ids])
    fill = np.mean([1 if str(s.get(i, "")).strip() else 0 for i in ids])
    return raw, fld, fill


for name, s in [("PaddleOCR", padd.to_dict()), ("VietOCR", viet.to_dict())]:
    raw, fld, fill = metrics(s)
    print(f"{name:>10}: CER {raw:.4f} (ocr_term {1-raw:.4f}) | folded CER {fld:.4f} | fill {fill:.1%}")
