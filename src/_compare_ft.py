"""Compare fine-tuned VietOCR vs base VietOCR vs PaddleOCR CER on val images
present in the quick-run FT cache (ocr_vietocr_ft_all.parquet)."""
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

ft = pd.read_parquet(cache_path("vietocr_ft", "all")).set_index("image_id")["ocr_text"].fillna("")
base = pd.read_parquet(cache_path("vietocr", "all")).set_index("image_id")["ocr_text"].fillna("")
padd = pd.read_parquet(cache_path("paddleocr", "all")).set_index("image_id")["ocr_text"].fillna("")

# val images that the FT quick-run actually OCR'd
ids = sorted(va_ids & set(ft.index) & set(base.index) & set(padd.index))
print(f"FT cache rows: {len(ft)} | comparing on {len(ids)} val images\n")


def stats(s):
    raw = np.mean([cer(gt[i], s.get(i, "")) for i in ids])
    fld = np.mean([cer(fold(gt[i]), fold(s.get(i, ""))) for i in ids])
    return raw, fld


for name, s in [("PaddleOCR", padd.to_dict()), ("base VietOCR", base.to_dict()),
                ("FT VietOCR", ft.to_dict())]:
    raw, fld = stats(s)
    print(f"{name:>14}: CER {raw:.4f} (ocr_term {1-raw:.4f}) | folded CER {fld:.4f}")

print("\n--- sample GT vs base vs FT (first 6) ---")
shown = 0
for i in ids:
    if gt[i].strip() and shown < 6:
        print(f"[{i}]")
        print("  GT  :", gt[i][:90])
        print("  base:", base.get(i, '')[:90])
        print("  FT  :", ft.get(i, '')[:90])
        shown += 1
