"""Eyeball OCR cache vs GT + CER. Usage: python _inspect_ocr.py <engine> <split>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from data import load_train_labels
from scoring import cer
from run_ocr import cache_path

engine = sys.argv[1] if len(sys.argv) > 1 else "easyocr"
split = sys.argv[2] if len(sys.argv) > 2 else "val"

cache = pd.read_parquet(cache_path(engine, split))
gt = load_train_labels().set_index("image_id")
cers = []
for _, r in cache.iterrows():
    g = gt.loc[r.image_id, "ocr_text"]
    c = cer(g, r.ocr_text)
    cers.append(c)
    print(f"ID {r.image_id} | CER {c:.2f} | conf {r.mean_conf:.2f}")
    print("  GT  :", (g[:100] or "(empty)"))
    print("  PRED:", (r.ocr_text[:100] or "(empty)"))
print(f"\nmean CER on {len(cers)}: {sum(cers)/len(cers):.3f}  (ocr_term {1-sum(cers)/len(cers):.3f})")
