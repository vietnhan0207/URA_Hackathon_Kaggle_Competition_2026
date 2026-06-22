"""Lightweight domain accent-restoration: build a map fold(word)->modal accented form
from TRAIN GT ocr_text, apply to OCR output to fix Vietnamese diacritics.
Self-contained (no downloads), domain-matched. Measure CER on val.
Tests on both PaddleOCR (best base) and VietOCR.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from data import load_train_labels, load_split_ids
from scoring import cer
from product_extract import fold
from run_ocr import cache_path

labels = load_train_labels()
tr_ids, va_ids = load_split_ids("train"), load_split_ids("val")
gt = labels.set_index("image_id")["ocr_text"]

# build accent map from TRAIN GT only (no val leakage)
word_map = defaultdict(Counter)
for t in labels[labels.image_id.isin(tr_ids)]["ocr_text"]:
    for w in str(t).split():
        word_map[fold(w)][w] += 1
accent = {k: c.most_common(1)[0][0] for k, c in word_map.items() if c}
print(f"accent map: {len(accent)} folded-word keys from train GT\n")


def restore(text: str) -> str:
    out = []
    for w in str(text).split():
        key = fold(w)
        # only restore alphabetic Vietnamese-ish tokens we know; keep others as-is
        out.append(accent.get(key, w) if key in accent else w)
    return " ".join(out)


def eval_engine(engine, ids_set):
    allc = pd.read_parquet(cache_path(engine, "all")).set_index("image_id")["ocr_text"].fillna("")
    ids = sorted(ids_set & set(allc.index))
    base = np.mean([cer(gt[i], allc[i]) for i in ids])
    rest = np.mean([cer(gt[i], restore(allc[i])) for i in ids])
    print(f"{engine:>10} (val {len(ids)}): CER {base:.4f} -> {rest:.4f} "
          f"(ocr_term {1-base:.4f} -> {1-rest:.4f}, delta {base-rest:+.4f})")


eval_engine("paddleocr", va_ids)
eval_engine("vietocr", va_ids)
