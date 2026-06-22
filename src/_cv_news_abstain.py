"""Test friend's NEWS-CONTEXT ABSTENTION on our head, train CV. When OCR shows
news/scandal markers, blank Halong/Pate predictions (those are news articles about
the brand, not product posts -> GT empty). Fully train-validatable (no LB needed).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data import load_train_labels
from empty_gate import EmptyGate
from product_calibrated import CalibratedRuleHead
from run_ocr import cache_path
from scoring import token_f1


def fold(s):
    s = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", " ", s)


NEWS = ["batkhan", "giamdoc", "tonggiamdoc", "cong ty", "khoito", "khoi to",
        "tintuc", "tin tuc", "news", "thutuong", "thu tuong", "phathien",
        "phat hien", "dichta", "chauphi", "chau phi", "lentieng", "len tieng",
        "dantri", "vtv", "bao ", "vietnamnet", "trach nhiem", "quan ly nha nuoc"]
HALONG_PATE = re.compile(r"ha long|canfoco|cot den|pate|do hop")


def is_news(folded_ocr):
    return any(m in folded_ocr for m in NEWS)


def _gkey(text, image_id):
    t = " ".join(str(text).lower().split())
    return "empty_" + str(image_id) if not t else hashlib.md5(t.encode("utf-8")).hexdigest()


labels = load_train_labels()
ocr = pd.read_parquet(cache_path("vietocr_ft", "all"))
df = labels.merge(ocr, on="image_id", suffixes=("_gt", "_ocr")).reset_index(drop=True)
df["ocr_text_ocr"] = df["ocr_text_ocr"].fillna("")
groups = [_gkey(t, i) for i, t in zip(df.image_id, df.ocr_text_gt)]
gkf = GroupKFold(n_splits=5)

gt, base_pred, abst_pred = [], [], []
n_blanked, n_blanked_gtempty = 0, 0
for tr_idx, va_idx in gkf.split(df, groups=groups):
    trd, vad = df.iloc[tr_idx], df.iloc[va_idx]
    head = CalibratedRuleHead(use_classifier_fallback=True, min_pprod=0.55,
                              gate_threshold=0.75).fit(
        trd.rename(columns={"ocr_text_gt": "ocr_text"})[["image_id", "ocr_text", "product_name"]])
    tg = trd.copy(); tg["gt_empty"] = (tg.ocr_text_gt.str.strip() == "").astype(int)
    eg = EmptyGate(threshold=0.6).fit(tg.rename(columns={"ocr_text_ocr": "ocr_text"}), tg.gt_empty)
    mask = eg.is_empty(vad.rename(columns={"ocr_text_ocr": "ocr_text"}))
    ocr_in = list(vad.ocr_text_ocr.where(~np.asarray(mask), ""))
    preds = list(head.predict_batch(ocr_in))
    base_pred += preds
    new = []
    for t, p, g in zip(ocr_in, preds, vad.product_name):
        f = fold(t)
        if str(p).strip() and HALONG_PATE.search(fold(p)) and is_news(f):
            new.append("")
            n_blanked += 1
            if str(g).strip() == "":
                n_blanked_gtempty += 1
        else:
            new.append(p)
    abst_pred += new
    gt += list(vad.product_name)

f1_base = np.mean([token_f1(g, p) for g, p in zip(gt, base_pred)])
f1_abst = np.mean([token_f1(g, p) for g, p in zip(gt, abst_pred)])
print("News-context abstention on our calibrated head (train CV, prod-F1 only):\n")
print(f"  baseline               prod_F1 = {f1_base:.4f}")
print(f"  + news abstention      prod_F1 = {f1_abst:.4f}   ({f1_abst-f1_base:+.4f})")
print(f"\nblanked {n_blanked} Halong/Pate preds on news-context rows | "
      f"of those, GT was actually EMPTY on {n_blanked_gtempty} "
      f"({n_blanked_gtempty/max(1,n_blanked):.0%} correct blanks)")
print(f"composite delta ~ {0.6*(f1_abst-f1_base):+.4f}")
