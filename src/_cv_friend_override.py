"""Cherry-pick test: take OUR winning calibrated head, then graft ONLY friend's
evidence_override_product_from_ocr on top (conditional canonical emission). Does
that one idea add anything to our 0.6685 head? Same OCR -> composite gap = pure F1.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data import load_train_labels
from empty_gate import EmptyGate
from product_calibrated import CalibratedRuleHead
from run_ocr import cache_path
from scoring import cer, token_f1

import friend_dispatcher as fd


def _gkey(text, image_id):
    t = " ".join(str(text).lower().split())
    return "empty_" + str(image_id) if not t else hashlib.md5(t.encode("utf-8")).hexdigest()


labels = load_train_labels()
ocr = pd.read_parquet(cache_path("vietocr_ft", "all"))
df = labels.merge(ocr, on="image_id", suffixes=("_gt", "_ocr")).reset_index(drop=True)
df["ocr_text_ocr"] = df["ocr_text_ocr"].fillna("")
groups = [_gkey(t, i) for i, t in zip(df.image_id, df.ocr_text_gt)]
gkf = GroupKFold(n_splits=5)

gt, base_pred, ovr_pred, pooled_cer = [], [], [], []
for tr_idx, va_idx in gkf.split(df, groups=groups):
    trd, vad = df.iloc[tr_idx], df.iloc[va_idx]
    tr_fit = trd.rename(columns={"ocr_text_gt": "ocr_text"})[["image_id", "ocr_text", "product_name"]]
    head = CalibratedRuleHead(use_classifier_fallback=True, min_pprod=0.55,
                              gate_threshold=0.75).fit(tr_fit)
    tg = trd.copy(); tg["gt_empty"] = (tg.ocr_text_gt.str.strip() == "").astype(int)
    eg = EmptyGate(threshold=0.6).fit(tg.rename(columns={"ocr_text_ocr": "ocr_text"}), tg.gt_empty)
    mask = eg.is_empty(vad.rename(columns={"ocr_text_ocr": "ocr_text"}))
    ocr_in = list(vad.ocr_text_ocr.where(~np.asarray(mask), ""))
    preds = head.predict_batch(ocr_in)
    base_pred += list(preds)
    ovr_pred += [fd.evidence_override_product_from_ocr(t, p) for t, p in zip(ocr_in, preds)]
    gt += list(vad.product_name)
    pooled_cer += [cer(g, p) for g, p in zip(vad.ocr_text_gt, ocr_in)]

ot = 1 - np.mean(pooled_cer)
f1_base = np.mean([token_f1(g, p) for g, p in zip(gt, base_pred)])
f1_ovr = np.mean([token_f1(g, p) for g, p in zip(gt, ovr_pred)])
n_changed = sum(1 for a, b in zip(base_pred, ovr_pred) if str(a) != str(b))

print("Our calibrated head, with vs without friend's evidence_override graft:\n")
print(f"{'variant':<28}{'composite':>10}{'prod_F1':>9}")
print(f"{'ours (baseline)':<28}{0.6*f1_base+0.4*ot:>10.4f}{f1_base:>9.4f}")
print(f"{'ours + evidence_override':<28}{0.6*f1_ovr+0.4*ot:>10.4f}{f1_ovr:>9.4f}")
print(f"\noverride changed {n_changed} predictions | "
      f"composite delta {0.6*(f1_ovr-f1_base):+.4f}")
