"""Optimal canonical surface form per product family, derived from TRAIN.

Problem: our ProductExtractor groups canonical labels by RAW token-set key, so
'patê cột đèn hải phòng' and 'pate cột đèn hải phòng' are different classes
(fragmentation). And it emits the modal surface form, not the token-F1-optimal one.

Fix (train-validated, no LB guessing):
  1. group products by DIACRITIC-FOLDED token-set key (merges ê/e, đ/d variants)
  2. for each group, choose the canonical output string S* that maximizes the
     expected token-F1 over that group's actual GT label distribution:
        S* = argmax_S  sum_g count(g) * token_f1(g, S)
     candidate S ranges over the group's observed surface forms.

Then CV: does remapping our classifier's output to S* raise product F1?
  - oracle (GT OCR) : isolates the canonicalization effect
  - full composite  : deployed estimate on real vietocr_ft OCR
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data import load_train_labels
from empty_gate import EmptyGate
from product_extract import ProductExtractor
from run_ocr import cache_path
from scoring import cer, token_f1


def fold_key(s: str) -> str:
    """Diacritic-folded, sorted-token key (merges spelling/diacritic variants)."""
    s = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = sorted(re.sub(r"\s+", " ", s).strip().split())
    return " ".join(toks)


def build_optimal_canon(train: pd.DataFrame) -> dict[str, str]:
    """fold_key -> token-F1-optimal surface form, from train label distribution."""
    nz = train[train.product_name.str.strip() != ""].copy()
    nz["fk"] = nz.product_name.map(fold_key)
    canon = {}
    for fk, grp in nz.groupby("fk"):
        forms = Counter(grp.product_name.str.strip())
        # candidate outputs = observed surface forms in this family
        best_s, best_score = None, -1.0
        for cand in forms:
            exp = sum(c * token_f1(g, cand) for g, c in forms.items())
            if exp > best_score:
                best_score, best_s = exp, cand
        canon[fk] = best_s
    return canon


def remap(label: str, canon: dict[str, str]) -> str:
    if not str(label).strip():
        return ""
    return canon.get(fold_key(label), label)


def _gkey(text, image_id):
    t = " ".join(str(text).lower().split())
    return "empty_" + str(image_id) if not t else hashlib.md5(t.encode("utf-8")).hexdigest()


def run():
    labels = load_train_labels()

    # show what the optimal map does for the dominant families
    canon_full = build_optimal_canon(labels)
    print("Optimal canonical forms for dominant families (fold_key -> S*):")
    for fk in sorted(canon_full, key=lambda k: -len(k))[:0]:
        pass
    nz = labels[labels.product_name.str.strip() != ""]
    top_fk = Counter(nz.product_name.map(fold_key))
    for fk, c in top_fk.most_common(10):
        forms = Counter(nz[nz.product_name.map(fold_key) == fk].product_name.str.strip())
        print(f"  [{c:>4}] -> '{canon_full[fk]}'   (variants: {dict(list(forms.items())[:3])})")

    # ---------- CV: oracle product F1, current vs remapped ----------
    groups = [_gkey(t, i) for i, t in zip(labels.image_id, labels.ocr_text)]
    gkf = GroupKFold(n_splits=5)
    pg, p_cur, p_opt = [], [], []
    for tr_idx, va_idx in gkf.split(labels, groups=groups):
        trd, vad = labels.iloc[tr_idx], labels.iloc[va_idx]
        ext = ProductExtractor(min_class_count=12, gate_threshold=0.55).fit(trd)
        canon = build_optimal_canon(trd)
        cur = ext.predict_batch(vad.ocr_text)         # GT OCR
        opt = [remap(c, canon) for c in cur]
        pg += list(vad.product_name); p_cur += cur; p_opt += opt
    f1_cur = np.mean([token_f1(g, p) for g, p in zip(pg, p_cur)])
    f1_opt = np.mean([token_f1(g, p) for g, p in zip(pg, p_opt)])
    print(f"\nORACLE product F1 (GT OCR): current {f1_cur:.4f} -> optimal-canon {f1_opt:.4f}  ({f1_opt-f1_cur:+.4f})")

    # ---------- CV: full composite on real vietocr_ft OCR ----------
    ocr = pd.read_parquet(cache_path("vietocr_ft", "all"))
    df = labels.merge(ocr, on="image_id", suffixes=("_gt", "_ocr")).reset_index(drop=True)
    df["ocr_text_ocr"] = df["ocr_text_ocr"].fillna("")
    groups = [_gkey(t, i) for i, t in zip(df.image_id, df.ocr_text_gt)]
    for tag, use_opt in [("current", False), ("optimal-canon", True)]:
        pooled_gt, pooled_pred, pooled_cer = [], [], []
        for tr_idx, va_idx in gkf.split(df, groups=groups):
            trd, vad = df.iloc[tr_idx], df.iloc[va_idx]
            ext = ProductExtractor(min_class_count=12, gate_threshold=0.55).fit(
                trd.rename(columns={"ocr_text_gt": "ocr_text"})[["image_id", "ocr_text", "product_name"]])
            canon = build_optimal_canon(trd.rename(columns={"ocr_text_gt": "ocr_text"}))
            tg = trd.copy(); tg["gt_empty"] = (tg.ocr_text_gt.str.strip() == "").astype(int)
            eg = EmptyGate(threshold=0.6).fit(tg.rename(columns={"ocr_text_ocr": "ocr_text"}), tg.gt_empty)
            mask = eg.is_empty(vad.rename(columns={"ocr_text_ocr": "ocr_text"}))
            ocr_in = vad.ocr_text_ocr.where(~np.asarray(mask), "")
            preds = ext.predict_batch(ocr_in)
            if use_opt:
                preds = [remap(p, canon) for p in preds]
            pooled_pred += list(preds); pooled_gt += list(vad.product_name)
            pooled_cer += [cer(g, p) for g, p in zip(vad.ocr_text_gt, ocr_in)]
        f1 = np.mean([token_f1(g, p) for g, p in zip(pooled_gt, pooled_pred)])
        ot = 1 - np.mean(pooled_cer)
        print(f"  {tag:<14}: composite {0.6*f1+0.4*ot:.4f} | F1 {f1:.4f} | ocr_term {ot:.4f}")


if __name__ == "__main__":
    run()
