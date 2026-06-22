"""Exact reimplementation of the competition metric.

    Score = 0.6 * mean(token_F1_product) + 0.4 * (1 - mean(CER_ocr))

- token_F1: token-level, case-insensitive set F1 on product_name.
            both empty -> 1.0 ; exactly one empty -> 0.0
- CER:      char error rate (Levenshtein / len(GT)), clamped to 1.0.
            GT empty & pred empty -> 0.0 ; GT empty & pred non-empty -> 1.0

Mirrors `_inline_composite_score` in the baseline notebook (Cell 7).
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from config import W_OCR, W_PRODUCT_F1


def _clean(val) -> str:
    return "" if pd.isna(val) else str(val).strip()


def token_f1(gt, pred) -> float:
    gt, pred = _clean(gt), _clean(pred)
    if not gt and not pred:
        return 1.0
    gt_tokens = set(gt.lower().split())
    pred_tokens = set(pred.lower().split())
    if not gt_tokens or not pred_tokens:
        return 0.0
    common = gt_tokens & pred_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def cer(gt, pred) -> float:
    gt, pred = _clean(gt), _clean(pred)
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    m, n = len(gt), len(pred)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if gt[i - 1] == pred[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return min(dp[n] / len(gt), 1.0)


def composite_score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = "image_id",
    return_components: bool = False,
):
    """Score a submission against ground truth. Returns float, or dict if
    return_components=True (composite, product_f1, ocr_term, cer)."""
    required = {"ocr_text", "product_name"}
    if not required.issubset(solution.columns) or not required.issubset(submission.columns):
        raise ValueError("Both frames must contain ocr_text and product_name")
    if submission[row_id_column_name].duplicated().any():
        raise ValueError("Duplicate image_id in submission")
    if set(submission[row_id_column_name]) != set(solution[row_id_column_name]):
        miss = len(set(solution[row_id_column_name]) - set(submission[row_id_column_name]))
        extra = len(set(submission[row_id_column_name]) - set(solution[row_id_column_name]))
        raise ValueError(f"IDs must match exactly (missing {miss}, extra {extra})")

    merged = solution.merge(submission, on=row_id_column_name, suffixes=("_gt", "_pred"))
    if merged.empty:
        raise ValueError("No matching rows after merge")

    product_f1 = merged.apply(
        lambda r: token_f1(r["product_name_gt"], r["product_name_pred"]), axis=1
    ).mean()
    avg_cer = merged.apply(
        lambda r: cer(r["ocr_text_gt"], r["ocr_text_pred"]), axis=1
    ).mean()

    ocr_term = 1.0 - avg_cer
    score = W_PRODUCT_F1 * product_f1 + W_OCR * ocr_term

    if return_components:
        return {
            "composite": round(float(score), 4),
            "product_f1": round(float(product_f1), 4),
            "ocr_term": round(float(ocr_term), 4),
            "avg_cer": round(float(avg_cer), 4),
        }
    return round(float(score), 4)


def per_row_scores(solution: pd.DataFrame, submission: pd.DataFrame,
                   row_id_column_name: str = "image_id") -> pd.DataFrame:
    """Return per-image f1 / cer for error analysis."""
    merged = solution.merge(submission, on=row_id_column_name, suffixes=("_gt", "_pred"))
    merged["f1"] = merged.apply(lambda r: token_f1(r["product_name_gt"], r["product_name_pred"]), axis=1)
    merged["cer"] = merged.apply(lambda r: cer(r["ocr_text_gt"], r["ocr_text_pred"]), axis=1)
    return merged
