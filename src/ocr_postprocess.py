"""OCR text post-processing: normalization, dedupe, and empty-gate.

Kept engine-agnostic so every OCR backend shares the same cleaning, and so the
final Kaggle notebook can inline these pure functions unchanged.
"""
from __future__ import annotations

import re
import unicodedata

from config import MAX_OCR_LEN


def normalize_text(text: str) -> str:
    """NFC unicode (preserve Vietnamese diacritics), collapse whitespace,
    strip newlines/tabs. Does NOT change letters/casing."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedupe_consecutive_tokens(text: str) -> str:
    """Drop consecutive duplicate tokens (common in thumbnails where the same
    word is detected twice). Case-insensitive comparison."""
    tokens = text.split()
    if not tokens:
        return ""
    out = [tokens[0]]
    for tok in tokens[1:]:
        if tok.lower() != out[-1].lower():
            out.append(tok)
    return " ".join(out)


def clean_ocr(text: str, max_len: int = MAX_OCR_LEN) -> str:
    """Full cleaning pipeline applied to every engine's raw output."""
    text = normalize_text(text)
    text = dedupe_consecutive_tokens(text)
    if max_len and len(text) > max_len:
        text = text[:max_len].rstrip()
    return text


def is_empty_text(
    text: str,
    mean_conf: float | None = None,
    n_boxes: int | None = None,
    *,
    min_chars: int = 2,
    min_conf: float = 0.0,
    min_boxes: int = 0,
) -> bool:
    """Empty-gate: decide whether to emit '' for ocr_text.

    Emitting '' when GT is empty scores CER=0 (perfect); emitting garbage on a
    blank/noise frame scores CER=1. Thresholds tuned on val in Phase 3 — defaults
    here are permissive (only truly empty / sub-min_chars text is gated)."""
    t = (text or "").strip()
    if len(t) < min_chars:
        return True
    if mean_conf is not None and min_conf > 0 and mean_conf < min_conf:
        return True
    if n_boxes is not None and min_boxes > 0 and n_boxes < min_boxes:
        return True
    return False
