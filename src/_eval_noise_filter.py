"""Prototype OCR noise filters and measure CER on val cache (no re-OCR).
Also reports how often GT contains each noise pattern (safety check before stripping).
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from data import load_train_labels, load_split_ids
from scoring import cer
from run_ocr import cache_path

labels = load_train_labels()
va = labels[labels.image_id.isin(load_split_ids("val"))][["image_id", "ocr_text"]]
ocr = pd.read_parquet(cache_path("paddleocr", "val"))[["image_id", "ocr_text"]]
df = va.merge(ocr, on="image_id", suffixes=("_gt", "_pred"))
df["ocr_text_pred"] = df["ocr_text_pred"].fillna("")

# ---- noise patterns ----
RE_HANDLE = re.compile(r"^@\w+")
RE_URL = re.compile(r"(www\.|https?://|\.vn$|\.com|\.net)", re.I)
RE_DATE = re.compile(r"^\d{1,2}[/.-]\d{1,2}([/.-]\d{2,4})?$|^\d{1,2}:\d{2}$")
RE_CODE = re.compile(r"^(iso|fssc|issn)\d|^\d{4,}$", re.I)


def gt_pattern_rates(gt_series):
    out = {}
    for name, fn in [
        ("@handle", lambda t: any(RE_HANDLE.match(w) for w in t.split())),
        ("url/domain", lambda t: bool(RE_URL.search(t))),
        ("date/time", lambda t: any(RE_DATE.match(w) for w in t.split())),
        ("code/iso", lambda t: any(RE_CODE.match(w) for w in t.split())),
    ]:
        out[name] = sum(1 for t in gt_series if t and fn(str(t))) / max((gt_series != "").sum(), 1)
    return out


def collapse_repeats(text: str) -> str:
    """Collapse repeated 1- to 3-grams (e.g. 'BRAINZ MEAT BRAINZ MEAT')."""
    toks = text.split()
    for n in (3, 2, 1):
        i = 0
        out = []
        while i < len(toks):
            if (i + 2 * n <= len(toks)
                    and [w.lower() for w in toks[i:i+n]] == [w.lower() for w in toks[i+n:i+2*n]]):
                # skip the duplicate block; keep advancing past further repeats
                out.extend(toks[i:i+n])
                i += n
                while (i + n <= len(toks)
                       and [w.lower() for w in toks[i-n:i]] == [w.lower() for w in toks[i:i+n]]):
                    i += n
            else:
                out.append(toks[i]); i += 1
        toks = out
    return " ".join(toks)


def filter_tokens(text: str) -> str:
    keep = []
    for w in text.split():
        if RE_HANDLE.match(w) or RE_URL.search(w) or RE_DATE.match(w) or RE_CODE.match(w):
            continue
        keep.append(w)
    return " ".join(keep)


def full_filter(text: str) -> str:
    return collapse_repeats(filter_tokens(text))


print("GT noise-pattern rates (how often GT has them — want LOW to strip safely):")
for k, v in gt_pattern_rates(df.ocr_text_gt).items():
    print(f"  {k:>12}: {v:.2%}")

base_cer = df.apply(lambda r: cer(r.ocr_text_gt, r.ocr_text_pred), axis=1).mean()
tok_cer = df.apply(lambda r: cer(r.ocr_text_gt, filter_tokens(r.ocr_text_pred)), axis=1).mean()
rep_cer = df.apply(lambda r: cer(r.ocr_text_gt, collapse_repeats(r.ocr_text_pred)), axis=1).mean()
full_cer = df.apply(lambda r: cer(r.ocr_text_gt, full_filter(r.ocr_text_pred)), axis=1).mean()

print(f"\nCER  baseline (cached)        : {base_cer:.4f}  (ocr_term {1-base_cer:.4f})")
print(f"CER  + token filter           : {tok_cer:.4f}  (ocr_term {1-tok_cer:.4f})")
print(f"CER  + repeat collapse        : {rep_cer:.4f}  (ocr_term {1-rep_cer:.4f})")
print(f"CER  + both (full filter)     : {full_cer:.4f}  (ocr_term {1-full_cer:.4f})")
