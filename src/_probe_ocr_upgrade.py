"""Local CPU probe: does (preprocessing) and/or (beam search) lower CER vs our
cached vietocr_ft OCR? Replicates the ft pipeline (CRAFT detect + vgg_transformer
recognizer @ cache/vietocr_ft.pth) on a train subset and measures CER vs GT.

Variants on the SAME images:
  base       : reproduce ft (no prep, beamsearch off)        -> sanity vs cache
  prep       : contrast x1.35 + sharpen before detect+rec
  beam       : beamsearch on
  prep+beam  : both

Run (env `ura`):  PYTHONIOENCODING=utf-8 python _probe_ocr_upgrade.py [N]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --- NumPy 2.x shims VietOCR/imgaug expect ---
if not hasattr(np, "sctypes"):
    np.sctypes = {"int": [np.int8, np.int16, np.int32, np.int64],
                  "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
                  "float": [np.float16, np.float32, np.float64],
                  "complex": [np.complex64, np.complex128],
                  "others": [bool, object, bytes, str, np.void]}
for _n, _t in [("bool", bool), ("object", object), ("int", int), ("float", float), ("str", str), ("complex", complex)]:
    if not hasattr(np, _n):
        setattr(np, _n, _t)

import cv2
from PIL import Image, ImageEnhance, ImageFilter

from config import TRAIN_IMAGES_DIR
from data import load_train_labels
from ocr_postprocess import clean_ocr
from scoring import cer

N = int(sys.argv[1]) if len(sys.argv) > 1 else 250
WEIGHTS = str(Path(__file__).resolve().parent.parent / "cache" / "vietocr_ft.pth")

import easyocr
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

print("loading detector (CRAFT) + recognizers ...")
detector = easyocr.Reader(["vi", "en"], gpu=False, verbose=False)


def make_rec(beam: bool):
    cfg = Cfg.load_config_from_name("vgg_transformer")
    cfg["device"] = "cpu"
    cfg["weights"] = WEIGHTS
    cfg["cnn"]["pretrained"] = False
    cfg["predictor"]["beamsearch"] = beam
    return Predictor(cfg)


rec_greedy = make_rec(False)
rec_beam = make_rec(True)


def detect_boxes(img):
    try:
        horiz, _ = detector.detect(img, min_size=15, text_threshold=0.6)
        return [tuple(int(v) for v in b) for b in (horiz[0] if horiz else [])]
    except Exception:
        return []


def preprocess(bgr, max_dim=1280):
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    w, h = img.size
    if max(w, h) > max_dim:
        r = max_dim / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.35).filter(ImageFilter.SHARPEN)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def ocr_one(path, prep: bool, beam: bool):
    bgr = cv2.imread(str(path))
    if bgr is None:
        return ""
    if prep:
        bgr = preprocess(bgr)
    items = []
    for (x0, x1, y0, y1) in detect_boxes(bgr):
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(bgr.shape[1], x1), min(bgr.shape[0], y1)
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue
        crop = Image.fromarray(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        items.append((y0, x0, crop))
    if not items:
        return ""
    rec = rec_beam if beam else rec_greedy
    if beam:
        texts = [rec.predict(it[2]) for it in items]   # beam: one at a time
    else:
        try:
            texts = rec.predict_batch([it[2] for it in items])
        except Exception:
            texts = [rec.predict(it[2]) for it in items]
    ys = [it[0] for it in items]
    band = max(8.0, (max(ys) - min(ys)) / 40.0)
    order = sorted(range(len(items)), key=lambda i: (round(items[i][0] / band), items[i][1]))
    return clean_ocr(" ".join(texts[i] for i in order if str(texts[i]).strip()))


labels = load_train_labels()
# evaluate on non-empty-GT images (CER is meaningful there); reproducible sample
pool = labels[labels.ocr_text.str.strip() != ""].sample(N, random_state=42).reset_index(drop=True)
print(f"probe on {len(pool)} non-empty-GT train images (CPU)\n")

variants = {"base": (False, False), "prep": (True, False),
            "beam": (False, True), "prep+beam": (True, True)}
acc = {k: [] for k in variants}
t0 = time.time()
for i, row in pool.iterrows():
    p = TRAIN_IMAGES_DIR / f"{row.image_id}.jpg"
    for name, (prep, beam) in variants.items():
        pred = ocr_one(p, prep, beam)
        acc[name].append(cer(row.ocr_text, pred))
    if (i + 1) % 25 == 0:
        el = time.time() - t0
        cur = {k: round(1 - np.mean(v), 4) for k, v in acc.items()}
        print(f"  [{i+1}/{len(pool)}] {el/60:.1f}min | ocr_term so far: {cur}")

print("\n=== PROBE RESULT (ocr_term = 1 - CER, higher better) ===")
for name in variants:
    c = np.mean(acc[name])
    print(f"  {name:<10}: CER {c:.4f} | ocr_term {1-c:.4f}")
base_term = 1 - np.mean(acc["base"])
for name in ("prep", "beam", "prep+beam"):
    d = (1 - np.mean(acc[name])) - base_term
    print(f"  delta {name:<10} vs base: {d:+.4f}")
print("\nNote: cached ft CV ocr_term = 0.6102 (full train). 'base' here should be close on this subset.")
