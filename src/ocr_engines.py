"""Pluggable OCR backends with a common interface.

Each engine implements `transcribe(image_path) -> OcrResult`. Imports are lazy
so a missing engine never breaks the others. Reading order is normalized
(top->bottom, then left->right) before joining lines.

Engines:
  - easyocr   : det+rec end-to-end, vi+en (baseline reference)
  - paddleocr : PP-OCR, strong on Vietnamese diacritics (accuracy track)
  - tesseract : light CPU fallback (lightweight track), needs `vie` traineddata
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ocr_postprocess import clean_ocr


@dataclass
class OcrResult:
    text: str                       # cleaned, reading-order joined
    raw_text: str                   # cleaned but pre-dedupe join of raw lines
    mean_conf: float = 0.0
    n_boxes: int = 0
    lines: list = field(default_factory=list)

    @property
    def n_chars(self) -> int:
        return len(self.text)


def _order_and_join(items: list[tuple[float, float, str, float]]) -> tuple[str, float, int]:
    """items: (y, x, text, conf). Sort top->bottom, left->right; join with space.
    Uses a coarse row-banding so words on the same line read left->right."""
    if not items:
        return "", 0.0, 0
    ys = [it[0] for it in items]
    band = max(8.0, (max(ys) - min(ys)) / 40.0)  # adaptive row height
    items = sorted(items, key=lambda it: (round(it[0] / band), it[1]))
    text = " ".join(it[2] for it in items if it[2].strip())
    confs = [it[3] for it in items if it[2].strip()]
    return text, (float(np.mean(confs)) if confs else 0.0), len(items)


class BaseEngine:
    name = "base"

    def transcribe(self, image_path) -> OcrResult:  # pragma: no cover
        raise NotImplementedError


class EasyOcrEngine(BaseEngine):
    name = "easyocr"

    def __init__(self, gpu: bool = True, conf_threshold: float = 0.35,
                 langs=("vi", "en")):
        import easyocr  # lazy
        self.reader = easyocr.Reader(list(langs), gpu=gpu, verbose=False)
        self.conf_threshold = conf_threshold

    def transcribe(self, image_path) -> OcrResult:
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            return OcrResult("", "", 0.0, 0)
        results = self.reader.readtext(img, detail=1, paragraph=False)
        items = []
        for box, txt, conf in results:
            if conf < self.conf_threshold:
                continue
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            items.append((min(ys), min(xs), txt, float(conf)))
        text, mean_conf, n = _order_and_join(items)
        raw = " ".join(it[2] for it in items)
        return OcrResult(clean_ocr(text), clean_ocr(raw), mean_conf, n,
                         [it[2] for it in items])


class PaddleOcrEngine(BaseEngine):
    name = "paddleocr"

    def __init__(self, gpu: bool = True, conf_threshold: float = 0.5,
                 lang: str = "vi"):
        from paddleocr import PaddleOCR  # lazy
        # use_angle_cls handles rotated text in thumbnails
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang,
                                 use_gpu=gpu, show_log=False)
        except TypeError:
            # newer PaddleOCR API drops some kwargs
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        self.conf_threshold = conf_threshold

    def transcribe(self, image_path) -> OcrResult:
        res = self.ocr.ocr(str(image_path), cls=True)
        items = []
        if res and res[0]:
            for box, (txt, conf) in res[0]:
                if conf < self.conf_threshold:
                    continue
                ys = [p[1] for p in box]
                xs = [p[0] for p in box]
                items.append((min(ys), min(xs), txt, float(conf)))
        text, mean_conf, n = _order_and_join(items)
        raw = " ".join(it[2] for it in items)
        return OcrResult(clean_ocr(text), clean_ocr(raw), mean_conf, n,
                         [it[2] for it in items])


class TesseractEngine(BaseEngine):
    name = "tesseract"

    def __init__(self, lang: str = "vie+eng", psm: int = 6):
        import pytesseract  # lazy
        self.pytesseract = pytesseract
        self.lang = lang
        self.config = f"--psm {psm}"

    def transcribe(self, image_path) -> OcrResult:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        data = self.pytesseract.image_to_data(
            img, lang=self.lang, config=self.config,
            output_type=self.pytesseract.Output.DICT)
        items = []
        for i, txt in enumerate(data["text"]):
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = -1.0
            if not txt.strip() or conf < 0:
                continue
            items.append((float(data["top"][i]), float(data["left"][i]),
                          txt, conf / 100.0))
        text, mean_conf, n = _order_and_join(items)
        raw = " ".join(it[2] for it in items)
        return OcrResult(clean_ocr(text), clean_ocr(raw), mean_conf, n,
                         [it[2] for it in items])


class RapidOcrEngine(BaseEngine):
    """PP-OCR (v4) models via ONNXRuntime — lightweight, low-memory, CPU-native.
    Often more accurate than the PaddleOCR 2.6 default models on this dataset."""
    name = "rapidocr"

    def __init__(self, gpu: bool = False, conf_threshold: float = 0.5):
        from rapidocr_onnxruntime import RapidOCR  # lazy
        self.ocr = RapidOCR()
        self.conf_threshold = conf_threshold

    def transcribe(self, image_path) -> OcrResult:
        result, _elapse = self.ocr(str(image_path))
        items = []
        if result:
            for box, txt, conf in result:
                try:
                    conf = float(conf)
                except (ValueError, TypeError):
                    conf = 1.0
                if conf < self.conf_threshold:
                    continue
                ys = [p[1] for p in box]
                xs = [p[0] for p in box]
                items.append((min(ys), min(xs), txt, conf))
        text, mean_conf, n = _order_and_join(items)
        raw = " ".join(it[2] for it in items)
        return OcrResult(clean_ocr(text), clean_ocr(raw), mean_conf, n,
                         [it[2] for it in items])


_REGISTRY = {
    "easyocr": EasyOcrEngine,
    "paddleocr": PaddleOcrEngine,
    "tesseract": TesseractEngine,
    "rapidocr": RapidOcrEngine,
}


def get_engine(name: str, **kwargs) -> BaseEngine:
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown engine '{name}'. Options: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
