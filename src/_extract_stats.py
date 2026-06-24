"""Compute every EDA stat locally and dump to presentation/eda_stats.json, so the
self-contained Kaggle/Colab notebook can plot without any data or src deps."""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data import load_train_labels
from product_calibrated import CalibratedRuleHead
from run_ocr import cache_path
from scoring import cer, token_f1

ROOT = config.CACHE_DIR.parent
labels = load_train_labels()
labels["ocr_text"] = labels["ocr_text"].fillna("")
labels["product_name"] = labels["product_name"].fillna("").str.strip()


def diac_frac(s):
    d = unicodedata.normalize("NFD", str(s))
    letters = sum(1 for c in d if c.isalpha())
    marks = sum(1 for c in d if unicodedata.category(c) == "Mn")
    return marks / letters if letters else 0.0


S = {}

# 1 product top-15
nz = labels[labels.product_name != ""]["product_name"].value_counts()
S["product_top15"] = [[k, int(v)] for k, v in nz.head(15).items()]
S["product_nonempty_total"] = int(nz.sum())
S["product_distinct"] = int(nz.size)

# 2 empty rates
S["empty"] = {
    "product": float((labels.product_name == "").mean()),
    "ocr": float((labels.ocr_text.str.strip() == "").mean()),
    "both": float(((labels.product_name == "") & (labels.ocr_text.str.strip() == "")).mean()),
}

# 3 ocr length hist
L = labels.ocr_text.str.len().values
cnt, edges = np.histogram(L, bins=50, range=(0, 520))
S["ocr_len"] = {"counts": cnt.tolist(), "edges": edges.tolist(),
                "mean": float(L.mean()), "median": float(np.median(L))}

# 4 diacritic density hist (text-bearing)
dd = labels[labels.ocr_text.str.strip() != ""].ocr_text.map(diac_frac).values
cnt, edges = np.histogram(dd, bins=40, range=(0, 1))
S["diac"] = {"counts": cnt.tolist(), "edges": edges.tolist(), "mean": float(dd.mean())}

# 5 engine bake-off
ENGINES = {"PaddleOCR": "paddleocr", "VietOCR (base)": "vietocr", "VietOCR-FT (ours)": "vietocr_ft"}
gt = labels.set_index("image_id")["ocr_text"]
series = {}
for nm, eng in ENGINES.items():
    f = cache_path(eng, "all")
    if Path(f).exists():
        series[nm] = pd.read_parquet(f)[["image_id", "ocr_text"]].set_index("image_id")["ocr_text"].fillna("")
common = set(gt.index)
for s in series.values():
    common &= set(s.index)
common = sorted(common)
S["engines"] = [[nm, float(np.mean([cer(gt[i], s.get(i, "")) for i in common])),
                 float(np.mean([diac_frac(s.get(i, "")) for i in common]))] for nm, s in series.items()]
S["engines_n"] = len(common)
S["gt_diac_ref"] = float(dd.mean())

# 6 CER bands (vietocr_ft vs GT, text-bearing)
ft = pd.read_parquet(cache_path("vietocr_ft", "all"))[["image_id", "ocr_text"]].rename(columns={"ocr_text": "pred"})
m = labels.merge(ft, on="image_id"); m["pred"] = m["pred"].fillna("")
m = m[m.ocr_text.str.strip() != ""]
cers = np.array([cer(g, p) for g, p in zip(m.ocr_text, m.pred)])
S["cer_bands"] = [float(((cers >= lo) & (cers < hi)).mean()) for lo, hi in [(0, .1), (.1, .3), (.3, .6), (.6, 1.01)]]

# 7 oracle vs real (in-sample)
head = CalibratedRuleHead(use_classifier_fallback=True, min_pprod=0.55, gate_threshold=0.75).fit(
    labels[["image_id", "ocr_text", "product_name"]])
oracle = head.predict_batch(labels["ocr_text"])
S["f1_oracle"] = float(np.mean([token_f1(g, p) for g, p in zip(labels.product_name, oracle)]))
real = head.predict_batch(m["pred"])
S["f1_real"] = float(np.mean([token_f1(g, p) for g, p in zip(m.product_name, real)]))

# 8 score progression (documented public LB)
S["progression"] = [["v8 classifier", 0.6232], ["v10 calibrated\n(ours)", 0.6685],
                    ["our OCR +\nteammate product", 0.6959]]
S["rival"] = 0.6495

# 9 experiment log (documented CV composite deltas)
S["experiments"] = [["prep+beam OCR", 0.0024], ["highlands-first reorder", 0.0005],
                    ["friend canonicalizer", -0.0025], ["aggressive multi-scale detect", -0.0164],
                    ["friend evidence_override", -0.0226], ["friend full head", -0.0342],
                    ["diacritic restoration", -0.0440], ["news-context abstention", -0.0599]]

# 10 phase shift
def fold(s):
    s = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return re.sub(r"[^a-z0-9 ]", " ", s)
MARK = r"canfoco|cot den|\bnan\b|nestle|highland|pate|ha long|do hop"
ps = []
for lab, f in [("Phase 1 (public test)", cache_path("vietocr_ft", "test")),
               ("Phase 2 (private test)", config.CACHE_DIR / "ocr_vietocr_ft_phase2test.parquet")]:
    if Path(f).exists():
        frac = pd.read_parquet(f)["ocr_text"].fillna("").map(fold).str.contains(MARK, regex=True).mean()
        ps.append([lab, float(frac)])
S["phase_shift"] = ps

out = ROOT / "presentation" / "eda_stats.json"
out.write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out)
print("engines:", S["engines"], "| oracle/real:", round(S["f1_oracle"], 3), round(S["f1_real"], 3))
