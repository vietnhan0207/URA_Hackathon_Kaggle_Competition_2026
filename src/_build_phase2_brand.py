"""Phase-2 submission with SEPARATE brand_name + product_name, both extracted from
OCR text via rules (post-processing, no model run). Follows host spec (Description
lines 282-372): brand_name = brand only; product_name = Brand + Product Line, in
the documented clean format (Ha Long Canfoco, Nestlé Milo, ...). Uses friend's
NORMALIZED_BRAND_RULES (they encode exactly this format + spelling normalization).
"""
import csv
import re
import unicodedata

import pandas as pd

import config
import friend_dispatcher as fd

P2 = str(config.CACHE_DIR / "ocr_vietocr_ft_phase2test.parquet")

# canonical product (from rules) -> BRAND only
BRAND_OF = {
    "Ha Long Canfoco Pate Cột Đèn": "Ha Long Canfoco",
    "Ha Long Canfoco Pate": "Ha Long Canfoco",
    "Ha Long Canfoco": "Ha Long Canfoco",
    "Đồ Hộp Hạ Long": "Ha Long Canfoco",
    "Pate Cột Đèn Hải Phòng": "Ha Long Canfoco",
    "Vinamilk": "Vinamilk", "TH True Milk": "TH True Milk", "Dutch Lady": "Dutch Lady",
    "Nutifood": "Nutifood", "Abbott Ensure": "Abbott", "Abbott PediaSure": "Abbott",
    "Abbott Similac": "Abbott", "Abbott Glucerna": "Abbott",
    "Nestlé Milo": "Nestlé", "Nestlé": "Nestlé", "Aptamil": "Aptamil",
    "Friso": "Friso", "Meiji": "Meiji", "Ba Vì": "Ba Vì", "Lothamilk": "Lothamilk",
    "Yomost": "Yomost", "Đà Lạt Milk": "Đà Lạt Milk", "Kun": "Kun", "Fami": "Fami",
    "Anlene": "Anlene", "Anchor": "Anchor", "Vissan": "Vissan", "Hafi": "Hafi",
    "Ba Huân": "Ba Huân", "San Hà": "San Hà", "CP": "CP", "Long Biên": "Long Biên",
    "Pate": "", "Highlands Coffee": "Highlands Coffee", "Coffee House": "Coffee House",
}
_KEYS = sorted(BRAND_OF, key=len, reverse=True)


def brand_of(product):
    if not product:
        return ""
    if product in BRAND_OF:
        return BRAND_OF[product]
    for k in _KEYS:                       # product = "Brand Line" -> match brand prefix
        if product.startswith(k):
            return BRAND_OF[k]
    return product


def extract(ocr_text):
    """(brand_name, product_name) from OCR; '' when no FMCG brand present."""
    product = fd.extract_product(ocr_text)   # host clean format: Brand [Line]
    return brand_of(product), product


ocr = pd.read_parquet(P2)
ocr["ocr_text"] = ocr["ocr_text"].fillna("")
bp = [extract(t) for t in ocr["ocr_text"]]
out = pd.DataFrame({
    "image_id": ocr["image_id"],
    "ocr_text": ocr["ocr_text"],
    "brand_name": [b for b, _ in bp],
    "product_name": [p for _, p in bp],
})

# sanity-check product_name vs phase-1 train GT (we have no phase-2 GT)
from data import load_train_labels
from run_ocr import cache_path
from scoring import token_f1
lab = load_train_labels().merge(
    pd.read_parquet(cache_path("vietocr_ft", "all"))[["image_id", "ocr_text"]],
    on="image_id", suffixes=("_gt", "_ocr"))
tr_prod = [fd.extract_product(t) for t in lab["ocr_text_ocr"].fillna("")]
f1 = sum(token_f1(g, p) for g, p in zip(lab["product_name"], tr_prod)) / len(lab)
print(f"[sanity] friend extract_product vs phase-1 train GT: prod-F1 = {f1:.4f}")

print(f"\nphase-2: brand fill {(out.brand_name.str.strip()!='').mean():.1%} | "
      f"product fill {(out.product_name.str.strip()!='').mean():.1%}")
print("top brands:", dict(out[out.brand_name.str.strip()!=''].brand_name.value_counts().head(8)))
print("top products:", dict(out[out.product_name.str.strip()!=''].product_name.value_counts().head(8)))

# write: ' ' for blanks, QUOTE_ALL, 4 cols
out = out[["image_id", "ocr_text", "brand_name", "product_name"]]
for c in ["ocr_text", "brand_name", "product_name"]:
    out[c] = out[c].fillna("").astype(str)
    out.loc[out[c].str.strip() == "", c] = " "
path = config.SUBMISSIONS_DIR / "submission_phase2_brand.csv"
out.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
print(f"\nwrote {path} | rows {len(out)} | cols {list(out.columns)}")
