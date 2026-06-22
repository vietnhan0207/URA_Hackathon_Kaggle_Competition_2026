"""Ported product-extraction logic from the 0.6495 friend notebook (v16).

Key ideas his pipeline has that our pure classifier lacks:
  1. OCR-tolerant brand dictionary (encodes OCR corruptions explicitly, e.g.
     Pate -> Rate/Fate/Late, Canfoco -> canfuco/canfood/ganfoco).
  2. Canonicalization to the EXACT competition label string (token-F1 rewards
     reconstructing the full multi-token canonical name).
  3. Evidence gating (`product_supported_by_ocr`) to block hallucinations.

This module is self-contained: import `extract_product`, `safe_product`,
`canonicalize_product_name`, `product_supported_by_ocr`.
"""
from __future__ import annotations

import re
import unicodedata


def fold_text(s: str) -> str:
    """Lowercase + remove Vietnamese accents for OCR-tolerant matching."""
    s = "" if s is None else str(s).lower()
    s = s.replace("đ", "d")
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


BRAND_RULES = [
    # ----- Strong Hạ Long / Canfoco variants (prepended: run first) -----
    (
        r"(halong|ha\s*long|ha\s*lonc|h[ạa]\s*long|h\s*long|hl)"
        r".{0,80}"
        r"(canfoco|cafoco|cafo?co|canfuco|canfood|canoco|ganfoco|halor\s*canfoco)",
        "Ha Long Canfoco", []
    ),
    (
        r"(cong\s*ty|c[oô]ng\s*ty|cty)"
        r".{0,40}"
        r"(cp|c[oô]\s*phan|cổ\s*phần)"
        r".{0,60}"
        r"(d[o0ổ]\s*h[oộeệ]p|do\s*hop)"
        r".{0,30}"
        r"(h[ạa]\s*long|ha\s*long|h\s*long)",
        "Đồ Hộp Hạ Long", []
    ),
    (
        r"(d[o0ổ]\s*h[oộeệ]p|do\s*hop|đồ\s*hộp|đổ\s*hộp)"
        r".{0,25}"
        r"(h[ạa]\s*long|ha\s*long|h\s*long|hl)",
        "Đồ Hộp Hạ Long", []
    ),
    (
        r"(pate|pat[eê]|pa\s*te|rat[eê]|fat[eê]|lat[eê]|bat[eê]|eat[eê]|zat[eê])"
        r".{0,50}"
        r"(c[o0]t\s*den|c[ộo]t\s*đ[èe]n|hai\s*phong|h[ảa]i\s*ph[òo]ng)",
        "Pate Cột Đèn Hải Phòng", []
    ),
    (
        r"(c[o0]t\s*den|c[ộo]t\s*đ[èe]n)"
        r".{0,50}"
        r"(pate|pat[eê]|pa\s*te|rat[eê]|fat[eê]|lat[eê]|bat[eê]|eat[eê]|zat[eê])",
        "Pate Cột Đèn Hải Phòng", []
    ),
    (r"hoa\s*sen\s*home", "HOA SEN HOME", []),
    (r"highlands?\s*coffee|highland\s*coffee|higlands?\s*coffee", "Highlands Coffee", []),
    (r"coffee\s*house|the\s*coffee\s*house", "The Coffee House", []),
    (r"ph[uú]c\s*long|phuc\s*long", "Phúc Long", []),

    # ----- OCR-tolerant Hạ Long / Canfoco / Pate Cột Đèn -----
    (
        r"(ha\s*long|halong|d[o0]\s*h[o0]p\s*ha\s*long|do\s*hop\s*ha\s*long|"
        r"canf[o0]c[o0]|canf[uoa]c[o0]|canfood|canoco|ganfoco)"
        r".{0,80}"
        r"(pate|pa\s*te|pat[eê]|rat[eê]|fat[eê])"
        r".{0,40}"
        r"(cot\s*den|c[o0]t\s*den|c[ộo]t\s*đ[èe]n)",
        "Ha Long Canfoco Pate Cột Đèn", []
    ),
    (
        r"(pate|pa\s*te|pat[eê]|rat[eê]|fat[eê])"
        r".{0,40}"
        r"(cot\s*den|c[o0]t\s*den|c[ộo]t\s*đ[èe]n|hai\s*phong|h[ảa]i\s*ph[òo]ng)",
        "Pate Cột Đèn Hải Phòng", []
    ),
    (
        r"(cot\s*den|c[o0]t\s*den|c[ộo]t\s*đ[èe]n)"
        r".{0,40}"
        r"(pate|pa\s*te|pat[eê]|rat[eê]|fat[eê])",
        "Pate Cột Đèn Hải Phòng", []
    ),
    (
        r"ha\s*long\s*canfoco|halong\s*canfoco|ha\s*long\s*canf[uoa]co|"
        r"canf[o0]co|canf[uoa]co|canfood|canoco|ganfoco",
        "Ha Long Canfoco", []
    ),
    (
        r"d[o0]\s*h[o0]p\s*h[aạ]\s*long|do\s*hop\s*ha\s*long|"
        r"cong\s*ty\s*(co\s*phan|cp)\s*do\s*hop\s*ha\s*long",
        "Đồ Hộp Hạ Long", []
    ),
    (r"ha\s*long\s*canfoco.*pate.*c[ộo]t|c[ộo]t\s*đ[èe]n.*ha\s*long\s*canfoco", "Ha Long Canfoco Pate Cột Đèn", []),
    (r"ha\s*long\s*canfoco.*pate|canfoco.*pate\s*c[ộo]t|pate\s*c[ộo]t\s*đ[èe]n.*canfoco", "Ha Long Canfoco Pate", []),
    (r"ha\s*long\s*canfoco|halong\s*canfoco|canfood|canfoco", "Ha Long Canfoco", []),
    (r"đ[ồo]\s*h[ộo]p\s*h[ạa]\s*long|do\s*hop\s*ha\s*long", "Đồ Hộp Hạ Long", []),
    (r"pate\s*c[ộo]t\s*đ[èe]n|pate\s*cot\s*den|c[ộo]t\s*đ[èe]n\s*h[ảa]i\s*ph[òo]ng", "Pate Cột Đèn Hải Phòng", []),
    (r"h[ạa]\s*long\s*pate|pate\s*h[ạa]\s*long", "Ha Long Canfoco Pate", []),

    # ----- Nestlé / NAN / Milo -----
    (r"(nestl[eé]|n[eé]stle|nan).{0,80}(infini\s*pro|infinipro|ifini\s*pro|ifinipro|infi\s*pro|a2)", "Nestlé NAN Infinipro A2", []),
    (r"(nestl[eé]|n[eé]stle|nan).{0,80}(opti\s*pro|optipro|opti\s*pro\s*plus|optiproplus)", "Nestlé NAN Optipro Plus", []),
    (r"(nestl[eé]|n[eé]stle|nan).{0,80}(supreme\s*pro|supremepro)", "Nestlé NAN Supremepro", []),
    (r"(nestl[eé]|n[eé]stle|alfamino).{0,80}(alfamino|infant)", "Nestlé Alfamino Infant", []),
    (r"(nestl[eé]|n[eé]stle|beba).{0,80}\bbeba\b", "Nestlé BEBA", []),
    (r"\bnan\b|s[uư][aã]\s*nan|sua\s*nan|n[a4]\s*n", "Nestlé NAN", []),
    (r"\bmilo\b|mi1o|miio", "Nestlé Milo", []),

    # ----- Other milk / infant formula -----
    (r"aptamil.{0,60}profutura|profutura.{0,60}aptamil", "Aptamil Profutura", []),
    (r"\baptamil\b", "Aptamil", []),
    (r"\bhipp\b|hi\s*pp|combiotic", "HiPP Combiotic", []),
    (r"optimum\s*gold|0ptimum\s*gold", "Optimum Gold", []),

    # ----- Coffee / beverage -----
    (r"highlands?\s*coffee|highlann?ds|higlands?|highlands?", "Highlands Coffee", []),
    (r"the\s*coffee\s*house|coffee\s*house", "The Coffee House", []),
    (r"ph[uú]c\s*long|phuc\s*long", "Phúc Long", []),

    # ----- Sauce / other FMCG -----
    (r"chinsu|chin\s*su|t[uư][oơ]ng\s*[oớ]t\s*chinsu|tuong\s*ot\s*chinsu", "Chinsu", []),

    # ----- Milk / dairy -----
    (r"vinamilk", "Vinamilk", ["flex", "adm gold", "sure", "canxi",
                               "colosbaby", "colos baby", "ong tho", "ông thọ", "dielac", "grow"]),
    (r"th true|thtrue", "TH True Milk", ["true yogurt", "grow", "school milk", "true butter"]),
    (r"dutch lady|cô gái", "Dutch Lady", ["grow", "complete", "canxi", "mom"]),
    (r"nutifood|nuti\b", "Nutifood", ["growplus", "grow plus", "pedia", "iq"]),
    (r"ensure\b", "Abbott Ensure", ["gold", "original", "plus"]),
    (r"pediasure", "Abbott PediaSure", []),
    (r"similac", "Abbott Similac", []),
    (r"glucerna", "Abbott Glucerna", []),
    (r"milo\b", "Nestlé Milo", []),
    (r"nestle|nestlé", "Nestlé", ["milo", "coffee mate", "carnation", "nestea", "nan", "sữa bột"]),
    (r"aptamil", "Aptamil", []),
    (r"friso\b", "Friso", ["gold", "comfort", "prestige"]),
    (r"meiji\b", "Meiji", ["growing", "step"]),
    (r"ba vi\b|bavi\b|ba vì", "Ba Vì", ["gold"]),
    (r"lothamilk", "Lothamilk", ["canxi"]),
    (r"yomost", "Yomost", []),
    (r"dalat milk|đà lạt", "Đà Lạt Milk", []),
    (r"kun\b|kun milk", "Kun", ["chocolate", "strawberry"]),
    (r"fami\b", "Fami", ["canxi", "kid"]),
    (r"anlene", "Anlene", ["gold", "concentrate"]),
    (r"anchor\b", "Anchor", ["butter", "cream"]),

    # ----- Pate / canned meat (other brands) -----
    (r"vissan", "Vissan", ["pate heo", "pate ga", "pate gà",
                           "xuc xich", "xúc xích", "lap xuong", "lạp xưởng"]),
    (r"hafi\b", "Hafi", ["pate"]),
    (r"ba huan|ba huân", "Ba Huân", ["pate"]),
    (r"san ha\b|san hà", "San Hà", ["pate"]),
    (r"\bc\.p\.|cp\s*foods|cp\s*food|cp\s*vi[eệ]t\s*nam|c\.p\s*vi[eệ]t\s*nam", "CP", ["pate", "xúc xích", "xuc xich"]),
    (r"long bien|long biên", "Long Biên", ["pate"]),
]


LINE_CANONICAL = {
    "flex": "Flex", "adm gold": "ADM Gold", "sure": "Sure", "canxi": "Canxi",
    "colosbaby": "ColosBaby", "colos baby": "ColosBaby", "ong tho": "Ông Thọ",
    "ông thọ": "Ông Thọ", "dielac": "Dielac", "grow": "Grow", "grow+": "Grow+",
    "growplus": "Grow Plus", "grow plus": "Grow Plus", "optipro": "Optipro",
    "opti pro": "Optipro", "infinipro": "Infinipro", "infini pro": "Infinipro",
    "pate heo": "Pate Heo", "pate ga": "Pate Gà", "pate gà": "Pate Gà",
    "pate gan": "Pate Gan", "xuc xich": "Xúc Xích", "xúc xích": "Xúc Xích",
    "lap xuong": "Lạp Xưởng", "lạp xưởng": "Lạp Xưởng",
}


def extract_product(text: str) -> str:
    """Return 'Brand ProductLine', 'Brand', or '' if no rule matches."""
    if not text or not text.strip():
        return ""
    raw = text.lower().replace("patê", "pate")
    tl = raw + " " + fold_text(raw)
    for pattern, brand, lines in BRAND_RULES:
        if re.search(pattern, tl, re.IGNORECASE):
            for line in lines:
                if re.search(line, tl, re.IGNORECASE):
                    line_key = line.lower().strip()
                    line_canon = LINE_CANONICAL.get(line_key, line.strip().title())
                    return f"{brand} {line_canon}"
            return brand
    return ""


def canonicalize_product_name(product_name: str, ocr_text: str = "") -> str:
    """Clean noisy product labels after rules/classifier prediction."""
    name = "" if product_name is None else str(product_name).strip()
    if not name:
        return ""

    folded = fold_text(ocr_text)
    name_folded = fold_text(name)

    has_pate = re.search(r"pate|pa\s*te|rat[eê]|fat[eê]", folded)
    has_cot_den = re.search(r"cot\s*den|hai\s*phong", folded)
    has_canfoco = re.search(r"ha\s*long\s*canfoco|halong\s*canfoco|canfoco|canfuco|canfood|canoco|ganfoco", folded)
    has_do_hop_ha_long = re.search(r"do\s*hop\s*ha\s*long|cong\s*ty\s*(co\s*phan|cp)\s*do\s*hop\s*ha\s*long", folded)

    if has_canfoco and has_pate and has_cot_den:
        return "Ha Long Canfoco Pate Cột Đèn"
    if has_pate and has_cot_den:
        return "Pate Cột Đèn Hải Phòng"
    if has_canfoco:
        return "Ha Long Canfoco"
    if has_do_hop_ha_long:
        return "Đồ Hộp Hạ Long"

    if re.search(r"\bnan\b", folded):
        if re.search(r"opti\s*pro|optipro", folded):
            return "Nestlé NAN Optipro Plus"
        if re.search(r"infini\s*pro|infinipro|infipro", folded):
            return "Nestlé NAN Infinipro A2"
        if re.search(r"alfamino", folded):
            return "Nestlé Alfamino Infant"
        return "Nestlé NAN"
    if re.search(r"milo", folded):
        return "Nestlé Milo"
    if re.search(r"nestle|nestlé", folded):
        return "Nestlé"

    if folded in {"pate", "pa te", "patê"}:
        return ""

    replacements = {
        "halong canfoco": "Ha Long Canfoco",
        "halong cafoco": "Ha Long Canfoco",
        "ha long canfoco": "Ha Long Canfoco",
        "do hop ha long halong canfoco": "Ha Long Canfoco",
        "cong ty co phan do hop ha long": "Đồ Hộp Hạ Long",
        "do hop cong ty cp do hop ha long": "Đồ Hộp Hạ Long",
        "do hop ha long": "Đồ Hộp Hạ Long",
        "pate cot den": "Pate Cột Đèn Hải Phòng",
        "pate cot den hai phong": "Pate Cột Đèn Hải Phòng",
        "pate ha long": "Pate Hạ Long",
        "pate cot den ha long canfoco": "Ha Long Canfoco Pate Cột Đèn",
        "nan": "Nestlé NAN",
        "sua nan": "Nestlé NAN",
    }
    return replacements.get(name_folded, name)


def normalize_for_evidence(s: str) -> str:
    """Fold accents and normalize common OCR confusions for evidence matching."""
    s = fold_text("" if s is None else str(s))
    s = re.sub(r"\b(rat[eê]?|fat[eê]?|lat[eê]?|bat[eê]?|eat[eê]?|zat[eê]?|pat[eê]?|paj[eê]?)\b", "pate", s)
    s = s.replace("canfuco", "canfoco").replace("canfood", "canfoco")
    s = s.replace("canoco", "canfoco").replace("ganfoco", "canfoco").replace("cafoco", "canfoco")
    s = s.replace("halor canfoco", "ha long canfoco").replace("halong", "ha long").replace("ha lonc", "ha long")
    s = re.sub(r"\bh\s*long\b", "ha long", s)
    s = re.sub(r"\bdo\s*hep\b|\bdo\s*hop\b", "do hop", s)
    s = s.replace("neste", "nestle").replace("nestlé", "nestle")
    s = s.replace("optiproplus", "optipro plus").replace("opti proplus", "optipro plus")
    s = re.sub(r"\bopti\s*pro\b", "optipro", s)
    s = re.sub(r"\binfini\s*pro\b|\binfi\s*pro\b|\bifini\s*pro\b|\bifinipro\b", "infinipro", s)
    s = re.sub(r"\bn\s*a\s*n\b", "nan", s)
    s = s.replace("cofee", "coffee").replace("coffe", "coffee")
    s = re.sub(r"\bhighlann?ds\b|\bhiglands\b|\bhighland\b", "highlands", s)
    s = re.sub(r"\bhi\s*pp\b", "hipp", s)
    s = re.sub(r"\bchin\s*su\b", "chinsu", s)
    s = re.sub(r"\bphuc\s*long\b", "phuc long", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


GENERIC_PRODUCT_NAMES = {
    "pate", "pa te", "thit heo", "thit hop", "san pham do hop", "do hop", "sua", "sua bot",
}
PRODUCT_STOP_TOKENS = {
    "nestle", "ha", "long", "do", "hop", "cong", "ty", "co", "phan",
    "hai", "phong", "sua", "pate", "patê", "milk", "food", "foods",
}


def product_supported_by_ocr(product_name: str, ocr_text: str) -> bool:
    """True only if the predicted product has visible OCR evidence."""
    name_norm = normalize_for_evidence(product_name)
    ocr_norm = normalize_for_evidence(ocr_text)
    if not name_norm or not ocr_norm:
        return False
    if name_norm in GENERIC_PRODUCT_NAMES:
        return False
    if name_norm in ocr_norm:
        return True

    name_tokens = name_norm.split()
    ocr_tokens = set(ocr_norm.split())

    if "nan" in name_tokens:
        return "nan" in ocr_tokens
    if "milo" in name_tokens:
        return "milo" in ocr_tokens
    if "cot" in name_tokens and "den" in name_tokens:
        return "pate" in ocr_tokens and ({"cot", "den"}.issubset(ocr_tokens) or {"hai", "phong"}.issubset(ocr_tokens))
    if "canfoco" in name_tokens:
        return "canfoco" in ocr_tokens or {"ha", "long"}.issubset(ocr_tokens) or {"do", "hop", "ha", "long"}.issubset(ocr_tokens)
    if name_norm == "do hop ha long":
        return {"do", "hop", "ha", "long"}.issubset(ocr_tokens)
    if name_norm == "cp":
        if {"do", "hop", "ha", "long"}.issubset(ocr_tokens):
            return False
        return any(x in ocr_norm for x in ("cp foods", "cp food", "cp vietnam", "cp viet nam", "c p vietnam", "c p viet nam"))
    if name_norm == "ba vi":
        return {"ba", "vi"}.issubset(ocr_tokens) or "bavi" in ocr_tokens
    if name_norm == "phuc long":
        return {"phuc", "long"}.issubset(ocr_tokens)
    if name_norm == "the coffee house":
        return "coffee" in ocr_tokens and "house" in ocr_tokens
    if name_norm == "highlands coffee":
        return "highlands" in ocr_tokens
    if name_norm == "hipp combiotic":
        return "hipp" in ocr_tokens or "combiotic" in ocr_tokens
    if name_norm.startswith("aptamil"):
        return "aptamil" in ocr_tokens
    if name_norm == "optimum gold":
        return {"optimum", "gold"}.issubset(ocr_tokens)

    distinctive = [t for t in name_tokens if len(t) >= 3 and t not in PRODUCT_STOP_TOKENS]
    if not distinctive:
        return False
    return any(t in ocr_tokens for t in distinctive)


def safe_product(product_name: str, ocr_text: str) -> str:
    """Canonicalize, then reject the product if OCR does not support it."""
    name = canonicalize_product_name(product_name, ocr_text)
    if not name:
        return ""
    return name if product_supported_by_ocr(name, ocr_text) else ""
