from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_dispatcher import MISSING_PRODUCT, normalize_match_text
from scripts.apply_product_canonicalization import load_canonicalizer
from scripts.notebook_product_logic import NotebookProductPredictor, extract_product


STRONG_TMP_PRODUCTS = {
    "do hop ha long",
    "pate cot den",
    "pate cot den hai phong",
    "nan",
    "nestle nan",
    "nestle nan optiproplus 1",
    "coffee house",
    "highland coffee",
    "highlands coffee",
}

HALONG_TMP_PRODUCTS = {
    "do hop ha long",
    "pate cot den",
    "pate cot den hai phong",
}

HALONG_NAN_TMP_PRODUCTS = HALONG_TMP_PRODUCTS | {
    "nan",
    "nestle nan",
    "nestle nan optiproplus 1",
}


def evidence_product_from_ocr(text: object) -> str:
    lowered = "" if text is None else str(text).lower()
    normalized = normalize_match_text(lowered)
    compact = normalized.replace(" ", "")
    exact_markers = [
        ("bebopblitz", "BEBOP BLITZ"),
        ("epicinspiration", "EPIC INSPIRATION"),
        ("imcvmusic", "IMCV MUSIC"),
        ("firststory", "FIRST STORY"),
        ("newsofficial", "NEWS OffiCial"),
    ]
    for marker, product in exact_markers:
        if marker in compact:
            return product
    if compact == "capcut":
        return "CapCut"
    if "canfogo" in compact:
        return "Ha Long Canfoco"
    if any(marker in compact for marker in ["dohophalong", "dohopha", "congtydohophalong", "ctydohophalong"]) and any(
        marker in compact
        for marker in [
            "batkhancap",
            "bokhancap",
            "tonggiamdoc",
            "tgd",
            "giamdoc",
            "congty",
            "phathien",
            "dichtalon",
            "khoito",
            "top50",
        ]
    ):
        return "Đồ Hộp Hạ Long"
    if "highland" in lowered and ("coffee" in lowered or "cf " in lowered or " cf" in lowered):
        return "Highlands Coffee"
    if "coffee house" in lowered:
        return "Coffee House"
    return ""


def evidence_cotden_product_from_ocr(text: object, strict: bool = True) -> str:
    lowered = "" if text is None else str(text).lower()
    compact = normalize_match_text(lowered).replace(" ", "")
    has_cotden = any(
        marker in compact
        for marker in [
            "cotden",
            "c0tden",
            "tecotden",
            "chocotden",
        ]
    )
    if not has_cotden:
        return ""
    if strict:
        has_support = any(
            marker in compact
            for marker in [
                "haiphong",
                "150g",
                "1509",
                "canfoco",
                "cnocanfoco",
                "ngcaifoco",
                "ngcan",
                "ungcan",
            ]
        )
        if not has_support:
            return ""
    return "Pate Cột Đèn Hải Phòng"


def evidence_halong_product_from_ocr(text: object, broad: bool = False) -> str:
    lowered = "" if text is None else str(text).lower()
    compact = normalize_match_text(lowered).replace(" ", "")
    news_markers = [
        "batkhan",
        "giamdoc",
        "tonggiamdoc",
        "congty",
        "khoito",
        "thutuong",
        "tintuc",
        "news",
        "tram tin",
        "mangten",
        "nhungthuoc",
        "thuoc",
    ]
    if any(marker in compact for marker in news_markers):
        return ""
    if "halong" in compact and any(marker in compact for marker in ["canfoco", "cahfoco", "canf"]):
        return "Ha Long Canfoco"
    do_hop_markers = ["dohop", "d6hp", "dohp"]
    long_markers = ["hlong", "holong", "hqlong", "halong", "halong"]
    product_markers = ["pate", "cotden", "150g", "90g", "netweight", "hop"]
    if (
        any(marker in compact for marker in do_hop_markers)
        and any(marker in compact for marker in long_markers)
        and any(marker in compact for marker in product_markers)
    ):
        return "Đồ Hộp Hạ Long"
    if broad and "halong" in compact and any(marker in compact for marker in ["haiphong", "congty", "thuoc"]):
        return "Đồ Hộp Hạ Long"
    return ""


def evidence_override_product_from_ocr(text: object, current_product: object) -> str:
    normalized = normalize_match_text(text)
    compact = normalized.replace(" ", "")
    current = "" if current_product is None else str(current_product)
    current_norm = normalize_match_text(current)

    has_highlands = "highlandscoffee" in compact or "highlandcoffee" in compact or "highlands" in compact
    has_do_hop_halong = any(marker in compact for marker in ["dohophalong", "d6hophalong", "dohopha"])
    has_canfoco = any(marker in compact for marker in ["canfoco", "canfoc", "cahfoco", "canfogo", "canpoco"])
    has_pate = "pate" in compact or "bate" in compact
    has_cotden = any(marker in compact for marker in ["cotden", "cotoen", "cotdp", "ctden", "cotsen"])
    has_haiphong = "haiphong" in compact or "hai phong" in normalized
    has_news_company_context = any(
        marker in compact
        for marker in [
            "phathien",
            "virusdichta",
            "dichtalon",
            "chauphi",
            "congtydohophalong",
            "congtycophan",
            "tonggiamdoc",
            "batkhancap",
            "khongsudung",
            "lientieng",
            "dantri",
            "news",
        ]
    )

    if (
        current_norm == "highlands coffee"
        and has_highlands
        and has_do_hop_halong
        and any(
            marker in compact
            for marker in [
                "ngungban",
                "cungcap",
                "lienquan",
                "phanhoi",
                "tamngungban",
                "doitactieuthu",
                "baocaotaichinh",
                "khongsudung",
                "goiten",
                "sudungvai",
            ]
        )
    ):
        return "Đồ Hộp Hạ Long"
    if current_norm == "pate cot den hai phong" and has_do_hop_halong and has_news_company_context:
        return "Đồ Hộp Hạ Long"

    if has_canfoco and has_pate and has_cotden:
        return "Ha Long Canfoco Pate Cột Đèn"
    if has_canfoco and has_pate and current_norm in {"do hop ha long", "ha long canfoco", "pate cot den hai phong"}:
        return "Ha Long Canfoco Pate"
    if has_canfoco and current_norm in {"do hop ha long", "nan", "pate", "pate cot den hai phong"}:
        return "Ha Long Canfoco"

    if (
        has_pate
        and has_cotden
        and has_haiphong
        and current_norm in {"do hop ha long", "nan", "coffee house", "highlands coffee"}
    ):
        return "Pate Cột Đèn Hải Phòng"

    if current_norm in {"nestle milo", "nestle sua bot"} and "nestle" in normalized:
        return "Nestlé"

    return current


def rule_product_from_ocr(text: object) -> str:
    product = extract_product(text)
    if not product:
        return ""
    normalized = normalize_match_text(text)
    compact = normalized.replace(" ", "")
    news_markers = [
        "batkhan",
        "giamdoc",
        "tonggiamdoc",
        "congty",
        "khoito",
        "tintuc",
        "news",
    ]
    pate_noise_markers = [
        "dichtachauphi",
        "tiktok",
        "cotheleng",
        "cothelentieng",
    ]
    if normalize_match_text(product) in HALONG_TMP_PRODUCTS and any(marker in compact for marker in news_markers):
        return ""
    if normalize_match_text(product) == "pate" and any(marker in compact for marker in pate_noise_markers):
        return ""
    return product


def _nonempty(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)


def _merge_ocr(base: pd.DataFrame, ocr: pd.DataFrame) -> pd.DataFrame:
    out = base[["image_id", "product_name"]].copy()
    ocr_cols = ocr[["image_id", "ocr_text"]].copy()
    out = out.merge(ocr_cols, on="image_id", how="left")
    out["ocr_text"] = out["ocr_text"].fillna("")
    return out[["image_id", "ocr_text", "product_name"]]


def _fill_empty(base: pd.DataFrame, fills: pd.Series) -> pd.DataFrame:
    out = base.copy()
    old_empty = ~_nonempty(out["product_name"])
    fill_nonempty = fills.fillna("").astype(str).str.strip().ne("")
    out.loc[old_empty & fill_nonempty, "product_name"] = fills[old_empty & fill_nonempty]
    out["product_name"] = out["product_name"].replace("", MISSING_PRODUCT)
    return out[["image_id", "ocr_text", "product_name"]]


def _canonicalize_df(df: pd.DataFrame, canonicalizer) -> pd.DataFrame:
    if canonicalizer is None:
        return df
    out = df.copy()
    out["product_name"] = [
        result.canonical
        for result in canonicalizer.canonicalize_batch(out["product_name"].fillna("").astype(str).tolist())
    ]
    return out


def _override_products_from_ocr(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["product_name"] = [
        evidence_override_product_from_ocr(text, product)
        for text, product in zip(out["ocr_text"].tolist(), out["product_name"].tolist())
    ]
    return out[["image_id", "ocr_text", "product_name"]]


def build_variants(
    base_submission: pd.DataFrame,
    ocr_submission: pd.DataFrame,
    train_labels: pd.DataFrame,
    output_dir: Path,
    tmp_submission: pd.DataFrame | None = None,
    canonical_map: Path | None = None,
    canonicalize_before_rules: bool = False,
    canonicalize_after_rules: bool = False,
) -> dict[str, Path]:
    base = _merge_ocr(base_submission, ocr_submission)
    canonicalizer = load_canonicalizer(canonical_map) if canonical_map is not None and canonical_map.exists() else None
    if canonicalize_before_rules:
        base = _canonicalize_df(base, canonicalizer)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    rule_fills = base["ocr_text"].map(rule_product_from_ocr)
    variants = {"rules_fill": _fill_empty(base, rule_fills)}

    for threshold in [0.70, 0.80, 0.90]:
        predictor = NotebookProductPredictor(prob_threshold=threshold).fit(train_labels)
        decisions = base["ocr_text"].map(predictor.predict_with_decision)
        fills = decisions.map(lambda item: item.product_name)
        variants[f"notebook_predictor_p{int(threshold * 100)}"] = _fill_empty(base, fills)

    if tmp_submission is not None:
        tmp = tmp_submission[["image_id", "product_name"]].rename(columns={"product_name": "tmp_product_name"})
        work = base.merge(tmp, on="image_id", how="left")
        tmp_products = work["tmp_product_name"].fillna("").astype(str).str.strip()
        halong_mask = tmp_products.map(lambda value: normalize_match_text(value) in HALONG_TMP_PRODUCTS)
        halong_nan_mask = tmp_products.map(lambda value: normalize_match_text(value) in HALONG_NAN_TMP_PRODUCTS)
        strong_mask = tmp_products.map(lambda value: normalize_match_text(value) in STRONG_TMP_PRODUCTS)
        variants["tmp_halong_fill"] = _fill_empty(base, tmp_products.where(halong_mask, ""))
        variants["tmp_halong_nan_fill"] = _fill_empty(base, tmp_products.where(halong_nan_mask, ""))
        variants["tmp_strong_fill"] = _fill_empty(base, tmp_products.where(strong_mask, ""))

        combined_halong_nan = _fill_empty(base, tmp_products.where(halong_nan_mask, ""))
        combined_rule_fills = combined_halong_nan["ocr_text"].map(rule_product_from_ocr)
        variants["tmp_halong_nan_plus_rules"] = _fill_empty(combined_halong_nan, combined_rule_fills)

        evidence_fills = combined_halong_nan["ocr_text"].map(evidence_product_from_ocr)
        variants["tmp_halong_nan_plus_evidence"] = _fill_empty(combined_halong_nan, evidence_fills)

        combined_rules = _fill_empty(combined_halong_nan, combined_rule_fills)
        evidence_fills = combined_rules["ocr_text"].map(evidence_product_from_ocr)
        variants["tmp_halong_nan_plus_rules_evidence"] = _fill_empty(combined_rules, evidence_fills)

        best_so_far = variants["tmp_halong_nan_plus_rules_evidence"]
        cotden_strict = best_so_far["ocr_text"].map(lambda text: evidence_cotden_product_from_ocr(text, strict=True))
        variants["tmp_halong_nan_rules_evidence_cotden_strict"] = _fill_empty(best_so_far, cotden_strict)
        cotden_all = best_so_far["ocr_text"].map(lambda text: evidence_cotden_product_from_ocr(text, strict=False))
        cotden_all_df = _fill_empty(best_so_far, cotden_all)
        variants["tmp_halong_nan_rules_evidence_cotden_all"] = cotden_all_df
        variants["tmp_halong_nan_rules_evidence_cotden_all_ocr_override"] = _override_products_from_ocr(cotden_all_df)

        halong_strict = cotden_all_df["ocr_text"].map(lambda text: evidence_halong_product_from_ocr(text, broad=False))
        variants["tmp_halong_nan_cotden_all_halong_strict"] = _fill_empty(cotden_all_df, halong_strict)
        halong_broad = cotden_all_df["ocr_text"].map(lambda text: evidence_halong_product_from_ocr(text, broad=True))
        variants["tmp_halong_nan_cotden_all_halong_broad"] = _fill_empty(cotden_all_df, halong_broad)

    metadata = {}
    for name, df in variants.items():
        if canonicalize_after_rules:
            df = _canonicalize_df(df, canonicalizer)
        path = output_dir / f"submission_{name}.csv"
        _write(df, path)
        outputs[name] = path
        metadata[name] = {
            "rows": int(len(df)),
            "empty_product": int(df["product_name"].astype(str).str.strip().eq("").sum()),
            "filled_vs_base": int(
                (
                    base["product_name"].astype(str).str.strip().eq("")
                    & df["product_name"].astype(str).str.strip().ne("")
                ).sum()
            ),
            "unique_product": int(df["product_name"].astype(str).str.strip().nunique()),
            "canonical_map": str(canonical_map) if canonical_map else "",
            "canonicalize_before_rules": bool(canonicalize_before_rules),
            "canonicalize_after_rules": bool(canonicalize_after_rules),
        }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--base-oldproduct", type=Path, default=None)
    parser.add_argument("--ocr-submission", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tmp-submission", type=Path, default=None)
    parser.add_argument("--canonical-map", type=Path, default=None)
    parser.add_argument("--canonicalize-before-rules", action="store_true")
    parser.add_argument("--canonicalize-after-rules", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = args.base_oldproduct or args.base_submission
    base = pd.read_csv(base_path, keep_default_na=False)
    ocr = pd.read_csv(args.ocr_submission, keep_default_na=False)
    train = pd.read_csv(args.train_labels, keep_default_na=False)
    tmp = pd.read_csv(args.tmp_submission, keep_default_na=False) if args.tmp_submission else None
    outputs = build_variants(
        base,
        ocr,
        train,
        args.output_dir,
        tmp_submission=tmp,
        canonical_map=args.canonical_map,
        canonicalize_before_rules=args.canonicalize_before_rules,
        canonicalize_after_rules=args.canonicalize_after_rules,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
 