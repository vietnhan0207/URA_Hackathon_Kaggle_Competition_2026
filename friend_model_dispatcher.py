from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


DATA_PATH = Path("data/raw/train_labels.csv")
MISSING_PRODUCT = " "


@dataclass(frozen=True)
class ProductPrediction:
    product_name: str
    source: str
    reason: str


def clean_text(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def normalize_match_text(value: object) -> str:
    text = clean_text(value).replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


NORMALIZED_BRAND_RULES = [
    (r"ha long canfoco.*pate.*cot|cot den.*ha long canfoco", "Ha Long Canfoco Pate Cột Đèn", []),
    (r"ha long canfoco.*pate|canfoco.*pate.*cot|pate.*cot den.*canfoco", "Ha Long Canfoco Pate", []),
    (r"ha long canfoco|halong canfoco|halongcanfoco|canfood|canfoco", "Ha Long Canfoco", []),
    (r"do hop ha long|dohop ha long|do hop h long|do hop ho long|do hp hq long|h p h long", "Đồ Hộp Hạ Long", []),
    (r"pate.*cot den|pate cot den|cot den hai phong|cotden hai phong", "Pate Cột Đèn Hải Phòng", []),
    (r"ha long pate|pate ha long", "Ha Long Canfoco Pate", []),
    (r"vinamilk", "Vinamilk", ["flex", "adm gold", "sure", "canxi", "colosbaby", "colos baby", "ong tho", "dielac", "grow"]),
    (r"th true|thtrue", "TH True Milk", ["true yogurt", "grow", "school milk", "true butter"]),
    (r"dutch lady|co gai", "Dutch Lady", ["grow", "complete", "canxi", "mom"]),
    (r"nutifood|nuti\b", "Nutifood", ["growplus", "grow plus", "pedia", "iq"]),
    (r"ensure\b", "Abbott Ensure", ["gold", "original", "plus"]),
    (r"pediasure", "Abbott PediaSure", []),
    (r"similac", "Abbott Similac", []),
    (r"glucerna", "Abbott Glucerna", []),
    (r"\bmilo\b", "Nestlé Milo", []),
    (r"nestle|nestle", "Nestlé", ["milo", "coffee mate", "carnation", "nestea", "nan", "sua bot"]),
    (r"aptamil", "Aptamil", []),
    (r"friso\b", "Friso", ["gold", "comfort", "prestige"]),
    (r"meiji\b", "Meiji", ["growing", "step"]),
    (r"ba vi\b|bavi\b", "Ba Vì", ["gold"]),
    (r"lothamilk", "Lothamilk", ["canxi"]),
    (r"yomost", "Yomost", []),
    (r"dalat milk|da lat", "Đà Lạt Milk", []),
    (r"kun\b|kun milk", "Kun", ["chocolate", "strawberry"]),
    (r"fami\b", "Fami", ["canxi", "kid"]),
    (r"anlene", "Anlene", ["gold", "concentrate"]),
    (r"anchor\b", "Anchor", ["butter", "cream"]),
    (r"vissan", "Vissan", ["pate heo", "pate ga", "xuc xich", "lap xuong"]),
    (r"hafi\b", "Hafi", ["pate"]),
    (r"ba huan", "Ba Huân", ["pate"]),
    (r"san ha\b", "San Hà", ["pate"]),
    (r"\bcp\b", "CP", ["pate", "xuc xich"]),
    (r"long bien", "Long Biên", ["pate"]),
    (r"\bpate\b|pate", "Pate", []),
    (r"highlands? coffee|highland coffee|highland.*cf", "Highlands Coffee", []),
    (r"coffee house|the coffee", "Coffee House", []),
]

HALONG_TMP_PRODUCTS = {
    "do hop ha long",
    "pate cot den",
    "pate cot den hai phong",
}

HALONG_NAN_TMP_PRODUCTS = HALONG_TMP_PRODUCTS | {
    "nan",
    "nestle nan",
    "nestle nan optiproplus 1",
    "coffee house",
    "highland coffee",
    "highlands coffee",
}

def _contains_keyword(normalized: str, keyword: str) -> bool:
    compact = normalized.replace(" ", "")
    key = normalize_match_text(keyword)
    if key in {"g", "kg", "ml"}:
        return bool(re.search(rf"\d+\s*{key}\b", normalized))
    if key in {"gia", "ban", "mua", "hop", "lon", "goi", "chai"}:
        return bool(re.search(rf"\b{re.escape(key)}\b", normalized))
    return key in normalized or key.replace(" ", "") in compact

def extract_product(text: str) -> str:
    if not text or not text.strip():
        return ""
    normalized = normalize_match_text(text)
    for pattern, brand, lines in NORMALIZED_BRAND_RULES:
        if re.search(pattern, normalized, re.IGNORECASE):
            for line in lines:
                if _contains_keyword(normalized, line):
                    return f"{brand} {line.title()}"
            return brand
    return ""

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

def evidence_product_from_ocr(text: object) -> str:
    lowered = "" if text is None else str(text).lower()
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

def evidence_override_product_from_ocr(text: object, current_product: object) -> str:
    normalized = normalize_match_text(text)
    compact = normalized.replace(" ", "")
    current = "" if current_product is None else str(current_product)
    current_norm = normalize_match_text(current)
    current_compact = current_norm.replace(" ", "")

    has_highlands = "highlandscoffee" in compact or "highlandcoffee" in compact or "highlands" in compact
    has_do_hop_halong = any(marker in compact for marker in ["dohophalong", "d6hophalong", "dohopha"])
    has_canfoco = any(marker in compact for marker in ["canfoco", "canfoc", "cahfoco", "canfogo", "canpoco"])
    has_pate = "pate" in compact or "bate" in compact or "tate" in compact
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
        current_compact in {"highlandscoffee", "highlandcoffee"}
        and has_highlands
        and has_do_hop_halong
        and any(marker in compact for marker in ["ngungban", "cungcap", "lienquan", "phanhoi", "tamngungban"])
    ):
        return "Đồ Hộp Hạ Long"
    if current_compact in {"patecotdenhaiphong", "patecotden"} and has_do_hop_halong and has_news_company_context:
        return "Đồ Hộp Hạ Long"

    if has_canfoco and has_pate and has_cotden:
        return "Ha Long Canfoco Pate Cột Đèn"
    if has_canfoco and has_pate and current_compact in {"dohophalong", "halongcanfoco", "patecotdenhaiphong", "patecotden"}:
        return "Ha Long Canfoco Pate"
    if has_canfoco and current_compact in {"dohophalong", "nan", "pate", "patecotdenhaiphong", "patecotden"}:
        return "Ha Long Canfoco"

    if (
        has_pate
        and has_cotden
        and has_haiphong
        and current_compact in {"dohophalong", "nan", "coffeehouse", "highlandscoffee", "patecotden"}
    ):
        return "Pate Cột Đèn Hải Phòng"

    if current_norm in {"nestle milo", "nestle sua bot"} and "nestle" in normalized:
        return "Nestlé"

    return current



def _token_f1(gt: object, pred: object) -> float:
    gt_text = clean_text(gt)
    pred_text = clean_text(pred)
    if not gt_text and not pred_text:
        return 1.0
    gt_tokens = set(gt_text.split())
    pred_tokens = set(pred_text.split())
    if not gt_tokens or not pred_tokens:
        return 0.0
    common = gt_tokens & pred_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def product_f1_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean([_token_f1(gt, pred) for gt, pred in zip(y_true, y_pred)]))


@dataclass
class HybridProductExtractor:
    top_k: int = 50
    fuzzy_threshold: float = 85.0
    has_product_threshold: float = 0.75
    prob_threshold: float = 0.60
    max_features: int = 5000
    min_class_count: int = 2
    short_exact_margin: int = 0
    random_state: int = 42
    enable_lookup: bool = True
    enable_dictionary_exact: bool = True
    enable_fuzzy: bool = True
    enable_lightgbm: bool = True
    enable_short_override: bool = True
    enable_empty_gate: bool = True

    def __post_init__(self) -> None:
        self.top_products_: list[str] = []
        self.ocr_text_lookup_: dict[str, str] = {}
        self.normalized_ocr_text_lookup_: dict[str, str] = {}
        self.top_product_lookup_: dict[str, str] = {}
        self.top_product_normalized_lookup_: dict[str, str] = {}
        self.known_products_: list[str] = []
        self.known_product_lookup_: dict[str, str] = {}
        self.label_encoder_: LabelEncoder | None = None
        self.has_product_classifier_: Pipeline | None = None
        self.classifier_: Pipeline | None = None
        self.constant_product_: str | None = None

    def fit(self, train_df: pd.DataFrame) -> "HybridProductExtractor":
        df = train_df.copy()
        df["ocr_text"] = df["ocr_text"].map(clean_text)
        df["product_name"] = df["product_name"].fillna("").astype(str).str.strip()
        positive = df[(df["ocr_text"] != "") & (df["product_name"] != "")]
        gate_rows = df[df["ocr_text"] != ""]
        if gate_rows["product_name"].ne("").nunique() == 2:
            self.has_product_classifier_ = Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            analyzer="char",
                            ngram_range=(2, 4),
                            max_features=self.max_features,
                            min_df=1,
                        ),
                    ),
                    (
                        "lgbm",
                        LGBMClassifier(
                            max_depth=3,
                            num_leaves=7,
                            n_estimators=100,
                            learning_rate=0.08,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            random_state=self.random_state,
                            n_jobs=1,
                            verbose=-1,
                        ),
                    ),
                ]
            )
            self.has_product_classifier_.fit(gate_rows["ocr_text"], gate_rows["product_name"].ne("").astype(int))

        counts = positive["product_name"].value_counts()
        lookup_rows = df[df["ocr_text"] != ""].copy()
        lookup_rows["normalized_ocr_text"] = lookup_rows["ocr_text"].map(normalize_match_text)
        if not lookup_rows.empty:
            ocr_counts = lookup_rows.groupby(["ocr_text", "product_name"]).size().reset_index(name="count")
            best_ocr = ocr_counts.sort_values("count").drop_duplicates("ocr_text", keep="last")
            self.ocr_text_lookup_ = dict(zip(best_ocr["ocr_text"], best_ocr["product_name"]))
            normalized_counts = (
                lookup_rows[lookup_rows["normalized_ocr_text"] != ""]
                .groupby(["normalized_ocr_text", "product_name"])
                .size()
                .reset_index(name="count")
            )
            best_normalized = normalized_counts.sort_values("count").drop_duplicates(
                "normalized_ocr_text", keep="last"
            )
            self.normalized_ocr_text_lookup_ = dict(
                zip(best_normalized["normalized_ocr_text"], best_normalized["product_name"])
            )

        self.top_products_ = counts.head(self.top_k).index.tolist()
        self.top_product_lookup_ = {clean_text(name): name for name in self.top_products_}
        self.top_product_normalized_lookup_ = {normalize_match_text(name): name for name in self.top_products_}

        self.known_products_ = counts.index.tolist()
        self.known_product_lookup_ = {clean_text(name): name for name in self.known_products_}

        clf_rows = positive[positive["product_name"].isin(counts[counts >= self.min_class_count].index)]
        n_classes = clf_rows["product_name"].nunique()
        if n_classes == 1:
            self.constant_product_ = str(clf_rows["product_name"].iloc[0])
            return self
        if n_classes < 2:
            return self

        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(clf_rows["product_name"])
        self.classifier_ = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(2, 4),
                        max_features=self.max_features,
                        min_df=1,
                    ),
                ),
                (
                    "lgbm",
                    LGBMClassifier(
                        max_depth=3,
                        num_leaves=7,
                        n_estimators=120,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=self.random_state,
                        n_jobs=1,
                        verbose=-1,
                    ),
                ),
            ]
        )
        self.classifier_.fit(clf_rows["ocr_text"], y)
        return self

    def predict(self, ocr_text: object) -> str:
        return self.predict_with_reason(ocr_text).product_name

    def _base_predict_with_reason(self, ocr_text: object) -> ProductPrediction:
        text = clean_text(ocr_text)
        if not text:
            return ProductPrediction(MISSING_PRODUCT, "empty_gate", "empty_ocr_text")
        if self.enable_lookup:
            text_match = self._tier1_ocr_text_with_source(text)
            if text_match:
                return text_match

        if self.enable_short_override:
            short_exact = self._tier1_short_exact_product(text)
            if short_exact:
                return ProductPrediction(short_exact, "short_exact_override", "normalized_text_equals_or_near_product")

        if self.enable_empty_gate and not self._has_product(text):
            return ProductPrediction(MISSING_PRODUCT, "empty_gate", "has_product_classifier_below_threshold")

        if self.enable_dictionary_exact:
            tier1 = self._tier1_exact(text)
            if tier1:
                return ProductPrediction(tier1, "dictionary_exact", "top_product_substring_match")

        if self.enable_fuzzy:
            tier2 = self._tier2_fuzzy(text)
            if tier2:
                return ProductPrediction(tier2, "fuzzy", "rapidfuzz_ratio_above_threshold")

        if self.enable_lightgbm:
            tier3 = self._tier3_classifier(text)
            if tier3:
                return ProductPrediction(tier3, "lightgbm", "tfidf_lightgbm_probability_above_threshold")
        return ProductPrediction(MISSING_PRODUCT, "empty_gate", "no_tier_confident")

    def predict_with_reason(self, ocr_text: object) -> ProductPrediction:
        text = clean_text(ocr_text)
        if not text:
            return ProductPrediction(MISSING_PRODUCT, "empty_gate", "empty_ocr_text")

        pred_obj = self._base_predict_with_reason(ocr_text)
        pred = pred_obj.product_name

        if str(pred).strip() == "":
            fill = rule_product_from_ocr(text)
            if not fill:
                fill = evidence_product_from_ocr(text)
            if not fill:
                fill = evidence_cotden_product_from_ocr(text, strict=False)
            if fill:
                pred = fill

        pred = evidence_override_product_from_ocr(text, pred)

        if str(pred).strip() == "":
            pred = MISSING_PRODUCT

        return ProductPrediction(pred, pred_obj.source, pred_obj.reason)

    def predict_many(self, texts: pd.Series) -> list[str]:
        return [self.predict(text) for text in texts]

    def _has_product(self, text: str) -> bool:
        if self.has_product_classifier_ is None:
            return True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            proba = self.has_product_classifier_.predict_proba([text])[0]
        classes = list(self.has_product_classifier_.classes_)
        if 1 not in classes:
            return True
        return float(proba[classes.index(1)]) >= self.has_product_threshold

    def _tier1_ocr_text(self, text: str) -> str | None:
        prediction = self._tier1_ocr_text_with_source(text)
        return prediction.product_name if prediction else None

    def _tier1_ocr_text_with_source(self, text: str) -> ProductPrediction | None:
        if text in self.ocr_text_lookup_:
            return ProductPrediction(
                self.ocr_text_lookup_[text] or MISSING_PRODUCT,
                "ocr_text_lookup",
                "exact_ocr_text_majority_lookup",
            )
        normalized_text = normalize_match_text(text)
        if normalized_text in self.normalized_ocr_text_lookup_:
            return ProductPrediction(
                self.normalized_ocr_text_lookup_[normalized_text] or MISSING_PRODUCT,
                "normalized_lookup",
                "normalized_ocr_text_majority_lookup",
            )
        return None

    def _tier1_short_exact_product(self, text: str) -> str | None:
        normalized_text = normalize_match_text(text)
        if not normalized_text:
            return None
        for normalized_name, canonical_name in sorted(
            self.top_product_normalized_lookup_.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if not normalized_name:
                continue
            if normalized_text == normalized_name:
                return canonical_name
            if normalized_name in normalized_text and len(normalized_text) <= len(normalized_name) + self.short_exact_margin:
                return canonical_name
        return None

    def _tier1_exact(self, text: str) -> str | None:
        for cleaned_name, canonical_name in sorted(
            self.top_product_lookup_.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if cleaned_name and cleaned_name in text:
                return canonical_name
        normalized_text = normalize_match_text(text)
        for normalized_name, canonical_name in sorted(
            self.top_product_normalized_lookup_.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if normalized_name and normalized_name in normalized_text:
                return canonical_name
        return None

    def _tier2_fuzzy(self, text: str) -> str | None:
        if not self.known_product_lookup_:
            return None
        choices = list(self.known_product_lookup_.keys())
        match = process.extractOne(text, choices, scorer=fuzz.ratio)
        if match is None:
            return None
        cleaned_name, score, _ = match
        if score > self.fuzzy_threshold:
            return self.known_product_lookup_[cleaned_name]
        return None

    def _tier3_classifier(self, text: str) -> str | None:
        if self.constant_product_ is not None:
            return self.constant_product_
        if self.classifier_ is None or self.label_encoder_ is None:
            return None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            proba = self.classifier_.predict_proba([text])[0]
        best_idx = int(np.argmax(proba))
        if float(proba[best_idx]) <= self.prob_threshold:
            return None
        label = int(self.classifier_.classes_[best_idx])
        return str(self.label_encoder_.inverse_transform([label])[0])


@dataclass
class EnsembleProductExtractor:
    top_k: int = 50
    fuzzy_threshold: float = 85.0
    has_product_threshold: float = 0.75
    prob_threshold: float = 0.60
    max_features: int = 5000
    min_class_count: int = 2
    short_exact_margin: int = 0
    random_state: int = 42

    def __post_init__(self) -> None:
        self.extractor_gt = HybridProductExtractor(
            top_k=self.top_k,
            fuzzy_threshold=self.fuzzy_threshold,
            has_product_threshold=self.has_product_threshold,
            prob_threshold=self.prob_threshold,
            max_features=self.max_features,
            min_class_count=self.min_class_count,
            short_exact_margin=self.short_exact_margin,
            random_state=self.random_state
        )
        self.extractor_pred = HybridProductExtractor(
            top_k=self.top_k,
            fuzzy_threshold=self.fuzzy_threshold,
            has_product_threshold=self.has_product_threshold,
            prob_threshold=self.prob_threshold,
            max_features=self.max_features,
            min_class_count=self.min_class_count,
            short_exact_margin=self.short_exact_margin,
            random_state=self.random_state
        )

    def fit(self, train_df: pd.DataFrame, train_cache_df: pd.DataFrame | None = None) -> "EnsembleProductExtractor":
        self.extractor_gt.fit(train_df)
        if train_cache_df is not None:
            pred_train_df = train_df.copy()
            pred_train_df = pred_train_df.merge(train_cache_df[["image_id", "pred_ocr_text"]], on="image_id", how="left")
            pred_train_df["ocr_text"] = pred_train_df["pred_ocr_text"].fillna("")
            self.extractor_pred.fit(pred_train_df)
        else:
            self.extractor_pred.fit(train_df)
        return self

    def predict(self, ocr_text: object) -> str:
        return self.predict_with_reason(ocr_text).product_name

    def predict_with_reason(self, ocr_text: object) -> ProductPrediction:
        text = clean_text(ocr_text)
        if not text:
            return ProductPrediction(MISSING_PRODUCT, "empty_gate", "empty_ocr_text")

        pred_obj_gt = self.extractor_gt._base_predict_with_reason(ocr_text)
        pred_gt = pred_obj_gt.product_name

        pred_obj_pred = self.extractor_pred._base_predict_with_reason(ocr_text)
        pred_pred = pred_obj_pred.product_name

        pred = pred_gt
        if str(pred).strip() == "":
            if normalize_match_text(pred_pred) in HALONG_NAN_TMP_PRODUCTS:
                pred = pred_pred

        if str(pred).strip() == "":
            fill = rule_product_from_ocr(text)
            if not fill:
                fill = evidence_product_from_ocr(text)
            if not fill:
                fill = evidence_cotden_product_from_ocr(text, strict=False)
            if fill:
                pred = fill

        pred = evidence_override_product_from_ocr(text, pred)

        if str(pred).strip() == "":
            pred = MISSING_PRODUCT

        return ProductPrediction(pred, pred_obj_gt.source, pred_obj_gt.reason)

    def predict_many(self, texts: pd.Series) -> list[str]:
        return [self.predict(text) for text in texts]


def run_oof_cv(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> tuple[pd.DataFrame, float]:
    work = df.copy()
    work["ocr_text"] = work["ocr_text"].fillna("").astype(str)
    work["product_name"] = work["product_name"].fillna("").astype(str).str.strip()

    oof = pd.Series([MISSING_PRODUCT] * len(work), index=work.index, dtype=object)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(work), start=1):
        extractor = HybridProductExtractor(random_state=random_state + fold)
        extractor.fit(work.iloc[train_idx])
        preds = extractor.predict_many(work.iloc[valid_idx]["ocr_text"])
        oof.iloc[valid_idx] = preds
        fold_score = product_f1_score(work.iloc[valid_idx]["product_name"], oof.iloc[valid_idx])
        print(f"Fold {fold}: product_f1={fold_score:.4f}")

    cv_score = product_f1_score(work["product_name"], oof)
    result = work[["image_id", "ocr_text", "product_name"]].copy()
    result["pred_product_name"] = oof
    return result, cv_score


def main() -> None:
    df = pd.read_csv(DATA_PATH, keep_default_na=False)
    oof, score = run_oof_cv(df)
    out_path = Path("oof_hybrid_product_predictions.csv")
    oof.to_csv(out_path, index=False, encoding="utf-8")
    print(f"OOF CV Score (product_f1): {score:.4f}")
    print(f"Saved OOF predictions: {out_path}")


if __name__ == "__main__":
    main()
 
 