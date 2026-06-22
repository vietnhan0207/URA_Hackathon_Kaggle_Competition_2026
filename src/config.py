"""Central paths and constants for the URA Hackathon pipeline.

Import this everywhere instead of hardcoding paths.
"""
from __future__ import annotations

from pathlib import Path

# --- Roots -------------------------------------------------------------------
# This file lives in <PROJECT_ROOT>/src/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Input data (given) ------------------------------------------------------
TRAIN_CSV = PROJECT_ROOT / "train.csv"
TRAIN_LABELS_CSV = PROJECT_ROOT / "train_labels.csv"
TEST_CSV = PROJECT_ROOT / "test.csv"
SAMPLE_SUBMISSION_CSV = PROJECT_ROOT / "sample_submission.csv"

TRAIN_IMAGES_DIR = PROJECT_ROOT / "train_images" / "train_images"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images" / "images"

# --- Generated artifacts -----------------------------------------------------
CACHE_DIR = PROJECT_ROOT / "cache"            # cached OCR outputs (parquet)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"    # gazetteer, trained heads
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SPLITS_DIR = PROJECT_ROOT / "splits"          # train/val id lists

for _d in (CACHE_DIR, ARTIFACTS_DIR, SUBMISSIONS_DIR, SPLITS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Metric weights (fixed by competition) -----------------------------------
W_PRODUCT_F1 = 0.6
W_OCR = 0.4
MAX_OCR_LEN = 500  # train labels truncated here; good practice to cap

# --- Validation --------------------------------------------------------------
VAL_FRACTION = 0.20
RANDOM_SEED = 42


def train_image_path(image_id: str) -> Path:
    return TRAIN_IMAGES_DIR / f"{image_id}.jpg"


def test_image_path(image_id: str) -> Path:
    return TEST_IMAGES_DIR / f"{image_id}.jpg"
