# Overview

## Goal
Build a lightweight, in-house AI pipeline that automatically extracts OCR text and product names from social media images (TikTok, Instagram, etc.), reducing reliance on costly cloud OCR APIs.

## Constraints (Lightweight Requirement)
To align with the in-house deployment goal, solutions will be evaluated on practical efficiency during the Final Presentation round. Participants are strongly encouraged to build solutions that:

- Run efficiently on CPU (no GPU dependency)
- Minimize model size and memory footprint
- Achieve fast inference speed per image

Specific thresholds will be announced before the Final Presentation round.

# Description

## Background
Social media platforms such as TikTok and Instagram generate massive volumes of product-related visual content daily. Businesses increasingly rely on this content for market intelligence, competitor analysis, and product cataloging. However, manually extracting structured product information from images is costly, slow, and unscalable.

This challenge invites participants to build an automated, lightweight AI pipeline capable of reading text from social media images and identifying product names — deployable in-house without dependency on expensive cloud OCR services.

## Problem Statement
Given a collection of images sourced from social media posts, participants must:

- **OCR Text Extraction** — Recognize and transcribe all visible text in the image
- **Product Name Extraction** — Identify and return named product entities from the recognized text (e.g., brand names, product lines, model names)

### Domain Context
Images may contain:
- Promotional banners with overlaid text
- Product packaging with brand/model information
- Sale announcements with pricing and product details
- Mixed Vietnamese–English text

## Data Description

| File/Folder | Description |
| :--- | :--- |
| `train.csv` | Image IDs with ground truth OCR text and product names |
| `test.csv` | Image IDs without labels — your model must predict these |
| `sample_submission.csv` | Example of the required submission format |
| `/images/` | Folder containing all JPG/PNG image files |

## Baseline
A simple baseline using PaddleOCR + regex extraction is provided in the Starter Notebook (which is the `lightweight-baseline-reference-starter.ipynb` in this folder). Participants are encouraged to beat this baseline as a first milestone.

## What Makes a Good Solution?
Beyond accuracy, the best solutions will be:

- **Lightweight** — small model size, fast inference
- **Generalizable** — performs well on unseen private test data
- **Practical** — deployable on standard CPU hardware without GPU

## Evaluation
Submissions are evaluated based on the quality of both OCR transcription and product name extraction against a hidden ground truth.

The final leaderboard score combines multiple metrics with weights determined by the organizers. Higher is better.

## Submission File
For each `image_id` in `test.csv`, predict the OCR text and product name.
The file should contain a header and have the following format: 

```csv
image_id,ocr_text,product_name
img_301,"Kem dưỡng da Neutrogena giảm 50%","Neutrogena"
img_302,"Giày Nike Air Max mới về","Nike Air Max"
img_303,"Flash sale hôm nay - Laptop Dell XPS 15","Dell XPS 15"
```

### Rules:
- `image_id` must match exactly with IDs in `test.csv`
- `ocr_text`: full transcribed text visible in the image
- `product_name`: primary product entity identified (single best match)
- Leave `product_name` as empty string `""` if no product is detected
- File must be UTF-8 encoded (important for Vietnamese text)

# Submission Format

## 1. File format
Teams submit one CSV file via Kaggle: Submit Predictions.

| Property | Requirement |
| :--- | :--- |
| **Columns** | Exactly 3: `image_id`, `ocr_text`, `product_name` |
| **Rows** | 2,006 — one row per image in `test.csv` |
| **Encoding** | UTF-8 (required for Vietnamese) |
| **Header** | Required: `image_id,ocr_text,product_name` |

## 2. Column definitions

| Column | Meaning |
| :--- | :--- |
| `image_id` | ID from `test.csv`, format img_XXXX (e.g., `img_2934`) |
| `ocr_text` | All visible text in the image, joined with spaces (no `\n` / `\t`) |
| `product_name` | Primary product: [Brand] [Product Line]; empty if none |

**Example (valid row):**
```csv
"image_id","ocr_text","product_name"
"img_2934","Sữa tươi Vinamilk Flex 180ml không đường","Vinamilk Flex"
```

**Example (no product / no text):**
```csv
"img_2935"," "," "
```

## 3. Empty fields
Kaggle often treats blank CSV cells as null and rejects the file.

| In your pipeline | In the CSV you upload |
| :--- | :--- |
| Empty string `""` | Single space `" "` |

*The metric strips whitespace before scoring, so `" "` and `""` score the same.*

**Recommended export (as in baseline notebook):**
```python
# empty ocr_text / product_name → " " before writing CSV
out.loc[out[col] == "", col] = " "
out.to_csv("submission.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
```

## 4. ID rules (metric will error if violated)

| Rule | Requirement |
| :--- | :--- |
| **Exact match** | Set of `image_id` must match `test.csv` exactly |
| **No missing** | All 2,006 IDs present |
| **No extra** | No IDs outside `test.csv` |
| **No duplicates** | Each `image_id` appears once |
| **Order** | Any order is fine; metric joins on `image_id` |

*Invalid submissions show a visible error on Kaggle (no silent partial score).*

## 5. Content rules

### `ocr_text`
- Full visible text, left-to-right / top-to-bottom
- Keep Vietnamese diacritics (Sữa, dưỡng, …)
- No newlines or tabs — use a single space between lines
- No readable text → empty (`" "` in CSV)
- Train labels are truncated at 500 chars; keeping OCR ≤ 500 is good practice (not a hard submit limit in the metric)

### `product_name`
- Format: Brand + Product Line (e.g. Vinamilk Flex, Ha Long Canfoco Pate Cột Đèn)
- Do not include: price, discount, weight, quantity, hashtags, slogans
- Brand only when no line is visible → e.g. Vinamilk
- No product → empty (`" "` in CSV)

## 6. Pre-submit checklist

| # | Check |
| :--- | :--- |
| **AC-1** | Row count = 2,006 |
| **AC-2** | No extra `image_id` |
| **AC-3** | No missing `image_id` |
| **AC-4** | No duplicate `image_id` |
| **AC-5** | No null/NaN cells |
| **AC-6** | No `\n` or `\t` in `ocr_text` |
| **AC-7** | Columns = `image_id`, `ocr_text`, `product_name` only |

*Baseline notebook Cell 6 runs these checks before writing `submission.csv`.*

## 7. How to submit on Kaggle
1. Go to **Competition** → **Submit Predictions**
2. Upload `submission.csv`
3. Wait for score on Public Leaderboard (50% of test set)

**Limits (from rules):**
- 5 submissions / day
- Select 2 final submissions for Private Leaderboard judging
- **Deadline**: 2026-06-24 23:59 (UTC+7)
- Code tab may be disabled — teams submit CSV only, not a Kaggle notebook run.

## 8. Reference scores (sanity check)

| Submission | Expected score |
| :--- | :--- |
| `sample_submission.csv` (all empty) | ~0.25 |
| Baseline reference notebook | ~0.5 |
| Perfect ground truth | 1.0 |

*All-empty is not 0 because ~14% of test rows have empty GT for both OCR and product.*

## 9. Minimal valid template
Copy structure from `sample_submission.csv`:

```csv
"image_id","ocr_text","product_name"
"img_2934","your ocr here","Vinamilk Flex"
"img_2935"," "," "
...
```
*Fill all 2,006 rows from `test.csv`, use `" "` for empty text fields, UTF-8, quote fields if they contain commas.*


## Dataset Description

The competition dataset consists of JPEG thumbnail images collected from TikTok videos in the Vietnamese market, covering two FMCG product categories: Pate (canned meat paste) and Milk (dairy products). Images are sourced via the YouNet Media API and represent four thumbnail types: cover, origin_cover, music_cover, and dynamic_cover.

Text visible in images is primarily Vietnamese. Diacritics must be preserved in your submission (e.g. Sữa, not Sua).

The dataset is split into a training set (~4,892 images) and a test set (2,006 images). Image IDs do not overlap between train and test.

### What am I predicting?
For each `image_id` in `test.csv`, submit:

| Field | Description |
| :--- | :--- |
| `ocr_text` | All visible text in the image, read left→right, top→bottom. Use `""` if no readable text. |
| `product_name` | Primary product or brand (brand + model). Exclude prices and promotions. Use `""` if none. |

**Scoring:**
`Score = 0.6 × F1_product + 0.4 × (1 − CER)`

---

### Files

| File | Description |
| :--- | :--- |
| `images.zip` | Test set images (2,006 JPEGs). Extract to get `images/img_XXXX.jpg`. |
| `train_images.zip` | Training set images (~4,892 JPEGs). Extract to get `train_images/img_XXXX.jpg`. |
| `train.csv` | Training set — one column `image_id`. Use with `train_labels.csv` for supervised learning. |
| `train_labels.csv` | Training ground truth — `image_id`, `ocr_text`, `product_name` for all training images. |
| `test.csv` | Test set — one column `image_id`. No labels provided. |
| `sample_submission.csv` | Submission template. Use as format reference for your test predictions. |

---

### Columns

#### `train.csv`
| Column | Type | Description |
| :--- | :--- | :--- |
| `image_id` | string | Unique training image identifier, e.g. `img_0001`. Maps to `train_images/img_0001.jpg`. ID range: `img_0001` – `img_6516`. |

#### `train_labels.csv`
| Column | Type | Description |
| :--- | :--- | :--- |
| `image_id` | string | Matches every ID in `train.csv`. |
| `ocr_text` | string | Ground-truth OCR — all visible text in the image. UTF-8. Empty string `""` if none. |
| `product_name` | string | Ground-truth primary product or brand name. Empty string `""` if none. |

#### `test.csv`
| Column | Type | Description |
| :--- | :--- | :--- |
| `image_id` | string | Unique test image identifier. Maps to `images/img_XXXX.jpg`. ID range: `img_2934` – `img_6900`. |

#### `sample_submission.csv` / your submission
| Column | Type | Description |
| :--- | :--- | :--- |
| `image_id` | string | Must match every ID in `test.csv` — same count, no extras, no missing. |
| `ocr_text` | string | Predicted visible text. UTF-8. Use `""` if none. |
| `product_name` | string | Predicted primary product or brand name. Use `""` if none. |

---

### Dataset summary

| Split | Images |
| :--- | :--- |
| **Train** | ~4,892 |
| **Test** | 2,006 |

### Submission rules
- CSV must be UTF-8 encoded
- Exactly 3 columns: `image_id`, `ocr_text`, `product_name`
- All 2,006 test rows required — no missing or duplicate IDs
- Empty fields use `""`, not null or NaN




# Discussion From The Competition Host

## Product Name & Brand Name - Definition

In our The 2nd URA Hackathon 2026, `product_name` and brand name are not separate columns — you submit only one `product_name` field, but its content is usually structured as brand + product line.

### Definitions

**Brand name**
The company or label name recognized on the image or in the OCR text.
*Examples:* Vinamilk, Nestlé, Vissan, Ha Long Canfoco, TH True Milk, Dutch Lady

**Product name (in this task)**
The primary product entity the thumbnail is about — typically in the form:
`[Brand] [Product Line / Model / Variant]`

This is what you predict in the `product_name` column — not the full OCR transcript.

### What counts as a valid `product_name`?
A valid `product_name` should satisfy:

| Criterion | Correct | Incorrect |
| :--- | :--- | :--- |
| **Product info only** | Vinamilk Flex | Sữa tươi Vinamilk Flex 180ml không đường giảm 20% |
| **No price / promotion** | Nestlé Milo | Milo giảm 30% mua 2 tặng 1 |
| **No weight / quantity** | Vissan Pate Heo | Vissan Pate Heo 170g combo 3 hộp |
| **No slogans / hashtags** | TH True Milk | TH True Milk #sale #tiktok |
| **One primary product** | the largest/most prominent product on screen | listing multiple products |
| **Brand + line when both are clear** | Ha Long Canfoco Pate Cột Đèn | only Pate (too generic when brand is visible) |
| **Brand logo only, no product line** | Vinamilk | guessing a line not shown in the image |
| **No product identifiable** | `""` (empty string) | random guess |

**Scoring:** the metric uses token-level F1 (case-insensitive) on `product_name`.
*Example:* GT = `Vinamilk Flex`, prediction = `vinamilk flex` → tokens still match.

---

## Examples

### 1. Product packaging / ads

| OCR (abbreviated) | Brand | Valid `product_name` |
| :--- | :--- | :--- |
| Sữa tươi tiệt trùng Vinamilk Flex 180ml không đường Mua 10 tặng 2 | Vinamilk | Vinamilk Flex |
| NESTLÉ MILO Chocolate Malt Drink 3in1 | Nestlé | Nestlé Milo |
| Vissan PATE HEO 170g | Vissan | Vissan Pate Heo |
| Dutch Lady Grow+ 900g | Dutch Lady | Dutch Lady Grow+ |
| Ba Vì Gold 1L | Ba Vì | Ba Vì Gold |

### 2. News / TikTok headline thumbnails (important in this dataset)
Many images are not packaging photos but news or scandal headlines. You should still extract `product_name` when the text mentions a brand or product.

| OCR (abbreviated) | Valid `product_name` |
| :--- | :--- |
| HA LONG CANFOCO ... PATE CỘT ĐÈN HẢI PHÒNG | Ha Long Canfoco Pate Cột Đèn |
| ĐỒ HỘP HẠ LONG TẠM DỪNG SẢN XUẤT | Ha Long Canfoco or Đồ Hộp Hạ Long |
| Vinamilk EST 1976 ... urgent announcement | Vinamilk |
| LỜI NHẮN NHỦ VỀ NHÂN QUẢ VÀ DI SẢN TRONG KINH DOANH | `""` (no brand/product) |

### 3. When is brand-only enough?
When the image only clearly shows the brand, with no specific product line:

| OCR | `product_name` |
| :--- | :--- |
| Vinamilk EST 1976 ... national dairy brand... | Vinamilk |
| Large Milo logo, no variant visible | Nestlé Milo |

### 4. When to use empty `""`?
- No readable text and no brand/product mention
- Memes, people, scenery, generic news unrelated to FMCG
- Examples in `solution.csv`: `img_2935`, `img_2940` — empty OCR or no product entity → `product_name` = empty
- Roughly ~14% of test images have empty ground-truth OCR and product — which is why an all-empty `sample_submission` still scores ~0.25.

---

## Brand vs product line - quick split

| Brand | Product Line | Valid `product_name` |
| :--- | :--- | :--- |
| Vinamilk | Flex | Vinamilk Flex |
| Nestlé | Milo | Nestlé Milo |
| Vissan | Pate Heo | Vissan Pate Heo |
| Ha Long Canfoco | Pate Cột Đèn | Ha Long Canfoco Pate Cột Đèn |

## Spelling normalization (recommended)

| On image | Prefer |
| :--- | :--- |
| CANFOCO, CANFOOD, HALONG | Ha Long Canfoco |
| HẠ LONG | Ha Long |
| Patê | Pate |

---

## Takeaway
- **Brand name** = the label/company name.
- **Product name** (submission column) = a short phrase describing the primary product as `Brand + Product Line`, taken from visible image/OCR content, without price, promotion, volume, or slogans; use empty string when no product is actually mentioned.