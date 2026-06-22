# URA Hackathon 2026 — Implementation Plan

> Living reference document. We follow this; we update it when decisions change.
> Created: 2026-06-15 · Deadline: **2026-06-24 23:59 (UTC+7)** · ~9 days.

---

## 1. Objective & Ground Truth

**Goal:** For each `image_id` in `test.csv` (2,006 images), predict:
- `ocr_text` — all visible text, diacritics preserved, no `\n`/`\t`, single spaces.
- `product_name` — primary product as `Brand + Product Line`; no price/promo/weight/slogan; `""` if none.

**Metric:** `Score = 0.6 × F1_product + 0.4 × (1 − CER)`
- `F1_product`: **token-level, case-insensitive** set F1 on product_name. Both-empty → 1.0.
- `CER`: char error rate on ocr_text (Levenshtein / len(GT)), clamped to 1.0. GT empty & pred empty → 0.0 (perfect); GT empty & pred non-empty → 1.0 (worst).

**Reference scores:** all-empty ≈ 0.25 · baseline ≈ 0.50 · perfect = 1.0.

**Strategy locked with user (2026-06-15):**
- Optimize **leaderboard first**, trim to lightweight thresholds later (Round 3 thresholds still unannounced).
- **Open-source pretrained OCR allowed** (PaddleOCR / VietOCR / Tesseract), no cloud APIs.
- Hardware: i7-12700H (20 threads) + RTX 3050 Ti 4GB.

**Update 2026-06-15 — CPU dev pivot:** the 2.4 GB GPU torch (cu121) wheel repeatedly
failed mid-download (unstable connection, `IncompleteRead`). Pivoted to **CPU torch**
(~200 MB) + **PaddleOCR-CPU** as the primary engine. Dev runs on CPU (20 threads), OCR
is cached once, and our timing numbers become real CPU numbers for the Round-3
efficiency table. GPU can be revisited if a reliable download path appears.

---

## 2. Key Data Facts (verified on disk)

| Fact | Value |
|---|---|
| Train images / labels | 4,892 / 4,892 |
| Test images | 2,006 |
| Empty `ocr_text` in train | **20.2%** |
| Empty `product_name` in train | **40.7%** |
| Unique product strings | 495 (very head-heavy) |
| Median OCR length / max | ~100 / 500 chars |
| Domain | FMCG: Pate + Milk; many TikTok news/scandal headline thumbnails |
| Top entities | Đồ Hộp Hạ Long, Pate Cột Đèn Hải Phòng, NAN, Nestlé, Highlands Coffee, Vinamilk |

**Implications:**
- ~40% of product credit is won by **correctly predicting empty**. An empty-gate is high-leverage.
- Product label space is small & concentrated → a **gazetteer + fuzzy matcher** covers most cases cheaply.
- CER battle = Vietnamese diacritics → OCR engine choice dominates the 0.4 term.

---

## 3. Local Project Structure (to build)

```
URA_Hackathon_Kaggle_Competition_2026/
├── Description.md                         # given
├── IMPLEMENTATION_PLAN.md                 # this file
├── lightweight-baseline-reference-starter.ipynb
├── train.csv / train_labels.csv / test.csv / sample_submission.csv
├── train_images/train_images/*.jpg
├── test_images/images/*.jpg
├── src/
│   ├── config.py            # paths, constants, thresholds
│   ├── scoring.py           # exact composite metric + token-F1 + CER
│   ├── data.py              # load csvs/images, train/val split
│   ├── ocr_engines.py       # pluggable: paddle / vietocr / tesseract / easyocr
│   ├── ocr_postprocess.py   # normalize, dedupe, empty-gate
│   ├── product_extract.py   # gazetteer + fuzzy + ML head
│   ├── run_ocr.py           # batch OCR -> cache parquet/json
│   ├── build_submission.py  # assemble + validate + export CSV
│   └── evaluate.py          # offline scoring on val split
├── cache/
│   ├── ocr_<engine>_train.parquet
│   └── ocr_<engine>_test.parquet
├── artifacts/
│   ├── gazetteer.json
│   └── product_head.joblib
├── submissions/
│   └── submission_<tag>.csv
└── notebooks/
    └── 01_eda.ipynb / 02_ocr_bakeoff.ipynb
```

Submission/inference also delivered as a clean notebook for the competition write-up.

---

## 4. Phased Plan

### Phase 0 — Environment & EDA  *(Day 1)*
- [ ] Verify Python env; create `requirements.txt`. Confirm GPU/CUDA visible to PaddlePaddle/torch.
- [ ] Install & smoke-test OCR engines: **PaddleOCR (PP-OCRv4/v5, vi)**, **VietOCR**, **Tesseract (vie)**, keep EasyOCR as baseline reference.
- [ ] EDA notebook: empty-rate by thumbnail type, OCR length distribution, brand frequency table, product token-length distribution, duplicate/near-duplicate images (TikTok thumbnails repeat).
- [ ] **Deliverable:** verified stats + a brand/alias frequency list feeding the gazetteer.

### Phase 1 — Scoring Harness & Validation  *(Day 1)*
- [ ] Implement `scoring.py` = exact competition metric (copy formula from baseline Cell 7, unit-test on known cases).
- [ ] Build a **train/val split** (e.g. 80/20, grouped so near-duplicate thumbnails don't leak across split).
- [ ] `evaluate.py`: given predictions df → composite + component scores (F1, CER) + breakdown by empty/non-empty.
- [ ] **Deliverable:** one command scores any candidate offline. No blind submissions.

### Phase 2 — OCR Engine Bake-off  *(Days 2–3)*
- [ ] `ocr_engines.py` with a common `transcribe(image)->str` interface, GPU on for dev.
- [ ] `run_ocr.py`: batch over train+test, **cache results to parquet** (engine, image_id, raw_text, boxes, conf). Parallelize across CPU/GPU.
- [ ] Score each engine's raw OCR CER on val (with current post-proc). Compare: Paddle vs VietOCR vs Tesseract vs EasyOCR.
- [ ] Consider **detector + recognizer** combos (Paddle det + VietOCR rec) if it wins CER.
- [ ] **Deliverable:** chosen primary OCR engine + cached OCR for all images. **Milestone target: CER term beating baseline.**

### Phase 3 — OCR Post-processing & Empty Gate  *(Day 3)*
- [ ] Reading-order sort (top→bottom, left→right), whitespace normalize, strip `\n`/`\t`, dedupe repeated tokens/lines.
- [ ] Confidence threshold tuning per engine (baseline used 0.35).
- [ ] **Empty-gate**: classify "no readable text" → output `""` (avoid CER=1 on blank frames). Use box count + mean confidence + total char length; tune threshold on val to maximize CER term.
- [ ] Diacritic/Unicode NFC normalization; optional Vietnamese spell/diacritic post-correction (only if it lowers val CER).
- [ ] Truncate to ≤500 chars (matches GT truncation; good practice).
- [ ] **Deliverable:** post-processed OCR maximizing the 0.4 CER term on val.

### Phase 4 — Product Name Extraction  *(Days 4–5)*  ← biggest lever (0.6 weight)
- [ ] **Gazetteer**: normalized brand → aliases + product lines, derived from `train_labels` + domain knowledge (extend baseline `BRAND_RULES`). Handle spelling variants (CANFOCO/CANFOOD/HALONG → Ha Long Canfoco; Patê→Pate; HẠ LONG→Ha Long).
- [ ] **Matcher**: fuzzy/alias match over OCR text → canonical `Brand + Line`. Token-aware so partial OCR still hits.
- [ ] **ML head** (fallback when rules miss): TF-IDF char n-grams + classifier (baseline pattern), trained on `train_labels`; tune `prob_threshold` / `min_class_count`.
- [ ] **Empty-product gate**: when no brand evidence, output `""` (wins ~41% of cases). Tune precision/recall on val.
- [ ] Token-F1 oracle ablation: run extractor on **GT ocr_text** to find the F1 ceiling of the extractor alone (isolates OCR error vs extractor error).
- [ ] **Deliverable:** product extractor maximizing val F1. **Milestone: product F1 well above baseline 0.44 fill.**

### Phase 5 — Integration & First Full Submission  *(Day 5–6)*
- [ ] `build_submission.py`: run full pipeline on test cache → assemble → run **AC-1..AC-7 checks** → empty→`" "` → UTF-8, `QUOTE_ALL`.
- [ ] Produce `submission_v1.csv`; record predicted val score and the exact config that made it.
- [ ] Sanity vs reference (all-empty ≈0.25, baseline ≈0.5).
- [ ] **Deliverable:** valid submission beating baseline on val; first real upload.

### Phase 6 — Iteration & Lightweight Track  *(Days 6–8)*
- [ ] Error analysis on val: worst CER images, product false-pos/neg. Targeted fixes (gazetteer gaps, det/rec tweaks, gate thresholds).
- [ ] **Lightweight variant**: swap primary OCR → PP-OCR mobile / Tesseract / quantized; measure score drop AND **CPU-only latency + model size** (build the Round-3 efficiency table: MB, ms/image, peak RAM).
- [ ] Keep both an "accuracy" and a "lightweight" config; document the trade curve.
- [ ] **Deliverable:** two tuned configs + measured efficiency numbers.

### Phase 7 — Finalize  *(Days 8–9)*
- [ ] Pick **2 final submissions** (per Kaggle rule): typically best-accuracy + best lightweight-compliant.
- [ ] Freeze code; clean inference notebook for the write-up; verify reproducibility from cache-free cold start.
- [ ] Final CPU benchmark + checklist pass; submit.

---

## 5. Milestones & Target Scores (val-measured)

| Milestone | Target composite | Gate |
|---|---|---|
| M1 Harness + baseline reproduced | ≈ 0.50 | scoring trusted |
| M2 Better OCR engine | 0.40 term ↑ (lower CER) | beats EasyOCR CER |
| M3 Strong product extractor | 0.6·F1 ↑ | F1 > baseline |
| M4 Full integrated v1 | **> 0.55** | beats baseline end-to-end |
| M5 Tuned + lightweight track | **0.60+** target | efficiency table ready |

---

## 6. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Round-3 lightweight thresholds unknown | May invalidate heavy OCR | Maintain trimmable lightweight track from Phase 6 |
| PaddleOCR/Paddle install flaky on Windows | Blocks OCR | Test Day 1; VietOCR/Tesseract fallbacks; CPU wheel if GPU fails |
| Hallucinated OCR on blank frames | CER penalty | Empty-gate (Phase 3) tuned on val |
| Product label noise / token mismatch | F1 ceiling | Canonical gazetteer; oracle ablation to separate causes |
| Near-duplicate thumbnails leak across split | Optimistic val | Group split in Phase 1 |
| Overfitting public LB (50% of test) | Private LB drop | Trust val; pick robust final 2 |

---

## 7. Working Conventions
- Cache OCR once; never re-run OCR to test product/post-proc changes.
- Every change measured on val **before** any submission.
- Tag each submission with its exact config in `submissions/` + a one-line log.
- Keep diacritics in NFC; CSV always UTF-8 + `QUOTE_ALL`; empty → `" "`.

---

## 8. Immediate Next Actions (Phase 0 + 1)
1. Create `src/` skeleton + `config.py` + `requirements.txt`.
2. Implement & unit-test `scoring.py`.
3. Smoke-test PaddleOCR / VietOCR / Tesseract on 5 sample images (GPU + CPU).
4. EDA: confirm stats + emit brand/alias frequency list.
5. Build grouped train/val split.

---

## 9. Progress Log

**2026-06-15**
- Phase 0/1 DONE: env `ura` (py3.11), `src/{config,scoring,data}.py`, scoring verified
  exact (oracle=1.0, all-empty=0.325), grouped val split 3912/980 (0 leakage).
- Env: CPU torch (GPU wheel undownloadable). Two envs — `ura` (torch/easyocr),
  `ura_paddle` (torch-free paddle; avoids OpenMP/shm.dll WinError 127 clash).
- Phase 2 DONE: `ocr_{engines,postprocess}.py`, `run_ocr.py` (parquet cache+resume),
  `evaluate.py`. Bake-off on 300 val → **PaddleOCR ocr_term 0.520 @1.34s/img** beats
  EasyOCR 0.496 @3.1s/img → PaddleOCR PICKED. Full val+test cache running.
- CER analysis: **GT = clean Vietnamese headline, not full transcription.** Lose CER to
  (1) mangled diacritics (paddle `vi`=Latin model), (2) ~16 noise tokens/img
  (@handles/dates/ISO/fine-print/repeats), (3) ~20 missed headline tokens.

**Re-sequencing decision:** do **Phase 4 (product F1, 0.6 weight) BEFORE heavy Phase 3
CER work** — product extraction is robust to OCR noise (brand keywords survive
diacritic errors), higher leverage, lower risk. Then Phase 3: VietOCR (diacritics) +
noise filtering to cut CER.

- Phase 4 DONE: `product_extract.py` — canonicalize labels to MODAL form per token-set
  group; diacritic-folded TF-IDF char n-grams + LogReg gate & classifier. Val (oracle
  OCR) F1 **0.571** (min_class_count=5, gate=0.45, 59 classes). On REAL PaddleOCR val
  text F1 only drops to **0.523** (folding works).
- **MILESTONE — first end-to-end val composite = 0.5088** (product_f1 0.523, ocr_term
  0.488). Already ≈ baseline, before any CER work. Refs on val: all-empty 0.155,
  OCR-only 0.350.
- Weak half = CER (ocr_term 0.488). Next cheap lever: **noise-filter the cached OCR**
  (handles/dates/ISO/repeats — ~16 extra tokens/img) with no re-OCR; then VietOCR.
- TODO: OCR train split (3912) → retrain product head on OCR text (kill train/test
  mismatch); tune gate to val empty-rate.

**2026-06-15 (cont.)**
- **Fixed val-split bug**: empties were lumped into one group → val had 0% empty. Now
  per-row groups → val 21.8% empty-OCR / 41% empty-prod (representative). Re-OCR not
  needed (per-image cache); use `all` cache filtered by split.
- Local all-train PaddleOCR **OOM-killed at 2980/4892** (machine has only 15.7GB RAM,
  ~1.2GB free → swapping). Test 100% cached; train/val ~60% — enough to tune. Don't
  fight local RAM; heavy OCR → Kaggle GPU.
- Disk: reclaimed 4.6GB (pip cache purge). Real project footprint ~2.85GB (two envs).
- **Empty-gate** (`empty_gate.py`): LogReg on [log_boxes, mean_conf, log_chars] →
  +0.006 ocr_term @thr 0.6 (small; paddle rarely hallucinates on empties).
- **Product head: train on clean GT > OCR'd text** (val F1 0.561 vs 0.527).
- **Honest val composite = 0.540** (v2 = PaddleOCR + GT product head + empty-gate).
  product_f1 0.561 (strong), ocr_term 0.509 (weak → VietOCR is next lever).
- **Kaggle GPU VietOCR notebook** written (`notebooks/kaggle_vietocr_gpu.ipynb`):
  EasyOCR-CRAFT detect + VietOCR recog, outputs `ocr_vietocr_{test,all}.parquet`.
  User running it. On return → compare CER vs PaddleOCR, integrate.
- Submissions: v1 (incomplete, void), v1b (0.538), **v2 (0.540, best)**.
