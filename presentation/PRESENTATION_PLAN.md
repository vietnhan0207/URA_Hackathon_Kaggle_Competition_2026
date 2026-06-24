# URA Hackathon 2026 — Presentation Plan
### "Competing the reliable way: a CV-disciplined OCR + product-extraction pipeline"

> **Meta-narrative (the hook for the supervisor):** the competition was executed
> as a faithful application of the *Kaggle CV Playbook*. Every slide ties an action
> we took to a principle in the deck. The story is not "we got a high score" — it
> is **"we built a score we can trust, and we can prove why each decision was made."**

---

## 0. How this document maps to the Playbook

| Playbook slide | Our section | Evidence we present |
| :-- | :-- | :-- |
| The metric is law | §2 Metric | Metric decomposition + per-term optimization |
| Understand → Validate → Baseline → Iterate → Ensemble | whole doc | Section order follows the loop |
| Public board is a trap / trust CV | §4, §7, §9 | CV→LB table, never probed |
| Make CV mirror the test split | §4 | GroupKFold-by-content rationale |
| Don't leak the answer | §4 | Fold-safe refitting of emit strings + gate |
| When CV & LB disagree, trust CV | §7, §9 | The 0.6142 CV → 0.6685 LB jump, explained |
| Mind the shake-up | §9 | Phase-1 public vs Phase-2 private analysis |

---

## Notebooks (the live evidence)

| Notebook | Purpose | Runs on |
|:--|:--|:--|
| `notebooks/eda_competition.ipynb` | **Main EDA** — mounts competition data, runs Data Understanding → Preprocessing → EDA (10 charts) | Kaggle (attach dataset) / Colab |
| `notebooks/eda_report_kaggle.ipynb` | Self-contained reference — embedded stats, no data needed | Anywhere |
| `notebooks/eda_report.ipynb` | Local full version — reads from cache parquets | `ura` env |

## Execution phases (how we build this, together)

- **Phase A — EDA notebook.** ✅ Done. `eda_competition.ipynb` mounts the competition
  dataset and produces 10 EDA charts covering: data understanding, preprocessing,
  product distribution, family grouping, empty rate, OCR length, diacritic analysis,
  word cloud, n-grams, product structure, feature correlations.
  *Charts saved to `figures/` in Kaggle working directory.*
- **Phase B — Methodology core.** Write §1–§4 (problem, metric math, data findings,
  CV design) with the theory boxes. Output: `presentation/02_methodology.md`.
- **Phase C — Models.** Write §5–§6 (OCR engine, product head) incl. the emit-gain
  derivation and the oracle/noise decomposition; build the engine comparison chart.
  Output: `presentation/03_models.md`.
- **Phase D — Experiment log & results.** Write §7–§8 (CV-gated experiment table,
  score progression) and results charts (score progression line, experiment Δ bar).
  Output: `presentation/04_results.md`.
- **Phase E — Generalization & conclusion.** Write §9–§10 (shake-up analysis,
  lessons, playbook checklist mapping). Output: `presentation/05_conclusion.md`.
- **Phase F — Slide outline.** Condense all .md → slide-by-slide outline
  (title + 3 bullets + 1 visual per slide) for the .pptx.

Each phase ends with concrete artifacts (charts + prose) so the deck assembles itself.

---

## 1. Title & Problem Framing  *(Understand)*
- One-line problem: extract **OCR text** + **product name** from Vietnamese TikTok
  FMCG thumbnails — lightweight, CPU-deployable, no cloud OCR.
- Why it's hard: mixed Vi/En, diacritics, news/scandal headlines, promo overlays,
  multi-region text, ~14% images with no product at all.
- **Deliverable framing:** a pipeline, not a single model — OCR engine + a
  decision-theoretic product head.

## 2. The Metric Is Law  *(theory-heavy)*
- **Score** = `0.6 · F1_product + 0.4 · (1 − CER)`.
- **token-F1** (product): case-insensitive set-overlap F1 on tokens; both-empty → 1,
  one-empty → 0. *(write the precision/recall/F1 equations).*
- **CER**: `Levenshtein(pred, gt) / len(gt)`, clamped to 1 — **char-level,
  case- AND diacritic-sensitive** (this single fact drives the OCR-engine choice).
- **Key consequence we exploit:** product and OCR are scored on **independent
  columns** → we can optimize each with the best tool for it (this is why mixing
  the best OCR with the best product later yields a jump — §7).
- Theory box: the metric defines a **per-image expected-reward objective** that we
  later solve in closed form (emit-gain, §5).

## 3. Data Deep-Dive / EDA  *(Understand)*  → **Phase A notebook**
Charts & findings to produce:
1. **Split sizes** — train ≈ 4,892 / test 2,006 (Phase 1); ID ranges; no overlap.
2. **Product-frequency distribution** (long-tail bar) — show the **dominant-family
   concentration**: a handful of families (Đồ Hộp Hạ Long, Pate Cột Đèn, NAN,
   Nestlé, Highlands) cover the majority of non-empty labels.
3. **Empty-GT rate** (~14%) — why all-empty already scores ~0.25 (baseline floor).
4. **OCR length distribution** — truncation at 500 chars; text-dense vs sparse.
5. **Diacritic prevalence** — fraction of GT chars that are diacritic marks →
   motivates a diacritic-preserving recognizer.
6. **CER-band histogram** (on our OCR vs GT): [0,.1)/[.1,.3)/[.3,.6)/[.6,1] — expose
   the **27% total-failure band** and tie it to *region-selection mismatch*.
7. **GT fragmentation** — the same product appears under **token-disjoint** surface
   forms (`Đồ hộp Hạ Long` vs `Halong Canfoco`) → motivates §5 canonical-form math.
- Finding callouts: concentration, fragmentation, empty floor, news-headline images.

## 4. Building a CV You Can Trust  *(Validate)*
- **Scheme:** 5-fold **GroupKFold**, grouped by **MD5 of normalized GT OCR text**.
  - *Why grouping?* near-duplicate thumbnails (same text, different crop) would
    otherwise straddle train/val → leakage → optimistic CV. Playbook: *"keep each
    group entirely within one fold."*
- **Fold-safe everything:** the calibrated head's emit strings, the empty-gate
  threshold, and the classifier are **refit on the training fold only** each split.
  Playbook: *"fit every transform inside the fold."*
- **CV→LB correlation:** present the measured offset (CV 0.6142 → LB 0.6685) and
  argue it is a **train→test distribution shift** (test is more family-concentrated),
  not overfitting — this sets up §9.
- Theory box: why CV variance across folds is our error bar; why we never tuned on
  the public board.

## 5. Models I — OCR Engine  *(Baseline)*  → **Phase C charts**
- **Engine bake-off:** PaddleOCR / RapidOCR (ONNX, fast) **strip diacritics** →
  structurally capped on the CER term; **VietOCR `vgg_transformer`** preserves them.
  - Chart: CER + diacritic-preservation by engine. Finding: VietOCR is the CER
    winner *because* of the diacritic-sensitive metric (§2).
- **Fine-tuning:** `vietocr_ft.pth` (vgg_transformer, **37.9M params, fp32, 152 MB**,
  verified pure weights — no optimizer bloat). Detector = CRAFT (EasyOCR).
- **Pipeline:** preprocess (contrast×1.35 + sharpen) → detect → recognize (greedy
  batched) → reading-order reconstruct → dedup/clean ≤500.
- **Lightweight angle** (supervisor cares — Description §Constraints): param count,
  CPU inference timing, greedy-vs-beam tradeoff (+0.0009 ocr_term not worth 2× time).

## 5.5 Lightweight / CPU Deployability  *(the competition's explicit constraint)*
The Description's *Constraints* section asks for CPU-friendly, small, fast solutions.
We address it head-on:
- **Model footprint:** VietOCR vgg_transformer = 37.9M params / 152 MB fp32 — a
  *modest* recognizer; no GPU dependency for inference.
- **CPU inference budget:** measured ~1.5–3 h for 2,006 imgs on CPU (greedy batched),
  dominated by detection + crop count, not model size; GPU only ~20 min — but the
  pipeline *runs* on CPU as required.
- **Speed levers we chose deliberately:** greedy over beam (−0.0009 ocr_term for
  ~2× speed), detection `max_dim` 1280→960 on CPU, batched recognition. Each is a
  documented accuracy/latency tradeoff.
- **The product head is essentially free** — rule + small TF-IDF classifier, ms-level.
- Chart: footprint/latency table; greedy-vs-beam accuracy-vs-time point plot.

## 6. Models II — Product Extraction Head  *(Iterate / theory-heavy)*
- **Baseline:** char TF-IDF + classifier (LGBM/LogReg), empty-gate → CV 0.5992.
- **CalibratedRuleHead (our winner):** ordered family **signatures**; each emits the
  **train-optimal token-F1 canonical string**; classifier fallback for the long tail;
  empty-gate to abstain. CV **0.6142**.
- **Emit-gain theorem (the centerpiece math):**
  - For an image, emitting string `S` beats abstaining iff
    `E[token_f1(GT, S)] > P(GT empty)`.
  - The per-cluster optimum is `S* = argmax_S  E_{GT∼cluster}[token_f1(GT, S)]` —
    derive why this picks the most-frequent surface form when forms are nested, and
    why **token-disjoint** forms impose a hard F1 ceiling no rule can break.
- **Oracle vs real (OCR-noise decomposition):** product F1 with **GT text as input**
  = **0.6904** vs real-OCR **0.6161** → **OCR-noise penalty 0.0743**. Chart: oracle
  vs real bar; interpret as the recoverable headroom that OCR quality gates.

## 7. The Experiment Log  *(Disciplined iteration — the "I tested everything" slide)*
A single CV-gated table — **one change, one CV number, kept or rejected:**

| Experiment | CV Δ (composite/F1) | Verdict | Why |
| :-- | :-- | :-- | :-- |
| prep+beam OCR | +0.006 ocr_term | marginal | small, kept as option |
| aggressive multi-scale/CLAHE detect | −0.041 CER | ❌ | recall adds noise vs bounded GT |
| diacritic restoration | −0.11 CER | ❌ | moves away from exact GT |
| brand-splice OCR correction | −0.004 comp | ❌ | GT is exact transcription |
| highlands-first rule reorder | +0.0005 | ~ | below LB resolution |
| news-context abstention (friend) | −0.10 F1 | ❌ | scandal ⇒ product present |
| friend full HybridExtractor on our OCR | −0.034 | ❌ | tuned to RapidOCR/LB |
| friend evidence_override | −0.023 | ❌ | long fixed forms mismatch train |
| friend canonicalizer | −0.004 | ❌ | head already train-optimal |

- Punchline: **everything was judged by one fixed CV** — exactly the playbook's
  "disciplined iteration." Negative results are *findings*, not failures.

## 8. Results & The Independent-Scoring Insight  *(Iterate → payoff)*
- **Score progression chart:** v8 0.6232 → **v10 calibrated 0.6685** (beat a 0.6495
  rival) → mixing **our OCR + a teammate's product = 0.6959** (proof of §2's
  column-independence; analyzed honestly, incl. why it doesn't transfer to our CV).
- Residual analysis chart: where product F1 is lost (dominant families, fragmentation),
  and the "over-fill on weak evidence" finding (abstention discipline).

## 9. Generalization & The Shake-up  *(Mind the shake-up — trust CV)*
- The CV→LB jump explained: **distribution shift, not overfit** (we never probed).
- **Public vs private (Phase 2):** private test is more **diverse** (28% dominant-
  family vs 53%); LB-tuned approaches lose their edge; **train-grounded calibration
  is the robust choice** — a live demonstration of the playbook's shake-up warning.
- Theory box: why a model selected by CV (not public rank) is the safe final pick.

## 10. Lessons & Conclusion  *(the Checklist slide)*
- Map each playbook checklist item to a concrete thing we did (✓ optimize exact
  metric, ✓ CV mirrors split, ✓ fold-safe, ✓ trust CV, ✓ OOF/abstention discipline,
  ✓ honest about ensembling).
- Top 3 findings restated: diacritic-metric coupling, emit-gain abstention, GT-
  fragmentation ceiling.
- What we'd do next with more time / compute.

---

## Visualization checklist

### Phase A — EDA (`eda_competition.ipynb`, runs live on competition data)
1. `00_sample_images` — sample training images with product labels
2. `01_product_longtail` — top-15 product labels, long-tail bar
3. `02_product_families` — family grouping + GT fragmentation (surface forms per family)
4. `03_empty_rate` — empty rate bar + label status breakdown
5. `04_ocr_length` — OCR text length histogram (500-char truncation)
6. `05_length_vs_product` — OCR length/word-count box plot by product presence
7. `06_diacritic_analysis` — diacritic density histogram + top diacritic characters
8. `07_wordcloud` — word cloud of OCR corpus
9. `08_ngrams` — top unigrams / bigrams / trigrams in OCR
10. `09_product_structure` — product word frequency + token-length distribution
11. `10_feature_correlations` — point-biserial correlations + Pearson matrix

### Phase D — Results (in `04_results.md`, charts embedded or separate)
12. Score-progression line (v8 → v10 → mixed)
13. Experiment-log Δ bar (kept vs rejected, one fixed CV)

### Phase E — Generalization
14. Phase-1 vs Phase-2 family-concentration bar (distribution shift)

## Theory/math inventory (boxes to write)
- token-F1 (precision/recall/F1 on token sets) · CER (Levenshtein/len) ·
  Emit-gain decision rule + per-cluster argmax · token-disjoint ceiling proof ·
  GroupKFold leakage argument · oracle−real noise decomposition · CV-as-estimator
  (bias/variance, why trust over a few-hundred-row public slice).
