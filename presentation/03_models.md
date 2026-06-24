# §5–§6 Models
### URA Hackathon 2026 — OCR engine, product extraction head, and theoretical derivations

---

## §5  Models I — OCR Engine

### 5.1  Pipeline architecture

The OCR pipeline is a two-stage system: **detection** (find where text is) then **recognition** (read what it says).

```
Image
  │
  ▼
[CRAFT Detector]  ──── finds bounding boxes of text regions
  │
  ▼
[Crop + Preprocess]  ── contrast ×1.35, unsharp mask (sharpen)
  │
  ▼
[VietOCR vgg_transformer]  ── recognises each crop → string
  │
  ▼
[Reading-order reconstruct]  ── sort by (row, x) to linearise layout
  │
  ▼
[Dedup + clean]  ── remove near-duplicates, truncate to 500 chars
  │
  ▼
  ocr_text  (→ CER scored)
```

**Design decision — detector fixed, recogniser swapped.** CRAFT (Character Region Awareness For Text) is a strong general-purpose text detector and was not changed across experiments. All OCR engine comparisons below hold the detector constant — only the recogniser was varied. This isolates the recogniser's contribution to CER.

---

### 5.2  Engine bake-off — full 8-engine comparison

Eight engine configurations were evaluated. The same `CalibratedRuleHead` (fit on all training labels) was applied to each engine's OCR output to isolate the OCR contribution — product F1 differences here reflect how well each engine's text feeds the rule patterns, not product head tuning.

GT diacritic reference: **0.285** (28.5% of letters in ground-truth text carry combining marks).

| Engine | Detector | n | CER ↓ | Diac density | Prod F1 | Composite ↑ | Notes |
|:--|:--|--:|--:|--:|--:|--:|:--|
| **VietOCR-FT (ours)** ★ | CRAFT | 4,892 | **0.4015** | 0.215 | 0.6316 | **0.6184** | Fine-tuned on ext. Vi dataset + competition train |
| VietOCR-FTup2 | CRAFT + upscale v2 | 4,891 | 0.4121 | 0.219 | 0.6311 | 0.6138 | Stronger upscale of small thumbnails |
| VietOCR-FTup | CRAFT + upscale | 4,891 | 0.4167 | 0.216 | 0.6358 | 0.6148 | Upscale small thumbnails before detection |
| RapidOCR (E2E) | DBNet ONNX | 300 | 0.4319 | 0.001 | **0.6842** | 0.6378 | val-only sample (300 imgs) — see anomaly note |
| VietOCR base | CRAFT | 4,892 | 0.4373 | 0.179 | 0.6314 | 0.6039 | Pre-trained, no fine-tuning |
| VietOCR-FTdb | RapidOCR DB ONNX | 4,891 | 0.4688 | 0.221 | 0.6055 | 0.5758 | Detector swapped CRAFT → DB; regression |
| PaddleOCR (E2E) | PP-OCR det | 2,980 | 0.4933 | 0.024 | 0.6338 | 0.5830 | Strips Vietnamese diacritics |
| EasyOCR (E2E) | CRAFT (built-in) | 300 | 0.5040 | 0.225 | 0.5523 | 0.5298 | val-only sample (300 imgs) |

*(Figures 11–12: `presentation/figures/11_engine_cer_diac.png`, `12_engine_composite.png`)*

**Reading the diacritic density column:** GT reference is 0.285. PaddleOCR outputs 0.024 — it is discarding 91% of diacritic marks because its ONNX recogniser was trained on a Latin/Chinese corpus with no Vietnamese combining marks in its output vocabulary. VietOCR-FT at 0.215 is closest to GT, meaning it actually transcribes tone marks rather than silently dropping them.

**Why the CER gap is structural, not tunable:** PaddleOCR physically cannot output `ạ`, `ổ`, `ươ` — they are absent from its character vocabulary. Every such character in the GT is an irrecoverable substitution. No amount of preprocessing or postprocessing can fix a missing output class.

> **Plain-language version (for presentation):**
> Imagine transcribing Vietnamese text but your keyboard only has English letters — you physically cannot type `ạ` or `ổ`. No matter how carefully you read the image, every accented character becomes an error. That is exactly PaddleOCR's situation. VietOCR was designed for Vietnamese — it has the right "keyboard."

**Key findings from the bake-off:**

**1. VietOCR-FT wins overall (CER 0.4015, composite 0.6184 on 4,892 images).**
The full training set coverage and best CER make it the clear choice.

**2. RapidOCR anomaly — prodF1 0.6842 despite diac=0.001.**
RapidOCR strips all diacritics yet achieves the highest product F1. Why? The `CalibratedRuleHead` applies `unicodedata.normalize("NFD")` + diacritic stripping (`fold()`) before pattern matching — so rules still fire correctly even on diacritic-free text. However, this score is on **only 300 val images** (not the full 4,892), making it unreliable and likely upward-biased by sample composition. Its CER of 0.4319 still heavily penalises composite on a full run.

**3. FTdb is a regression (CER 0.4688 > base 0.4373).**
Swapping CRAFT for the DB (DBNet) detector made performance worse, not better. CRAFT is superior for TikTok thumbnails: it handles arbitrary text orientations, curved overlays, and stylised FMCG fonts. DB detectors are tuned for horizontal, high-contrast document text. This is a negative result that confirms the detector choice matters as much as the recogniser.

**4. FTup / FTup2 — upscaling is not the lever (CER 0.4167 / 0.4121 vs FT 0.4015).**
Upscaling small thumbnails before detection slightly improves CER over the base model but cannot close the gap to plain fine-tuning. The bottleneck is recogniser domain adaptation, not image resolution.

---

### 5.3  VietOCR architecture — why vgg_transformer

VietOCR offers two backbone choices:

| Backbone | Params | Speed | Accuracy on Vietnamese |
|:--|:--|:--|:--|
| `vgg_seq2seq` | ~13M | Faster | Lower — RNN decoder, weaker long-range context |
| `vgg_transformer` *(chosen)* | **37.9M** | Slower | Higher — attention decoder, handles multi-word text |

**Why transformer over seq2seq:** FMCG thumbnails often contain full product names like `Đồ Hộp Hạ Long 3 sao Cao Cấp` — multi-word, capitalisation-varying, mixed Vietnamese/English. A seq2seq RNN decoder suffers from error accumulation over long sequences; an attention-based transformer decoder can attend globally to the feature map and is more robust to long, complex text. The extra 24M params cost inference time but pay off in CER.

**Model checkpoint:** `vietocr_ft.pth` — 152 MB, fp32 weights only (no optimizer state). The checkpoint is pure inference weight — no training overhead is included.

---

### 5.4  Fine-tuning strategy

The base VietOCR vgg_transformer was originally pre-trained on a large external Vietnamese OCR dataset. We then continued training on the **competition's own training images** to adapt to the specific visual domain (TikTok thumbnails, promotional overlays, specific font styles).

**Why fine-tuning helps:**

- The base model was trained on printed documents and street signs; TikTok thumbnails have stylised fonts, colour overlays, and heavy compression artefacts.
- Fine-tuning on competition train images brings the recogniser's error distribution closer to the test domain.
- Measured improvement: VietOCR base CER 0.4373 → VietOCR-FT CER **0.4015** — a **0.036 absolute reduction**, equivalent to +0.014 in the composite score's OCR term alone (weight 0.4 × 0.036 = 0.014).

**Caveat:** fine-tuning on train images risks overfitting the recogniser to train-domain visual styles. However, since OCR is not evaluated via the CV (it is pre-computed), this risk is contained — we cannot measure OCR overfitting from the CV alone. The LB CER improvement is the observed evidence that fine-tuning generalised.

---

### 5.5  Preprocessing pipeline

Two preprocessing steps are applied to each detected crop before recognition:

**1. Contrast enhancement (×1.35):**
Promotional text on FMCG thumbnails is often low-contrast (white text on light background, or vice versa). Boosting contrast sharpens the character boundaries that the CNN feature extractor relies on.

**2. Unsharp mask (sharpening):**
TikTok video frames are compressed with H.264/H.265, which blurs fine strokes. Sharpening partially reverses this, recovering stroke edges that carry diacritic mark information (the hook above `ỉ`, the dot below `ọ` are thin strokes easily blurred).

**Reading-order reconstruction:**
CRAFT detects boxes in arbitrary order. We sort them by approximate reading order: quantise by vertical position (rows), then sort left-to-right within each row. This produces natural top-to-bottom, left-to-right text linearisation — matching the linear string that the GT labeller would have read.

**Deduplication and truncation:**
Adjacent crops that contain the same text (re-detected text, watermark repetitions) are removed. The final string is truncated to 500 characters — the competition format limit.

---

### 5.6  Greedy decode vs beam search — the speed/accuracy tradeoff

VietOCR's transformer decoder can be run in two modes:

| Decode mode | CER | Relative speed |
|:--|:--|:--|
| Greedy (argmax at each step) | baseline | 1× |
| Beam search (width=4) | ~−0.003 lower CER | ~2× slower |

From the CV experiment log: beam search yielded **+0.0024 on the OCR term** (composite contribution: +0.0010) — small enough to be below the fold-variance noise floor. We chose greedy as the production decode.

**Decision rationale:** the OCR term has weight 0.4, so a 0.003 CER improvement moves composite score by only 0.4 × 0.003 = **0.0012**. Against a 2× inference time cost (from ~20 min GPU to ~40 min, or ~3h CPU to ~6h), the trade-off is poor. The product F1 term (weight 0.6) has far higher leverage per unit of effort.

> **Plain-language version (for presentation):**
> Beam search is like double-checking your answer before writing it — it considers multiple possibilities and picks the best. But here, the extra checking only improves the score by 0.001, while taking twice as long. We chose speed over this marginal gain, and spent that time improving the product extraction head instead, which has a much larger impact on the final score.

---

### 5.7  Lightweight / CPU deployability

The competition description explicitly asks for CPU-friendly solutions. Our pipeline satisfies this:

| Component | Size / cost |
|:--|:--|
| CRAFT detector | ~30MB, runs on CPU |
| VietOCR vgg_transformer | 152 MB fp32, no GPU dependency |
| CalibratedRuleHead | <1 MB (rules + small TF-IDF matrix) |
| **Total pipeline** | **~182 MB** |

**CPU inference time:** approximately 1.5–3 hours for 2,006 test images (greedy, batched crops, detection `max_dim=960` on CPU). GPU reduces this to ~20 minutes. The pipeline *runs* on CPU — it is simply slower, as expected for a seq2seq attention model without quantisation.

**Speed levers available if CPU time is a hard constraint:**

1. Reduce `max_dim` from 1280 → 960 on detection (fewer crops per image → faster)
2. Greedy decode (already chosen)
3. Batch recognition crops together (already implemented — amortises model load overhead)
4. Quantise the recogniser to int8 (not tested — estimated ~2× speedup with minor CER degradation)

The product head adds negligible cost — regex matching + TF-IDF inference is milliseconds per image.

---

## §6  Models II — Product Extraction Head

### 6.1  Baseline: ProductExtractor (TF-IDF + Logistic Regression)

The baseline product head (`product_extract.py`) is a fully data-driven approach:

**Architecture:**
1. **Canonicalise** training labels: group by token-sorted key, pick the modal surface form per group — this collapses near-duplicate labels into one canonical class.
2. **Binary gate:** char n-gram TF-IDF (2–5 grams, 5000 features) + LogisticRegression → P(has product). If P < 0.5, abstain.
3. **Multiclass classifier:** same TF-IDF features → predict the canonical product form (only classes with ≥3 training examples).

**Features:** diacritic-stripped (via `unidecode`) lowercased OCR text. This makes the classifier robust to OCR diacritic errors — if the recogniser writes `Ha Long` instead of `Hạ Long`, the folded feature is the same.

**CV result:** composite **0.5992**

**Why it falls short:**
- The classifier sees all classes equally — it cannot exploit the strong prior that certain families dominate.
- Char n-gram features work well for frequent classes but fail on the long tail, where training examples are sparse.
- The binary gate threshold (0.5) is a single global cutoff, not calibrated per-family.

---

### 6.2  CalibratedRuleHead — design rationale

The key insight from EDA: **five families cover ~60% of non-empty labels, and each leaves a distinct lexical fingerprint in the OCR text** (n-gram analysis, Figure 08). A rule that fires on `"canfoco"` in the OCR text is extremely high-precision — almost every such image has the Hạ Long canned food product.

This motivates replacing the single-classifier approach with **ordered family signatures**: hard rules that fire with high precision on the dominant families, with the classifier as a fallback for everything else.

**CV result:** composite **0.6142** (+0.015 over baseline)

---

### 6.3  SIG_PATTERNS — the nine ordered rules

```python
SIG_PATTERNS = [
    # Rule 1 — most specific: image mentions BOTH canfoco AND pate/cot den
    ("halong_canfoco_pate_cotden",
     r"(canfoco|canfuco|cafoco).*(pate|cot den).*(cot den|hai phong)|"
     r"(pate|cot den).*(canfoco|canfuco)"),

    # Rules 2–3 — NAN sub-variants (must come before bare NAN)
    ("nan_optipro",    r"\bnan\b.*opti ?pro|opti ?pro.*\bnan\b"),
    ("nan_infinipro",  r"\bnan\b.*infini ?pro|infini ?pro.*\bnan\b"),

    # Rule 4 — Pate Cột Đèn (without canfoco)
    ("pate_cotden",    r"\bpate\b.*\b(cot den|hai phong)\b|"
                       r"\b(cot den|hai phong)\b.*\bpate\b|\bcot den\b"),

    # Rule 5 — Ha Long Canfoco (without pate)
    ("halong_canfoco", r"\bcanfoco\b|\bcanfuco\b|\bcafoco\b|"
                       r"ha long canfoco|halong canfoco"),

    # Rule 6 — Đồ Hộp Hạ Long (without canfoco)
    ("do_hop_ha_long", r"do hop ha long|do hop.*ha long|"
                       r"cong ty.*do hop.*ha long"),

    # Rule 7 — bare NAN (after sub-variants consumed)
    ("nan",            r"\bnan\b"),

    # Rule 8 — Highlands Coffee
    ("highlands",      r"highlands? coffee|highlands"),

    # Rule 9 — Nestlé
    ("nestle",         r"\bnestle\b"),
]
```

**All patterns operate on folded (diacritic-stripped, lowercase, alnum-only) OCR text**, so `"Canfoco"`, `"CANFOCO"`, `"cànfôcô"` all match `\bcanfoco\b`.

**Three important design choices:**

**1. Most-specific first (sequential consumption):**
After a rule fires and claims its images, those images are removed from `remaining` before the next rule runs. This prevents contamination: if an image mentions both "canfoco" and "pate cot den", Rule 1 claims it — neither Rule 4 (pate alone) nor Rule 5 (canfoco alone) ever sees it. Without this ordering, Rules 4 and 5 would be calibrated on a contaminated set that includes the compound images, producing suboptimal emit strings.

**2. Precision gate (min_pprod = 0.55):**
A rule only activates if, in the training fold, at least 55% of the images it would match have a non-empty product label. This prevents rules from firing on OCR text that merely mentions a brand in passing (news headlines, competitor mentions).

**3. Train-optimal emit string:**
For each rule's matched cluster, `fit()` tries all candidate surface forms with ≥3 training examples and picks the one that maximises mean token-F1 against all images in the cluster:

```python
best_form = argmax_{cand in candidates}  mean_{i in cluster} token_f1(gt_i, cand)
```

This is the exact solution to the emit-gain problem (§6.4 below). The emit string is not manually chosen — it is derived from training data.

> **Plain-language version (for presentation):**
> Think of the rules like a specialist team. The most expert specialist (who recognises images mentioning BOTH "canfoco" AND "pate/cot den") gets first pick of images. Once they claim their images, the next specialist (who recognises pate alone) only sees images the first specialist didn't take. This prevents any image being claimed by the wrong specialist. Each specialist also learns, from training data, exactly which product name to output — not hardcoded by hand.

---

### 6.4  The emit-gain theorem — formal derivation

**Setup.** Consider a cluster $C$ of images that match a given signature. For each image $i \in C$, the GT product label $g_i$ is drawn from the cluster's label distribution $\{\pi_v\}$ where $\pi_v = P(\text{GT} = v \mid C)$.

We must decide: emit some string $S$, or abstain (emit `""`)?

**Payoff of abstaining:**

$$\text{reward}(\text{abstain}) = \mathbb{E}_{i \sim C}[F1("", g_i)]$$

By the token-F1 special cases:
- If $g_i = ""$ (image has no product): $F1("", "") = 1$
- If $g_i \neq ""$: $F1("", g_i) = 0$

$$\text{reward}(\text{abstain}) = P(g_i = "" \mid C) = \pi_\emptyset$$

**Payoff of emitting string $S$:**

$$\text{reward}(S) = \mathbb{E}_{i \sim C}[F1(S, g_i)] = \sum_v \pi_v \cdot F1(S, v)$$

**Decision rule (emit-gain theorem):**

$$\text{emit } S \iff \mathbb{E}_{i \sim C}[F1(S, g_i)] > \pi_\emptyset$$

In words: emit $S$ if and only if the expected token-F1 of emitting $S$ against the cluster's true label distribution exceeds the probability of the GT being empty (the abstention baseline).

**Optimal emission:**

$$S^* = \underset{S}{\arg\max}\ \mathbb{E}_{i \sim C}[F1(S, g_i)]$$

In `fit()`, this argmax is computed exactly over the set of candidate surface forms (those with ≥3 training examples in the cluster). The `best_val - empty_base > 0` check is precisely the emit-gain condition — a rule only activates if the best emit string beats abstention.

**Why the argmax picks the dominant surface form:** for a family where 90% of labels are `"Đồ hộp Hạ Long"` and 10% are `"HALONG CANFOCO"` (token-disjoint), the expected F1 of emitting `"Đồ hộp Hạ Long"` is approximately 0.90 × 1.0 + 0.10 × 0 = 0.90, while emitting `"HALONG CANFOCO"` gives ≈ 0.10. The dominant form wins. When forms share tokens (e.g., `"NAN"` vs `"sữa NAN"` — the token `nan` overlaps), the argmax may pick a form that achieves partial F1 against multiple variants, which is better than 0.

> **Plain-language version (for presentation):**
> The key question for each rule is: should I say something, or stay quiet? If I stay quiet and the image has no product, I score 1 (correct silence). If I say something wrong, I score 0. The theorem formalises this: only speak if the best answer you can give is more likely to be right than the image having no product at all. And among all the things you could say, pick whichever one, on average across training images, gets the highest F1.

---

### 6.5  The token-disjoint ceiling — why no rule can fully solve fragmentation

Let family $k$ have surface forms $\mathcal{V}_k = \{v_1, \ldots, v_m\}$ with probabilities $\{\pi_j\}$, and suppose some pair $(v_a, v_b)$ is token-disjoint: $\text{tokens}(v_a) \cap \text{tokens}(v_b) = \emptyset$.

Then for **any** emission $S$:

$$F1(S, v_a) > 0 \implies F1(S, v_b) = 0$$

*Proof:* $F1(S, v_a) > 0$ requires $\text{tokens}(S) \cap \text{tokens}(v_a) \neq \emptyset$, i.e., $S$ shares at least one token with $v_a$. Since $\text{tokens}(v_a) \cap \text{tokens}(v_b) = \emptyset$, that token is not in $v_b$, so $\text{tokens}(S) \cap \text{tokens}(v_b) = \emptyset$, giving $F1(S, v_b) = 0$. $\square$

**Consequence:** the best single emission achieves:

$$F1_{\max}^{(k)} = \max_j \sum_l \pi_l \cdot F1(v_j, v_l) \leq 1 - \min(\pi_a, \pi_b) < 1$$

For the Hạ Long family where the disjoint pair `{đồ, hộp, hạ, long}` / `{halong, canfoco}` has combined probability ~15%, the ceiling is ≤ 0.85 regardless of any rule or classifier. **This is an annotation artefact that requires fixing at the labelling level, not the model level.**

---

### 6.6  Classifier fallback — the long tail

Images that do not match any of the 9 signatures (i.e., they survive all rules with `remaining`) are passed to the `ProductExtractor` classifier fallback. This handles:
- Products outside the top-5 families
- Novel brand mentions not covered by the patterns
- Ambiguous OCR text where no brand fingerprint is detectable

The classifier uses character n-gram TF-IDF (2–5 grams, 5000 features) on diacritic-stripped OCR text — robust to OCR diacritic errors. LogisticRegression with `class_weight="balanced"` handles the long-tail class imbalance.

**Why the fallback matters less than it seems:** the long tail has high entropy — many classes each with few training examples. The classifier's accuracy here is naturally lower than on the dominant families. In practice, the fallback contributes mostly through correct **abstentions** (gate fires at P < threshold) rather than correct predictions.

---

### 6.7  Oracle vs real F1 — OCR-noise decomposition

The head can be evaluated in two modes:

| Mode | Input to head | Product F1 | Interpretation |
|:--|:--|:--|:--|
| **Oracle** | GT `ocr_text` (perfect OCR) | **~0.69** | Upper bound: how good could rules be if OCR were perfect? |
| **Real** | VietOCR-FT predictions | **~0.62** (CV) | Actual deployed performance |
| **Gap** | — | **~0.07** | OCR-noise penalty: F1 lost due to imperfect OCR |

**Reading the gap:** the head's rule patterns are designed for clean text (e.g., `\bcanfoco\b`). When VietOCR-FT misreads a character — say, outputting `"canfôco"` instead of `"canfoco"` — the folded form becomes `"canfoco"` (diacritics stripped), so the rule still fires. But when the recogniser misreads `"canfoco"` as `"carnfoco"` (consonant error), the pattern fails entirely. The 0.07 gap represents these hard OCR errors where the brand fingerprint is corrupted beyond recovery.

**Interpretation for the supervisor:** even with perfect OCR, the head scores ~0.69 — not 1.0. The remaining ~0.31 gap is due to GT fragmentation (§6.5), news-context images (§1), and long-tail products the classifier handles poorly. The OCR noise penalty accounts for only about one-quarter of the total gap from perfect. Further improving the head's F1 beyond 0.69 would require solving the annotation fragmentation problem.

> **Plain-language version (for presentation):**
> We ran our product extraction rules on the perfect, hand-typed text (oracle) and on our OCR engine's output (real). The rules score 0.69 with perfect input and 0.62 with real OCR. The 0.07 gap is the price of imperfect text recognition — about 7 out of 100 predictions are wrong because the OCR made an error that corrupted the brand keyword. Importantly, even with perfect OCR we only reach 0.69, not 1.0 — the remaining gap is due to the labelling inconsistency problem described in §3.3, which no OCR improvement can fix.

---

## Summary — model decision map

| Decision | Options considered | Chosen | Reason |
|:--|:--|:--|:--|
| OCR recogniser | EasyOCR, RapidOCR, PaddleOCR, VietOCR base, VietOCR-FT | **VietOCR-FT** | Best CER (0.4015) on full 4,892-image set; only engine preserving diacritics at scale |
| OCR backbone | `vgg_seq2seq` (~13M params), `vgg_transformer` (~37.9M params) | **vgg_transformer** | Attention decoder handles multi-word product names; RNN error accumulation hurts long sequences |
| OCR detector | CRAFT vs DBNet (FTdb experiment) | **CRAFT** | DBNet swap → CER 0.4688 (worse than base); CRAFT superior for stylised/angled TikTok text |
| Thumbnail upscaling | No upscale (FT), upscale v1 (FTup), upscale v2 (FTup2) | **No upscale** | FTup2 CER 0.4121 / FTup 0.4167 — both worse than plain FT 0.4015; upscale is not the lever |
| Decode strategy | Greedy, Beam(4) | **Greedy** | Beam +0.003 CER improvement = +0.001 composite; negligible vs 2× inference cost |
| Product head | TF-IDF classifier only | **Rules + classifier** | +0.045 LB gain (v8→v10); rules exploit known family concentration with high precision |
| Rule ordering | Alphabetical, random, specific-first | **Most-specific first** | Sequential consumption prevents rule contamination (e.g. `pate_cotden` must fire before `halong_canfoco`) |
| Emit string | Hardcoded, modal form, argmax-F1 | **Argmax token-F1** | Train-optimal per cluster; emit-gain theorem guarantees improvement over abstaining |
| Abstention | Never abstain, fixed threshold, calibrated per rule | **Calibrated per rule** | `min_pprod=0.55` gate prevents false-positive on scandal/news images where brand appears in negative context |

---

*Next: §7–§8 Experiment log + Results — the CV-gated iteration table, score progression, and residual analysis → `04_results.md`*
