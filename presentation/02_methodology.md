# §1–§4 Methodology Core
### URA Hackathon 2026 — for supervisor presentation (technical ML audience, ~20 min)

---

## §1  Problem Framing

**Task.** Given a Vietnamese TikTok FMCG thumbnail image, produce two strings:

| Output column | What it is | Scored by |
|:--|:--|:--|
| `ocr_text` | Verbatim text visible in the image | CER (char-level edit distance, diacritic-sensitive) |
| `product_name` | FMCG product/brand extracted from that text | token-F1 (case-insensitive set overlap) |

The two columns are **scored independently** — this is the single most exploitable structural property of the metric (§2.3).

**Why it is hard.**

1. **Script complexity.** Vietnamese uses a Latin base augmented by two independent orthographic layers: *vowel modifiers* (circumflex `^`, breve `˘`, horn `̛`) and *tone marks* (acute/grave/hook/tilde/dot-below — six tones). The product labels use fully accented text; a CER metric that is character-exact over Unicode penalises every missing or wrong combining mark.

2. **Domain noise.** TikTok thumbnails combine product packaging, promotional text overlays, creator watermarks, and — critically — **news/scandal headlines** that mention brand names in a negative context (e.g., *"Highlands Coffee bị phạt…"*). A naive rule that fires on brand keywords extracts a product where the true label is empty.

3. **GT fragmentation.** The same physical product appears under token-disjoint surface forms set by different annotators: `Đồ hộp Hạ Long`, `ĐỒ HỘP HẠ LONG`, `HALONG CANFOCO`, `Ha Long Canfoco`. No single string can achieve F1 = 1 against all variants simultaneously — this imposes a hard ceiling (§6 derivation).

4. **Multi-region detection.** CRAFT detects every text region in the image. When the labeller read a different region than the detector, the OCR text and GT ocr_text refer to disjoint text — CER ≈ 1 regardless of recogniser quality. This accounts for the ~27% "total-failure" CER band (§3.6).

**Pipeline decomposition.**

```
Image
  └─► Detect (CRAFT)
        └─► Crop text regions
              └─► Preprocess (contrast ×1.35, sharpen)
                    └─► Recognise (VietOCR vgg_transformer)
                          └─► Reading-order reconstruct + dedup → ocr_text
                                └─► CalibratedRuleHead → product_name
```

The OCR stage optimises the CER term; the product head optimises the F1 term. Both are trained/calibrated on the same training split, validated by the same 5-fold CV (§4).

---

## §2  The Metric Is Law

### 2.1  Composite score

$$\boxed{\text{Score} = 0.6 \cdot F1_{\text{product}} + 0.4 \cdot (1 - \text{CER})}$$

The 0.6 / 0.4 weighting makes product extraction the dominant term. A 10-point gain in product F1 is worth 1.5× a 10-point gain in OCR quality.

### 2.2  token-F1 (product)

Let $\hat{p}$ be the predicted product string and $g$ the ground-truth string.

Define the token multisets (case-insensitive, whitespace-split):

$$\hat{P} = \text{tokens}(\text{lower}(\hat{p})), \quad G = \text{tokens}(\text{lower}(g))$$

$$\text{precision} = \frac{|\hat{P} \cap G|}{|\hat{P}|}, \qquad \text{recall} = \frac{|\hat{P} \cap G|}{|G|}$$

$$F1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

**Special cases (implementation contract):**

| $\hat{p}$ | $g$ | $F1$ |
|:--|:--|:--|
| empty | empty | **1** (both correct to abstain) |
| non-empty | empty | **0** (false positive) |
| empty | non-empty | **0** (miss) |

**Key consequence:** token-F1 is case-insensitive and operates on a *set overlap* — word order does not matter, and case differences are ignored. This is strictly more lenient than CER, which is *char-exact and order-sensitive*. The two metrics therefore reward different model properties.

### 2.3  CER (character error rate)

$$\text{CER}(\hat{o}, o) = \min\!\left(1,\ \frac{\text{Lev}(\hat{o},\, o)}{|o|}\right)$$

where $\text{Lev}$ is the standard Levenshtein edit distance (insert/delete/substitute, unit cost). The score term uses $1 - \text{CER}$, so lower edit distance is better.

**Critical constraint:** CER is **character-level, case-sensitive, and diacritic-sensitive**. A prediction `Ha Long` vs ground truth `Hạ Long` incurs a substitution on the dot-below mark `̣` — a single character of edit distance. For a 7-character GT string this alone gives CER ≥ 0.14.

**Structural implication.** Any OCR engine that strips Vietnamese diacritics (PaddleOCR, RapidOCR ONNX) cannot reach CER < ~0.2 on this dataset regardless of any other improvement, because every diacritic mark in the GT becomes an irreducible substitution error. This is not a tuning problem — it is architectural. VietOCR's seq2seq recogniser with a Vietnamese-trained vocabulary is therefore the only viable choice (engine comparison in §5).

### 2.4  Independent-column property

Because product_name and ocr_text occupy **separate columns**, the score decomposes exactly:

$$\text{Score} = 0.6 \cdot F1_{\text{product}}(\hat{p}, g_p) + 0.4 \cdot (1 - \text{CER}(\hat{o}, g_o))$$

There is no cross-term. This means:

- The optimal $\hat{p}$ depends only on $g_p$, not on $g_o$.
- The optimal $\hat{o}$ depends only on $g_o$, not on $g_p$.
- **Mixing submissions** (taking $\hat{p}$ from the model that maximises F1 and $\hat{o}$ from the model that minimises CER) is guaranteed to be at least as good as either alone.
- Here the competition accidentally made it so there are no tradeoffs. You can have two completely independent specialists:

+ Model A:  best at OCR  →  fills ocr_text column
+ Model B:  best at product extraction  →  fills product_name column

Combined submission:  takes ocr_text from A, product_name from B

This property was exploited in §8: combining our OCR text (low CER) with a teammate's product extraction (high F1) yielded 0.6959, strictly above either individual submission.

### 2.5  Per-image expected-reward formulation

Each image defines a Bernoulli-like decision: *emit a string or abstain*. Define the per-image reward:

$$r_i(\hat{p}) = 0.6 \cdot F1(\hat{p},\ g_{p,i}) + 0.4 \cdot (1 - \text{CER}(\hat{o}_i,\ g_{o,i}))$$

The CER term is fixed once OCR is run. The actionable decision is $\hat{p}$. For a cluster of images $C$ that match a known signature, the optimal string is:

$$S^* = \underset{S}{\arg\max}\ \mathbb{E}_{i \sim C}\!\left[F1(S,\ g_{p,i})\right]$$

This is the **emit-gain theorem** formalised in §6.

---

## §3  Data Deep-Dive (EDA findings)

*All figures from `notebooks/eda_competition.ipynb`. Numbers are from the training split (4,892 images).*

### 3.1  Dataset structure

| Split | Images | Source |
|:--|:--|:--|
| Train | 4,892 | Competition training set, labelled |
| Test (Phase 1) | 2,006 | Public leaderboard (hidden GT) |
| Test (Phase 2) | 1,000 | Private leaderboard (hidden until 2026-06-24) |

Image IDs are non-overlapping integers; train and test are strictly disjoint by ID range. No data augmentation or cross-split contamination is possible from the raw IDs alone.

### 3.2  Product label distribution (Figure 01)

The distribution is strongly long-tailed. The top-15 exact surface forms cover ~60% of all non-empty labels, while the remaining 480+ distinct forms cover the rest. The **dominant families** are:

- Đồ Hộp Hạ Long / HALONG CANFOCO (canned seafood)
- Pate Cột Đèn Hải Phòng (pâté)
- NAN / Nestlé NAN (infant formula)
- Nestlé / Sữa Nestle (dairy)
- Highlands Coffee

These five families are the primary targets for rule-based extraction. Everything outside this top-5 is handled by a classifier fallback or abstention.

### 3.3  GT fragmentation (Figure 02)

A **product family** is defined by normalising the surface form: lowercase, strip diacritics, collapse whitespace. The Hạ Long family, for example, has **7+ distinct surface forms** in the training set:

```
ĐỒ HỘP HẠ LONG   (uppercase, full)
Đồ hộp Hạ Long   (title case)
đồ hộp Hạ Long   (lowercase prefix)
HALONG CANFOCO    (romanised, brand name)
Ha Long Canfoco   (mixed)
Hạ Long           (short form)
...
```

These are **token-disjoint**: `{"do", "hop", "ha", "long"}` ∩ `{"halong", "canfoco"}` = ∅. For a rule that emits `"Đồ Hộp Hạ Long"`, F1 against `"HALONG CANFOCO"` (after lowercasing) = 0. The family-level F1 ceiling is strictly below 1.0 regardless of emission strategy. This is a labelling artefact that cannot be resolved without canonical form collapse — and the competition's own metric does not collapse forms.

**Formal ceiling.** Let $\mathcal{V}_k = \{v_1, \ldots, v_m\}$ be the set of surface forms for family $k$, and $\pi_j = P(\text{GT} = v_j | \text{family } k)$. The optimal single emission $S^*$ achieves:

$$F1_{\max}^{(k)} = \max_j \sum_l \pi_l \cdot F1(v_j, v_l) < 1 \quad \text{if any pair is token-disjoint}$$

### 3.4  Empty-GT rate (Figure 03)

| Category | Fraction |
|:--|:--|
| product_name empty | **40.7%** |
| ocr_text empty | **20.2%** |
| both empty | **19.6%** |
| Has product label | **59.3%** |

The high product-empty rate (40.7%) means that a model which always emits something incurs a large F1 = 0 penalty on ~41% of images. Abstaining when uncertain is not just conservative — it is **Bayes-optimal** when the probability of the GT being empty exceeds the expected F1 of the proposed emission (§6.1).

### 3.5  OCR text length (Figure 04)

The GT ocr_text is truncated to 500 characters by the competition format. Mean length is ~99 chars, median ~70 chars, with a bimodal-ish shape: a spike at 0 (empty OCR, 20% of images) and a broad mode peaking around 50–150 chars. The 500-char tail captures images with dense promotional overlays.

### 3.6  Text length vs product presence (Figure 05)

`r` is the point-biserial correlation coefficient — a special case of Pearson correlation used when one variable is binary (0/1) and the other is continuous.

Point-biserial correlation (r) between each feature and `has_product` (binary 0/1):

| Feature | r | p-value | interpretation |
|:--|:--|:--|:--|
| `ocr_len` | **+0.415** | 3.59e-203 | longer text → more likely to have product label |
| `ocr_word_count` | **+0.420** | 1.12e-208 | same signal at word level — consistent |
| `diac_frac` | **+0.477** | 3.98e-277 | **strongest**: higher diacritic density → product more likely present |

All three are highly statistically significant (p ≪ 0.001).

The `diac_frac` result is the most informative finding: product-label images tend to be **packaging shots** containing dense, fully-accented Vietnamese product names (`Đồ Hộp Hạ Long`, `Pate Cột Đèn`). Empty-label images tend to be news headlines or scenic content with sparser diacritics. Diacritic density is therefore a proxy for "this image looks like product packaging, not a news clip" — a signal the classifier implicitly exploits.

The full correlation matrix (including `has_ocr`) is shown in Figure 10. `has_ocr` remains the single strongest gate signal overall: if the detector found no text at all, product is almost surely absent — directly motivating the **empty-gate** (§6).

### 3.7  Vietnamese diacritics (Figure 06)

In NFD decomposition, Vietnamese text carries **two stacked layers** of combining marks:

| Unicode mark | Vietnamese name | Linguistic role | Example |
|:--|:--|:--|:--|
| U+0301 Acute ´ | sắc | Rising tone | á, ó, ú |
| U+0300 Grave ` | huyền | Falling tone | à, è, ù |
| U+0303 Tilde ~ | ngã | Tumbling tone | ã, õ |
| U+0309 Hook above | hỏi | Asking tone | ả, ỉ, ủ |
| U+0323 Dot below | nặng | Heavy tone | ạ, ọ, ụ |
| U+0302 Circumflex ^ | — | Vowel modifier | â, ê, ô |
| U+0306 Breve ˘ | — | Vowel modifier | ă |
| U+031B Horn | — | Vowel modifier | ơ, ư |

~25% of all letters in the OCR text carry at least one combining mark. PaddleOCR and RapidOCR's ONNX recognisers discard all combining marks (they output ASCII-only or NFC-simplified Latin). Every such mark that appears in the GT becomes an irrecoverable substitution error in the CER calculation — structurally capping those engines below VietOCR.

### 3.8  Word cloud & n-grams (Figures 07–08)

The top unigrams in OCR text are dominated by brand-adjacent words: `highland`, `coffee`, `nan`, `nestle`, `long`, `canfoco`, `pate`, `halong`. This confirms that brand pattern matching on the raw OCR text is a high-precision strategy — the dominant product families leave clear lexical fingerprints. The trigram analysis shows compound phrases like `do hop ha` (Đồ hộp Hạ), `highlands coffee`, `pate cot den` appearing frequently — these are the exact signatures used by the CalibratedRuleHead (§6).

---

## §4  Building a CV You Can Trust

### 4.1  The leakage risk: near-duplicate images

TikTok content is reposted. Two images can show the same text at slightly different crops, resolutions, or aspect ratios. If image A (crop variant 1) lands in the training fold and image B (crop variant 2) lands in the validation fold:

- Both have identical GT ocr_text → same group key.
- The model has effectively "seen" the GT for image B during training.
- CV metric is optimistically biased.

The leakage is not in the model weights — it is in the **calibration data** (emit strings, gate threshold) which are computed from the training fold.

### 4.2  GroupKFold by content hash

To prevent this, we assign each image to a group:

```python
def group_key(gt_ocr_text, image_id):
    t = " ".join(gt_ocr_text.lower().split())   # normalise
    if not t:
        return f"empty_{image_id}"              # empty OCR → unique group
    return hashlib.md5(t.encode()).hexdigest()  # hash of normalised text
```

Then use `sklearn.model_selection.GroupKFold(n_splits=5)`.

**Guarantee:** all images with identical (normalised) GT OCR text are in the **same fold** — either all in train or all in validation for any given split. No near-duplicate can straddle a fold boundary.

Empty-OCR images each get a unique group key (they are all structurally different in terms of what was undetected) to avoid artificially concentrating them in one fold.

> **Plain-language version (for presentation):**
> Imagine a class exam where the teacher accidentally includes two photos of the same question — one in the practice set and one in the real test. The student recognises it and gets it right, but not because they understood it. GroupKFold is how we prevent that: we convert each image's text into a fingerprint (MD5 hash), then make sure every image with the same fingerprint is always on the same side — either all in training or all in validation, never split. So the model is always tested on genuinely unseen text.

### 4.3  Fold-safe protocol — what must be fit inside the fold

The following operations are **fit on the training fold only** and applied to the validation fold:

| Component | Fit on | What is learned |
|:--|:--|:--|
| Emit strings | Train fold labels | Optimal canonical surface form per signature cluster |
| Empty-gate threshold | Train fold `has_product` distribution | P(product empty | signature) threshold |
| TF-IDF + classifier | Train fold OCR texts + labels | Feature weights for long-tail fallback |

If any of these were fit on the full dataset before splitting (a common mistake), the CV would be inflated because the emit strings and gate thresholds would incorporate validation-set label information.

**Implementation check:** each of the above is instantiated fresh inside the `for tr_idx, va_idx in gkf.split(...)` loop. No global state is shared across folds.

> **Plain-language version (for presentation):**
> Think of it like a cooking competition judge who has already tasted all the dishes before scoring — even if they try to be objective, they've been influenced. "Fold-safe" means the model is only allowed to learn from the training portion of each fold, then is tested blind on the validation portion it has never seen. We enforce this for three things: (1) the **emit strings** — which exact product name to output for a given brand pattern, (2) the **empty-gate threshold** — when to abstain rather than guess, and (3) the **classifier** — the fallback for brands not covered by rules. All three are re-learned from scratch inside every fold, never from the full dataset. This is what makes the CV score trustworthy enough to use as the sole decision criterion for keeping or rejecting an experiment.

### 4.4  Five-fold CV variance as an error bar

With 5 folds and ~4,892 images, each validation fold has ~978 images. The standard error of the mean F1 across folds gives us a ±1σ confidence interval on the true composite score. In practice, fold variance was ~0.008–0.012 composite units, making differences below ~0.003 statistically insignificant at the fold level. We used this to cull small experiments (e.g., highlands-first reordering gave +0.0005 — below the noise floor).

### 4.5  CV→LB gap: distribution shift, not overfitting

| Split | Composite score |
|:--|:--|
| 5-fold CV (train) | 0.6142 |
| Public leaderboard (test) | 0.6685 |
| Gap | **+0.054** |

A naive reading would attribute this to overfitting to the public board. But we **never tuned on the public board** — the 0.6685 was the first submission of the calibrated head, submitted once.

The correct explanation is **distribution shift**: the public test set (Phase 1) is more family-concentrated than the training set.

| Split | Dominant-family marker fraction |
|:--|:--|
| Training set | ~35% |
| Phase-1 test | **~44%** |
| Phase-2 test | **~28%** |

The CalibratedRuleHead performs best precisely when dominant-family images are present (it fires with high precision on those). The test set has more of them → higher F1 → higher composite.

This is confirmed by the Phase-2 result: Phase-2 is **less** family-concentrated (28%), so LB-probed rules that overfit to Phase-1 concentration will degrade. Our train-grounded calibration is expected to generalise better to Phase-2 (§9 in the full plan).

**The Playbook principle:** *"When CV and LB disagree, trust CV."* We held to this: the model was selected by CV rank, not LB rank. The LB number is treated as a noisy estimate on a different distribution, not ground truth.

---

*Next: §5–§6 Models — OCR engine, fine-tuning, emit-gain derivation, oracle/noise decomposition → `03_models.md`*
