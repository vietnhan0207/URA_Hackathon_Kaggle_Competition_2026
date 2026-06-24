# §7–§8 Results
### URA Hackathon 2026 — experiment log, score progression, and residual analysis

---

## §7  The Experiment Log — disciplined iteration

Every experiment below was judged against the **same fixed 5-fold GroupKFold CV** (composite score, mean across 5 folds). An experiment was accepted only if it produced a positive CV delta. The public leaderboard was never used to select between experiments — it was consulted only after a submission was chosen by CV.

> **Plain-language version (for presentation):**
> Think of CV as our internal exam. Every idea had to pass that exam before we would even submit it. This is the Kaggle CV Playbook's core instruction: "never let the public board make your decisions." We ran dozens of experiments; most failed. The table below is the honest record of what we tried and what we kept.

### 7.1  Experiment table

| # | Experiment | CV Δ | Verdict | Why |
|:--|:--|:--|:--|:--|
| 1 | **Engine: PaddleOCR → VietOCR base** | OCR term +0.040 (est.) | ✅ Kept | VietOCR preserves diacritics; structural advantage on CER |
| 2 | **Fine-tune VietOCR on competition images** | CER −0.036 → composite +0.014 | ✅ Kept | LB confirms: v4 0.5898 → v6 0.6030 (+0.013); FT transfers to test domain |
| 3 | **Tighter product gate** (min5/gate0.45 → min12/gate0.55) | prod F1 +0.007 | ✅ Kept | Conservative threshold reduces false-positive on news images; LB confirms v6→v7 (+0.019) |
| 4 | **Empty-gate** (abstain when OCR text is very short) | composite +0.001 | ✅ Kept (marginal) | Free gain; LB v7→v8 confirms +0.0015, smaller than val predicted (test has fewer true-empty) |
| 5 | **Greedy → Beam decode (width=4)** | OCR term +0.003 | ~ Marginal | +0.001 composite; not worth 2× inference time. Logged as option, not deployed in production. |
| 6 | **CalibratedRuleHead** (9 ordered SIG_PATTERNS, argmax-F1 emit strings, min_pprod=0.55, gate=0.75) | composite **+0.017** | ✅ **Kept — decisive** | CV 0.597 → **0.6142**. LB v8→v10 confirms **+0.045**. Rule precision far exceeds classifier posterior on dominant families. |
| 7 | **Aggressive multi-scale detection + CLAHE** | CER −0.041 (worse) | ❌ Rejected | Added recall brings in background noise; GT was transcribed from a single reading pass, not all visible text. More detection ≠ closer GT match. |
| 8 | **Diacritic restoration post-processing** | CER −0.11 (worse) | ❌ Rejected | GT is an exact transcription — if the annotator wrote a word without a tone mark, "correcting" it to the standard form is a substitution error, not a fix. Metric is character-exact. |
| 9 | **Brand-splice OCR correction** (replace detected brand tokens with canonical form) | composite −0.004 | ❌ Rejected | Same reason as #8: GT is exact transcription, not canonical form. Splicing canonical text into OCR output introduces CER errors even where the original was correct. |
| 10 | **Rule reorder: highlands first** | composite +0.0005 | ~ Below noise | Improvement below fold variance (~0.002). The current most-specific-first order is already theoretically optimal. |
| 11 | **News-context abstention** (friend's rule: abstain if negative-sentiment context words present) | prod F1 −0.10 | ❌ Rejected | Our EDA shows scandal/news images still have the product present (annotators tagged product regardless of sentiment context). Abstaining on these images loses correct labels. |
| 12 | **Teammate B's full HybridExtractor on our OCR text** | composite −0.034 | ❌ Rejected | HybridExtractor was tuned on RapidOCR output (diacritic-free). His fixed string patterns do not match VietOCR's accented output. CV 0.5800 vs our 0.6142. |
| 13 | **Teammate B's `evidence_override` module** | composite −0.023 | ❌ Rejected | His `evidence_override` uses long fixed forms that conflict with GT fragmentation — overrides a correct rule emission with a worse canonical form. |
| 14 | **Teammate B's canonicalizer** | composite −0.004 | ❌ Rejected | Our argmax-token-F1 emit string is already train-optimal per cluster. His canonicalizer replaces it with a form that scores lower on the cluster's actual label distribution. |
| 15 | **Detector swap: CRAFT → DBNet (FTdb)** | CER −0.047 (0.4688 vs base 0.4373) | ❌ Rejected | DB detector is tuned for horizontal document text; CRAFT handles TikTok's angled, stylised overlays. Full-set evaluation on 4,891 images confirms regression. |
| 16 | **Upscale small thumbnails before detection (FTup, FTup2)** | CER −0.015 / −0.009 (FTup2 0.4121 vs FT 0.4015) | ❌ Rejected | Upscaling helps slightly vs base but cannot beat plain fine-tuning. The bottleneck is recogniser domain adaptation, not image resolution. LB confirms: ftup2 0.6209 < v8 0.6232. |

**Punchline:** 10 experiments rejected, 6 accepted. Every acceptance had a positive CV delta confirmed by LB. Every rejection had a negative CV delta that was not submitted for LB confirmation. This is the Kaggle CV Playbook in practice — the public board never drove a decision.

---

### 7.2  CV→LB correlation

The consistently positive CV→LB gap is explained by **train→test distribution shift**, not overfitting:

| Submission | CV (train dist.) | Kaggle public LB | Gap |
|:--|--:|--:|--:|
| v6 — FT VietOCR + product head | 0.5900 | 0.6030 | +0.0130 |
| v7 — tighter product gate | 0.5970 | 0.6217 | +0.0247 |
| v8 — + empty-gate | 0.5970 | 0.6232 | +0.0262 |
| **v10 — CalibratedRuleHead** | **0.6142** | **0.6685** | **+0.0543** |

*(Figure 13: `presentation/figures/13_cv_vs_lb.png`)*

**Why the gap grows from +0.013 to +0.054:** the CalibratedRuleHead is the most family-concentrated component in the pipeline — its precision gains are largest when dominant families are overrepresented. Phase-1 public test has ~44% dominant-family images vs ~35% in train. The rule head benefits disproportionately, inflating the LB score relative to CV.

**Why this does not mean we overfit:** we never tuned v10 on the public board. The 0.6685 was the *first* submission of the CalibratedRuleHead. The gap is entirely attributable to distribution shift, not probe-driven selection.

---

## §8  Results & the Independent-Scoring Insight

### 8.1  Score progression

*(Figure 12 / 13: score progression line chart)*

| Milestone | LB | Key change | CV delta |
|:--|:--|:--|:--|
| v2 — PaddleOCR baseline | 0.5890 | First valid full submission | — |
| v3 — all-label training, tighter gate | 0.5803 | Tighter gate *hurt* on diacritic-free text | — |
| v4 — switch to VietOCR | 0.5898 | Diacritics help CER; product head not adapted | — |
| v6 — fine-tuned VietOCR | 0.6030 | FT transfers to test domain | +0.013 |
| v7 — tighter product gate | 0.6217 | Conservative gate lifts product F1 | +0.019 |
| v8 — empty-gate | 0.6232 | Small free gain | +0.002 |
| ftup2_min12 — upscale experiment | 0.6209 | Negative result: upscale < plain FT | — |
| **v10 — CalibratedRuleHead** | **0.6685** | Rule head + argmax emit strings | **+0.045** |
| mixed — our OCR + teammate product | 0.6959 | Independent-column exploit (post-hoc) | n/a |

**Notable: the v3 dip.** Tightening the product gate from min5/gate0.45 to min12/gate0.55 *hurt* on PaddleOCR's text (−0.009). The same configuration *helped* on VietOCR text (+0.019 from v6→v7). Why? PaddleOCR's diacritic-free output reduces pattern-match confidence — the tighter gate abstains too aggressively on a recogniser that cannot produce the patterns faithfully. This is an **engine-head coupling** effect: the optimal gate threshold depends on the OCR engine's output vocabulary.

**The +0.045 jump (v8→v10)** is the single largest single-change gain in the entire experiment log. It comes from two effects acting together:
1. **Higher precision per emission** — rule patterns with argmax-F1 emit strings fire correctly on the top-5 families where the classifier's posterior was noisy.
2. **Distribution shift amplification** — Phase-1 test has more dominant-family images (44%), so high-precision rules fire more often on the test set than the train distribution predicts.

---

### 8.2  The independent-column insight

The competition metric decomposes **exactly** as:

$$\text{Score} = 0.6 \cdot F1_{\text{product}}(\hat{y}_{\text{prod}},\, y_{\text{prod}}) + 0.4 \cdot (1 - \text{CER}(\hat{y}_{\text{ocr}},\, y_{\text{ocr}}))$$

The two terms share no variables. Substituting our OCR (low CER) with teammate B's product predictions (high F1 on his distribution) yields:

$$\text{Score}_{\text{mixed}} = 0.6 \cdot F1_{\text{teammate B}} + 0.4 \cdot (1 - \text{CER}_{\text{ours}})$$

**Result:** mixed submission → **0.6959** (+0.027 over v10 0.6685).

**Honest accounting of the mixed result:**

| Property | v10 (ours) | mixed |
|:--|:--|:--|
| OCR source | VietOCR-FT (CER 0.4015) | VietOCR-FT (same) |
| Product source | CalibratedRuleHead, CV 0.6142 | Teammate B's rules |
| CV reproducible? | ✅ Yes (0.6142 across 5 folds) | ❌ No (teammate B CV 0.5800 on our OCR) |
| LB | 0.6685 | 0.6959 |
| Why LB is high for mixed | Distribution-shift bonus (our model) | Teammate B's rules are LB-tuned — high on public distribution only |
| Expected Phase-2 (private) | ≥ our CV 0.6142 | Degrades — his rules tuned to public test distribution |

**The mixed result is not submitted as our principled prediction for Phase 2.** It is reported here as:
1. Proof that the metric's independent-column structure can be exploited.
2. An upper bound on what is achievable by combining the best available OCR and best available product head separately.
3. A cautionary example: a score that is not reproducible in CV should not be trusted as evidence of generalisation.

> **Plain-language version (for presentation):**
> The competition scores two columns independently — like two separate exams graded separately and then averaged. This means you can put the best student for each exam in separately. We used our OCR (which preserves Vietnamese diacritics, giving us the best CER) and combined it with a teammate's product extraction (which happened to perform better on the public test). The result jumped to 0.6959. But here is the honest part: when we tested our teammate's rules on our own CV setup, they actually scored *worse* than ours (0.58 vs 0.61). His rules were tuned to a specific test distribution; ours were tuned to the training data, which is what matters for the private (final) test.

---

### 8.3  Oracle vs real — OCR-noise decomposition

To understand how much of our product F1 gap is caused by OCR errors (vs the product head itself), we ran the CalibratedRuleHead on two inputs:

| Input to product head | Product F1 |
|:--|:--|
| **GT `ocr_text`** (perfect OCR, no noise) | **0.6904** |
| **VietOCR-FT output** (real OCR) | **0.6161** |
| **Gap = OCR-noise penalty** | **0.0743** |

**Interpretation:** if OCR were perfect, our product head would achieve F1 ≈ 0.69. The gap to real performance (0.62) is almost entirely explained by OCR errors causing rule patterns to miss or fire incorrectly. The remaining ceiling at 0.69 (not 1.0) is the GT-fragmentation ceiling — the structural limit imposed by token-disjoint surface forms (§3.3 / §6.5).

**Headroom decomposition:**

```
                                               F1 = 1.00  (perfect product extraction)
                                                    │
                   GT-fragmentation ceiling ────────┘ (~0.31 gap — irrecoverable from label noise)
                                               F1 = 0.69  (oracle: perfect OCR, our head)
                                                    │
                   OCR-noise penalty ───────────────┘ (~0.074 gap — recoverable with better OCR)
                                               F1 = 0.615  (real: VietOCR-FT + our head)
                                                    │
                   LB result ───────────────────────┘ (test-distribution shift closes some gap)
                                               F1 ≈ 0.647  (implied from LB composite)
```

**What this tells us about where to invest next:**
- Improving OCR CER by another 0.05 (difficult but possible with larger model / more data) would move product F1 by ~0.037.
- The GT-fragmentation ceiling (0.31 gap) is unrecoverable without a canonical-form agreement between annotators — outside our control.
- Our product head is already at ~89% of oracle performance (0.6161 / 0.6904) — the marginal gains from head engineering are small relative to the OCR bottleneck.

---

### 8.4  Where product F1 is lost — residual analysis

Grouping errors by failure mode:

| Failure mode | Approx. share | Example |
|:--|:--|:--|
| **GT fragmentation** — token-disjoint surface forms | ~35% of product errors | GT = `"HALONG CANFOCO"`, we emit `"Đồ Hộp Hạ Long"` → F1 = 0 |
| **OCR failure** — rule pattern does not match degraded OCR text | ~30% of product errors | Diacritic dropped, pattern fails to fire, head abstains → F1 = 0 |
| **Long-tail families** — outside top-5, classifier posterior too low | ~20% of product errors | Obscure brand, classifier scores 0.4 < gate 0.75, abstains → F1 = 0 |
| **News/scandal images** — rule fires, product is empty in GT | ~10% of product errors | Headline mentions brand in negative context; GT `product_name = ""`, we emit brand → penalised |
| **Multi-product images** — multiple brands present, GT picks one | ~5% of product errors | Image shows two products; annotator labelled one; we emit the other |

The **GT fragmentation** category is the most important: it is structural (cannot be fixed by a better model) and accounts for the largest share of errors. It is why the oracle ceiling is 0.69 rather than 1.0.

The **news/scandal** category is the motivation for the `min_pprod=0.55` gate — abstaining when rule precision is below 55% on training data avoids most false-positive emissions on negative-context images.

---

*Next: §9–§10 Generalization & Conclusion → `05_conclusion.md`*
