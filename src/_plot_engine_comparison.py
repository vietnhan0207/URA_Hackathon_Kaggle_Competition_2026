"""Plot engine comparison charts from engine_comparison.json.
Run AFTER _compare_engines.py has produced the JSON.

    python src/_plot_engine_comparison.py

Saves 3 PNGs to presentation/figures/:
  11_engine_cer_diac.png   — CER + diacritic density grouped bar
  12_engine_composite.png  — composite score bar (head applied to each OCR)
  13_cv_vs_lb.png          — CV vs public LB scatter/line
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT   = Path(__file__).resolve().parents[1]
FIG    = ROOT / "presentation" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DATA   = ROOT / "presentation" / "engine_comparison.json"

if not DATA.exists():
    raise FileNotFoundError(f"Run _compare_engines.py first → {DATA}")

payload = json.loads(DATA.read_text(encoding="utf-8"))
engines = payload["engines"]
cv_lb   = payload["cv_lb"]
gt_diac = payload["gt_diac_ref"]

PALETTE = ["#e94b27", "#6c761b", "#ffb8a7", "#324712"]
ORANGE, OLIVE, PEACH, GREEN = PALETTE

mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "text.color": "black",
    "axes.labelcolor": "black", "axes.titlecolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "axes.edgecolor": "black", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110,
})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"  saved {name}.png")

# Sort engines by CER ascending (best first)
engines_sorted = sorted(engines, key=lambda e: e["cer"])
names  = [e["name"].replace(" ★", "") for e in engines_sorted]
cers   = [e["cer"]      for e in engines_sorted]
diacs  = [e["eng_diac"] for e in engines_sorted]
comps  = [e["composite"] for e in engines_sorted]
is_ours = ["FT (ours)" in e["name"] or "★" in e["name"] for e in engines_sorted]

x = np.arange(len(names))
w = 0.35

# ── Figure 1: CER + diacritic density ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# CER bar
bar_colors = [ORANGE if not o else GREEN for o in is_ours]
b1 = axes[0].bar(x, cers, color=bar_colors, edgecolor="black", linewidth=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(names, rotation=35, ha="right", fontsize=9)
axes[0].set_ylabel("Mean CER (lower = better)")
axes[0].set_title("OCR Engine CER Comparison\n(green = our chosen engine)")
for i, v in enumerate(cers):
    axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8,
                 fontweight="bold" if is_ours[i] else "normal")

# Diacritic density grouped bar
b2 = axes[1].bar(x - w/2, diacs, w, label="Engine output",
                  color=[GREEN if o else ORANGE for o in is_ours],
                  edgecolor="black", linewidth=0.5)
b3 = axes[1].bar(x + w/2, [gt_diac] * len(x), w, label=f"GT reference ({gt_diac:.3f})",
                  color=PEACH, edgecolor=OLIVE, linewidth=0.5, alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(names, rotation=35, ha="right", fontsize=9)
axes[1].set_ylabel("Diacritic marks per letter")
axes[1].set_title("Diacritic Preservation vs GT Reference\n(closer to GT = better)")
axes[1].legend(fontsize=9)
for i, v in enumerate(diacs):
    axes[1].text(i - w/2, v + 0.003, f"{v:.3f}", ha="center", fontsize=7)

plt.suptitle("OCR Engine Bake-off: CER and Diacritic Preservation", fontsize=14,
             fontweight="bold")
save(fig, "11_engine_cer_diac")
plt.show()

# ── Figure 2: Composite score per engine (same product head applied to all) ───
fig, ax = plt.subplots(figsize=(11, 5))
bar_cols = [GREEN if o else ORANGE for o in is_ours]
bars = ax.bar(x, comps, color=bar_cols, edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Composite score (same CalibratedRuleHead)")
ax.set_title("Composite Score by OCR Engine\n"
             "(CalibratedRuleHead applied to each engine's output — isolates OCR contribution)")
for i, v in enumerate(comps):
    ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=8,
            fontweight="bold" if is_ours[i] else "normal",
            color=GREEN if is_ours[i] else "black")
ax.set_ylim(min(comps) * 0.97, max(comps) * 1.04)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GREEN, label="VietOCR-FT (chosen)"),
                   Patch(color=ORANGE, label="Other engines")],
          fontsize=9, loc="lower right")
save(fig, "12_engine_composite")
plt.show()

# ── Figure 3: CV vs public LB ──────────────────────────────────────────────────
cv_pts = [(d["tag"], d["cv"], d["lb"]) for d in cv_lb if d["cv"] is not None]
tags_cv  = [p[0] for p in cv_pts]
cvs      = [p[1] for p in cv_pts]
lbs      = [p[2] for p in cv_pts]

# also include the mix point (no CV) as a separate marker
mix = next((d for d in cv_lb if d["tag"] == "mix"), None)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: CV vs LB scatter
ax = axes[0]
ax.scatter(cvs, lbs, color=ORANGE, s=120, zorder=5)
for tag, cv_v, lb_v in cv_pts:
    ax.annotate(tag, (cv_v, lb_v), textcoords="offset points",
                xytext=(6, 4), fontsize=9)
# diagonal y=x reference
lo, hi = min(cvs + lbs) - 0.01, max(cvs + lbs) + 0.01
ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, label="CV = LB")
ax.set_xlabel("5-fold CV score (train distribution)")
ax.set_ylabel("Kaggle public LB score (test distribution)")
ax.set_title("CV vs Public LB\n(points above diagonal = test favoured model)")
ax.legend(fontsize=9)

# Right: progression line chart (all milestones including mix)
all_tags = [d["tag"] for d in cv_lb]
all_lbs  = [d["lb"]  for d in cv_lb]
all_cvs  = [d["cv"]  if d["cv"] else None for d in cv_lb]
x_idx    = np.arange(len(all_tags))

ax2 = axes[1]
ax2.plot(x_idx, all_lbs, "-o", color=ORANGE, lw=2.5, ms=9, label="Public LB")
cv_mask = [(i, v) for i, v in enumerate(all_cvs) if v is not None]
if cv_mask:
    ax2.plot([i for i, _ in cv_mask], [v for _, v in cv_mask],
             "s--", color=OLIVE, lw=1.8, ms=8, label="CV (train)")
for i, v in enumerate(all_lbs):
    ax2.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=8, color=ORANGE)
for i, v in cv_mask:
    ax2.text(i, v - 0.006, f"{v:.4f}", ha="center", fontsize=8, color=OLIVE)

ax2.set_xticks(x_idx)
ax2.set_xticklabels(all_tags, rotation=20, ha="right")
ax2.set_ylabel("Score")
ax2.set_title("Score Progression: CV vs Public LB\n"
              "(gap = distribution shift, not overfit)")
ax2.legend(fontsize=9)

plt.suptitle("Cross-validation vs Leaderboard Analysis", fontsize=14, fontweight="bold")
save(fig, "13_cv_vs_lb")
plt.show()

print("\nAll 3 figures saved to", FIG)
