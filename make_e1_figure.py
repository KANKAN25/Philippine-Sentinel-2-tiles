"""Slide 7 figure: per-class F1 of the E1 baseline, mean +/- sd over seeds.

Run from the repo root after rebuild_e1_tables.py:  python make_e1_figure.py
Reads  results/tables/e1_val_per_seed.csv
Writes results/figures/e1_val_per_class_f1.png (300 dpi) and the CSV behind it.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

per_seed = pd.read_csv("results/tables/e1_val_per_seed.csv")
cls_cols = [c for c in per_seed.columns if c.startswith("f1_")]
n = len(per_seed)

data = pd.DataFrame({
    "class": [c[3:] for c in cls_cols],
    "mean": per_seed[cls_cols].mean().values,
    "sd": per_seed[cls_cols].std(ddof=1).values,
}).sort_values("mean")  # weakest at the bottom
mf1, mf1_sd = per_seed["macro_f1"].mean(), per_seed["macro_f1"].std(ddof=1)

out = Path("results/figures"); out.mkdir(parents=True, exist_ok=True)
data.to_csv(out / "e1_val_per_class_f1.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
weak = data["mean"] < mf1
ax.barh(data["class"], data["mean"], xerr=data["sd"], capsize=3,
        color=["#d9822b" if w else "#4c78a8" for w in weak])
for i, c in enumerate(data["class"]):  # each seed as a dot, so the spread is visible
    ax.scatter(per_seed[f"f1_{c}"], [i] * n, s=12, color="black", zorder=3)
ax.axvline(mf1, ls="--", color="gray", label=f"macro-F1 {mf1:.3f} ± {mf1_sd:.3f}")
ax.legend(loc="lower right", frameon=True)
ax.set_xlim(0, 1.05)
ax.set_xlabel("F1 score (validation set)")
ax.set_title(f"E1 baseline: small CNN, EuroSAT RGB, {n} seeds (preliminary)")
fig.text(0.01, 0.01, f"Bars: mean ± sd over {n} seeds; dots: individual seeds. Orange: below macro-F1. "
         "Spatial-block split; test set not yet used.", fontsize=7, color="gray")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(out / "e1_val_per_class_f1.png", dpi=300)
print(f"macro-F1 {mf1:.3f} ± {mf1_sd:.3f} (n={n})")
print(data.round(3).to_string(index=False))
