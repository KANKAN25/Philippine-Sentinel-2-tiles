"""Rebuild the E1 result tables from the saved predictions of every seed.

Run from the repo root:  python rebuild_e1_tables.py
Reads results/preds/e1_smallcnn_rgb_seed<N>_val.npz and writes
results/tables/e1_val_seed<N>.csv, e1_val_per_seed.csv and e1_val_summary.csv.
Uses the same metric code and bootstrap seed as train_baseline.py, so the numbers
match what each training run printed.
"""
import re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, "src")
from evaluate import bootstrap_ci, macro_f1

SPLIT = "val"
files = sorted(Path("results/preds").glob(f"e1_smallcnn_rgb_seed*_{SPLIT}.npz"))
if not files:
    raise SystemExit("No prediction files found in results/preds/")

rows = []
for f in files:
    seed = int(re.search(r"seed(\d+)_", f.name).group(1))
    z = np.load(f)
    y_true, y_pred, classes = z["y_true"], z["y_pred"], [str(c) for c in z["classes"]]
    mf1, lo, hi = bootstrap_ci(y_true, y_pred, macro_f1, seed=seed)
    per_cls = f1_score(y_true, y_pred, labels=range(len(classes)), average=None, zero_division=0)
    row = {"seed": seed, "macro_f1": mf1, "ci_low": lo, "ci_high": hi}
    row.update({f"f1_{c}": v for c, v in zip(classes, per_cls)})
    rows.append(row)
    pd.DataFrame([row]).to_csv(f"results/tables/e1_{SPLIT}_seed{seed}.csv", index=False)
    print(f"seed {seed}: macro-F1 {mf1:.3f} [95% CI {lo:.3f}, {hi:.3f}]  (n={len(y_true)})")

per_seed = pd.DataFrame(rows).sort_values("seed")
per_seed.to_csv(f"results/tables/e1_{SPLIT}_per_seed.csv", index=False)
cols = ["macro_f1"] + [f"f1_{c}" for c in classes]
summary = pd.DataFrame({"metric": cols, "mean": per_seed[cols].mean().values,
                        "sd": per_seed[cols].std(ddof=1).values, "n_seeds": len(per_seed)})
summary.to_csv(f"results/tables/e1_{SPLIT}_summary.csv", index=False)
print(summary.round(3).to_string(index=False))
