"""E1 baseline: small CNN on EuroSAT RGB 64x64, one run per seed.

Usage (from the repo root):
  python src/train_baseline.py --data data --seeds 1 2 3

--data is the folder holding the RGB PNGs as <split>/<label>/<stem>.png.
--split-csv (default data_card/splits.csv) has columns path,label,block,split, where
`path` is member 1's original Windows path to the .tif, e.g.
C:\\Users\\User\\Downloads\\EuroSAT_MS\\AnnualCrop\\AnnualCrop_1.tif. Only the file stem
is taken from it; the PNG is looked up under --data.
"""
import argparse, json, platform, subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import f1_score

from common import small_cnn
from evaluate import bootstrap_ci, macro_f1

IMG = (64, 64)


def png_path(data_root, split, label, tif_path):
    """<data_root>/<split>/<label>/<stem>.png from a Windows .tif path (works on any OS)."""
    name = tif_path.split("\\")[-1]
    return Path(data_root) / split / label / Path(name).with_suffix(".png").name


def load_split_csv(csv_path, data_root):
    """Read splits.csv, add a `png` column, and stop if any expected PNG is missing."""
    df = pd.read_csv(csv_path)
    df["png"] = [png_path(data_root, s, l, p) for s, l, p in zip(df["split"], df["label"], df["path"])]
    missing = [p for p in df["png"] if not p.exists()]
    if missing:
        examples = "\n  ".join(str(p) for p in missing[:5])
        raise SystemExit(f"{len(missing)} of {len(df)} PNGs listed in {csv_path} are missing "
                         f"under {data_root}. First few:\n  {examples}")
    return df


def read_split(df, name):
    rows = df[df["split"] == name]
    return [str(p) for p in rows["png"]], rows["label"].tolist()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_ds(files, file_labels, class_names, batch=64, shuffle=False, seed=1):
    idx = {c: i for i, c in enumerate(class_names)}
    labels = np.array([idx[c] for c in file_labels], dtype="int32")

    def load(path, y):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG)          # float32 in 0-255; the model rescales
        img.set_shape((*IMG, 3))
        return img, y

    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).cache()
    if shuffle:  # only for training; evaluation order must match labels
        ds = ds.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-csv", default="data_card/splits.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-split", default="val", choices=["val", "test"],
                    help="Keep 'val' until all decisions are made; 'test' is used once.")
    args = ap.parse_args()

    df = load_split_csv(args.split_csv, args.data)

    for d in ("runs", "models", "results/tables", "results/preds"):
        Path(d).mkdir(parents=True, exist_ok=True)

    train_files, train_labels = read_split(df, "train")
    val_files, val_labels = read_split(df, "val")
    eval_files, eval_labels = read_split(df, args.eval_split)
    classes = sorted(df["label"].unique())
    print(f"{len(classes)} classes | train {len(train_files)} | val {len(val_files)} | "
          f"eval ({args.eval_split}) {len(eval_files)}")

    val_ds, _ = make_ds(val_files, val_labels, classes, args.batch)
    eval_ds, y_eval = make_ds(eval_files, eval_labels, classes, args.batch)

    rows = []
    for seed in args.seeds:
        keras.utils.set_random_seed(seed)
        name = f"e1_smallcnn_rgb_seed{seed}"
        train_ds, _ = make_ds(train_files, train_labels, classes, args.batch, shuffle=True, seed=seed)

        model = small_cnn((*IMG, 3), len(classes))
        model.compile(optimizer=keras.optimizers.Adam(args.lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2, shuffle=False,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                           restore_best_weights=True),
                             keras.callbacks.CSVLogger(f"runs/{name}.csv")])
        model.save(f"models/{name}.keras")

        probs = model.predict(eval_ds, verbose=0)
        y_pred = probs.argmax(1)
        np.savez(f"results/preds/{name}_{args.eval_split}.npz",
                 y_true=y_eval, y_pred=y_pred, probs=probs, classes=np.array(classes))

        mf1, lo, hi = bootstrap_ci(y_eval, y_pred, macro_f1, seed=seed)
        per_cls = f1_score(y_eval, y_pred, labels=range(len(classes)), average=None, zero_division=0)
        row = {"seed": seed, "macro_f1": mf1, "ci_low": lo, "ci_high": hi}
        row.update({f"f1_{c}": v for c, v in zip(classes, per_cls)})
        rows.append(row)
        print(f"seed {seed}: macro-F1 {mf1:.3f} [95% CI {lo:.3f}, {hi:.3f}]")

    tag = f"e1_{args.eval_split}"
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(f"results/tables/{tag}_per_seed.csv", index=False)

    metric_cols = ["macro_f1"] + [f"f1_{c}" for c in classes]
    summary = pd.DataFrame({"metric": metric_cols,
                            "mean": per_seed[metric_cols].mean().values,
                            "sd": per_seed[metric_cols].std(ddof=1).values,
                            "n_seeds": len(rows)})
    summary.to_csv(f"results/tables/{tag}_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    Path(f"runs/{tag}_meta.json").write_text(json.dumps({
        "seeds": args.seeds, "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
        "img_size": IMG, "classes": classes, "split_csv": str(args.split_csv),
        "data": str(args.data), "eval_split": args.eval_split, "git_commit": git_commit(),
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "keras": keras.__version__}, indent=2))


if __name__ == "__main__":
    main()
