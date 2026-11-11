"""E1 baseline: small CNN on EuroSAT RGB 64x64, one run per seed.

Usage (from the repo root):
  python src/train_baseline.py --data data --seeds 1 2 3

--data is the folder holding the RGB PNGs as <split>/<label>/<stem>.png.
--split-csv (default data_card/splits.csv) has columns path,label,block,split, where
`path` is member 1's original Windows path to the .tif, e.g.
C:\\Users\\User\\Downloads\\EuroSAT_MS\\AnnualCrop\\AnnualCrop_1.tif. Only the file stem
is taken from it; the PNG is looked up under --data.

Images are decoded once and cached as uint8 arrays in <data>/_cache/<split>.npz. The
cache stores a hash of the split's ordered file list and is rebuilt if the split changes.

Each seed writes its own row to results/tables/e1_<split>_seed<N>.csv, and the summary
(mean +/- sd) is rebuilt from all seed files present. So seeds can be run in separate
calls (useful on Colab) and still end up in one summary table.
"""
import argparse, hashlib, json, platform, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from PIL import Image
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


def _read_png(path):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if arr.shape[:2] != IMG:
        arr = np.asarray(Image.fromarray(arr).resize(IMG[::-1], Image.BILINEAR), dtype=np.uint8)
    return arr


def load_arrays(df, split, classes, data_root):
    """Decode a split once into (N, 64, 64, 3) uint8 + int32 labels, cached on disk."""
    rows = df[df["split"] == split]
    rel = [p.relative_to(data_root).as_posix() for p in rows["png"]]
    key = hashlib.sha256("\n".join(rel).encode()).hexdigest()
    y = np.array([classes.index(c) for c in rows["label"]], dtype="int32")

    cache = Path(data_root) / "_cache" / f"{split}.npz"
    if cache.exists():
        with np.load(cache) as z:
            if str(z["key"]) == key:
                print(f"{split}: loaded {len(z['x'])} images from cache")
                return z["x"], y
        print(f"{split}: cache is stale (split changed), rebuilding")

    files = [str(p) for p in rows["png"]]
    x = np.empty((len(files), *IMG, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, arr in enumerate(pool.map(_read_png, files)):
            x[i] = arr
            if (i + 1) % 2000 == 0 or i + 1 == len(files):
                print(f"{split}: decoded {i + 1}/{len(files)}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, key=np.array(key))
    return x, y


def make_ds(x, y, batch=64, shuffle=False, seed=1):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:  # only for training; evaluation order must match labels
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32), b),  # 0-255; the model rescales
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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

    classes = sorted(str(c) for c in df["label"].unique())
    x_tr, y_tr = load_arrays(df, "train", classes, args.data)
    x_va, y_va = load_arrays(df, "val", classes, args.data)
    x_ev, y_ev = (x_va, y_va) if args.eval_split == "val" else load_arrays(df, "test", classes, args.data)
    print(f"{len(classes)} classes | train {len(x_tr)} | val {len(x_va)} | "
          f"eval ({args.eval_split}) {len(x_ev)}")

    val_ds = make_ds(x_va, y_va, args.batch)
    eval_ds = make_ds(x_ev, y_ev, args.batch)
    tag = f"e1_{args.eval_split}"

    for seed in args.seeds:
        keras.utils.set_random_seed(seed)
        name = f"e1_smallcnn_rgb_seed{seed}"
        train_ds = make_ds(x_tr, y_tr, args.batch, shuffle=True, seed=seed)

        model = small_cnn((*IMG, 3), len(classes))
        model.compile(optimizer=keras.optimizers.Adam(args.lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                           restore_best_weights=True),
                             keras.callbacks.CSVLogger(f"runs/{name}.csv")])
        model.save(f"models/{name}.keras")

        probs = model.predict(eval_ds, verbose=0)
        y_pred = probs.argmax(1)
        np.savez(f"results/preds/{name}_{args.eval_split}.npz",
                 y_true=y_ev, y_pred=y_pred, probs=probs, classes=np.array(classes))

        mf1, lo, hi = bootstrap_ci(y_ev, y_pred, macro_f1, seed=seed)
        per_cls = f1_score(y_ev, y_pred, labels=range(len(classes)), average=None, zero_division=0)
        row = {"seed": seed, "macro_f1": mf1, "ci_low": lo, "ci_high": hi}
        row.update({f"f1_{c}": v for c, v in zip(classes, per_cls)})
        pd.DataFrame([row]).to_csv(f"results/tables/{tag}_seed{seed}.csv", index=False)
        print(f"seed {seed}: macro-F1 {mf1:.3f} [95% CI {lo:.3f}, {hi:.3f}]")

    # Rebuild the summary from every seed file present, so separate calls combine.
    seed_files = sorted(Path("results/tables").glob(f"{tag}_seed*.csv"))
    per_seed = pd.concat([pd.read_csv(f) for f in seed_files]).sort_values("seed")
    per_seed.to_csv(f"results/tables/{tag}_per_seed.csv", index=False)
    metric_cols = ["macro_f1"] + [f"f1_{c}" for c in classes]
    summary = pd.DataFrame({"metric": metric_cols,
                            "mean": per_seed[metric_cols].mean().values,
                            "sd": per_seed[metric_cols].std(ddof=1).values,
                            "n_seeds": len(per_seed)})
    summary.to_csv(f"results/tables/{tag}_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    Path(f"runs/{tag}_meta.json").write_text(json.dumps({
        "seeds_in_summary": per_seed["seed"].tolist(), "epochs": args.epochs, "lr": args.lr,
        "batch": args.batch, "img_size": IMG, "classes": classes,
        "split_csv": str(args.split_csv), "data": str(args.data),
        "eval_split": args.eval_split, "git_commit": git_commit(),
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "keras": keras.__version__}, indent=2))


if __name__ == "__main__":
    main()
