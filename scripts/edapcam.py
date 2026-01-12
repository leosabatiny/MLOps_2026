from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("/gpfs/scratch1/shared/scur2360/datasets/pcam/pcam_data/surfdrive")
OUT_DIR = Path("assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_X = DATA_DIR / "camelyonpatch_level_2_split_train_x.h5"
TRAIN_Y = DATA_DIR / "camelyonpatch_level_2_split_train_y.h5"
VALID_X = DATA_DIR / "camelyonpatch_level_2_split_valid_x.h5"
VALID_Y = DATA_DIR / "camelyonpatch_level_2_split_valid_y.h5"


def first_key(h5file):
    return list(h5file.keys())[0]


def load_labels(path):
    with h5py.File(path, "r") as f:
        y = np.asarray(f[first_key(f)]).squeeze()
    return y.astype(np.int64)


def sample_means(x_path, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(x_path, "r") as f:
        x = f[first_key(f)]
        idx = rng.choice(len(x), size=min(n, len(x)), replace=False)
        means = np.array([x[i].mean() for i in idx], dtype=np.float32)
    return means


def save_label_barplot(y, outpath, title):
    counts = np.bincount(y, minlength=2)
    plt.figure(figsize=(4, 3))
    plt.bar([0, 1], counts)
    plt.xticks([0, 1], ["0", "1"])
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_hist(values, outpath, title, xlabel):
    plt.figure(figsize=(5, 3))
    plt.hist(values, bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


if __name__ == "__main__":
    y_train = load_labels(TRAIN_Y)
    y_valid = load_labels(VALID_Y)

    save_label_barplot(y_train, OUT_DIR / "q7_eda_train_label_counts.png", "PCAM train label counts")
    save_label_barplot(y_valid, OUT_DIR / "q7_eda_valid_label_counts.png", "PCAM valid label counts")

    train_means = sample_means(TRAIN_X, n=500, seed=0)
    save_hist(train_means, OUT_DIR / "q7_eda_train_mean_intensity_hist.png",
              "Train mean pixel intensity (sample)", "Mean intensity (0-255-ish)")

    valid_means = sample_means(VALID_X, n=500, seed=1)
    save_hist(valid_means, OUT_DIR / "q7_eda_valid_mean_intensity_hist.png",
              "Valid mean pixel intensity (sample)", "Mean intensity (0-255-ish)")

    print("Saved plots to:", OUT_DIR.resolve())
