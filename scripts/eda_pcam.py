import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt


def infer_keys(h5file):
    keys = [k for k in h5file.keys() if isinstance(h5file[k], h5py.Dataset)]
    if not keys:
        raise RuntimeError("No datasets found in the .h5 file.")
    # Heuristic: images usually have higher ndim (e.g., N,H,W,C or N,C,H,W), labels are 1D/2D.
    keys_sorted = sorted(keys, key=lambda k: h5file[k].ndim)
    label_key = keys_sorted[0]
    image_key = keys_sorted[-1]
    return image_key, label_key, keys


def load_labels(path):
    with h5py.File(path, "r") as f:
        image_key, label_key, keys = infer_keys(f)
        y = f[label_key][:]
    y = np.asarray(y).reshape(-1).astype(int)
    return y, (image_key, label_key, keys)


def sample_mean_intensity(path, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        image_key, label_key, keys = infer_keys(f)
        X = f[image_key]
        y = np.asarray(f[label_key][:]).reshape(-1).astype(int)

        N = len(y)
        idx = rng.choice(N, size=min(n, N), replace=False)

        means = np.empty(len(idx), dtype=np.float32)
        labels = y[idx]

        # Read lazily to avoid loading the whole dataset
        for j, i in enumerate(idx):
            img = X[i]
            means[j] = float(np.mean(img))
    return means, labels, (image_key, label_key, keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="Path to a PCAM .h5 file (train or test).")
    ap.add_argument("--outdir", default="assets", help="Where to save plots.")
    ap.add_argument("--n", type=int, default=5000, help="Number of images to sample for intensity stats.")
    args = ap.parse_args()

    y, meta1 = load_labels(args.h5)
    counts = np.bincount(y, minlength=2)

    means, labels, meta2 = sample_mean_intensity(args.h5, n=args.n)

    # Plot 1: class distribution
    plt.figure(figsize=(6, 4))
    plt.bar([0, 1], counts, width=0.6)
    plt.xticks([0, 1], ["0 (neg)", "1 (pos)"])
    plt.ylabel("Count")
    plt.title("PCAM label distribution")
    plt.tight_layout()
    p1 = f"{args.outdir}/pcam_label_distribution.png"
    plt.savefig(p1, dpi=200)
    plt.close()

    # Plot 2: intensity / outlier check
    plt.figure(figsize=(7, 4))
    plt.hist(means, bins=50)
    plt.xlabel("Mean pixel intensity (per image)")
    plt.ylabel("Frequency")
    plt.title(f"PCAM mean intensity histogram (sample n={len(means)})")
    plt.tight_layout()
    p2 = f"{args.outdir}/pcam_intensity_hist.png"
    plt.savefig(p2, dpi=200)
    plt.close()

    # Plot 3: intensity by class (optional but useful)
    plt.figure(figsize=(7, 4))
    data0 = means[labels == 0]
    data1 = means[labels == 1]
    plt.boxplot([data0, data1], tick_labels=["0 (neg)", "1 (pos)"])
    plt.ylabel("Mean pixel intensity")
    plt.title("PCAM intensity by label (boxplot)")
    plt.tight_layout()
    p3 = f"{args.outdir}/pcam_intensity_by_label.png"
    plt.savefig(p3, dpi=200)
    plt.close()

    print("Saved:", p1)
    print("Saved:", p2)
    print("Saved:", p3)
    print("Inferred keys (images/labels/all):", meta1)


if __name__ == "__main__":
    main()
