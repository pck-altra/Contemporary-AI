"""Rebuild publication-quality figures from persisted logs and predictions."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/homework3-matplotlib")
import csv
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
COLORS = ["#8c959f", "#d97706", "#2563eb", "#0d9488", "#9333ea", "#dc4267", "#344b67"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "#fafbfc",
        "savefig.dpi": 180,
    }
)


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def main():
    summaries = [
        json.loads(p.read_text()) for p in sorted((ROOT / "results").glob("*/summary.json"))
    ]
    selection = json.loads((ROOT / "results" / "selection.json").read_text())
    best = selection["selected"]
    data = {}
    tests = {}
    for s in summaries:
        name = s["name"]
        r = list(csv.DictReader((ROOT / "results" / name / "history.csv").open()))
        data[name] = {k: np.array([float(row[k]) for row in r]) for k in r[0]}
        tests[name] = json.loads((ROOT / "results" / name / "test_metrics.json").read_text())
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.5))
    for i, s in enumerate(summaries):
        name = s["name"]
        h = data[name]
        color = COLORS[i % len(COLORS)]
        for a, key in zip(ax.flat, ["train_loss", "val_loss", "train_acc", "val_acc"]):
            a.plot(h["epoch"], h[key], label=name, color=color, lw=1.7)
            a.set_xlabel("Epoch")
            a.grid(alpha=0.18)
        ax[0, 0].set_title("Training objective (different losses across recipes)")
        ax[0, 1].set_title("Validation cross-entropy (hard labels)")
        ax[1, 0].set_title("Training accuracy (Mixup: weighted target accuracy)")
        ax[1, 1].set_title("Validation accuracy (clean images)")
    ax[1, 1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    save(fig, "training_curves.png")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    h = data[best]
    ax[0].plot(h["epoch"], h["train_acc"], label="Train (augmented / mixed)")
    ax[0].plot(h["epoch"], h["val_acc"], label="Validation (clean)")
    ax[0].axvline(
        next(s["best_epoch"] for s in summaries if s["name"] == best),
        color="grey",
        ls=":",
        label="Chosen epoch",
    )
    ax[0].set_title(f"Selected model: {best}")
    ax[0].legend(fontsize=8)
    ax[0].set_xlabel("Epoch")
    ax[1].plot(h["epoch"], h["lr"], color="#0d9488")
    ax[1].set_title("2-epoch warmup + cosine learning rate")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Learning rate")
    for a in ax:
        a.grid(alpha=0.18)
    fig.tight_layout()
    save(fig, "selected_training.png")
    names = [s["name"] for s in summaries]
    xx = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.7))
    ax[0].bar(
        xx - 0.18,
        [s["best_val_acc"] * 100 for s in summaries],
        0.35,
        label="Best validation",
        color="#2563eb",
    )
    ax[0].bar(
        xx + 0.18,
        [tests[n]["accuracy"] * 100 for n in names],
        0.35,
        label="Test (post-selection)",
        color="#0d9488",
    )
    ax[0].set_ylim(60, 100)
    ax[0].set_ylabel("Accuracy (%)")
    ax[0].legend(fontsize=8)
    ax[0].set_xticks(xx, names, rotation=25, ha="right")
    ax[1].scatter(
        [s["total_seconds"] / 60 for s in summaries],
        [tests[n]["accuracy"] * 100 for n in names],
        c=[COLORS[i % 7] for i in xx],
        s=70,
    )
    for s, n in zip(summaries, names):
        ax[1].annotate(
            n,
            (s["total_seconds"] / 60, tests[n]["accuracy"] * 100),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    ax[1].set_xlabel("Training time (minutes)")
    ax[1].set_ylabel("Test accuracy (%)")
    ax[1].set_title("Accuracy and compute budget")
    fig.tight_layout()
    save(fig, "comparison.png")
    m = tests[best]
    cm = np.array(m["confusion"])
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm / 1000, vmin=0, vmax=1, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                f"{cm[i,j]}\n{cm[i,j]/10:.1f}%",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if cm[i, j] > 550 else "#243247",
            )
    ax.set_xticks(range(10), CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{best}: test confusion matrix (count / row %)")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Recall within true class")
    fig.tight_layout()
    save(fig, "confusion.png")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].barh(CLASSES, np.array(m["recall"]) * 100, color="#0d9488")
    ax[0].set_xlim(0, 100)
    ax[0].set_xlabel("Per-class accuracy / recall (%)")
    ax[0].invert_yaxis()
    for name in dict.fromkeys([names[0], "E4_smooth", "E5_mixup", best]):
        bins = tests[name]["calibration"]
        valid = [b for b in bins if b["count"]]
        ax[1].plot(
            [b["confidence"] for b in valid],
            [b["accuracy"] for b in valid],
            marker="o",
            markersize=3,
            label=f'{name} (ECE={tests[name]["ece15"]:.3f})',
        )
    ax[1].plot([0, 1], [0, 1], "k:", alpha=0.6)
    ax[1].legend(fontsize=7)
    ax[1].set_xlabel("Mean confidence")
    ax[1].set_ylabel("Empirical accuracy")
    ax[1].set_title("Reliability diagram (15 equal-width bins)")
    fig.tight_layout()
    save(fig, "class_calibration.png")
    cam = np.load(ROOT / "results" / "gradcam.npz")
    n = len(cam["ids"])
    for start, label in [
        (0, "high_confidence_errors"),
        (4, "typical_errors"),
        (8, "correct_examples"),
    ]:
        fig, axes = plt.subplots(3, 4, figsize=(10, 7))
        for col, k in enumerate(range(start, start + 4)):
            image = cam["images"][k]
            pred = int(cam["predictions"][k])
            true = int(cam["labels"][k])
            for row in range(3):
                axes[row, col].imshow(image, interpolation="nearest")
                axes[row, col].axis("off")
            axes[0, col].set_title(
                f'#{cam["ids"][k]}  true: {CLASSES[true]}\npred: {CLASSES[pred]} ({cam["confidence"][k]:.3f})',
                fontsize=9,
            )
            axes[1, col].imshow(cam["predicted_maps"][k], cmap="jet", alpha=0.48, vmin=0, vmax=1)
            axes[2, col].imshow(cam["true_maps"][k], cmap="jet", alpha=0.48, vmin=0, vmax=1)
            if col == 0:
                axes[0, col].text(
                    -0.15,
                    0.5,
                    "Original",
                    rotation=90,
                    va="center",
                    transform=axes[0, col].transAxes,
                )
                axes[1, col].text(
                    -0.15,
                    0.5,
                    "Predicted-class CAM",
                    rotation=90,
                    va="center",
                    transform=axes[1, col].transAxes,
                )
                axes[2, col].text(
                    -0.15,
                    0.5,
                    "True-class CAM",
                    rotation=90,
                    va="center",
                    transform=axes[2, col].transAxes,
                )
        fig.tight_layout()
        save(fig, label + ".png")
    occlusion = np.load(ROOT / "results/occlusion.npz")
    fig, ax = plt.subplots(1, 2, figsize=(10.8, 3.8))
    values = [occlusion[f"drop_{k}"] for k in ["high", "low", "random"]]
    labels = ["High CAM", "Low CAM", "Random"]
    ax[0].bar(labels, [v.mean() * 100 for v in values], color=["#dc4267", "#0d9488", "#8c959f"])
    ax[0].set_ylabel("Mean probability drop (percentage points)")
    ax[0].set_title("Erase 25% of image pixels (n=100)")
    ax[1].scatter(values[2] * 100, values[0] * 100, color="#2563eb", alpha=0.6, s=20)
    limits = [
        min(values[0].min(), values[2].min()) * 100 - 2,
        max(values[0].max(), values[2].max()) * 100 + 2,
    ]
    ax[1].plot(limits, limits, "k:")
    ax[1].set_xlabel("Probability drop: random erasure (pp)")
    ax[1].set_ylabel("Probability drop: high-CAM erasure (pp)")
    ax[1].set_title("Paired comparison; above line favors CAM")
    fig.tight_layout()
    save(fig, "occlusion.png")

    # A sample montage and train-only pixel distribution, using the saved split.
    raw = []
    ys = []
    for i in range(1, 6):
        batch = pickle.loads((ROOT / "data" / f"data_batch_{i}").read_bytes(), encoding="bytes")
        raw.append(batch[b"data"].reshape(-1, 3, 32, 32))
        ys.extend(batch[b"labels"])
    raw = np.concatenate(raw)
    ys = np.array(ys)
    sp = np.load(ROOT / "results" / f'split_seed{summaries[0]["args"]["seed"]}.npz')
    fig, axes = plt.subplots(5, 10, figsize=(11, 6))
    for c in range(10):
        ids = np.flatnonzero(ys == c)[:5]
        for r, i in enumerate(ids):
            axes[r, c].imshow(raw[i].transpose(1, 2, 0), interpolation="nearest")
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(CLASSES[c], fontsize=8)
    fig.tight_layout()
    save(fig, "dataset_samples.png")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for split_name, color in [("train", "#2563eb"), ("val", "#d97706")]:
        counts = np.bincount(ys[sp[split_name]])
        ax[0].bar(
            np.arange(10) + (-0.24 if split_name == "train" else 0),
            counts,
            0.23,
            label=split_name,
            color=color,
        )
    ax[0].bar(
        np.arange(10) + 0.24, np.repeat(1000, 10), 0.23, label="test (all 1000)", color="#0d9488"
    )
    ax[0].set_xticks(range(10), CLASSES, rotation=35, ha="right")
    ax[0].set_ylabel("Images")
    ax[0].legend(fontsize=8)
    samples = raw[sp["train"][::45]]
    for c, color in enumerate(["#e45756", "#38a169", "#3182ce"]):
        ax[1].hist(
            samples[:, c].ravel() / 255,
            bins=40,
            histtype="step",
            density=True,
            color=color,
            label="RGB"[c],
        )
    ax[1].set_xlabel("Pixel value before standardization")
    ax[1].set_ylabel("Density")
    ax[1].legend()
    fig.tight_layout()
    save(fig, "data_distribution.png")
    # Machine-readable analysis supports the prose, without hard-coding any metric.
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = []
    for idx in np.argsort(off.ravel())[::-1][:10]:
        i, j = np.unravel_index(idx, off.shape)
        pairs.append({"true": CLASSES[i], "pred": CLASSES[j], "count": int(off[i, j])})
    probs = np.load(ROOT / "results" / best / "predictions.npz")
    conf = probs["probabilities"].max(1)
    correct = probs["probabilities"].argmax(1) == probs["labels"]
    detail = {
        "selected": best,
        "top_confusions": pairs,
        "wrong_count": int((~correct).sum()),
        "wrong_confidence_ge_09": int(((~correct) & (conf >= 0.9)).sum()),
        "correct_mean_confidence": float(conf[correct].mean()),
        "wrong_mean_confidence": float(conf[~correct].mean()),
        "class_names": CLASSES,
    }
    (ROOT / "results" / "analysis.json").write_text(json.dumps(detail, indent=2))
    print("Figures and analysis rebuilt:", best)


if __name__ == "__main__":
    main()
