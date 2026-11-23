"""Compact figures for the two-column experiment report."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/homework3-matplotlib")
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "report"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.1,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 240,
        "axes.linewidth": 0.6,
    }
)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def main():
    name = json.loads((ROOT / "results/selection.json").read_text())["selected"]
    rows = list(csv.DictReader((ROOT / "results" / name / "history.csv").open()))
    h = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.0))
    for ax, keys, labels in [
        (axes[0], ["train_loss", "val_loss"], ["Train (smoothed CE)", "Validation (hard CE)"]),
        (axes[1], ["train_acc", "val_acc"], ["Train (augmented)", "Validation (clean)"]),
    ]:
        for key, label, color in zip(keys, labels, ["#355c83", "#b56b28"]):
            ax.plot(h["epoch"], h[key], label=label, color=color, lw=1.1)
        ax.legend()
        ax.grid(alpha=0.15)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Accuracy")
    axes[1].axvline(54, color="grey", linestyle=":", lw=0.8)
    fig.tight_layout(pad=0.55)
    save(fig, "curves.png")

    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    palette = ["#94999d", "#c17d23", "#406f9b", "#258a7d", "#8f65a7", "#bd5873"]
    for color, run in zip(
        palette, ["E0_plain", "E1_aug", "E2_residual", "E3_se", "E4_smooth", "E5_mixup"]
    ):
        r = list(csv.DictReader((ROOT / "results" / run / "history.csv").open()))
        ax.plot(
            [int(v["epoch"]) for v in r],
            [float(v["val_acc"]) * 100 for v in r],
            label=run[:2],
            color=color,
            lw=1.0,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.legend(ncol=3, loc="lower right")
    ax.grid(alpha=0.15)
    fig.tight_layout(pad=0.5)
    save(fig, "ablation_curves.png")

    m = json.loads((ROOT / "results" / name / "test_metrics.json").read_text())
    cm = np.array(m["confusion"])
    classes = ["airplane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    fig, ax = plt.subplots(figsize=(3.5, 3.45))
    ax.imshow(cm / 1000, vmin=0, vmax=1, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=6.1,
                color="white" if cm[i, j] > 550 else "#17232e",
            )
    ax.set_xticks(range(10), classes, rotation=45, ha="right")
    ax.set_yticks(range(10), classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout(pad=0.45)
    save(fig, "confusion.png")

    fig, ax = plt.subplots(figsize=(3.5, 2.15))
    for run, label, color in [
        ("E2_residual", "E2", "#406f9b"),
        ("E4_smooth", "E4", "#8f65a7"),
        ("E5_mixup", "E5", "#bd5873"),
        (name, "E4-60", "#258a7d"),
    ]:
        bins = json.loads((ROOT / "results" / run / "test_metrics.json").read_text())["calibration"]
        bins = [b for b in bins if b["count"]]
        ax.plot(
            [b["confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            marker="o",
            ms=2,
            lw=0.9,
            label=label,
            color=color,
        )
    ax.plot([0, 1], [0, 1], "k:", lw=0.7)
    ax.legend()
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    fig.tight_layout(pad=0.45)
    save(fig, "calibration.png")

    cam = np.load(ROOT / "results/gradcam.npz")
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.45))
    for ax, k in zip(axes.flat, range(4)):
        ax.imshow(cam["images"][k], interpolation="nearest")
        ax.axis("off")
        ax.set_title(
            f'#{cam["ids"][k]}  {classes[cam["labels"][k]]} -> {classes[cam["predictions"][k]]}\np={cam["confidence"][k]:.3f}',
            fontsize=7.5,
            pad=3,
        )
    fig.tight_layout(pad=0.35)
    save(fig, "errors.png")

    fig, axes = plt.subplots(3, 3, figsize=(3.5, 3.55))
    for col, k in enumerate([0, 3, 11]):
        for row in range(3):
            axes[row, col].imshow(cam["images"][k], interpolation="nearest")
            axes[row, col].axis("off")
        axes[0, col].set_title(
            f'#{cam["ids"][k]}\n{classes[cam["labels"][k]]} -> {classes[cam["predictions"][k]]}',
            fontsize=7,
            pad=3,
        )
        axes[1, col].imshow(cam["predicted_maps"][k], cmap="jet", alpha=0.48, vmin=0, vmax=1)
        axes[2, col].imshow(cam["true_maps"][k], cmap="jet", alpha=0.48, vmin=0, vmax=1)
    for row, label in enumerate(["Original", "Predicted CAM", "True CAM"]):
        axes[row, 0].text(
            -0.11,
            0.5,
            label,
            rotation=90,
            va="center",
            ha="right",
            fontsize=7,
            transform=axes[row, 0].transAxes,
        )
    fig.tight_layout(pad=0.4, h_pad=0.35, w_pad=0.3)
    save(fig, "cam.png")

    oc = np.load(ROOT / "results/occlusion.npz")
    fig, ax = plt.subplots(figsize=(3.5, 1.65))
    values = [oc[f"drop_{k}"].mean() * 100 for k in ["high", "low", "random"]]
    ax.bar(
        ["High CAM", "Low CAM", "Random"],
        values,
        color=["#b55b62", "#4e8c80", "#8a939c"],
        width=0.6,
    )
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_ylim(0, 73)
    ax.set_ylabel("Probability drop (pp)")
    fig.tight_layout(pad=0.4)
    save(fig, "occlusion.png")


if __name__ == "__main__":
    main()
