"""Cross-check saved metrics against raw probabilities and split indices."""

import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    selection = json.loads((ROOT / "results/selection.json").read_text())
    summaries = [
        json.loads(p.read_text()) for p in sorted((ROOT / "results").glob("*/summary.json"))
    ]
    expected = sorted(summaries, key=lambda s: (-s["best_val_acc"], s["name"]))[0]["name"]
    assert selection["selected"] == expected, "Selection is inconsistent with validation results"
    fields = [
        "name",
        "parameters",
        "epochs",
        "best_epoch",
        "best_val_accuracy",
        "test_accuracy",
        "macro_f1",
        "nll",
        "ece15",
        "brier",
        "minutes",
    ]
    rows = []
    for s in summaries:
        name = s["name"]
        base = ROOT / "results" / name
        history = list(csv.DictReader((base / "history.csv").open()))
        assert len(history) == s["epochs_ran"]
        assert max(float(r["val_acc"]) for r in history) == s["best_val_acc"]
        chosen = next(r for r in history if int(r["epoch"]) == s["best_epoch"])
        assert float(chosen["val_acc"]) == s["best_val_acc"]
        for r in history:
            assert all(np.isfinite(float(v)) for v in r.values())
        p = np.load(base / "predictions.npz")
        prob = p["probabilities"]
        y = p["labels"]
        pred = prob.argmax(1)
        assert prob.shape == (10000, 10)
        assert np.allclose(prob.sum(1), 1, atol=2e-6)
        assert np.isfinite(prob).all() and (prob >= 0).all()
        m = json.loads((base / "test_metrics.json").read_text())
        cm = np.zeros((10, 10), dtype=int)
        np.add.at(cm, (y, pred), 1)
        assert np.array_equal(cm, m["confusion"])
        assert np.all(cm.sum(1) == 1000)
        assert abs(np.trace(cm) / 10000 - m["accuracy"]) < 1e-12
        assert abs(-np.log(np.maximum(prob[np.arange(10000), y], 1e-12)).mean() - m["nll"]) < 1e-6
        sp = np.load(ROOT / "results" / f'split_seed{s["args"]["seed"]}.npz')
        tr, va = sp["train"], sp["val"]
        assert len(tr) == 45000 and len(va) == 5000 and len(np.intersect1d(tr, va)) == 0
        assert hashlib.sha256(tr.tobytes() + va.tobytes()).hexdigest() == s["split_sha256"]
        assert (ROOT / "checkpoints" / f"{name}.pt").stat().st_size > 1000000
        assert (
            hashlib.sha256((ROOT / "checkpoints" / f"{name}.pt").read_bytes()).hexdigest()
            == selection["checkpoint_sha256"][name]
        )
        rows.append(
            dict(
                name=name,
                parameters=s["parameters"],
                epochs=s["epochs_ran"],
                best_epoch=s["best_epoch"],
                best_val_accuracy=s["best_val_acc"],
                test_accuracy=m["accuracy"],
                macro_f1=m["macro_f1"],
                nll=m["nll"],
                ece15=m["ece15"],
                brier=m["brier"],
                minutes=s["total_seconds"] / 60,
            )
        )
    with (ROOT / "results/experiment_summary.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with (ROOT / "results/test_predictions.csv").open() as f:
        assert sum(1 for _ in csv.DictReader(f)) == 10000
    cam = np.load(ROOT / "results/gradcam.npz")
    assert cam["images"].shape == (12, 32, 32, 3)
    assert cam["predicted_maps"].shape == (12, 32, 32)
    assert np.isfinite(cam["predicted_maps"]).all() and np.isfinite(cam["true_maps"]).all()
    assert len(np.unique(cam["ids"])) == 12
    assert np.all(cam["predicted_maps"] >= 0) and np.all(cam["predicted_maps"] <= 1.00001)
    audit = np.load(ROOT / "results/occlusion.npz")
    info = json.loads((ROOT / "results/occlusion.json").read_text())
    assert len(np.unique(audit["ids"])) == 100
    assert np.array_equal(np.bincount(audit["labels"]), np.repeat(10, 10))
    for key in ["high", "low", "random"]:
        assert np.isfinite(audit[f"drop_{key}"]).all()
        assert abs(float(audit[f"drop_{key}"].mean()) - info["mean_probability_drop"][key]) < 1e-8
    print(
        f"PASS: {len(summaries)} runs, histories, split hash, 10000 predictions/run, metrics, checkpoints, 12 CAMs."
    )


if __name__ == "__main__":
    main()
