"""Freeze selection on validation accuracy, then evaluate the locked candidates."""

import argparse
import csv
import hashlib
import json
import numpy as np
import torch
import torch.nn.functional as F
from models import Classifier
from train import device_for
from data import ROOT, CLASSES, load_data


def metrics(y, probs):
    pred = probs.argmax(1)
    n = len(y)
    cm = np.zeros((10, 10), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    recall = np.diag(cm) / cm.sum(1)
    precision = np.diag(cm) / np.maximum(cm.sum(0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    acc = float((pred == y).mean())
    z = 1.96
    center = (acc + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    conf = probs.max(1)
    correct = pred == y
    bins = []
    ece = 0.0
    for i in range(15):
        mask = (conf > i / 15) & (conf <= (i + 1) / 15)
        count = int(mask.sum())
        ba = float(correct[mask].mean()) if count else 0.0
        bc = float(conf[mask].mean()) if count else 0.0
        ece += count / n * abs(ba - bc)
        bins.append(dict(count=count, accuracy=ba, confidence=bc))
    return dict(
        accuracy=acc,
        correct=int(correct.sum()),
        n=n,
        wilson95=[center - half, center + half],
        macro_f1=float(f1.mean()),
        nll=float(-np.log(np.maximum(probs[np.arange(n), y], 1e-12)).mean()),
        brier=float(((probs - np.eye(10)[y]) ** 2).sum(1).mean()),
        ece15=ece,
        precision=precision.tolist(),
        recall=recall.tolist(),
        f1=f1.tolist(),
        confusion=cm.tolist(),
        calibration=bins,
    )


def load_model(name, device):
    ck = torch.load(ROOT / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=False)
    recipe = ck["metadata"]["recipe"]
    model = Classifier(recipe["residual"], recipe["se"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def gradcam(model, x, target):
    """Grad-CAM at final 4x4 feature map, for a specified class logit."""
    model.zero_grad(set_to_none=True)
    feat = model.features(x)
    feat.retain_grad()
    logits = model.head(model.drop(model.pool(feat).flatten(1)))
    logits[0, target].backward()
    weights = feat.grad.mean((2, 3), keepdim=True)
    cam = torch.relu((weights * feat).sum(1, keepdim=True))
    cam = F.interpolate(cam, (32, 32), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = device_for(args.device)
    summaries = [
        json.loads(p.read_text()) for p in sorted((ROOT / "results").glob("*/summary.json"))
    ]
    if not summaries:
        raise ValueError("Train experiments before evaluating")
    # Lexical tie break makes selection deterministic. Test outcomes never enter this decision.
    best = sorted(summaries, key=lambda s: (-s["best_val_acc"], s["name"]))[0]
    selection = {
        "criterion": "maximum validation accuracy; lexical name tie break",
        "selected": best["name"],
        "val_accuracy": best["best_val_acc"],
        "candidates": {s["name"]: s["best_val_acc"] for s in summaries},
        "checkpoint_sha256": {
            s["name"]: hashlib.sha256(
                (ROOT / "checkpoints" / f'{s["name"]}.pt').read_bytes()
            ).hexdigest()
            for s in summaries
        },
        "test_policy": "No model or hyperparameter changes after opening test results.",
    }
    selpath = ROOT / "results" / "selection.json"
    if selpath.exists() and json.loads(selpath.read_text()) != selection:
        raise ValueError("Selection is locked. Do not tune on test results.")
    selpath.write_text(json.dumps(selection, indent=2))
    raw, y = load_data(test=True)
    for summary in summaries:
        name = summary["name"]
        out = ROOT / "results" / name
        if (
            (out / "test_metrics.json").exists()
            and (out / "predictions.npz").exists()
            and (
                name != best["name"]
                or (
                    (ROOT / "results/gradcam.npz").exists()
                    and (ROOT / "results/test_predictions.csv").exists()
                )
            )
        ):
            continue
        model, ck = load_model(name, device)
        mean = torch.tensor(ck["metadata"]["mean"], device=device, dtype=torch.float32)[
            None, :, None, None
        ]
        std = torch.tensor(ck["metadata"]["std"], device=device, dtype=torch.float32)[
            None, :, None, None
        ]
        x = (torch.from_numpy(raw).to(device, dtype=torch.float32) / 255.0 - mean) / std
        probs = []
        with torch.no_grad():
            for xb in x.split(256):
                probs.append(model(xb).softmax(1).cpu().numpy())
        probs = np.concatenate(probs)
        m = metrics(y, probs)
        (out / "test_metrics.json").write_text(json.dumps(m, indent=2))
        np.savez_compressed(out / "predictions.npz", labels=y, probabilities=probs)
        print(name, "test accuracy", m["accuracy"], flush=True)
        if name == best["name"]:
            pred = probs.argmax(1)
            confidence = probs.max(1)
            with (ROOT / "results" / "test_predictions.csv").open("w") as f:
                w = csv.writer(f)
                w.writerow(["index", "true", "pred", "confidence"] + [f"p_{c}" for c in CLASSES])
                for i in range(len(y)):
                    w.writerow(
                        [
                            i,
                            CLASSES[y[i]],
                            CLASSES[pred[i]],
                            float(confidence[i]),
                            *probs[i].tolist(),
                        ]
                    )
            # Deterministic examples: 4 high-confidence errors, 4 typical errors near median,
            # and 4 correct examples (one/class where possible); export both target maps.
            errors = np.flatnonzero(pred != y)
            ranked = errors[np.argsort(-confidence[errors])]
            ids = list(ranked[:4])
            mid = len(ranked) // 2
            ids += list(ranked[mid : mid + 4])
            for cls in [0, 1, 3, 5]:
                candidates = np.flatnonzero((y == cls) & (pred == y))
                ids.append(int(candidates[np.argmax(confidence[candidates])]))
            maps_pred = []
            maps_true = []
            for i in ids:
                maps_pred.append(gradcam(model, x[i : i + 1], int(pred[i])))
                maps_true.append(gradcam(model, x[i : i + 1], int(y[i])))
            np.savez_compressed(
                ROOT / "results" / "gradcam.npz",
                ids=np.array(ids),
                images=raw[ids].transpose(0, 2, 3, 1),
                predicted_maps=np.array(maps_pred),
                true_maps=np.array(maps_true),
                labels=y[ids],
                predictions=pred[ids],
                confidence=confidence[ids],
            )
            # Raw montage for later human interpretation, rather than inferring from heatmaps alone.
    print("Locked final model:", best["name"], flush=True)


if __name__ == "__main__":
    main()
