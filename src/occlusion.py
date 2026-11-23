"""Post-selection faithfulness audit; not used for selecting or tuning any model."""

import argparse
import json
import numpy as np
import torch
from data import ROOT, load_data
from evaluate import load_model, gradcam
from train import device_for


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    torch.set_num_threads(4)
    device = device_for(args.device)
    name = json.loads((ROOT / "results/selection.json").read_text())["selected"]
    model, ck = load_model(name, device)
    raw, y = load_data(test=True)
    # Fixed balanced sample independent of confidence/correctness: first 10 indices per true class.
    ids = np.concatenate([np.flatnonzero(y == cls)[:10] for cls in range(10)])
    mean = torch.tensor(ck["metadata"]["mean"], device=device, dtype=torch.float32)[
        None, :, None, None
    ]
    std = torch.tensor(ck["metadata"]["std"], device=device, dtype=torch.float32)[
        None, :, None, None
    ]
    x = (torch.from_numpy(raw[ids]).to(device, dtype=torch.float32) / 255.0 - mean) / std
    with torch.no_grad():
        original = model(x).softmax(1)
        targets = original.argmax(1)
        original_conf = original.gather(1, targets[:, None])[:, 0].cpu().numpy()
    cams = np.array([gradcam(model, x[i : i + 1], int(targets[i])) for i in range(len(x))])
    ranking = np.argsort(cams.reshape(len(x), -1), axis=1, kind="stable")
    rng = np.random.default_rng(2026)
    drops = {}
    for mode in ["high", "low", "random"]:
        if mode == "high":
            pixels = ranking[:, -256:]
        elif mode == "low":
            pixels = ranking[:, :256]
        else:
            pixels = np.array([rng.choice(1024, 256, replace=False) for _ in x])
        mask = np.ones((len(x), 1024), dtype=np.float32)
        np.put_along_axis(mask, pixels, 0, axis=1)
        perturbed = x * torch.from_numpy(mask.reshape(-1, 1, 32, 32)).to(device)
        with torch.no_grad():
            probs = model(perturbed).softmax(1)
            conf = probs.gather(1, targets[:, None])[:, 0].cpu().numpy()
        drops[mode] = original_conf - conf
    paired = drops["high"] - drops["random"]
    bootstrap = (
        np.random.default_rng(2026).choice(paired, (5000, len(paired)), replace=True).mean(1)
    )
    result = {
        "model": name,
        "n": len(ids),
        "selection": "first 10 official test indices per true class; selected before viewing outcomes",
        "mask": "25% pixels set to training channel mean (zero after standardization)",
        "target": "original predicted-class probability, fixed after occlusion",
        "mean_probability_drop": {k: float(v.mean()) for k, v in drops.items()},
        "median_probability_drop": {k: float(np.median(v)) for k, v in drops.items()},
        "high_minus_random_mean": float(paired.mean()),
        "high_minus_random_bootstrap95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "high_exceeds_random_fraction": float((paired > 0).mean()),
        "limitations": "Small balanced sample; masks can be out of distribution; bootstrap does not cover training randomness or prove causality.",
    }
    (ROOT / "results/occlusion.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        ROOT / "results/occlusion.npz",
        ids=ids,
        labels=y[ids],
        targets=targets.cpu().numpy(),
        original_confidence=original_conf,
        **{f"drop_{k}": v for k, v in drops.items()},
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
