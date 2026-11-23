"""Classify an RGB image with a supplied or validation-selected checkpoint."""

import argparse
import json
import numpy as np
import torch
from PIL import Image
from evaluate import load_model
from train import device_for
from data import ROOT, CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    name = args.checkpoint or json.loads((ROOT / "results/selection.json").read_text())["selected"]
    device = device_for(args.device)
    model, ck = load_model(name, device)
    image = Image.open(args.image).convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
    x = (
        torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().unsqueeze(0).to(device)
        / 255.0
    )
    mean = torch.tensor(ck["metadata"]["mean"], device=device)[None, :, None, None]
    std = torch.tensor(ck["metadata"]["std"], device=device)[None, :, None, None]
    with torch.no_grad():
        probs = model((x - mean) / std).softmax(1)[0].cpu().numpy()
    print(
        json.dumps(
            {
                "model": name,
                "top3": [
                    {"class": CLASSES[i], "probability": float(probs[i])}
                    for i in np.argsort(probs)[::-1][:3]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
