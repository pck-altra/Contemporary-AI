"""Exact pixel duplicate audit for the fixed training/validation split."""

import hashlib
import json
import numpy as np
from data import ROOT, load_data, split


def main():
    raw, y = load_data()
    tr_idx, va_idx = split(y)
    tr = {hashlib.sha256(raw[i].tobytes()).hexdigest() for i in tr_idx}
    va = {hashlib.sha256(raw[i].tobytes()).hexdigest() for i in va_idx}
    result = {
        "train_unique_images": len(tr),
        "val_unique_images": len(va),
        "train_val_exact_duplicate_hashes": len(tr & va),
        "policy": "Index-disjoint stratified split; no claim of semantic or near-duplicate disjointness.",
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/data_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
