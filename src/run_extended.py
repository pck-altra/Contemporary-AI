"""After the six controlled runs finish, extend the validation-best recipe."""

import json
import subprocess
import sys
from data import ROOT
from train import RECIPES


def main():
    summaries = []
    for name in RECIPES:
        path = ROOT / "results" / name / "summary.json"
        if not path.exists():
            raise RuntimeError(
                "Complete all six controlled runs before selecting extended training"
            )
        summaries.append(json.loads(path.read_text()))
    chosen = sorted(summaries, key=lambda s: (-s["best_val_acc"], s["name"]))[0]
    args = [
        sys.executable,
        str(ROOT / "src/train.py"),
        "--runs",
        chosen["name"],
        "--epochs",
        "60",
        "--batch-size",
        "256",
        "--seed",
        "2026",
        "--suffix",
        "_long60",
        "--patience",
        "12",
        "--device",
        "auto",
    ]
    text = " ".join(args) + "\nChosen from validation results only: " + chosen["name"] + "\n"
    (ROOT / "results/extra_run_command.txt").write_text(text)
    with (ROOT / "results/extended_training.log").open("w") as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
