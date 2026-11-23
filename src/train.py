"""Run controlled validation-only experiments. No test data are loaded here."""

import argparse
import csv
import hashlib
import json
import platform
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from models import Classifier
from data import ROOT, load_data, split, augment

# E0 -> E1 changes augmentation; E1 -> E2 residual; E2 -> E3 SE;
# E3 -> E4 smoothing; E4 -> E5 Mixup. All other hyperparameters fixed.
RECIPES = {
    "E0_plain": dict(residual=False, se=False, aug=False, smoothing=0.0, mixup=0.0),
    "E1_aug": dict(residual=False, se=False, aug=True, smoothing=0.0, mixup=0.0),
    "E2_residual": dict(residual=True, se=False, aug=True, smoothing=0.0, mixup=0.0),
    "E3_se": dict(residual=True, se=True, aug=True, smoothing=0.0, mixup=0.0),
    "E4_smooth": dict(residual=True, se=True, aug=True, smoothing=0.1, mixup=0.0),
    "E5_mixup": dict(residual=True, se=True, aug=True, smoothing=0.1, mixup=0.2),
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def evaluate(model, x, y, batch):
    model.eval()
    total_loss = torch.zeros((), device=x.device)
    correct = torch.zeros((), device=x.device)
    with torch.no_grad():
        for start in range(0, len(x), batch):
            logits = model(x[start : start + batch])
            target = y[start : start + batch]
            total_loss += F.cross_entropy(logits, target, reduction="sum")
            correct += (logits.argmax(1) == target).sum()
    return total_loss.item() / len(x), correct.item() / len(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=list(RECIPES))
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="auto")
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--suffix", default="")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    device = device_for(args.device)
    raw, labels = load_data()
    tr, va = split(labels, args.seed)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    splitfile = ROOT / "results" / f"split_seed{args.seed}.npz"
    np.savez(splitfile, train=tr, val=va)
    mean = raw[tr].mean((0, 2, 3)) / 255.0
    std = raw[tr].std((0, 2, 3)) / 255.0
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)[None, :, None, None]
    std_t = torch.tensor(std, dtype=torch.float32, device=device)[None, :, None, None]
    x = torch.from_numpy(raw).to(device, dtype=torch.float32) / 255.0
    x = (x - mean_t) / std_t
    y = torch.from_numpy(labels).to(device)
    xtr, ytr = x[torch.from_numpy(tr).to(device)], y[torch.from_numpy(tr).to(device)]
    xv, yv = x[torch.from_numpy(va).to(device)], y[torch.from_numpy(va).to(device)]
    del x, y, raw
    for run in args.runs:
        recipe = RECIPES[run]
        name = run + args.suffix
        out = ROOT / "results" / name
        out.mkdir(exist_ok=True)
        if (out / "summary.json").exists() and not args.force:
            print(f"{name}: complete, skipping", flush=True)
            continue
        seed_all(args.seed)
        model = Classifier(recipe["residual"], recipe["se"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0005)
        cpu_rng = torch.Generator().manual_seed(args.seed)
        aug_rng = torch.Generator().manual_seed(args.seed + 1)
        mix_rng = np.random.default_rng(args.seed + 2)
        if args.benchmark:
            for _ in range(3):
                loss = F.cross_entropy(model(xtr[: args.batch_size]), ytr[: args.batch_size])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            sync(device)
            start = time.perf_counter()
            for _ in range(20):
                xb = augment(xtr[: args.batch_size], aug_rng)
                loss = F.cross_entropy(model(xb), ytr[: args.batch_size])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            sync(device)
            print(
                json.dumps(
                    {
                        "device": str(device),
                        "batch": args.batch_size,
                        "seconds_per_step": (time.perf_counter() - start) / 20,
                        "params": sum(p.numel() for p in model.parameters()),
                    }
                ),
                flush=True,
            )
            return
        metadata = {
            "name": name,
            "recipe": recipe,
            "args": vars(args),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "split_sha256": hashlib.sha256(tr.tobytes() + va.tobytes()).hexdigest(),
            "train_size": len(tr),
            "val_size": len(va),
            "train_per_class": np.bincount(labels[tr]).tolist(),
            "val_per_class": np.bincount(labels[va]).tolist(),
            "determinism": "Seeded RNG; accelerator kernels may be nondeterministic; not a cross-device bitwise guarantee.",
        }
        (out / "config.json").write_text(json.dumps(metadata, indent=2))
        fields = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"]
        best = -1
        best_epoch = 0
        history = []
        total_start = time.perf_counter()
        with (out / "history.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                # 2-epoch linear warmup followed by cosine, fixed horizon per run.
                if epoch <= 2:
                    lr = args.lr * epoch / 2
                else:
                    lr = 1e-5 + (args.lr - 1e-5) * 0.5 * (
                        1 + np.cos(np.pi * (epoch - 2) / max(1, args.epochs - 2))
                    )
                for group in optimizer.param_groups:
                    group["lr"] = float(lr)
                model.train()
                start = time.perf_counter()
                order = torch.randperm(len(xtr), generator=cpu_rng).to(device)
                loss_sum = torch.zeros((), device=device)
                acc_sum = torch.zeros((), device=device)
                for ids in order.split(args.batch_size):
                    xb, yb = xtr[ids], ytr[ids]
                    if recipe["aug"]:
                        xb = augment(xb, aug_rng)
                    optimizer.zero_grad(set_to_none=True)
                    if recipe["mixup"]:
                        lam = float(mix_rng.beta(recipe["mixup"], recipe["mixup"]))
                        perm = torch.randperm(len(ids), generator=cpu_rng).to(device)
                        xb = lam * xb + (1 - lam) * xb[perm]
                        other = yb[perm]
                        logits = model(xb)
                        loss = lam * F.cross_entropy(
                            logits, yb, label_smoothing=recipe["smoothing"]
                        ) + (1 - lam) * F.cross_entropy(
                            logits, other, label_smoothing=recipe["smoothing"]
                        )
                        acc = (
                            lam * (logits.argmax(1) == yb).sum()
                            + (1 - lam) * (logits.argmax(1) == other).sum()
                        )
                    else:
                        logits = model(xb)
                        loss = F.cross_entropy(logits, yb, label_smoothing=recipe["smoothing"])
                        acc = (logits.argmax(1) == yb).sum()
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.detach() * len(ids)
                    acc_sum += acc.detach()
                vl, va_acc = evaluate(model, xv, yv, args.batch_size)
                sync(device)
                row = dict(
                    epoch=epoch,
                    lr=float(lr),
                    train_loss=loss_sum.item() / len(xtr),
                    train_acc=acc_sum.item() / len(xtr),
                    val_loss=vl,
                    val_acc=va_acc,
                    seconds=time.perf_counter() - start,
                )
                writer.writerow(row)
                handle.flush()
                history.append(row)
                if va_acc > best:
                    best = va_acc
                    best_epoch = epoch
                    torch.save(
                        {
                            "model": model.cpu().state_dict(),
                            "metadata": metadata,
                            "epoch": epoch,
                            "val_acc": best,
                        },
                        ROOT / "checkpoints" / f"{name}.pt",
                    )
                    model.to(device)
                print(
                    f'{name} {epoch:02d}/{args.epochs} loss={row["train_loss"]:.4f} train={row["train_acc"]:.4f} val={va_acc:.4f} time={row["seconds"]:.1f}s',
                    flush=True,
                )
                if epoch - best_epoch >= args.patience:
                    print(f"{name}: early stop", flush=True)
                    break
        summary = {
            **metadata,
            "best_val_acc": best,
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            "total_seconds": time.perf_counter() - total_start,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
