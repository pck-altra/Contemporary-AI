"""Verified CIFAR batches and a single stratified, persisted split."""

from pathlib import Path
import hashlib
import pickle
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MD5 = {
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    "test_batch": "40351d587109b95175f43aff81a1287e",
    "batches.meta": "5ff9c542aee3614f3951f8cda6e48888",
}


def load_data(test=False, directory=None):
    path = Path(directory) if directory else ROOT / "data"
    files = ["test_batch"] if test else [f"data_batch_{i}" for i in range(1, 6)]
    images, labels = [], []
    for name in files:
        blob = (path / name).read_bytes()
        if hashlib.md5(blob).hexdigest() != MD5[name]:
            raise ValueError(f"CIFAR integrity check failed: {name}")
        obj = pickle.loads(blob, encoding="bytes")  # Only verified official batch files.
        images.append(obj[b"data"].reshape(-1, 3, 32, 32))
        labels.extend(obj[b"labels"])
    return np.concatenate(images), np.asarray(labels, dtype=np.int64)


def split(labels, seed=2026):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for cls in range(10):
        ids = rng.permutation(np.flatnonzero(labels == cls))
        val.extend(ids[:500])
        train.extend(ids[500:])
    return np.asarray(train), np.asarray(val)


def augment(x, generator):
    """Independent crop and flip per sample; random choices use a private CPU RNG."""
    n = len(x)
    offsets = torch.randint(9, (n, 2), generator=generator).to(x.device)
    rows = torch.arange(32, device=x.device)[None, :, None] + offsets[:, 0, None, None]
    cols = torch.arange(32, device=x.device)[None, None, :] + offsets[:, 1, None, None]
    indices = (rows * 40 + cols).reshape(n, 1, 1024).expand(n, 3, 1024)
    x = F.pad(x, (4, 4, 4, 4), mode="reflect").flatten(2).gather(2, indices).reshape(n, 3, 32, 32)
    flip = (torch.rand(n, generator=generator) < 0.5).to(x.device)
    return torch.where(flip[:, None, None, None], x.flip(-1), x)
