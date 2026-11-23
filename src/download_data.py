"""Download and verify official CIFAR-10; extract only expected batch files."""

import hashlib
import tarfile
import urllib.request
from data import ROOT, MD5, load_data


def main():
    directory = ROOT / "data"
    directory.mkdir(exist_ok=True)
    try:
        load_data()
        load_data(test=True)
        print("Verified dataset already present.")
        return
    except (FileNotFoundError, ValueError):
        pass
    archive = directory / "cifar-10-python.tar.gz"
    if not archive.exists():
        print("Downloading official CIFAR-10 (163 MB)...", flush=True)
        urllib.request.urlretrieve(
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", archive
        )
    if hashlib.md5(archive.read_bytes()).hexdigest() != "c58f30108f718f92721af3b95e74349a":
        raise ValueError("Archive MD5 mismatch. Remove the incomplete archive and retry.")
    with tarfile.open(archive, "r:gz") as tar:
        for name, md5 in MD5.items():
            member = tar.getmember("cifar-10-batches-py/" + name)
            blob = tar.extractfile(member).read()
            if hashlib.md5(blob).hexdigest() != md5:
                raise ValueError(f"Batch MD5 mismatch: {name}")
            (directory / name).write_bytes(blob)
    load_data()
    load_data(test=True)
    print("CIFAR-10 downloaded and verified.")


if __name__ == "__main__":
    main()
