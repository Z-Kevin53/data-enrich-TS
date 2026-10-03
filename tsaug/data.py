"""CIFAR-100 few-shot benchmark (CIFAR-FS style), fixed deterministic split.

Protocol (seed=42, identical for every experiment so all experiments share
the exact same validation set):
  - classes: the first N_CLASSES fine-label classes of CIFAR-100
  - per class: 50 images -> validation (never augmented, never trained on)
               k images   -> few-shot support set D_0
The raw train split is downloaded once via torchvision, parsed to a cached
.npy/.npz so per-experiment loading is fast.
"""
import os
import pickle
from pathlib import Path

import numpy as np

DATA_ROOT = Path(os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cifar100")))
SEED = 42
N_CLASSES = 20
VAL_PER_CLASS = 50
NPZ_NAME = "cifar100_train.npz"
CIFAR100_URLS = [
    "https://ossci-datasets.s3.amazonaws.com/cifar/cifar-100-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
]


def _download_cifar100_direct():
    """Direct HTTP download of the official CIFAR-100 tarball (no torchvision)."""
    import tarfile
    import urllib.request
    print("[DATA] downloading CIFAR-100 train split (direct) ...")
    tar_path = DATA_ROOT / "cifar-100-python.tar.gz"
    last_err = None
    for url in CIFAR100_URLS:
        try:
            urllib.request.urlretrieve(url, tar_path)
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"CIFAR-100 download failed: {last_err!r}")
    with tarfile.open(tar_path) as tf:
        tf.extractall(DATA_ROOT)
    tar_path.unlink(missing_ok=True)
    if not (DATA_ROOT / "cifar-100-python" / "train").exists():
        raise RuntimeError("CIFAR-100 train file missing after extraction")
    print("[DATA] direct download complete")


def _download_cifar100():
    """Download the CIFAR-100 train split.

    Prefers torchvision when it imports cleanly; falls back to a direct HTTP
    download of the official tarball so the pipeline stays usable even when
    the torchvision/torch build pair is mismatched.
    """
    if (DATA_ROOT / "cifar-100-python" / "train").exists():
        return
    try:
        from torchvision.datasets import CIFAR100
        print("[DATA] downloading CIFAR-100 train split via torchvision ...")
        CIFAR100(root=str(DATA_ROOT), train=True, download=True)
        print("[DATA] download complete")
        return
    except Exception as e:
        print(f"[DATA] torchvision download unavailable ({e!r}); "
              f"falling back to direct HTTP")
    _download_cifar100_direct()


def _parse_raw():
    """Parse the CIFAR-100 binary pickle -> (X uint8 [n,32,32,3], y int64).

    The official pickle is a Python-2 dump, so keys/strings may come back as
    bytes; unpickle with encoding="latin-1" to normalize them to str.
    """
    raw = DATA_ROOT / "cifar-100-python" / "train"
    if raw.exists():
        with open(raw, "rb") as f:
            d = pickle.load(f, encoding="latin-1")
        keys = {str(k) for k in d}
        if "data" in keys and "fine_labels" in keys:
            X = d["data"].astype(np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            y = np.asarray(d["fine_labels"], dtype=np.int64)
            return X, y
        raise RuntimeError(f"CIFAR-100 train pickle missing expected keys: {keys}")
    return None, None


def _iter_dataset():
    """Slow fallback: iterate the torchvision dataset object (PIL decoding)."""
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(root=str(DATA_ROOT), train=True, download=False)
    X = np.stack([np.asarray(im) for im, _ in ds])
    y = np.asarray([lb for _, lb in ds], dtype=np.int64)
    return X, y


def ensure_data() -> Path:
    """Guarantee the cached npz exists; download+cache on first call."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    npz = DATA_ROOT / NPZ_NAME
    if npz.exists():
        return npz
    _download_cifar100()
    X, y = _parse_raw()
    if X is None:
        X, y = _iter_dataset()
    np.savez_compressed(npz, X=X, y=y)
    print(f"[DATA] cached CIFAR-100 train split: {X.shape} @ {npz}")
    return npz


def build_fewshot(shots: int = 5, seed: int = SEED) -> dict:
    """Build the fixed few-shot benchmark: support set D_0 and shared validation.

    Returns dict with:
      seed_X  uint8 [C*shots, 32, 32, 3]
      seed_y  int64 [C*shots]            (class ids 0..C-1)
      val_X   uint8 [C*50, 32, 32, 3]
      val_y   int64 [C*50]
      classes list[int] (length C)
    """
    shots = max(2, int(shots))
    npz = ensure_data()
    with np.load(npz) as arr:
        X, y = arr["X"], arr["y"]
    rng = np.random.default_rng(seed)
    seed_idx, val_idx = [], []
    for c in range(N_CLASSES):
        idx = np.where(y == c)[0]
        perm = rng.permutation(idx)
        val_idx.append(perm[:VAL_PER_CLASS])
        rest = perm[VAL_PER_CLASS:]
        seed_idx.append(rest[:shots])
    seed_idx = np.concatenate(seed_idx)
    val_idx = np.concatenate(val_idx)
    return {
        "seed_X": np.ascontiguousarray(X[seed_idx]),
        "seed_y": np.repeat(np.arange(N_CLASSES, dtype=np.int64), shots),
        "val_X": np.ascontiguousarray(X[val_idx]),
        "val_y": np.repeat(np.arange(N_CLASSES, dtype=np.int64), VAL_PER_CLASS),
        "classes": list(range(N_CLASSES)),
    }


if __name__ == "__main__":
    d = build_fewshot(shots=5)
    print("seed:", d["seed_X"].shape, d["seed_y"].shape)
    print("val :", d["val_X"].shape, d["val_y"].shape)
