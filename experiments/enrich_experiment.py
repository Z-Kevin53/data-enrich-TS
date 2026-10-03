"""Data enrichment experiment on UCI Spambase (real open dataset).

Task: given only a SEED subset of labeled emails, find the enrichment
strategy (synthetic / oversample / noise / hybrid) that maximizes the
downstream validation accuracy of a small MLP. Enrichment targets the
minority class (non-spam, ~39% of Spambase).

Data: UCI Spambase - 4601 emails, 57 numeric features, binary label (1=spam).
Downloaded on first run, cached under data/spambase/ (git-ignored).

Split: fixed seed 42 -> 85% train / 15% val; seed = first int(seed_ratio *
len(train)) rows of the shuffled train split. Val is never touched.

Params: EXPERIMENT_PARAMS env var (orchestrator) -> --params -> defaults.
Output: final stdout line is "val_acc: X.XXXXXX".
GPU: physical GPU 0 only (pinned by nvidia-smi UUID), CPU fallback.
"""
import os
import subprocess


def _pin_to_gpu0():
    """Pin CUDA to physical GPU 0 (nvidia-smi index 0) only."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("AR_GPU"):
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True, timeout=10)
        first = out.strip().splitlines()[0].strip()
        if first:
            os.environ["CUDA_VISIBLE_DEVICES"] = first
    except Exception:
        pass  # no nvidia-smi / no GPU -> leave default behavior


_pin_to_gpu0()

import argparse
import io
import json
import tarfile
import time
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "spambase"))
CSV_NAME = "spambase.csv"
N_FEATURES = 57

MIRRORS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.csv",
    "https://archive.ics.uci.edu/static/public/22/spambase.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/00226_spambase_data.tar.gz",
]

DEFAULTS = {
    "enrich_method": "none",   # none | synthetic | oversample | noise | hybrid
    "seed_ratio": 0.25,        # fraction of the train split used as labeled seed
    "enrich_ratio": 1.0,       # #added minority rows = ratio * seed minority count
    "noise_std": 0.1,          # gaussian noise std (noise/hybrid/synthetic jitter)
    "hidden_dim": 64,
    "n_epochs": 5,
    "learning_rate": 0.001,
    "batch_size": 32,
    "dropout": 0.1,
    "weight_decay": 0.001,
    "optimizer": "adam",
    "seed": 42,
}


def _fetch_csv():
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, CSV_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        return dest
    last_err = None
    for url in MIRRORS:
        try:
            print(f"[DATA] downloading {url}")
            with urllib.request.urlopen(url, timeout=120) as r:
                blob = r.read()
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(blob)) as z:
                    name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
                    blob = z.read(name)
            elif url.endswith(".tar.gz"):
                with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as t:
                    mem = next(m for m in t.getmembers()
                               if m.name.lower().endswith((".csv", ".data")))
                    blob = t.extractfile(mem).read()
            text = blob.decode("utf-8", errors="replace")
            rows = [line for line in text.splitlines() if line.strip()]
            if len(rows) < 4000:
                raise RuntimeError(f"unexpected row count {len(rows)}")
            with open(dest, "w") as f:
                f.write("\n".join(rows) + "\n")
            print(f"[DATA] cached {len(rows)} rows -> {dest}")
            return dest
        except Exception as e:
            last_err = e
            print(f"[DATA] mirror failed: {e}")
    raise RuntimeError(f"could not download spambase: {last_err}")


def load_data():
    path = _fetch_csv()
    with open(path) as f:
        values = [line.split(",") for line in f if line.strip()]
    arr = np.array(values, dtype=np.float32)
    X, y = arr[:, :N_FEATURES], (arr[:, N_FEATURES] > 0.5).astype(np.int64)
    return X, y


def enrich(X_seed, y_seed, method, enrich_ratio, noise_std, rng):
    """Enrich the minority class per method; return (X, y)."""
    counts = np.bincount(y_seed, minlength=2)
    if method == "none" or counts.min() == 0:
        return X_seed, y_seed
    minority = int(np.argmin(counts))
    idx_m = np.where(y_seed == minority)[0]
    n_add = int(len(idx_m) * enrich_ratio)
    if n_add <= 0:
        return X_seed, y_seed

    def pick(n):
        return idx_m[rng.integers(len(idx_m), size=n)]

    def synth(n):
        a, b = pick(n), pick(n)
        u = rng.random((n, 1)).astype(np.float32)
        return X_seed[a] * u + X_seed[b] * (1.0 - u)

    def noisy(n):
        return (X_seed[pick(n)]
                + rng.normal(0.0, noise_std, size=(n, N_FEATURES)).astype(np.float32))

    label = lambda n: np.full(n, minority, dtype=np.int64)

    if method == "oversample":
        extra_x, extra_y = X_seed[pick(n_add)], label(n_add)
    elif method == "synthetic":
        jitter = rng.normal(0.0, noise_std, size=(n_add, N_FEATURES)).astype(np.float32)
        extra_x, extra_y = synth(n_add) + jitter, label(n_add)
    elif method == "noise":
        extra_x, extra_y = noisy(n_add), label(n_add)
    elif method == "hybrid":
        n_syn = n_add // 2
        n_ns = n_add - n_syn
        extra_x = np.vstack([synth(n_syn), noisy(n_ns)])
        extra_y = label(n_add)
    else:
        raise ValueError(f"unknown enrich_method: {method}")
    return np.vstack([X_seed, extra_x]), np.concatenate([y_seed, extra_y])


def main():
    params = dict(DEFAULTS)
    env = os.environ.get("EXPERIMENT_PARAMS")
    if env:
        params.update(json.loads(env))
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default=None)
    args = ap.parse_args()
    if args.params:
        params.update(json.loads(args.params))

    method = str(params.get("enrich_method", "none")).lower()
    seed_ratio = float(np.clip(params.get("seed_ratio", 0.25), 0.05, 0.95))
    enrich_ratio = float(np.clip(params.get("enrich_ratio", 1.0), 0.0, 3.0))
    noise_std = float(np.clip(params.get("noise_std", 0.1), 0.0, 1.0))
    hidden_dim = max(8, int(params.get("hidden_dim", 64)))
    n_epochs = max(1, int(params.get("n_epochs", 5)))
    lr = float(params.get("learning_rate", 0.001))
    batch_size = max(1, int(params.get("batch_size", 32)))
    dropout = float(np.clip(params.get("dropout", 0.1), 0.0, 0.8))
    weight_decay = max(0.0, float(params.get("weight_decay", 0.001)))
    optimizer_name = str(params.get("optimizer", "adam")).lower()
    seed = int(params.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[GPU] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[GPU] CPU (CUDA unavailable)")
    rng = np.random.default_rng(seed)

    X, y = load_data()
    perm = rng.permutation(len(y))
    n_val = int(0.15 * len(y))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    n_seed = max(20, int(seed_ratio * len(tr_idx)))
    seed_idx = rng.permutation(len(tr_idx))[:n_seed]
    X_seed, y_seed = X[tr_idx[seed_idx]], y[tr_idx[seed_idx]]
    counts = np.bincount(y_seed, minlength=2)
    print(f"[DATA] rows={len(y)} train={len(tr_idx)} val={len(val_idx)} "
          f"seed={n_seed} (class0={counts[0]}, class1={counts[1]})")

    X_en, y_en = enrich(X_seed, y_seed, method, enrich_ratio, noise_std, rng)
    print(f"[ENRICH] method={method} enrich_ratio={enrich_ratio} noise_std={noise_std} "
          f"-> rows={len(y_en)}")

    # standardize with seed-set statistics only (no leakage from enrichment)
    mu, sd = X_seed.mean(axis=0), X_seed.std(axis=0) + 1e-8
    Xs = torch.tensor((X_en - mu) / sd, dtype=torch.float32, device=device)
    yt = torch.tensor(y_en, dtype=torch.int64, device=device)
    Xv = torch.tensor((X[val_idx] - mu) / sd, dtype=torch.float32, device=device)
    yv = torch.tensor(y[val_idx], dtype=torch.int64, device=device)

    model = nn.Sequential(
        nn.Linear(N_FEATURES, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden_dim, 2)).to(device)
    opt_cls = {"adam": optim.Adam, "adamw": optim.AdamW, "sgd": optim.SGD}.get(optimizer_name, optim.Adam)
    opt = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xs, yt), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    model.train()
    t0 = time.time()
    for epoch in range(n_epochs):
        for xb, yb in loader:
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
        if epoch == n_epochs - 1 or (epoch + 1) % max(1, n_epochs // 2) == 0:
            print(f"[TRAIN] epoch {epoch + 1}/{n_epochs} t={time.time() - t0:.1f}s")

    model.eval()
    with torch.no_grad():
        acc = float((model(Xv).argmax(dim=1) == yv).float().mean().item())
    print(f"val_acc: {acc:.6f}")


if __name__ == "__main__":
    main()