"""AG News few-shot text benchmark with a fixed deterministic split.

Protocol (seed=42; every experiment shares the EXACT same validation set, so
all val_acc values in results are directly comparable):
  - dataset: AG News (4 classes: World, Sports, Business, Sci/Tech)
  - per class: 100 lines -> validation set (never augmented, never trained on)
               k lines   -> few-shot support set D_0 (k = shots)
  - the shared tokenizer (vocab, top-20k words) is built from the full
    training split - the text analog of a pretrained tokenizer; only the
    LABELS of non-support lines are withheld, matching the few-shot setup.

Raw data is downloaded once (with fallback mirrors) and cached as
data/ag_news/ag_news_cache.npz (git-ignored).
"""
import os
import re
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 42
MAX_LEN = 64          # hard cap for tokenization (model truncates to seq_len)
VAL_PER_CLASS = 100   # fixed validation lines per class (400 total)
VOCAB_CAP = 20_000

PAD, UNK, BOS = 1, 0, 2   # reserved ids: <PAD>, <UNK>, <BOS>

# Ordered download sources: (filename, url, kind)
SOURCES = [
    ("fasttext.zip", "http://mlg.ucd.edu.cn/projects/fasttext.zip", "fasttext"),
    ("AG_News.csv", "https://www.danielpnash.com/downloads/AG_News.csv", "csv"),
    ("ag_news_train.csv",
     "https://hf-mirror.com/datasets/qgfuestc/ag_news/resolve/main/train.csv", "csv"),
]

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ag_news"
CACHE_NPZ = DATA_DIR / "ag_news_cache.npz"

_LABEL_MAP = {
    "1": 0, "2": 1, "3": 2, "4": 3,
    "world": 0, "sports": 1, "business": 2, "sci/tech": 3, "sci-tech": 3,
}


def _http_get(url, dest, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = str(dest) + ".part"
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, dest)


def _parse_fasttext_zip(path):
    """ag.train / ag.test lines: '<label>__<TAB><text>' with labels 1..4."""
    texts, ys = [], []
    with zipfile.ZipFile(path) as zf:
        name = "ag.train" if "ag.train" in zf.namelist() else None
        if name is None:
            raise ValueError("fasttext.zip does not contain ag.train")
        with zf.open(name) as f:
            for raw in f:
                line = raw.decode("utf-8", "ignore")
                m = re.match(r"^(\S+)__\t(.+)$", line)
                if not m:
                    continue
                lab = _LABEL_MAP.get(m.group(1))
                if lab is None:
                    continue
                texts.append(m.group(2).strip())
                ys.append(lab)
    return texts, ys


def _parse_csv(path):
    """Lenient CSV parser: header must contain 'label' or 'class'; text is
    the concatenation of the remaining string columns (title, description...)."""
    import csv
    texts, ys = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
        f.seek(0)
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        label_col = next(c for c in cols if c.strip().lower() in ("label", "class"))
        text_cols = [c for c in cols if c is not label_col]
        for row in reader:
            lab = _LABEL_MAP.get(str(row.get(label_col, "")).strip().lower())
            if lab is None:
                continue
            text = " ".join(str(row[c]).strip() for c in text_cols if row.get(c))
            if not text:
                continue
            texts.append(text)
            ys.append(lab)
    return texts, ys


def _build_vocab(texts, cap=VOCAB_CAP):
    counts = Counter()
    for t in texts:
        for w in t.lower().split():
            counts[w] += 1
    words = [w for w, c in counts.most_common() if c >= 2][: cap - 3]
    return ["<UNK>", "<PAD>", "<BOS>"] + words


def _tokenize(text, word2id):
    return [word2id.get(w, UNK) for w in text.lower().split()]


def _pad_batch(ids_list, length):
    out = np.full((len(ids_list), length), PAD, dtype=np.int32)
    lens = np.zeros(len(ids_list), dtype=np.int32)
    for i, ids in enumerate(ids_list):
        ids = ids[:length]
        out[i, :len(ids)] = ids
        lens[i] = len(ids)
    return out, lens

def download_and_cache(verbose=True):
    """Ensure the cached npz exists; returns its path."""
    if CACHE_NPZ.exists():
        return CACHE_NPZ
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    texts, ys, used = None, None, None
    for fname, url, kind in SOURCES:
        try:
            if verbose:
                print(f"[DATA] downloading {url} ...", flush=True)
            dest = DATA_DIR / fname
            if not dest.exists():
                _http_get(url, dest)
            if kind == "fasttext":
                texts, ys = _parse_fasttext_zip(dest)
            else:
                texts, ys = _parse_csv(dest)
            if len(texts) < 10_000:
                raise ValueError(f"parsed only {len(texts)} lines; trying next source")
            used = url
            break
        except Exception as e:
            if verbose:
                print(f"[DATA] source failed ({e!r}); trying next...", flush=True)
    if texts is None:
        raise RuntimeError(
            "could not download AG News from any source: "
            + str([u for _, u, _ in SOURCES]))
    if verbose:
        print(f"[DATA] parsed {len(texts)} lines from {used}; building vocab...",
              flush=True)
    y = np.asarray(ys, dtype=np.int32)
    c = int(y.max()) + 1
    if len(set(y.tolist())) != 4 or c != 4:
        raise RuntimeError(f"expected 4 classes, found {c}")
    vocab = _build_vocab(texts)
    word2id = {w: i for i, w in enumerate(vocab)}
    all_ids = [_tokenize(t, word2id) for t in texts]

    # Fixed split: 100 val lines per class, deterministic (seed 42)
    rng = np.random.RandomState(SEED)
    val_idx, rest = [], {}
    for cl in range(c):
        cls_idx = np.where(y == cl)[0]
        perm = rng.permutation(cls_idx)
        val_idx.append(perm[:VAL_PER_CLASS])
        rest[cl] = perm[VAL_PER_CLASS:]
    val_idx = np.concatenate(val_idx)
    rest_idx = np.concatenate([rest[cl] for cl in range(c)])
    order = rng.permutation(rest_idx)          # shuffle the (huge) train pool
    train_ids, train_len = _pad_batch([all_ids[i] for i in order], MAX_LEN)
    val_ids, val_len = _pad_batch([all_ids[i] for i in val_idx], MAX_LEN)
    yv = y[val_idx]
    vp = np.argsort(yv, kind="stable")
    val_ids, val_len, yv = val_ids[vp], val_len[vp], yv[vp]

    np.savez_compressed(
        CACHE_NPZ,
        train_ids=train_ids, train_y=y[order], train_len=train_len,
        val_ids=val_ids, val_y=yv, val_len=val_len,
        vocab=np.array(vocab, dtype=object),
    )
    if verbose:
        print(f"[DATA] cache written: {CACHE_NPZ} (train={len(train_ids)}, "
              f"val={len(val_ids)}, vocab={len(vocab)})", flush=True)
    return CACHE_NPZ


def load_cache(verbose=True):
    download_and_cache(verbose=verbose)
    return np.load(CACHE_NPZ, allow_pickle=True)


def build_fewshot(shots=5, seq_len=MAX_LEN, verbose=True):
    """Return the shared few-shot arrays (truncated to seq_len):
      support_X [C*k, L] int32, support_y [C*k], val_X [400, L], val_y [400],
      vocab_size
    The per-class support rows are chosen deterministically from the cached
    train pool (same seed as the split), so every experiment sees identical D_0.
    """
    z = load_cache(verbose=verbose)
    vocab = [str(w) for w in z["vocab"]]
    c = int(z["val_y"].max()) + 1
    rng = np.random.RandomState(SEED)
    support_ids, support_y = [], []
    for cl in range(c):
        cls_rows = np.where(z["train_y"] == cl)[0]
        take = np.sort(rng.choice(cls_rows, shots, replace=False))
        for i in take:
            support_ids.append(z["train_ids"][i])
            support_y.append(cl)
    X = np.stack(support_ids)[:, :seq_len]
    y = np.asarray(support_y, dtype=np.int32)
    return {
        "support_X": X, "support_y": y,
        "val_X": z["val_ids"][:, :seq_len], "val_y": z["val_y"],
        "vocab_size": len(vocab), "n_classes": c,
    }
