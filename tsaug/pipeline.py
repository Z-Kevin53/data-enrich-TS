"""TS-augmentation pipeline: generate -> ensemble-score -> select -> iterate.

All images are float32 tensors [n, 3, 32, 32] in [0, 1] (CPU tensors for
storage; moved to `device` for compute).

Algorithm (docs/02_experimental_design.md, equations (1)-(4)):
  1. train Teacher T on current D
  2. N Students generate candidates x' = G_i(Enc_i(x), z), z ~ N(0, I)
  3. S(x') = [wT * P_T(c|x') + wS * mean_{j!=i} P_j(c|x')] / (wT + wS);
     accept per-class TopK with S >= theta
  4. D <- D union accepted; fine-tune T on D; repeat to target size
Final: ProtoNet trained on D, evaluated on the held-out val set (val_acc).
"""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .models import ProtoNet, StudentNet, TeacherCNN

LAMBDA_FEAT = 0.5  # feature-consistency weight in the student objective


# --------------------------------------------------------------------------
# data / transforms
# --------------------------------------------------------------------------
def to_tensor(x):
    """uint8 [n,32,32,3] -> float32 cpu [n,3,32,32] in [0,1]."""
    return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float()


def _affine_warp(imgs, rng, device):
    n = imgs.shape[0]
    ang = torch.tensor([math.radians(rng.uniform(-18.0, 18.0)) for _ in range(n)], device=device)
    sc = torch.tensor([rng.uniform(0.88, 1.12) for _ in range(n)], device=device)
    tx = torch.tensor([rng.uniform(-0.18, 0.18) for _ in range(n)], device=device)
    ty = torch.tensor([rng.uniform(-0.18, 0.18) for _ in range(n)], device=device)
    cos, sin = ang.cos(), ang.sin()
    row1 = torch.stack([sc * cos, -sc * sin, tx], dim=1)  # (n, 3)
    row2 = torch.stack([sc * sin, sc * cos, ty], dim=1)   # (n, 3)
    theta = torch.stack([row1, row2], dim=1)              # (n, 2, 3)
    grid = F.affine_grid(theta, list(imgs.shape), align_corners=False)
    return F.grid_sample(imgs, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _color_jitter(imgs, rng):
    n = imgs.shape[0]
    out = imgs.clone()
    bright = torch.tensor([rng.uniform(0.75, 1.25) for _ in range(n)],
                          device=imgs.device).view(-1, 1, 1, 1)
    out = out * bright
    chan = torch.tensor(rng.uniform(0.8, 1.2, size=3), dtype=torch.float32,
                        device=imgs.device).view(1, 3, 1, 1)
    out = out * chan
    out = out + float(rng.uniform(-0.08, 0.08))
    return out.clamp(0.0, 1.0)


def transform_batch(imgs, style, rng, device):
    """Style prior Tr(x): per-image random warp / color jitter / mixed."""
    n = imgs.shape[0]
    r = rng.random(n)
    if style == "warp":
        do_warp, do_jit = np.ones(n, bool), np.zeros(n, bool)
    elif style == "jitter":
        do_warp, do_jit = np.zeros(n, bool), np.ones(n, bool)
    else:
        do_warp, do_jit = r < 0.5, r >= 0.5
    out = imgs.clone()
    if do_warp.any():
        out[do_warp] = _affine_warp(out[do_warp], rng, device)
    if do_jit.any():
        out[do_jit] = _color_jitter(out[do_jit], rng)
    return out


# --------------------------------------------------------------------------
# training loops
# --------------------------------------------------------------------------
def train_teacher(model, X, y, lr, epochs, device, batch_size=64, seed=42):
    model.train()
    xt, yt = X.to(device), y.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=True, generator=g)
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model


def train_student(student, X, y, style, lr, epochs, device, latent_dim,
                  seed=42, batch_size=16):
    """L_S = ||x' - Tr(x)||_1 + CE(cls(x'), c) + lambda * (1 - cos(f(x'), f(x)))."""
    student.train()
    xt, yt = X.to(device), y.to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-5)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=True, generator=g)
    rng = np.random.default_rng(seed + 1)
    for _ in range(epochs):
        for xb, yb in dl:
            z = torch.randn(xb.shape[0], latent_dim, device=device)
            tgt = transform_batch(xb, style, rng, device)
            gen = student.generate(xb, z)
            cls_loss = F.cross_entropy(student.classify(gen), yb)
            recon = (gen - tgt).abs().mean()
            fg = student.encode(gen)
            fs = student.encode(xb).detach()
            feat = (1 - F.cosine_similarity(fg, fs, dim=1)).mean()
            loss = recon + cls_loss + LAMBDA_FEAT * feat
            opt.zero_grad()
            loss.backward()
            opt.step()
    student.eval()
    return student


def train_proto(model, X, y, lr, epochs, device, batch_size=64, seed=42):
    model.train()
    xt, yt = X.to(device), y.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=True, generator=g)
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            lossf(model.logits(xb), yb).backward()
            opt.step()
    model.eval()
    return model


def proto_eval(model, X, y, device):
    """Prototype inference: per-class mean features, cosine similarity."""
    model.eval()
    with torch.no_grad():
        f = F.normalize(model(X.to(device)), dim=1)
        yd = y.to(device)
        protos = torch.stack([f[yd == c].mean(0) for c in range(model.n_classes)])
        protos = F.normalize(protos, dim=1)
        pred = (f @ protos.T).argmax(1)
    return float((pred == yd).float().mean().item())


# --------------------------------------------------------------------------
# ensemble scoring
# --------------------------------------------------------------------------
def ensemble_score(teacher, students, G, cand_cls, src_idx, teacher_weight):
    """S(x') = [wT*P_T(c|x') + wS*mean_{j!=src} P_j(c|x')] / (wT + wS) in [0,1].

    G: [K,3,32,32] on device; cand_cls, src_idx: long [K] on device.
    Returns a CPU float tensor [K].
    """
    device = G.device
    with torch.no_grad():
        idx = torch.arange(G.shape[0], device=device)
        p_t = F.softmax(teacher(G).float(), dim=1)[idx, cand_cls]
        n_s = len(students)
        peer_sum = torch.zeros(G.shape[0], device=device)
        for j, s in enumerate(students):
            p_j = F.softmax(s.classify(G).float(), dim=1)[idx, cand_cls]
            peer_sum = peer_sum + torch.where(src_idx != j, p_j, torch.zeros_like(p_j))
        peer = peer_sum / max(1, n_s - 1)
        wT = max(0.5, float(teacher_weight))
        return ((wT * p_t + peer) / (wT + 1.0)).cpu()


# --------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------
def run_ts(params, data, device, seed=42):
    shots = max(2, int(params.get("shots", 5)))
    n_students = max(2, int(params.get("n_students", 3)))
    wT = max(0.5, float(params.get("teacher_weight", 3.0)))
    theta = float(np.clip(params.get("score_threshold", 0.6), 0.05, 0.95))
    iters = max(2, int(params.get("aug_iters", 4)))
    target = max(shots + 5, int(params.get("target_shots", 50)))
    M = max(2, int(params.get("candidates_per_source", 4)))
    ch = max(16, int(params.get("student_channels", 64)))
    lat = max(4, int(params.get("latent_dim", 16)))
    tlr = max(1e-5, float(params.get("teacher_lr", 0.001)))
    slr = max(1e-5, float(params.get("student_lr", 0.001)))
    style = str(params.get("student_style", "mixed")).lower()
    if style not in ("jitter", "warp", "mixed"):
        style = "mixed"
    retrain_stu = int(params.get("retrain_students", 0)) > 0
    rng = np.random.default_rng(seed)

    C = len(data["classes"])
    Xs = to_tensor(data["seed_X"]) / 255.0
    ys = torch.from_numpy(data["seed_y"]).long()
    Xv = to_tensor(data["val_X"]) / 255.0
    yv = torch.from_numpy(data["val_y"]).long()

    t0 = time.time()
    teacher = TeacherCNN(C).to(device)
    train_teacher(teacher, Xs, ys, tlr, 40, device, seed=seed)
    print(f"[TS] teacher init done (C={C}, shots={shots}) t={time.time() - t0:.1f}s", flush=True)

    students = []
    for i in range(n_students):
        s = StudentNet(C, channels=ch, latent_dim=lat, seed=seed + 101 * (i + 1)).to(device)
        train_student(s, Xs, ys, style, slr, 10, device, lat, seed=seed + 7 * (i + 1))
        students.append(s)
    print(f"[TS] {n_students} students trained (style={style}, ch={ch}, lat={lat}) "
          f"t={time.time() - t0:.1f}s", flush=True)

    D = [Xs[ys == c].contiguous() for c in range(C)]

    def tacc():
        teacher.eval()
        with torch.no_grad():
            p = teacher(Xv.to(device)).argmax(1)
        return float((p == yv.to(device)).float().mean().item())

    it_stats = []
    for t in range(1, iters + 1):
        gens, cand_cls, src = [], [], []
        for c in range(C):
            cur = D[c].shape[0]
            if cur >= target:
                continue
            n_src = min(4, cur)
            idx = rng.choice(cur, size=n_src, replace=False)
            for j in idx:
                block = D[c][j:j + 1].repeat(M, 1, 1, 1)
                for i, s in enumerate(students):
                    with torch.no_grad():
                        z = torch.randn(M, lat, device=device)
                        g = s.generate(block.to(device), z).cpu()
                    gens.append(g)
                    cand_cls += [c] * M
                    src += [i] * M
        if not gens:
            break
        G = torch.cat(gens, 0).to(device)
        cls_t = torch.tensor(cand_cls, dtype=torch.long, device=device)
        src_t = torch.tensor(src, dtype=torch.long, device=device)
        S = ensemble_score(teacher, students, G, cls_t, src_t, wT)
        S_np = S.numpy()
        ta = tacc()

        order = np.argsort(-S_np)
        take_by_class = {c: [] for c in range(C)}
        for k in order:
            if S_np[k] < theta:
                break
            take_by_class[cand_cls[k]].append(int(k))
        acc_idx = []
        for c in range(C):
            need = target - D[c].shape[0]
            if need > 0:
                acc_idx += take_by_class[c][:need]
        for c in range(C):
            take = [k for k in acc_idx if cand_cls[k] == c]
            if take:
                D[c] = torch.cat([D[c], G.cpu()[take]], 0)
        n_total = int(sum(d.shape[0] for d in D))
        s_acc = float(S_np[acc_idx].mean()) if acc_idx else float("nan")
        it_stats.append({
            "iter": t, "n_total": n_total, "n_candidates": len(cand_cls),
            "n_accepted": len(acc_idx),
            "mean_score_all": round(float(S_np.mean()), 4),
            "mean_score_accepted": round(s_acc, 4) if acc_idx else None,
            "min_score_accepted": round(float(S_np[acc_idx].min()), 4) if acc_idx else None,
            "teacher_val_acc": ta,
            "t": round(time.time() - t0, 1),
        })
        print(f"[TS] iter {t}/{iters}: n={n_total}/{target * C} cand={len(cand_cls)} "
              f"acc={len(acc_idx)} S_all={float(S_np.mean()):.3f} S_acc={s_acc:.3f} "
              f"tacc={ta:.3f} t={time.time() - t0:.1f}s", flush=True)

        if n_total >= target * C:
            break
        Xall = torch.cat(D, 0)
        yall = torch.cat([torch.full((d.shape[0],), c, dtype=torch.long)
                          for c, d in enumerate(D)], 0)
        train_teacher(teacher, Xall, yall, tlr, 10, device, seed=seed + t)
        if retrain_stu:
            for i, s in enumerate(students):
                train_student(s, Xall, yall, style, slr, 5, device, lat,
                              seed=seed + 31 * (t + 1) * (i + 1))

    Xf = torch.cat(D, 0)
    yf = torch.cat([torch.full((d.shape[0],), c, dtype=torch.long) for c, d in enumerate(D)], 0)
    proto = ProtoNet(C).to(device)
    train_proto(proto, Xf, yf, max(tlr, 0.001), 4, device, seed=seed)
    val_acc = proto_eval(proto, Xv, yv, device)
    base = _baseline_acc(shots, seed, device, C, Xs, ys, Xv, yv)
    print(f"[TS] final n={int(Xf.shape[0])} val_acc={val_acc:.4f} "
          f"baseline_acc={base:.4f} t={time.time() - t0:.1f}s", flush=True)
    return {
        "val_acc": val_acc, "baseline_acc": base, "iterations": it_stats,
        "final_n": int(Xf.shape[0]), "target_n": target * C,
        "time_seconds": round(time.time() - t0, 1),
    }


def _baseline_acc(shots, seed, device, C, Xs, ys, Xv, yv):
    """k-shot no-augmentation baseline (cached: depends only on shots/seed)."""
    cache_path = (Path(os.path.dirname(os.path.abspath(__file__))).parent
                  / "results" / "baseline_cache.json")
    key = f"shots={shots}|seed={seed}"
    cache = {}
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    if key in cache:
        return float(cache[key])
    proto = ProtoNet(C).to(device)
    train_proto(proto, Xs, ys, 0.001, 4, device, seed=seed)
    acc = proto_eval(proto, Xv, yv, device)
    cache[key] = acc
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=1)
    return acc
