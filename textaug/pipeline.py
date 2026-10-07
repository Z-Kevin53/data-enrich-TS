"""TS-augmentation pipeline for TEXT: generate -> ensemble-score -> select -> iterate.

Text analog of tsaug.pipeline (image). Sequences are long tensors [n, L],
right-padded with PAD=1. The style prior Tr(x) is token-level:
  denoise : input with 15-25% of content tokens deleted -> decoder rewrites
            a full clean sentence (the text version of "jitter")
  swap    : input x, target = x with ~20% local adjacent token pairs swapped
            (the text version of "warp" - a locally reordered variant)
  mixed   : per-sequence random choice among clean/denoise/swap
Student loss = token-CE(x' vs Tr(x)) + CE(class head) + 0.5 * (1 - cos(f(x'), f(x))).
Candidate selection uses the SAME weighted-ensemble score as the image task:
  S(x') = [w_T * P_T(c|x') + mean_{j != i} P_j(c|x')] / (w_T + 1)
i.e. the generating student is excluded from its own scoring (self-bias).
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .models import BOS, PAD, ProtoText, StudentText, TeacherText

LAMBDA_FEAT = 0.5
TEACHER_EPOCHS = 30      # teacher initial training on D_0
FINETUNE_EPOCHS = 10     # teacher re-fine-tune after each accepted batch
STUDENT_EPOCHS = 10      # student (re)training epochs
PROTO_EPOCHS = 4         # evaluator training on the final dataset
BATCH = 16


def apply_style(x, style, rng):
    """Token-level style prior. x: [B, L] long tensor (CPU or GPU).

    Returns (input_ids, target_ids): what the student encodes and what its
    decoder is trained to produce. 'mixed' picks clean/denoise/swap per row.
    """
    B, L = x.shape
    xin = x.clone()
    xt = x.clone()
    styles = ["clean", "denoise", "swap"]
    for i in range(B):
        s = styles[int(rng.random() * 3)] if style == "mixed" else (
            style if style in styles else "clean")
        if s == "denoise":
            keep = rng.uniform(0.75, 0.85)              # delete 15-25%
            content = (x[i] != PAD).nonzero().squeeze(-1)
            if content.dim() == 0:
                content = content.unsqueeze(0)
            if len(content):
                drop = content[rng.random(len(content)) > keep]
                xin[i, drop] = PAD
        elif s == "swap":
            pos = [j for j in range(L - 1)
                   if x[i, j].item() != PAD and x[i, j + 1].item() != PAD]
            rng.shuffle(pos)
            used, n_swap = set(), max(1, L // 5)
            for j in pos:
                if len(used) >= n_swap:
                    break
                if j in used or j + 1 in used:
                    continue
                used.add(j); used.add(j + 1)
                a, b = xt[i, j].item(), xt[i, j + 1].item()
                xt[i, j], xt[i, j + 1] = b, a
    return xin, xt


def _make_loader(X, y, batch_size=BATCH):
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def train_teacher(model, X, y, lr, epochs, device, seed=42,
                  batch_size=BATCH):
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb in _make_loader(X, y, batch_size):
            loss = F.cross_entropy(model(xb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()


def train_student(student, X, y, style, lr, epochs, device, seed=42,
                  batch_size=BATCH, z_scale=1.0):
    """Student objective: teacher-forced token CE against the style target,
    class CE and feature self-distillation on an ACTUAL (greedy) generation.
    The unrolled generation runs no-grad (cheap); gradients flow through the
    encoder/cls head on the generated sequence, mirroring the image design."""
    torch.manual_seed(seed)
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    L = X.shape[1]
    V = student.tok_head.out_features
    for _ in range(epochs):
        for xb, yb in _make_loader(X, y, batch_size):
            xb, yb = xb.to(device), yb.to(device)
            b = xb.shape[0]
            z = torch.randn(b, student.latent_dim, device=device) * z_scale
            xin, tgt = apply_style(xb.cpu(), style, np.random.RandomState(
                seed + int(torch.randint(1000, (1,)).item())))
            xin, tgt = xin.to(device), tgt.to(device)
            logits = student.teacher_forced_logits(xin, z, tgt)
            tok_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, V), tgt[:, 1:].reshape(-1),
                ignore_index=PAD)
            gen = student.generate(xin, z, max_len=L)     # no-grad sampling
            fgen = student.encode(gen)
            fsrc = student.encode(xin).detach()
            cls_loss = F.cross_entropy(student.classify(gen), yb)
            feat = (1 - F.cosine_similarity(fgen, fsrc, dim=1)).mean()
            loss = tok_loss + cls_loss + LAMBDA_FEAT * feat
            opt.zero_grad(); loss.backward(); opt.step()
    student.eval()



def train_proto(model, X, y, lr, epochs, device, seed=42,
                batch_size=BATCH):
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb in _make_loader(X, y, batch_size):
            loss = F.cross_entropy(model.logits(xb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()


@torch.no_grad()
def proto_eval(model, X, y, device):
    """Prototypical classification: per-class mean features, cosine scoring."""
    f = model(X.to(device))
    classes = sorted(set(y.tolist()))
    protos = torch.stack([f[y == c].mean(0) for c in classes])
    sims = F.cosine_similarity(f.unsqueeze(0), protos.unsqueeze(1), dim=2)  # [C, B]
    pred = sims.argmax(0).cpu()                                             # [B]
    return float((pred == torch.as_tensor(y)).float().mean().item())


@torch.no_grad()
def teacher_val_acc(model, X, y, device, batch_size=64):
    model.eval()
    X, y = X.to(device), y.to(device)
    correct = 0
    for i in range(0, len(X), batch_size):
        pred = model(X[i:i + batch_size]).argmax(1)
        correct += int((pred == y[i:i + batch_size]).sum().item())
    return correct / max(1, len(X))


@torch.no_grad()
def ensemble_score(teacher, students, G, cand_cls, src_idx, wT):
    """Weighted ensemble score of candidate x' for class c (image formula):
    S = [wT * P_T(c|x') + mean_{j != i} P_j(c|x')] / (wT + 1), where i = src_idx.
    G: [K, L] long tensor; cand_cls/src_idx: int arrays of length K.
    Returns a [K] tensor on G's device.
    """
    device = G.device
    pT = torch.softmax(teacher(G.to(device)), dim=1)
    K = len(cand_cls)
    peers = torch.zeros(K, len(students), device=device)
    for j, st in enumerate(students):
        p = torch.softmax(st.classify(G.to(device)), dim=1)
        peers[:, j] = p[torch.arange(K, device=device), cand_cls]
    S = torch.empty(K, device=device)
    for k in range(K):
        j = int(src_idx[k])
        sel = [peers[k, q] for q in range(len(students)) if q != j]
        S[k] = (wT * pT[k, cand_cls[k]] + (sum(sel) / max(1, len(sel)))) / (wT + 1)
    # pTc[k] = teacher probability of the CLAIMED class, exposed for diagnostics
    pTc = pT[torch.arange(K, device=device), cand_cls]
    return S, pTc



# --------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------
def run_ts(params, data, device, seed=42):
    shots = max(2, int(params.get("shots", 5)))
    n_students = max(2, int(params.get("n_students", 2)))
    wT = max(0.5, float(params.get("teacher_weight", 3.0)))
    theta = float(np.clip(params.get("score_threshold", 0.6), 0.05, 0.95))
    iters = max(2, int(params.get("aug_iters", 3)))
    target = max(shots + 5, int(params.get("target_shots", 30)))
    M = max(2, int(params.get("candidates_per_source", 2)))
    L = max(16, min(int(params.get("seq_len", 48)), 64))
    embed = max(32, int(params.get("embed_dim", 64)))
    hidden = max(32, int(params.get("student_hidden", 64)))
    lat = max(4, int(params.get("latent_dim", 8)))
    tlr = max(1e-5, float(params.get("teacher_lr", 0.001)))
    slr = max(1e-5, float(params.get("student_lr", 0.001)))
    style = str(params.get("text_style", "mixed")).lower()
    if style not in ("clean", "denoise", "swap", "mixed"):
        style = "mixed"
    retrain_stu = int(params.get("retrain_students", 0)) > 0
    # --- new TS architecture dimensions (all clamped to safe ranges) ---
    thid = max(32, int(params.get("teacher_hidden", hidden)))   # teacher width
    n_layers = int(np.clip(params.get("n_layers", 1), 1, 2))    # BiLSTM depth
    dropout = float(np.clip(params.get("dropout", 0.0), 0.0, 0.5))
    te_epochs = max(4, int(params.get("teacher_epochs", TEACHER_EPOCHS)))
    z_scale = float(np.clip(params.get("z_scale", 1.0), 0.25, 2.0))
    batch = max(8, int(params.get("batch_size", BATCH)))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    V = int(data["vocab_size"])
    C = int(data["n_classes"])
    Xs = torch.from_numpy(np.ascontiguousarray(data["support_X"])).long()[:, :L]
    ys = torch.from_numpy(data["support_y"]).long()
    Xv = torch.from_numpy(np.ascontiguousarray(data["val_X"])).long()[:, :L]
    yv = torch.from_numpy(data["val_y"]).long()

    t0 = time.time()
    teacher = TeacherText(V, C, embed, thid, dropout=dropout,
                          n_layers=n_layers).to(device)
    train_teacher(teacher, Xs, ys, tlr, te_epochs, device, seed=seed,
                  batch_size=batch)
    print(f"[TS] teacher init done (C={C}, shots={shots}, thid={thid}, "
          f"layers={n_layers}, drop={dropout}, te_epochs={te_epochs}) "
          f"t={time.time() - t0:.1f}s", flush=True)

    students = []
    for i in range(n_students):
        s = StudentText(V, C, embed, hidden, lat, seed=seed + 101 * (i + 1),
                        n_layers=n_layers, dropout=dropout).to(device)
        train_student(s, Xs, ys, style, slr, STUDENT_EPOCHS, device,
                      seed=seed + 7 * (i + 1), batch_size=batch,
                      z_scale=z_scale)
        students.append(s)
    print(f"[TS] {n_students} students trained (style={style}, L={L}, "
          f"emb={embed}, hid={hidden}, thid={thid}, layers={n_layers}, "
          f"drop={dropout}, lat={lat}, z_sigma={z_scale}, bs={batch}) "
          f"t={time.time() - t0:.1f}s", flush=True)

    D = [Xs[ys == c].contiguous() for c in range(C)]

    def tacc():
        return teacher_val_acc(teacher, Xv, yv, device)

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
                block = D[c][j:j + 1].repeat(M, 1)
                for i, s in enumerate(students):
                    with torch.no_grad():
                        z = torch.randn(M, lat, device=device) * z_scale
                        xin, _ = apply_style(block.cpu(), style,
                                             np.random.default_rng(
                                                 seed + 1000 * t + 17 * c + 3 * j))
                        g = s.generate(xin.to(device), z, max_len=L).cpu()
                    gens.append(g)
                    cand_cls += [c] * M
                    src += [i] * M
        if not gens:
            break
        G = torch.cat(gens, 0).to(device)
        cls_t = torch.tensor(cand_cls, dtype=torch.long)
        src_t = torch.tensor(src, dtype=torch.long)
        S, pTc = ensemble_score(teacher, students, G, cls_t, src_t, wT)
        S_np = S.cpu().numpy()
        pTc_np = pTc.cpu().numpy()
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
            "accept_rate": round(len(acc_idx) / max(1, len(cand_cls)), 4),
            "mean_score_all": round(float(S_np.mean()), 4),
            "mean_score_accepted": round(s_acc, 4) if acc_idx else None,
            "min_score_accepted": round(float(S_np[acc_idx].min()), 4) if acc_idx else None,
            "teacher_prob_mean": round(float(pTc_np.mean()), 4),
            "teacher_prob_std": round(float(pTc_np.std()), 4) if len(pTc_np) > 1 else 0.0,
            "per_class_n": [int(d.shape[0]) for d in D],
            "teacher_val_acc": ta,
            "t": round(time.time() - t0, 1),
        })
        print(f"[TS] iter {t}/{iters}: n={n_total}/{target * C} cand={len(cand_cls)} "
              f"acc={len(acc_idx)} ({100.0 * len(acc_idx) / max(1, len(cand_cls)):.0f}%) "
              f"S_all={float(S_np.mean()):.3f} S_acc={s_acc:.3f} "
              f"Pt={float(pTc_np.mean()):.3f} tacc={ta:.3f} t={time.time() - t0:.1f}s",
              flush=True)
        if n_total >= target * C:
            break
        Xall = torch.cat(D, 0)
        yall = torch.cat([torch.full((d.shape[0],), c, dtype=torch.long)
                          for c, d in enumerate(D)], 0)
        train_teacher(teacher, Xall, yall, tlr, FINETUNE_EPOCHS, device,
                      seed=seed + t, batch_size=batch)
        if retrain_stu:
            for i, s in enumerate(students):
                train_student(s, Xall, yall, style, slr, 5, device,
                              seed=seed + 31 * (t + 1) * (i + 1),
                              batch_size=batch, z_scale=z_scale)

    Xf = torch.cat(D, 0)
    yf = torch.cat([torch.full((d.shape[0],), c, dtype=torch.long)
                    for c, d in enumerate(D)], 0)
    proto = ProtoText(V, C, embed, hidden, n_layers=n_layers).to(device)
    train_proto(proto, Xf, yf, max(tlr, 0.001), PROTO_EPOCHS, device,
                seed=seed, batch_size=batch)
    val_acc = proto_eval(proto, Xv, yv, device)
    base = _baseline_acc(shots, seed, device, C, V, Xs, ys, Xv, yv)
    per_class = [int(d.shape[0]) for d in D]
    print(f"[TS] final n={int(Xf.shape[0])} per_class={per_class} "
          f"val_acc={val_acc:.4f} baseline_acc={base:.4f} "
          f"t={time.time() - t0:.1f}s", flush=True)
    return {
        "val_acc": val_acc, "baseline_acc": base, "iterations": it_stats,
        "final_n": int(Xf.shape[0]), "target_n": target * C,
        "final_per_class": per_class,
        "time_seconds": round(time.time() - t0, 1),
    }


def _baseline_acc(shots, seed, device, C, V, Xs, ys, Xv, yv):
    """k-shot no-augmentation baseline (cached: depends only on shots/seed)."""
    import json, os
    cache_path = (Path(os.path.dirname(os.path.abspath(__file__))).parent
                  / "results" / "baseline_cache_text.json")
    key = f"shots={shots}|seed={seed}|L={Xs.shape[1]}"
    cache = {}
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    if key in cache:
        return float(cache[key])
    proto = ProtoText(V, C, 64, 64).to(device)
    train_proto(proto, Xs, ys, 0.001, PROTO_EPOCHS, device, seed=seed)
    acc = proto_eval(proto, Xv, yv, device)
    cache[key] = acc
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=1)
    return acc

