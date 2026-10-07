"""TS (Teacher-Student) ensemble augmentation experiment for few-shot AG News.

AutoResearch entry point for the TEXT task (round 2 of the research).
The full TS pipeline (teacher training, N latent-conditioned sequence
generators, weighted-ensemble scoring, top-k selection, iteration to target
size, prototypical-network evaluation) lives in the textaug package.

Data: AG News (4 classes), k-shot protocol with a fixed 100-per-class
validation set (seed 42). Downloaded/cached on first run under
data/ag_news/ (git-ignored).

Params: EXPERIMENT_PARAMS env var (orchestrator) -> --params -> defaults.
Output: the final stdout line is "val_acc: X.XXXXXX" (higher is better).
Also writes results/curves_text/<sha1(params)>.json with per-iteration stats.
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
import hashlib
import json
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from textaug.data import build_fewshot
from textaug.pipeline import run_ts

DEFAULTS = {
    "shots": 5,                      # initial k-shot support set (per class)
    "target_shots": 30,              # target per-class dataset size
    "n_students": 2,                 # N student generators
    "teacher_weight": 3.0,           # w_T (peer weight w_S is fixed to 1)
    "score_threshold": 0.6,          # theta acceptance threshold on S in [0,1]
    "aug_iters": 3,                  # max augmentation iterations
    "candidates_per_source": 2,      # M candidates per source sentence per student
    "seq_len": 48,                   # max token length
    "embed_dim": 64,                 # token embedding dimension
    "student_hidden": 64,            # BiLSTM hidden width of the students
    "teacher_hidden": 64,            # BiLSTM hidden width of the teacher
                                     # (decoupled: scorer may be wider than
                                     #  the generators; default = student's)
    "n_layers": 1,                   # BiLSTM depth (teacher & student enc)
    "dropout": 0.0,                  # encoder/inter-layer dropout
    "teacher_epochs": 30,            # teacher initial training epochs on D_0
    "z_scale": 1.0,                  # latent prior scale, z ~ N(0, s^2 I)
    "batch_size": 16,                # training batch size
    "latent_dim": 8,                 # latent code dimension z
    "teacher_lr": 0.001,
    "student_lr": 0.001,
    "text_style": "mixed",           # clean | denoise | swap | mixed
    "retrain_students": 0,           # 1 = refresh students each iteration
    "seed": 42,
}


def _jsonable(obj):
    """Recursively convert numpy/torch scalars so json.dump never crashes."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, (int, float)):
        return obj
    # numpy / torch scalars
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    try:
        return float(obj)
    except Exception:
        return str(obj)


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

    print(f"[PARAMS] {json.dumps(params, sort_keys=True)}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GPU] {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}",
          flush=True)
    t0 = time.time()
    data = build_fewshot(shots=int(params["shots"]),
                         seq_len=int(params.get("seq_len", 64)))
    print(f"[DATA] seed={data['support_X'].shape[0]} (k={int(params['shots'])}/class) "
          f"val={data['val_X'].shape[0]} vocab={data['vocab_size']} "
          f"t={time.time() - t0:.1f}s", flush=True)

    res = run_ts(params, data, device, seed=int(params.get("seed", 42)))

    # per-experiment curves for the paper (keyed by a hash of the params).
    # A curve failure must NEVER kill the experiment: the metric line below
    # is what the evaluator parses, so guard this whole block.
    try:
        cur_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "results", "curves_text"))
        os.makedirs(cur_dir, exist_ok=True)
        h = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
        with open(os.path.join(cur_dir, h + ".json"), "w") as f:
            json.dump(_jsonable({"params": params,
                                 "val_acc": res["val_acc"],
                                 "baseline_acc": res["baseline_acc"],
                                 "iterations": res["iterations"],
                                 "final_n": res["final_n"],
                                 "target_n": res["target_n"],
                                 "time_seconds": res["time_seconds"]}),
                      f, indent=1)
    except Exception as e:
        print(f"[WARN] curve write failed (metric unaffected): {e!r}", flush=True)

    print(f"[TS] iters={len(res['iterations'])} final_n={res['final_n']}/{res['target_n']} "
          f"baseline_acc={res['baseline_acc']:.4f} total_t={time.time() - t0:.1f}s", flush=True)
    # LAST LINE: the evaluator parses the final number on this exact line.
    print(f"val_acc: {res['val_acc']:.6f}")


if __name__ == "__main__":
    main()
