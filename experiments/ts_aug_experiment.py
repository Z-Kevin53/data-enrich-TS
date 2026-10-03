"""TS (Teacher-Student) ensemble augmentation experiment for few-shot CIFAR-100.

AutoResearch entry point. The full TS pipeline (teacher training, N student
generators, weighted-ensemble scoring, top-k selection, iteration to target
size, ProtoNet evaluation) lives in the tsaug package.

Data: CIFAR-100, first 20 fine classes, CIFAR-FS style protocol with a fixed
50-per-class validation set (seed 42). Downloaded/cached on first run under
data/cifar100/ (git-ignored).

Params: EXPERIMENT_PARAMS env var (orchestrator) -> --params -> defaults.
Output: the final stdout line is "val_acc: X.XXXXXX" (higher is better).
Also writes results/curves/<sha1(params)>.json with per-iteration stats for
the IEEE paper figures.
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

from tsaug.data import build_fewshot
from tsaug.pipeline import run_ts

DEFAULTS = {
    "shots": 5,                      # initial k-shot support set (per class)
    "target_shots": 50,              # target per-class dataset size
    "n_students": 3,                 # N student generators
    "teacher_weight": 3.0,           # w_T (peer weight w_S is fixed to 1)
    "score_threshold": 0.6,          # theta acceptance threshold on S in [0,1]
    "aug_iters": 4,                  # max augmentation iterations
    "candidates_per_source": 4,      # M candidates per source image per student
    "student_channels": 64,          # student base channel width
    "latent_dim": 16,                # latent code dimension z
    "teacher_lr": 0.001,
    "student_lr": 0.001,
    "student_style": "mixed",        # jitter | warp | mixed
    "retrain_students": 0,           # 1 = refresh students each iteration
    "seed": 42,
}


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GPU] {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}",
          flush=True)
    t0 = time.time()
    data = build_fewshot(shots=int(params["shots"]), seed=42)
    print(f"[DATA] seed={data['seed_X'].shape[0]} (k={int(params['shots'])}/class) "
          f"val={data['val_X'].shape[0]} t={time.time() - t0:.1f}s", flush=True)

    res = run_ts(params, data, device, seed=int(params.get("seed", 42)))

    # per-experiment curves for the paper (keyed by a hash of the params)
    cur_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results", "curves"))
    os.makedirs(cur_dir, exist_ok=True)
    h = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    with open(os.path.join(cur_dir, h + ".json"), "w") as f:
        json.dump({"params": params,
                   "val_acc": res["val_acc"], "baseline_acc": res["baseline_acc"],
                   "iterations": res["iterations"], "final_n": res["final_n"],
                   "target_n": res["target_n"], "time_seconds": res["time_seconds"]},
                  f, indent=1)

    print(f"[TS] iters={len(res['iterations'])} final_n={res['final_n']}/{res['target_n']} "
          f"baseline_acc={res['baseline_acc']:.4f} total_t={time.time() - t0:.1f}s", flush=True)
    # LAST LINE: the evaluator parses the final number on this exact line.
    print(f"val_acc: {res['val_acc']:.6f}")


if __name__ == "__main__":
    main()
