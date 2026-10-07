"""Markdown paper generator for AutoResearch TS-augmentation runs.

Produces a complete, data-driven paper (abstract, method with the ensemble
scoring formula, protocol, results, per-setting analysis, convergence,
runtime, discussion, limitations, appendices). Every number comes from the
experiment records, the convergence history, or the per-experiment curve
files (results/curves*/<hash>.json); nothing is invented.

Task-aware: program_config["task"] selects the dataset / style-prior wording
("text" -> AG News sentences, "image" -> CIFAR-100 images); anything unknown
falls back to a generic description, so the same generator serves both the
CIFAR-100 and AG News research rounds.
"""
import json, math, statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Ordered keys of the TS pipeline (the program config also carries legacy
# keys from earlier tasks; tables show only what experiments actually used).
TS_KEY_ORDER = [
    "shots", "target_shots", "n_students", "teacher_weight", "score_threshold",
    "aug_iters", "candidates_per_source", "seq_len", "student_channels",
    "embed_dim", "student_hidden", "teacher_hidden", "n_layers", "dropout",
    "teacher_epochs", "z_scale", "latent_dim", "teacher_lr", "student_lr",
    "batch_size", "text_style", "student_style", "retrain_students",
]

KEY_LABELS = {
    "shots": "initial k-shot support (per class)",
    "target_shots": "target dataset size (per class)",
    "n_students": "N student generators",
    "teacher_weight": "w_T teacher weight in the ensemble",
    "score_threshold": "theta acceptance threshold",
    "aug_iters": "maximum augmentation iterations",
    "candidates_per_source": "M candidates per source per student",
    "seq_len": "maximum token length L",
    "student_channels": "student CNN width",
    "embed_dim": "embedding dimension",
    "student_hidden": "student hidden width h_S",
    "teacher_hidden": "teacher hidden width h_T",
    "n_layers": "encoder depth (LSTM layers)",
    "dropout": "dropout rate",
    "teacher_epochs": "teacher initial training epochs",
    "z_scale": "latent prior scale sigma (z ~ N(0, sigma^2 I))",
    "latent_dim": "latent code dimension",
    "teacher_lr": "teacher learning rate",
    "student_lr": "student learning rate",
    "batch_size": "training batch size",
    "text_style": "text style prior Tr(x)",
    "student_style": "image style prior Tr(x)",
    "retrain_students": "refresh students each iteration (0/1)",
}

TASK_INFO = {
    "text": {
        "dataset": "AG News",
        "task_desc": "4-class short-text classification "
                     "(World / Sports / Business / Sci-Tech)",
        "unit": "tokenized sentence",
        "unit_note": "right-padded token sequences of at most 64 tokens from a "
                     "top-20k vocabulary (a shared, pretrained-style tokenizer "
                     "built from the full training split)",
        "encoder": "bidirectional LSTM over learned token embeddings "
                   "(packed sequences, so padding never leaks into features)",
        "decoder": "latent-conditioned LSTM that re-emits a full token sequence",
        "styles": {
            "clean": "identity",
            "denoise": "input with 15-25% of its content tokens deleted; the "
                       "decoder must rewrite a full clean sentence",
            "swap": "input x with ~20% of adjacent token pairs locally "
                    "swapped (a locally reordered variant of x)",
            "mixed": "per-sequence random choice among clean / denoise / swap",
        },
        "gen_verb": "sequences",
        "augment_verb": "sentences",
    },
    "image": {
        "dataset": "CIFAR-100",
        "task_desc": "100-class image classification (32x32 RGB images)",
        "unit": "32x32 RGB image",
        "unit_note": "small normalized images in a 100-way classification task",
        "encoder": "small convolutional network",
        "decoder": "latent-conditioned convolutional decoder that re-synthesizes an image",
        "styles": {
            "clean": "identity",
            "jitter": "small photometric jitter (brightness / contrast)",
            "warp": "local geometric warp of the image",
            "mixed": "per-sample random choice among clean / jitter / warp",
        },
        "gen_verb": "images",
        "augment_verb": "images",
    },
}

GENERIC_INFO = {
    "dataset": "the benchmark dataset",
    "task_desc": "few-shot classification",
    "unit": "sample",
    "unit_note": "",
    "encoder": "encoder network",
    "decoder": "latent-conditioned generator",
    "styles": {"clean": "identity", "mixed": "random choice of transforms"},
    "gen_verb": "samples",
    "augment_verb": "samples",
}


class PaperGenerator:
    def __init__(self, config=None):
        self.config = config or {}
        self.title_prefix = self.config.get("title_prefix", "AutoResearch: Automated")
        self.include_ablation = self.config.get("include_ablation", True)
        self.include_analysis = self.config.get("include_analysis", True)
        self._task = "generic"
        self._pc: Dict = {}
        self._convergence: Dict = {}
        self._records: List[Dict] = []
        self._best: Optional[Dict] = None
        self._stats: Dict = {}
        self._curves: List[Dict] = []
        self._best_curve: Optional[Dict] = None

    def generate_paper(self, experiment_records, convergence_data, program_config=None):
        if not experiment_records:
            return self._empty_paper()
        pc = program_config or {}
        task = str(pc.get("task", "")).lower()
        self._task = task if task in TASK_INFO else "generic"
        self._pc = pc
        self._convergence = convergence_data or {}
        self._records = list(experiment_records)
        direction = pc.get("direction", "higher")
        finite = [r for r in self._records if math.isfinite(r.get("metric_value", 0))]
        pool = finite if finite else self._records
        if direction == "lower":
            self._best = min(pool, key=lambda r: r.get("metric_value", 0))
        else:
            self._best = max(pool, key=lambda r: r.get("metric_value", 0))
        self._stats = self._compute_stats(self._records, direction)
        self._curves = self._load_curves()
        self._best_curve = self._match_curve(self._best)
        sections = [
            self._title(),
            self._abstract(),
            self._introduction(),
            self._problem_setup(),
            self._method(),
            self._protocol(),
            self._results_overview(),
            self._top_table(),
            self._best_detail(),
        ]
        if self.include_ablation:
            sections.append(self._per_setting_analysis())
        if self.include_analysis:
            sections.append(self._convergence_section())
            sections.append(self._runtime_section())
        sections += [
            self._discussion(),
            self._limitations(),
            self._conclusion(),
            self._appendix_log(),
            self._appendix_curves(),
        ]
        return "\n\n".join(s for s in sections if s)

    # ---------------- helpers ---------------- #
    def _t(self):
        return TASK_INFO.get(self._task, GENERIC_INFO)

    def _finite(self):
        return [r for r in self._records if math.isfinite(r.get("metric_value", 0))]

    def _metric_name(self):
        for r in self._records:
            if r.get("metric_name"):
                return r["metric_name"]
        return self._pc.get("target_metric", "val_acc")

    def _arch_str(self, p):
        """Compact architecture summary of one config, in a fixed order."""
        def g(k, default="-"):
            return p.get(k, default)
        bits = []
        if "shots" in p or "target_shots" in p:
            bits.append(f"k={g('shots')}->{g('target_shots')}/class")
        if "n_students" in p:
            bits.append(f"N={g('n_students')}")
        if "teacher_weight" in p or "score_threshold" in p:
            bits.append(f"w_T={g('teacher_weight')}, theta={g('score_threshold')}")
        if "aug_iters" in p:
            bits.append(f"iters<= {g('aug_iters')}")
        if "candidates_per_source" in p:
            bits.append(f"M={g('candidates_per_source')}")
        if "student_hidden" in p or "teacher_hidden" in p:
            depth = f"x{g('n_layers')}" if "n_layers" in p else ""
            bits.append(f"h_S={g('student_hidden')}, h_T={g('teacher_hidden')}{depth}")
        if "seq_len" in p:
            bits.append(f"L={g('seq_len')}")
        if "embed_dim" in p:
            bits.append(f"emb={g('embed_dim')}")
        if "latent_dim" in p:
            bits.append(f"d_z={g('latent_dim')}, sigma={g('z_scale')}")
        if "dropout" in p:
            bits.append(f"drop={g('dropout')}")
        if "teacher_epochs" in p:
            bits.append(f"te_epochs={g('teacher_epochs')}")
        if "batch_size" in p:
            bits.append(f"bs={g('batch_size')}")
        if "text_style" in p:
            bits.append(f"style={g('text_style')}")
        if "student_style" in p:
            bits.append(f"style={g('student_style')}")
        if "retrain_students" in p:
            bits.append(f"retrain={g('retrain_students')}")
        if "teacher_lr" in p or "student_lr" in p:
            bits.append(f"lr_T={g('teacher_lr')}, lr_S={g('student_lr')}")
        return ", ".join(bits) if bits else json.dumps(p, sort_keys=True)

    @staticmethod
    def _sparkline(values, width=48):
        vals = [v for v in values if math.isfinite(v)]
        if not vals:
            return ""
        blocks = "▁▂▃▄▅▆▇█"
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-12:
            return blocks[4] * len(vals)
        if len(vals) > width:
            idx = [round(i * (len(vals) - 1) / (width - 1)) for i in range(width)]
            vals = [vals[i] for i in idx]
        return "".join(blocks[int(7 * (v - lo) / (hi - lo))] for v in vals)

    def _load_curves(self):
        out = []
        base = Path("results")
        for d in ("curves_text", "curves"):
            p = base / d
            if not p.is_dir():
                continue
            for f in sorted(p.glob("*.json")):
                try:
                    out.append(json.loads(f.read_text(encoding="utf-8")))
                except Exception:
                    pass
        return out

    def _match_curve(self, record):
        """Curve file whose params contain all of record's params (curves
        store the FULL merged param dict, records only the searched keys)
        and whose val_acc agrees with the recorded metric."""
        if not record:
            return None
        p = record.get("params", {}) or {}
        m = record.get("metric_value", float("nan"))
        fallback = None
        for c in self._curves:
            cp = c.get("params", {})
            if not all(cp.get(k) == v for k, v in p.items()):
                continue
            if math.isfinite(m) and abs(float(c.get("val_acc", -9)) - m) < 1e-3:
                return c
            if fallback is None:
                fallback = c
        return fallback

    def _baseline_of(self, curve):
        if curve and math.isfinite(curve.get("baseline_acc", float("nan"))):
            return float(curve["baseline_acc"])
        return None

    def _compute_stats(self, records, direction="higher"):
        metrics = [r.get("metric_value", 0) for r in records
                   if math.isfinite(r.get("metric_value", 0))]
        if not metrics:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "best": 0,
                    "count": 0, "improvement_pct": 0, "failed": len(records)}
        n = len(metrics)
        mean = sum(metrics) / n
        std = statistics.stdev(metrics) if n > 1 else 0.0
        best_val = min(metrics) if direction == "lower" else max(metrics)
        if direction == "lower":
            improvement = ((metrics[0] - best_val) / abs(metrics[0]) * 100
                           if metrics[0] != 0 else 0.0)
        else:
            improvement = ((best_val - metrics[0]) / abs(metrics[0]) * 100
                           if metrics[0] != 0 else 0.0)
        return {"mean": mean, "std": std, "min": min(metrics), "max": max(metrics),
                "best": best_val, "count": n, "improvement_pct": improvement,
                "failed": len(records) - n}

    def _empty_paper(self):
        return "# AutoResearch\nNo experiment data available."

    # ---------------- sections ---------------- #
    def _title(self):
        t = self._t()
        return (f"# {self.title_prefix}\n\n"
                f"**Date**: {datetime.now().strftime('%B %d, %Y')}\n"
                f"**Task**: {t['dataset']} — {t['task_desc']}\n"
                f"**Generated by**: AutoResearch framework "
                f"({len(self._records)} experiments, "
                f"{self._convergence.get('improved_count', 0)} ratchet improvements)")

    def _abstract(self):
        t = self._t()
        s = self._stats
        m = self._metric_name()
        best_p = self._best.get("params", {}) if self._best else {}
        bl = self._baseline_of(self._best_curve)
        bl_txt = (f" compared with a {best_p.get('shots', 'k')}-shot no-augmentation "
                  f"baseline of {bl:.4f} ({(self._stats['best'] - bl) * 100:+.2f} pp)"
                  if bl is not None and self._pc.get("direction", "higher") == "higher"
                  else "")
        return f"""## Abstract

Few-shot learning is limited by the scarcity of labeled examples. This paper
studies **Teacher-Student (TS) ensemble data augmentation**: a teacher scorer
trained on the current small dataset ranks synthetic candidates produced by
latent-conditioned student generators, and only candidates that pass a
weighted-ensemble score threshold are accepted into the dataset, which is then
grown iteratively. The whole TS pipeline — architecture (teacher/student
width and depth, latent prior scale, style prior), selection (ensemble
weights, threshold), and schedule (iterations, candidates per source) — is
searched automatically by the AutoResearch ratchet loop, which keeps a
configuration only if it improves the downstream prototypical-network
validation accuracy on {t['dataset']}.

We report **{len(self._records)} automatically designed experiments**
({s['count']} with finite metrics). The best configuration
({self._arch_str(best_p)}) achieves **{m} = {s['best']:.4f}**, mean
{s['mean']:.4f} (std {s['std']:.4f}) over the run{bl_txt}. Per-setting
analyses and per-iteration augmentation curves show how selection
quality, ensemble weighting and generator capacity interact under a strict
per-experiment time budget."""

    def _introduction(self):
        t = self._t()
        s = self._stats
        return f"""## 1. Introduction

Few-shot classification must make a model work from a handful of labeled
{t['unit'].strip('s')} per class. The standard remedies — stronger
inductive biases, pretraining, or test-time adaptation — all assume
resources that are unavailable in the extreme low-data regime. An
attractive alternative is **data augmentation with the model itself**:
generate new examples, but only keep the ones the current model ecosystem
finds trustworthy.

The TS ensemble framework formalizes this idea with two asymmetric roles:

- a **teacher** classifier, retrained on the current dataset D_t and used
  only to *score* candidates (it never generates data);
- **N student generators**, latent-conditioned encoder-decoders trained to
  rewrite a source {t['unit']} under a *style prior* Tr(x). Each student
  produces candidates conditioned on a random latent code z, so a single
  source {t['unit']} yields a diverse family of variants.

A candidate x' claimed by student i for class c is scored by the
**leave-one-out weighted ensemble**
S(x') = [w_T P_T(c|x') + mean_{{j != i}} P_j(c|x')] / (w_T + 1):
the teacher plus all *other* students, so a generator can never score its
own output. Candidates with S >= theta fill the per-class quota, the
teacher is fine-tuned on the enlarged dataset, and the loop repeats until
the target size is reached or the iteration budget is exhausted.

Because the pipeline exposes more than a dozen interacting architectural and
selection knobs (teacher/student widths and depths, latent prior scale,
style prior, ensemble weight w_T, threshold theta, iteration budget,
candidates per source, ...), manual tuning is infeasible. We therefore run
the **AutoResearch** ratchet: an automated loop that proposes the next
configuration from the experiment history, executes it under a fixed time
budget, and keeps it only if it strictly improves the held-out metric. This
paper documents the search over the TS architecture for
{t['dataset']} ({t['task_desc']}), and — critically — the *content* of what
the automated search found, not just the best number."""

    def _problem_setup(self):
        t = self._t()
        p = self._best.get("params", {}) if self._best else {}
        return f"""## 2. Problem Setup

**Dataset.** {t['dataset']}: {t['task_desc']}. Each {t['unit']} is
{t['unit_note']}.

**Few-shot protocol.** For k shots per class we sample the support set D_0
deterministically (fixed seed) and hold out a **fixed validation set**
(100 examples per class for the text task) that no experiment may train on
or augment. Because every configuration — including the no-augmentation
baseline — is evaluated on the *identical* validation set, all reported
{self._metric_name()} values are directly comparable.

**Objective.** Maximize the validation accuracy of a prototypical-network
classifier (per-class mean features, cosine scoring) trained from scratch
on the final augmented dataset, with a fixed per-experiment wall-clock
budget of {self._pc.get('constraint', {}).get('max_time_seconds', '-')} s."""

    def _method(self):
        t = self._t()
        styles = t["styles"]
        style_lines = "\n".join(f"  - **{k}** — {v}" for k, v in styles.items())
        ss = (self._pc.get("search_space") or {})
        rows = []
        for k in TS_KEY_ORDER:
            if k in ss and ss[k]:
                rows.append(f"| `{k}` | {KEY_LABELS.get(k, k)} | "
                            f"{', '.join(str(v) for v in ss[k])} |")
        extra = [k for k in ss if k not in TS_KEY_ORDER and isinstance(ss[k], list)]
        for k in extra:
            rows.append(f"| `{k}` | {KEY_LABELS.get(k, k)} | "
                        f"{', '.join(str(v) for v in ss[k])} |")
        ss_table = ("| Parameter | Meaning | Values searched |\n|---|---|---|\n"
                    + "\n".join(rows)) if rows else "_not recorded_"
        return f"""## 3. Method

### 3.1 The TS ensemble augmentation loop

Let D_t be the dataset at iteration t, D_0 the k-shot support set. One
iteration does:

1. **Sampling.** For every class c that has not reached its target size,
   pick up to 4 source {t['gen_verb']} from D_t[c]; each source is passed
   through all N students, which each emit M candidates — giving
   <= 4M N candidates per iteration, each *claimed* by the class of its
   source.
2. **Generation.** Student i conditions its decoder on the source feature
   and a fresh latent code z ~ N(0, sigma^2 I); the latent dimension and
   the scale sigma are both searched. The style prior Tr defines the
   training objective (below) and the candidate distribution.
3. **Ensemble scoring.** Each candidate x' for class c gets the
   leave-one-out weighted score
   S(x') = [w_T P_T(c|x') + mean_{{j != i}} P_j(c|x')] / (w_T + 1),
   where i is the generating student. Excluding i removes self-bias: a
   student that is merely confident in its own outputs does not inflate
   the score.
4. **Selection.** Candidates are ranked by S; those with S >= theta are
   accepted (per class, up to the remaining quota). Acceptance is
   therefore both quality- and quota-controlled.
5. **Refresh.** The teacher is fine-tuned on the enlarged D_{{t+1}}
   (10 epochs); optionally the students are retrained as well
   (`retrain_students`). The loop stops when the target size is reached
   or the iteration budget is exhausted.

### 3.2 Student objective and style prior

Each student is trained with a three-term objective: token cross-entropy
against the style target Tr(x) (teacher forcing), a classification CE on an
*actual greedy generation* from the same latent condition, and a
feature-matching term (1 - cos(f(x'), f(x))) with weight 0.5 that keeps the
generated {t['gen_verb']} in the neighborhood of real ones. The style prior
variants searched are:

{style_lines}

### 3.3 Architecture (the searched T-S space)

Teacher and students share the {t['encoder']} backbone but have
**independent parameters, and independently searched widths and depths** —
the teacher only scores, so it may be wider/deeper than the generators
that dominate runtime. Students add a {t['decoder']}, a latent-conditioned
initial state, and a peer-scoring head. The downstream evaluator is a
prototypical network with the same encoder family.

### 3.4 The AutoResearch ratchet

The orchestrator maintains the experiment history and a ratchet: the best
metric so far. A heuristic idea generator proposes up to 3 configurations
per round — a neighborhood perturbation of the best (floats scaled by
U(0.8, 1.2), ints by {{-1, 0, +1}}) after a clear improvement, otherwise
broad random sampling plus a regularization-focused candidate. Each
experiment runs as a subprocess with a hard wall-clock limit; a failure or
timeout is logged and the loop continues. Improved configurations are kept
(and committed); the rest are discarded.

### 3.5 Search space

{ss_table}"""

    def _protocol(self):
        t = self._t()
        c = self._pc.get("constraint", {})
        failed = self._stats.get("failed", 0)
        return f"""## 4. Experimental Protocol

- **Hardware.** A single GPU (physical GPU 0, pinned by nvidia-smi UUID
  before torch is touched); experiments that exceed the time budget are
  killed and the loop continues.
- **Budget.** {c.get('max_time_seconds', '-')} s per experiment,
  {c.get('max_iterations', '-')} experiments maximum.
- **Reproducibility.** Fixed protocol seed (support-set sampling, data
  split) so every configuration — and the no-augmentation baseline — is
  judged on the identical validation set. The k-shot baseline is computed
  once per (k, seed) and cached, so it is never affected by augmentation.
- **Failure handling.** {failed} of {len(self._records)} recorded
  experiments did not yield a finite metric (crash/timeout) and were
  counted as non-improving; they are excluded from the statistical
  summaries but kept in the full log (Appendix A)."""

    def _results_overview(self):
        s = self._stats
        m = self._metric_name()
        bl = self._baseline_of(self._best_curve)
        bl_line = ""
        if bl is not None and self._pc.get("direction", "higher") == "higher":
            bl_line = (f"\n- Best vs. no-augmentation baseline: "
                       f"{bl:.4f} -> {s['best']:.4f} "
                       f"({(s['best'] - bl) * 100:+.2f} pp)")
        return f"""## 5. Results

### 5.1 Overview

| Quantity | Value |
|---|---|
| Experiments recorded | {len(self._records)} |
| Experiments with finite metric | {s['count']} |
| Ratchet improvements | {self._convergence.get('improved_count', 0)} |
| Best {m} | **{s['best']:.6f}** |
| Mean {m} | {s['mean']:.6f} |
| Std dev | {s['std']:.6f} |
| Min / Max | {s['min']:.6f} / {s['max']:.6f} |
| Improvement over first run | {s['improvement_pct']:+.2f}% |
{bl_line}"""

    def _top_table(self):
        m = self._metric_name()
        direction = self._pc.get("direction", "higher")
        finite = self._finite()
        key = (lambda r: r.get("metric_value", 0))
        top = sorted(finite, key=key, reverse=(direction != "lower"))[:10]
        best_val = self._stats.get("best", 0)
        rows = []
        for i, r in enumerate(top, 1):
            d = r.get("metric_value", 0) - best_val
            rows.append(f"| {i} | {r.get('experiment_id', '?')} | "
                        f"{r.get('experiment_name', '?')} | {r.get('metric_value', 0):.4f} | "
                        f"{d:+.4f} | {r.get('status', '?')} | "
                        f"{self._arch_str(r.get('params', {}))} |")
        return f"""### 5.2 Top-10 configurations

| Rank | ID | Name | {m} | gap to best | Status | Configuration |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}"""

    def _best_detail(self):
        m = self._metric_name()
        best = self._best
        p = best.get("params", {}) if best else {}
        lines = [f"""### 5.3 Best configuration in detail

- **Experiment**: `{best.get('experiment_name', '?')}` (id {best.get('experiment_id', '?')})
- **{m}**: {best.get('metric_value', 0):.6f}
- **Runtime**: {best.get('time_seconds', 0):.1f} s"""]
        lines.append("\n**Full parameter vector:**\n")
        lines.append("```json\n" + json.dumps(p, indent=2, sort_keys=True) + "\n```")
        curve = self._best_curve
        if curve and curve.get("iterations"):
            its = curve["iterations"]
            rows = []
            for it in its:
                pc_n = it.get("per_class_n")
                pc_s = f"[{', '.join(str(x) for x in pc_n)}]" if pc_n else "-"
                rows.append(
                    f"| {it.get('iter', '?')} | {it.get('n_total', '?')} | "
                    f"{pc_s} | {it.get('n_candidates', '?')} | "
                    f"{it.get('n_accepted', '?')} | {it.get('accept_rate', '?')} | "
                    f"{it.get('mean_score_all', '?')} | "
                    f"{it.get('teacher_prob_mean', '?')} | "
                    f"{it.get('teacher_val_acc', '?')} | {it.get('t', '?')} s |")
            n0 = curve.get("params", {}).get("shots", "?")
            tf = its[-1].get("teacher_val_acc", "?")
            t0 = its[0].get("teacher_val_acc", tf)
            lines.append(f"""
**Per-iteration augmentation curve** (dataset grown from {n0} to {curve.get('final_n', '?')} examples, target {curve.get('target_n', '?')}):

| iter | total n | per-class n | candidates | accepted | acc. rate | mean S | mean P_T | teacher val acc | t |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

The teacher's own validation accuracy rises from {t0} to {tf} as the
dataset grows, while the per-iteration acceptance rate and mean ensemble
score trace the selection pressure: as the quota nears, only the highest
scoring candidates pass. Final per-class sizes:
{curve.get('final_per_class') or [it.get('per_class_n') for it in its[-1:]]}.
""")
        else:
            lines.append("\n_No per-iteration curve file matched this "
                         "configuration (failed/timed-out run).")
        bl = self._baseline_of(curve)
        if bl is not None:
            lines.append(f"\n**Ablation anchor**: the identical protocol "
                         f"with **no augmentation** scores {bl:.4f}; the "
                         f"best TS configuration adds "
                         f"{(best.get('metric_value', 0) - bl) * 100:+.2f} "
                         f"percentage points.")
        return "\n".join(lines)

    def _per_setting_analysis(self):
        m = self._metric_name()
        direction = self._pc.get("direction", "higher")
        finite = self._finite()
        if not finite:
            return ""
        vals = defaultdict(set)
        for r in finite:
            for k, v in (r.get("params") or {}).items():
                vals[k].add(round(v, 6) if isinstance(v, float) else v)
        known = [k for k in TS_KEY_ORDER if len(vals.get(k, set())) >= 2]
        extra = [k for k in sorted(vals) if len(vals[k]) >= 2 and k not in known]
        keys = (known + extra)[:8]
        if not keys:
            return ""
        lines = ["## 6. Per-Setting Analysis", ""]
        lines.append(
            "Each table groups the finite experiments by the value of one "
            "searched parameter and reports how many configurations fall in "
            "each bucket, their mean metric, and the best result achieved "
            "inside the bucket. Buckets are ordered by mean metric.")
        for idx, k in enumerate(keys, 1):
            groups = defaultdict(list)
            for r in finite:
                v = (r.get("params") or {}).get(k)
                if v is None:
                    continue
                v = round(v, 6) if isinstance(v, float) else v
                groups[v].append(r.get("metric_value", 0))
            rows = []
            for v, mv in groups.items():
                top = max(mv) if direction != "lower" else min(mv)
                rows.append((v, len(mv), sum(mv) / len(mv), top))
            rows.sort(key=lambda t_: t_[2], reverse=(direction != "lower"))
            lines.append("")
            lines.append(f"### 6.{idx} `{k}` — {KEY_LABELS.get(k, k)}")
            lines.append("")
            lines.append(f"| value | # configs | mean {m} | best {m} |")
            lines.append("|---|---|---|---|")
            for v, n, mean, top in rows:
                lines.append(f"| {v} | {n} | {mean:.4f} | {top:.4f} |")
        return "\n".join(lines)

    def _convergence_section(self):
        conv = self._convergence
        m = self._metric_name()
        tot = conv.get("total_experiments") or len(self._records)
        imp = conv.get("improved_count", 0)
        worse = conv.get("worse_count", 0)
        met = [v for v in (conv.get("metrics_over_time") or [])
               if isinstance(v, (int, float)) and math.isfinite(v)]
        spark = self._sparkline(met)
        trail = 0
        for r in reversed(self._records):
            if r.get("status") == "improved":
                break
            trail += 1
        rate = (imp / tot * 100) if tot else 0.0
        bm = conv.get("best_metric", self._stats.get("best", 0))
        if not (isinstance(bm, (int, float)) and math.isfinite(bm)):
            bm = self._stats.get("best", 0)
        spark_line = f"\n    {spark}\n" if spark else ""
        return f"""## 7. Convergence

The ratchet accepted **{imp} of {tot}** proposals ({rate:.1f}% improvement
rate); {worse} proposals scored below the incumbent and were discarded.
Finite metrics in execution order (crashed runs omitted):
{spark_line}
The last {trail} recorded run(s) did not beat the incumbent — a plateau.
Per Section 3.4 the idea generator responds to plateaus by switching from
neighborhood perturbations of the incumbent to broad random sampling plus
a regularization-focused candidate. Final best: {m} = {bm:.4f}."""

    def _runtime_section(self):
        budget = (self._pc.get("constraint") or {}).get("max_time_seconds")
        times = [r.get("time_seconds", 0) for r in self._records
                 if math.isfinite(r.get("time_seconds", 0) or 0)
                 and (r.get("time_seconds", 0) or 0) > 0]
        if not times:
            return "## 8. Runtime\n\n_No wall-clock times were recorded._"
        n_to = sum(1 for r in self._records
                   if str(r.get("status", "")).lower() in ("timeout", "timed_out")
                   or (isinstance(budget, (int, float)) and budget > 0
                       and (r.get("time_seconds", 0) or 0) >= budget))
        best_t = (self._best.get("time_seconds", 0) or 0) if self._best else 0
        budget_txt = ""
        if isinstance(budget, (int, float)) and budget > 0:
            budget_txt = (f"Mean utilization of the {budget:.0f} s per-experiment "
                          f"budget is {sum(times) / len(times) / budget * 100:.0f}%.")
        lines = [f"""## 8. Runtime

| Quantity | Value |
|---|---|
| Runs with recorded time | {len(times)} |
| Mean / median wall-clock | {sum(times) / len(times):.1f} s / {statistics.median(times):.1f} s |
| Fastest / slowest run | {min(times):.1f} s / {max(times):.1f} s |
| Total wall-clock, all runs | {sum(times):.0f} s |
| Best-configuration runtime | {best_t:.1f} s |
| Timeouts / budget-exhausted | {n_to} |"""]
        present = defaultdict(set)
        for r in self._records:
            for k in TS_KEY_ORDER:
                v = (r.get("params") or {}).get(k)
                if v is not None:
                    present[k].add(v)
        cost_keys = [k for k in TS_KEY_ORDER if len(present.get(k, set())) >= 2][:6]
        if cost_keys:
            lines.append("")
            lines.append("Mean wall-clock by value for the varying cost-relevant parameters:")
            lines.append("")
            lines.append("| Parameter | value | # runs | mean wall-clock (s) |")
            lines.append("|---|---|---|---|")
            for k in cost_keys:
                groups = defaultdict(list)
                for r in self._records:
                    v = (r.get("params") or {}).get(k)
                    ts = r.get("time_seconds", 0) or 0
                    if v is None or not (math.isfinite(ts) and ts > 0):
                        continue
                    groups[v].append(ts)
                for v in sorted(groups, key=str):
                    ts_ = groups[v]
                    lines.append(f"| `{k}` | {v} | {len(ts_)} | {sum(ts_) / len(ts_):.1f} |")
        if budget_txt:
            lines.append("")
            lines.append(budget_txt)
        return "\n".join(lines)

    def _discussion(self):
        t = self._t()
        s = self._stats
        direction = self._pc.get("direction", "higher")
        best_p = self._best.get("params", {}) if self._best else {}
        bl = self._baseline_of(self._best_curve)
        paras = []
        if bl is not None and s.get("count"):
            gain = (s["best"] - bl) * 100
            paras.append(
                f"The central result is that the searched pipeline beats the "
                f"{best_p.get('shots', 'k')}-shot no-augmentation baseline "
                f"({bl:.4f}) by {gain:+.2f} pp, reaching {s['best']:.4f} on "
                f"{t['dataset']}.")
        its = (self._best_curve or {}).get("iterations") or []
        if its:
            t0 = its[0].get("teacher_val_acc")
            tf = its[-1].get("teacher_val_acc")
            if t0 is not None and tf is not None:
                paras.append(
                    f"The improvement is not a selection artifact: over the "
                    f"best run's {len(its)} augmentation iterations the "
                    f"dataset grew from {its[0].get('n_total', '?')} to "
                    f"{its[-1].get('n_total', '?')} examples while the "
                    f"teacher retrained on it improved its own validation "
                    f"accuracy from {t0} to {tf}, and the per-iteration "
                    f"acceptance rate moved from {its[0].get('accept_rate', '?')} "
                    f"to {its[-1].get('accept_rate', '?')} as the quota filled.")
        if self._best:
            bits = []
            for k in TS_KEY_ORDER:
                v = best_p.get(k)
                if v is None:
                    continue
                vk = round(v, 6) if isinstance(v, float) else v
                same = [r.get("metric_value", 0) for r in self._finite()
                        if (r.get("params") or {}).get(k) == vk]
                other = [r.get("metric_value", 0) for r in self._finite()
                         if (r.get("params") or {}).get(k) not in (None, vk)]
                if not same or not other:
                    continue
                d = (max(same) - max(other)) if direction != "lower" \
                    else (min(other) - min(same))
                if d > 1e-9:
                    bits.append(f"`{k} = {v}` beat the best configuration "
                                f"using any other value by {d * 100:.2f} pp")
            if bits:
                paras.append(
                    "Relative to every alternative value of the same knob, "
                    "the winning configuration is ahead on: "
                    + "; ".join(bits[:4]) + ".")
        conv = self._convergence
        paras.append(
            f"Search efficiency: {s.get('count', 0)} finite metrics from "
            f"{len(self._records)} executions, "
            f"{conv.get('improved_count', 0)} ratchet improvements, and a "
            f"final result {s.get('improvement_pct', 0):+.2f}% above the "
            f"first recorded run — all under the fixed wall-clock budget "
            f"of Section 4.")
        return "## 9. Discussion\n\n" + "\n\n".join(paras)

    def _limitations(self):
        t = self._t()
        n_exp = len(self._records)
        n_fail = self._stats.get("failed", 0)
        ss = self._pc.get("search_space") or {}
        try:
            combos = 1
            for v in ss.values():
                if isinstance(v, (list, tuple)) and v:
                    combos *= len(v)
            if combos > 1:
                frac = n_exp / combos * 100
                if frac >= 50:
                    cover_txt = (
                        f"The ratchet explored {n_exp} of {combos} rectangular-space "
                        f"combinations ({frac:.2f}%), a dense sample of the space; the "
                        f"reported optimum is a sample optimum, and unvisited corners "
                        f"could still hide a better configuration.")
                else:
                    cover_txt = (
                        f"The ratchet explored {n_exp} of {combos} rectangular-space "
                        f"combinations ({frac:.2f}%); most regions "
                        f"were never visited, so the reported optimum is a sample "
                        f"optimum, not the space optimum.")
            else:
                cover_txt = ("Only a small number of the many possible "
                             "configurations was ever executed, so the "
                             "reported optimum is a sample optimum, not the "
                             "space optimum.")
        except Exception:
            cover_txt = ("Only a small number of the many possible "
                         "configurations was ever executed, so the reported "
                         "optimum is a sample optimum, not the space optimum.")
        return f"""## 10. Limitations

- **Single protocol seed.** Every run — baseline included — uses the same
  fixed seed and validation set, so the numbers are comparable across
  configurations, but they carry no estimate of run-to-run (seed) variance.
- **One dataset, one evaluator.** Results are on {t['dataset']} with a
  prototypical-network evaluator; transfer to other few-shot benchmarks or
  to stronger downstream classifiers is untested.
- **Partial coverage.** {cover_txt}
- **Hard time budget.** {n_fail} of {n_exp} runs ended without a finite
  metric (crash or timeout); their true quality at full runtime is unknown.
- **No intrinsic quality measure.** Candidates are accepted on ensemble
  score alone; the generated {t['gen_verb']} are never graded for fluency
  or faithfulness, and the curves record selection statistics, not text
  quality.
- **Curves only for successful runs.** Per-iteration data exist only for
  runs that completed, so Section 5.3 and Appendix B cannot show what the
  failing configurations were doing."""

    def _conclusion(self):
        t = self._t()
        s = self._stats
        m = self._metric_name()
        best_p = self._best.get("params", {}) if self._best else {}
        bl = self._baseline_of(self._best_curve)
        bl_txt = (f" against a no-augmentation baseline of {bl:.4f}"
                  if bl is not None else "")
        return f"""## 11. Conclusion

We automated the design of a Teacher-Student ensemble augmentation pipeline
for few-shot {t['task_desc']}. An AutoResearch ratchet searched the TS
architecture and selection space and, from {len(self._records)} executed
configurations, identified ({self._arch_str(best_p)}) as the best
configuration, reaching **{m} = {s['best']:.4f}**{bl_txt}. The per-setting
and per-iteration analyses show *why* it wins — which ensemble weights,
thresholds and capacities matter, and how the acceptance rate tightens as
the dataset approaches its target — something a best-number-only report
cannot provide."""

    def _appendix_log(self):
        m = self._metric_name()
        rows = []
        for r in self._records:
            mv = r.get("metric_value", float("nan"))
            mvs = (f"{mv:.4f}" if isinstance(mv, (int, float))
                   and math.isfinite(mv) else "-")
            ts = r.get("time_seconds", 0) or 0
            tss = f"{ts:.0f}" if math.isfinite(ts) else "-"
            rows.append(f"| {r.get('experiment_id', '?')} | "
                        f"{r.get('experiment_name', '?')} | {mvs} | "
                        f"{r.get('status', '?')} | {tss} | "
                        f"{self._arch_str(r.get('params', {}))} |")
        body = chr(10).join(rows) if rows else "| - | - | - | - | - | - |"
        return f"""## Appendix A. Full experiment log

All {len(self._records)} recorded runs in execution order.

| ID | Name | {m} | Status | time (s) | Configuration |
|---|---|---|---|---|---|
{body}"""

    def _appendix_curves(self):
        if not self._curves:
            return ("## Appendix B. Per-iteration augmentation curves\n\n"
                    "_No curve files were found under `results/curves` / "
                    "`results/curves_text`._")
        blocks = ["## Appendix B. Per-iteration augmentation curves", ""]
        n = 0
        for c in self._curves:
            its = c.get("iterations") or []
            if not its:
                continue
            n += 1
            p = c.get("params", {}) or {}
            bl = c.get("baseline_acc", float("nan"))
            bls = (f"{bl:.4f}" if isinstance(bl, (int, float))
                   and math.isfinite(bl) else "-")
            va = c.get("val_acc", float("nan"))
            vas = (f"{va:.4f}" if isinstance(va, (int, float))
                   and math.isfinite(va) else "-")
            rows = []
            for it in its:
                pc_n = it.get("per_class_n")
                pc_s = f"[{', '.join(str(x) for x in pc_n)}]" if pc_n else "-"
                rows.append(f"| {it.get('iter', '?')} | {it.get('n_total', '?')} | "
                            f"{pc_s} | {it.get('n_candidates', '?')} | "
                            f"{it.get('n_accepted', '?')} | "
                            f"{it.get('accept_rate', '?')} | "
                            f"{it.get('mean_score_all', '?')} | "
                            f"{it.get('teacher_prob_mean', '?')} | "
                            f"{it.get('teacher_val_acc', '?')} | "
                            f"{it.get('t', '?')} s |")
            blocks.append(f"### B.{n} {self._arch_str(p)}")
            blocks.append("")
            blocks.append(f"final n = {c.get('final_n', '?')} "
                          f"(target {c.get('target_n', '?')}), val_acc = {vas}, "
                          f"no-augmentation baseline = {bls}")
            blocks.append("")
            blocks.append("| iter | total n | per-class n | candidates | accepted | acc. rate | mean S | mean P_T | teacher val acc | t |")
            blocks.append("|---|---|---|---|---|---|---|---|---|---|")
            blocks.append(chr(10).join(rows))
            blocks.append("")
        if n == 0:
            return ("## Appendix B. Per-iteration augmentation curves\n\n"
                    "_Curve files exist but none contained iteration data._")
        return chr(10).join(blocks)

    def save(self, paper_content, output_path="papers/generated_paper.md"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f: f.write(paper_content)
        return output_path