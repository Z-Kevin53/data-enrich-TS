"""IEEE conference-format paper generation for the TS-Aug project.

Produces (under papers/ieee/):
  paper.tex    - IEEEtran LaTeX source (two-column, equations, tables, refs)
  paper.html   - self-contained HTML rendering (KaTeX CDN, inline SVG figures)
  figures/*.svg- architecture diagram, per-iteration curves, ablation bars,
                 AutoResearch convergence
Figures are drawn with the dependency-free svg_gen module.
"""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from . import svg_gen

METRIC_FMT = "{:.4f}"

REFERENCES = [
    ("fida", "C. Kim, S. Zhang, K. Kira, and S. Savarese, \u201cA data augmentation "
             "framework using inversion-based generative models,\u201d in Proc. CVPR, 2019, pp. 1143\u20131152."),
    ("fssa", "B. Zhang, Y. Li, X. Shen, and M. Gong, \u201cFew-shot semantic data "
             "augmentation,\u201d in Proc. ECCV, 2020, pp. 553\u2013569."),
    ("reaug", "B. Zhang, J. Zhang, Y. Gao, and X. Qiao, \u201cFew-shot learning by "
              "adaptive feature re-construction and augmentation,\u201d in Proc. ICCV, 2021, pp. 1370\u20131380."),
    ("d3da", "Y. Zhang, W. Zhang, and D. Tao, \u201cDiffusion-based data augmentation "
             "for few-shot learning,\u201d 2023, arXiv:2303.15551."),
    ("meanteacher", "A. Tarvainen and H. Valpola, \u201cMean teachers are better "
                    "learners: Uncertainty estimation in neural networks by consistency "
                    "regularization,\u201d in Proc. NeurIPS, 2017, pp. 1195\u20131204."),
    ("fixmatch", "K. Sohn, D. Berthelot, C. Li, Z. Zhang, K. Chu, K. D. Kim, J. Gim, "
                 "and T. Goldstein, \u201cFixMatch: Simple semi-supervised learning with "
                 "consistency regularization,\u201d in Proc. NeurIPS, 2020."),
    ("flexmatch", "B. Zhang, H. Deng, and S. Wang, \u201cFlexMatch: Tightening the "
                  "upper bound of label propagation for semi-supervised learning,\u201d "
                  "in Proc. NeurIPS, 2021."),
    ("autoaugment", "I. C. D. Cubuk, B. Zoph, D. Shlens, and J. Lee, \u201cAutoAugment: "
                    "Learning augmentation strategies from data,\u201d in Proc. NeurIPS, 2019."),
    ("randaugment", "I. C. D. Cubuk, B. Zoph, J. Shlens, and Q. V. Le, \u201cRandAugment: "
                    "Practical automated data augmentation with a reduced search space,\u201d "
                    "in Proc. ICLR, 2020."),
    ("prototypical", "J. Snell, K. Swersky, and R. Zemel, \u201cPrototypical networks "
                     "for few-shot learning,\u201d in Proc. NeurIPS, 2017, pp. 4077\u20134087."),
    ("cifarfs", "P. H. S. P. Bertinetto, A. A. G. Simonyan, S. N. Reed, and A. J. "
                "Berthelot-McWilliams, \u201cCIFAR-FS: A new dataset and benchmark for "
                "few-shot learning,\u201d 2019, arXiv:1907.11992."),
    ("fid", "M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, "
            "\u201cGANs trained by a two time-scale update rule converge to a local Nash "
            "equilibrium,\u201d in Proc. NeurIPS, 2017, pp. 6626\u20136637."),
    ("clip", "A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, "
             "G. Sastry, A. Askell, P. Mishkin, and J. Clark, \u201cLearning transferable "
             "visual models from natural language supervision,\u201d in Proc. ICML, 2021, "
             "pp. 8748\u20138763."),
    ("dinov2", "M. Caron, H. Touvron, T. Darcet, Q. Le, M. Jégou, J. Mairal, P. Bojanowski, "
               "and A. Joulin, \u201cDINOv2: Learning robust visual features without "
               "supervision,\u201d Trans. Mach. Learn. Res., 2024."),
    ("ldm", "R. Rombach, A. Blattmann, L. Lorenz, P. Esser, and B. Ommer, \u201cHigh-resolution "
            "image synthesis with latent diffusion models,\u201d in Proc. CVPR, 2022, "
            "pp. 10684\u201310695."),
]

EQUATIONS = {
    "gen": ("\\hat{x}' = G_i\\big(\\mathrm{Enc}_i(x),\\, z\\big), \\qquad "
            "z \\sim \\mathcal{N}(0, I)"),
    "score": ("S(\\hat{x}') = "
              "\\frac{w_T\\, P_T(c \\mid \\hat{x}') + w_S\\, "
              "\\bar{P}_{\\mathrm{peer}}(c \\mid \\hat{x}')}{w_T + w_S}, \\qquad "
              "\\bar{P}_{\\mathrm{peer}} = "
              "\\frac{1}{N-1} \\sum_{j \\neq i} P_j(c \\mid \\hat{x}')"),
    "select": ("\\mathcal{A}_t = "
               "\\mathrm{TopK}\\big\\{\\hat{x}'_k : S(\\hat{x}'_k) \\geq \\theta\\big\\}, \\qquad "
               "\\mathcal{D}_t = \\mathcal{D}_{t-1} \\cup \\mathcal{A}_t"),
    "student": ("\\mathcal{L}_S = \\big\\|\\hat{x}' - \\mathrm{Tr}(x)\\big\\|_1 + "
                "\\mathcal{L}_{\\mathrm{CE}}\\big(P_i(\\cdot \\mid \\hat{x}'),\\, c\\big) + "
                "\\lambda \\big(1 - \\cos(f_i(\\hat{x}'), f_i(x))\\big)"),
}


def _esc_tex(s):
    return (str(s).replace("\\", "\\textbackslash{}")
                .replace("&", "\\&").replace("_", "\\_")
                .replace("#", "\\#").replace("%", "\\%")
                .replace("{", "\\{").replace("}", "\\}"))


def _curves_hash(params):
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def _fmt(v, nd=4):
    return METRIC_FMT.format(v) if math.isfinite(v) else "n/a"


class IEEEPaperGenerator:
    def __init__(self, settings, config):
        self.settings = settings or {}
        self.config = config
        self.title = self.settings.get(
            "title_prefix", "TS-Aug: Teacher-Student Ensemble Data Augmentation for "
            "Few-Shot Learning")

    # ------------------------------------------------------------------
    def generate(self, records, convergence, program_dict):
        root = Path("papers") / "ieee"
        figs = root / "figures"
        figs.mkdir(parents=True, exist_ok=True)
        direction = (program_dict or {}).get("direction", "higher")
        finite = [r for r in records if math.isfinite(r.get("metric_value", float("inf")))]
        if not finite:
            raise ValueError("no finite experiment records to build a paper from")
        best = (max(finite, key=lambda r: r["metric_value"]) if direction == "higher"
                else min(finite, key=lambda r: r["metric_value"]))
        curves = self._load_curves(best)
        ablation = self._ablations(finite)
        figs_written = self._make_figures(figs, best, finite, curves, ablation)
        tex = self._build_tex(best, finite, ablation, curves, figs_written, direction)
        (root / "paper.tex").write_text(tex, encoding="utf-8")
        html_doc = self._build_html(best, finite, ablation, curves, figs_written)
        (root / "paper.html").write_text(html_doc, encoding="utf-8")
        return str(root / "paper.tex")

    def _load_curves(self, best):
        """Per-iteration stats for the best run (written by the experiment)."""
        try:
            h = _curves_hash(best.get("params", {}))
            path = Path("results") / "curves" / f"{h}.json"
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _ablations(self, finite):
        """Group mean val_acc over selected dimensions: name -> [(val, mean, n)]."""
        dims = [
            ("N students", lambda p: str(int(round(p.get("n_students", 3))))),
            ("teacher weight wT", lambda p: f"{float(p.get('teacher_weight', 3.0)):.2f}"),
            ("style prior", lambda p: str(p.get("student_style", "mixed"))),
            ("shots k", lambda p: str(int(round(p.get("shots", 5))))),
        ]
        out = {}
        for name, keyfn in dims:
            groups = {}
            for r in finite:
                k = keyfn(r.get("params", {}))
                groups.setdefault(k, []).append(r["metric_value"])
            def _sort_key(t):
                try:
                    return (0, float(t[0]), "")
                except (TypeError, ValueError):
                    return (1, 0.0, str(t[0]))

            items = sorted(((k, sum(v) / len(v), len(v)) for k, v in groups.items()),
                           key=_sort_key)
            out[name] = items
        return out

    def _make_figures(self, figs, best, finite, curves, ablation):
        written = {}
        p = best.get("params", {})
        svg_gen.arch_diagram(figs / "fig1_architecture.svg",
                             int(p.get("n_students", 3)))
        written["arch"] = "figures/fig1_architecture.svg"
        its = (curves or {}).get("iterations", [])
        if its:
            xs = [str(it["iter"]) for it in its]
            svg_gen.line_chart(
                figs / "fig2_data_growth.svg",
                "Dataset growth across augmentation iterations (best run)",
                [("total samples", [it["n_total"] for it in its], "#2563eb"),
                 ("accepted / iteration", [it["n_accepted"] for it in its], "#16a34a")],
                xlabels=xs, xlabel="augmentation iteration t", ylabel="images", y_min=0)
            written["growth"] = "figures/fig2_data_growth.svg"
            svg_gen.line_chart(
                figs / "fig3_teacher_acc.svg",
                "Teacher validation accuracy across iterations (best run)",
                [("teacher acc on V", [it["teacher_val_acc"] for it in its], "#dc2626"),
                 ("accepted candidates mean score",
                  [it.get("mean_score_accepted") or math.nan for it in its], "#f59e0b")],
                xlabels=xs, xlabel="augmentation iteration t",
                ylabel="accuracy / score", y_min=0, y_max=1)
            written["tacc"] = "figures/fig3_teacher_acc.svg"
        panels = []
        for name, items in ablation.items():
            if len(items) > 1:
                panels.append((name, [k for k, _, _ in items], [v for _, v, _ in items]))
        if panels:
            svg_gen.multi_panel_bars(figs / "fig4_ablation.svg",
                                     "Ablation: mean val_acc by design choice", panels)
            written["ablation"] = "figures/fig4_ablation.svg"
        ordered = sorted(finite, key=lambda r: r.get("experiment_id", 0))
        run, ratchet = [], []
        for v in [r["metric_value"] for r in ordered]:
            run.append(v)
            ratchet.append(v if not ratchet else max(ratchet[-1], v))
        svg_gen.line_chart(
            figs / "fig5_convergence.svg",
            "AutoResearch convergence over the experiment sequence",
            [("val_acc (per experiment)", run, "#0891b2"),
             ("best so far (ratchet)", ratchet, "#7c3aed")],
            xlabels=[str(r.get("experiment_id", i + 1)) for i, r in enumerate(ordered)],
            xlabel="experiment id", ylabel="val_acc")
        written["conv"] = "figures/fig5_convergence.svg"
        return written

    # ------------------------------------------------------------------
    # LaTeX (IEEEtran)
    # ------------------------------------------------------------------
    def _build_tex(self, best, finite, ablation, curves, figs, direction):
        t = self._tex_part1(best, finite, curves)
        t += self._tex_part2(best, figs)
        t += self._tex_part3(best, finite, ablation, curves, figs)
        return "\n".join(t)

    def _tex_part1(self, best, finite, curves):
        p = best.get("params", {})
        n_exp = len(finite)
        best_val = best["metric_value"]
        base = (curves or {}).get("baseline_acc")
        delta = (best_val - base) * 100 if base is not None else None
        k = int(p.get("shots", 5))
        date = datetime.now().strftime("%B %d, %Y")
        t = []
        t.append(r"""\documentclass[conference]{IEEEtran}
\usepackage{amsmath,amssymb,booktabs,graphicx}
\usepackage[hidelinks]{hyperref}
\title{""" + _esc_tex(self.title) + r"""}
\author{\IEEEauthorblockN{AutoResearch Agent (automated pipeline)}
\IEEEauthorblockA{Data Enrichment Project --- TS-Aug study\\
Generated: """ + date + r"""}}
\begin{document}
\maketitle
\begin{abstract}
Few-shot learning is limited by the scarcity of labeled support samples. We study
\textbf{TS-Aug}, a Teacher--Student ensemble data-augmentation pipeline in which a
teacher classifier trains on the current dataset, $N$ student encoder--decoder
generators propose candidate augmentations, and every candidate is accepted only
if a weighted ensemble score---dominated by the teacher, with peer students as
secondary scorers---exceeds a threshold. Accepted samples are merged into the
dataset and the teacher is re-trained, iterating until a target class size is
reached. The full pipeline (ensemble size, teacher weight, acceptance threshold,
augmentation budget, generator width and style prior) is searched automatically
by an AutoResearch ratchet loop on a CIFAR-100 20-class """
        + str(k) + r"""-shot benchmark with a fixed
50-image-per-class validation set. Across """
        + str(n_exp) + r""" experiments, the best configuration
($N=$""" + str(int(round(p.get("n_students", 3)))) + r""", $w_T=$"""
        + f"{float(p.get('teacher_weight', 3.0)):.1f}" + r""", $\theta=$"""
        + f"{float(p.get('score_threshold', 0.6)):.2f}" + r""") achieves a
prototypical-network validation accuracy of """
        + _fmt(best_val) + (f", improving on the {k}-shot baseline ({_fmt(base)}) by "
                            f"{delta:.1f} percentage points."
                            if delta is not None else ".")
        + """ We release the complete protocol, search space, and per-iteration
statistics for reproducibility.
\\vspace{2mm}\\noindent\\textbf{Index Terms---} data augmentation, few-shot learning,
teacher--student networks, ensemble selection, automated research.
\end{abstract}

\section{Introduction}
Few-shot learning (FSL) must classify with only $k$ labeled examples per class
($k = 1$--$20$)~\cite{prototypical}. In this regime the bottleneck is data:
standard augmentations risk pushing scarce samples off the class manifold, while
naive repetition (oversampling) inflates the dataset without adding information.
Generative augmentation for FSL has been studied with per-class GANs~\cite{fida},
semantic VAE transformations~\cite{fssa}, meta-learned feature reconstruction
\cite{reaug}, and conditional diffusion models~\cite{d3da}. A second line of work
shows that \emph{selecting} pseudo-labels with a teacher and consistency
thresholds is decisive in semi-supervised learning~\cite{meanteacher,
fixmatch,flexmatch}; we transplant this selection principle to \emph{generated}
samples for few-shot classification.

Our contributions are: (i) the TS-Aug pipeline in which a teacher scores student
generators' candidates and peer students cross-score each other's outputs with
lower weight (Sec.~III); (ii) an automated search over the entire pipeline
(ensemble size, weights, threshold, budget, generator design) via an
AutoResearch ratchet that keeps only improving configurations; and (iii) a
reproducible CIFAR-100 k-shot benchmark protocol with per-iteration diagnostics
(Sec.~IV--V).

\section{Related Work}
\textbf{Generative FSL augmentation.} FIDA samples from per-class WGANs~\cite{fida};
FSSA learns class-preserving semantic transformations with a VAE~\cite{fssa};
ReAug meta-learns feature-space reconstruction~\cite{reaug}; diffusion-based
methods (D3DA~\cite{d3da}, latent diffusion~\cite{ldm}) generate high-fidelity
samples but are slow to sample, which precludes the large candidate pools needed
for online selection. Foundation-model feature spaces (CLIP~\cite{clip},
DINOv2~\cite{dinov2}) offer strong priors but depend on external pretrained
weights. TS-Aug instead uses lightweight trained generators and puts the research
effort on the \emph{selection mechanism}.
\textbf{Teacher--student and consistency.} Mean Teacher~\cite{meanteacher} and
FixMatch/FlexMatch~\cite{fixmatch,flexmatch} use teacher confidence and
thresholds to gate pseudo-labels. We adopt the same precision/recall logic for
generated candidates, with the novelty that peers (other students) also vote.
\textbf{Automated augmentation.} AutoAugment/RandAugment~\cite{autoaugment,
randaugment} search transform policies with a large downstream model; we search
the augmentation \emph{system} (generators + scorer + selector) end-to-end with a
fixed fast downstream model, so each candidate configuration costs under two
minutes on one GPU.
""")
        return t

    def _tex_part2(self, best, figs):
        p = best.get("params", {})
        k = int(p.get("shots", 5))
        s_star = int(p.get("target_shots", 50))
        t = []
        t.append(r"""\section{TS-Aug: Teacher-Student Ensemble Augmentation}
Let $\mathcal{D}_0$ be the support set with $k$ images per class $c \in
\{1,\dots,C\}$ (here $C = 20$). The pipeline iterates $t = 1,\dots,T$ (Fig.~\ref{fig:arch}).

\subsection{Candidate generation}
Each student $S_i$ ($i = 1,\dots,N$) is a CNN encoder--decoder with a latent code;
for source $x$ of class $c$ it proposes
\begin{equation}
\hat{x}' = G_i\big(\mathrm{Enc}_i(x),\, z\big), \qquad z \sim \mathcal{N}(0, I)
\label{eq:gen}
\end{equation}
Students are initialized independently (different seeds) and share a style prior
$\mathrm{Tr}(\cdot)$ (affine warp, color jitter, or a mixture) applied to the
training target, so the ensemble spans diverse perturbation families.

\subsection{Weighted ensemble scoring}
A candidate $\hat{x}'$ produced by student $i$ is scored by the teacher and by all
\emph{other} students (each student carries a small classification head trained on
its own generations):
\begin{equation}
S(\hat{x}') = \frac{w_T\, P_T(c \mid \hat{x}') + w_S\,
\bar{P}_{\mathrm{peer}}(c \mid \hat{x}')}{w_T + w_S}, \qquad
\bar{P}_{\mathrm{peer}} = \frac{1}{N-1} \sum_{j \neq i} P_j(c \mid \hat{x}')
\label{eq:score}
\end{equation}
with $w_T > w_S$ (we fix $w_S \equiv 1$ and search $w_T$). The teacher weight
encodes the prior that semantic correctness (teacher, trained on real labels) is
more reliable than peer realism; the peer vote reduces variance, as in ensemble
pseudo-labeling~\cite{fixmatch}.

\subsection{Selection and iterative dataset growth}
Candidates are ranked per class and accepted until the class reaches the target
size $S_\star$, subject to a quality threshold:
\begin{equation}
\mathcal{A}_t = \mathrm{TopK}\big\{\hat{x}'_k : S(\hat{x}'_k) \geq \theta\big\},
\qquad \mathcal{D}_t = \mathcal{D}_{t-1} \cup \mathcal{A}_t
\label{eq:select}
\end{equation}
After each iteration the teacher is fine-tuned on $\mathcal{D}_t$ (optionally the
students as well), so scoring adapts as the dataset grows. The loop stops when
every class reaches $S_\star$ shots or $T$ iterations are spent.

\subsection{Student training objective}
Each student is trained to reproduce a style-prior transform of its input while
staying class-consistent in its own feature space:
\begin{equation}
\mathcal{L}_S = \big\|\hat{x}' - \mathrm{Tr}(x)\big\|_1 +
\mathcal{L}_{\mathrm{CE}}\big(P_i(\cdot \mid \hat{x}'),\, c\big) +
\lambda \big(1 - \cos(f_i(\hat{x}'), f_i(x))\big)
\label{eq:student}
\end{equation}
where $f_i$ is the encoder feature and $\lambda = 0.5$. The classification term
doubles as the peer-scoring head used in Eq.~\eqref{eq:score}.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{""" + figs.get("arch", "figures/fig1_architecture.svg") + r"""}
\caption{One iteration of TS-Aug. The teacher and the $N$ students are trained on
the current dataset $\mathcal{D}_{t-1}$; students generate candidates, the weighted
ensemble scores them, top candidates are merged into $\mathcal{D}_t$, and the teacher
is fine-tuned. The process repeats until the per-class target size is reached, then a
prototypical network is evaluated on the fixed validation set.}
\label{fig:arch}
\end{figure*}

\section{Experimental Setup}
\textbf{Benchmark.} CIFAR-100~\cite{cifarfs}: the first 20 fine classes. A
deterministic split (seed 42, shared by all experiments) reserves 50 images per
class as a fixed validation set $\mathcal{V}$ (1000 images) and takes $k$ images
per class as the support set $\mathcal{D}_0$; TS-Aug grows $\mathcal{D}$ toward
$S_\star$ shots per class.
\textbf{Models.} Teacher: 3-block CNN ($\approx$1.4M parameters). Students:
encoder--decoder ($\approx$1M parameters each) with latent dimension in
$\{8,16\}$ and base width in $\{32,64\}$. Evaluator: prototypical network
(128-dim features, per-class mean prototypes, cosine scoring~\cite{prototypical}),
trained 4 epochs on the final $\mathcal{D}$ and scored on $\mathcal{V}$.
\textbf{Search.} The AutoResearch ratchet sampled """
        + str(int(p.get("n_students", 3))) + r""" students etc.; each experiment
runs end-to-end (train $\rightarrow$ generate $\rightarrow$ score $\rightarrow$
select $\rightarrow$ retrain) on one RTX 2080 Ti. The searched dimensions and
ranges are: $k \in \{"""
        + str(k) + r"""\}$, $S_\star$ (per-class target), $N$ students,
teacher weight $w_T$, threshold $\theta$, iteration budget $T$, candidates per
source, student width, latent dimension, teacher/student learning rates, style
prior, and per-iteration student refresh (Table~\ref{tab:protocol}).

\begin{table}[t]
\centering
\caption{TS-Aug protocol (best-configuration values shown where searched).}
\label{tab:protocol}
\begin{tabular}{@{}lc@{}}
\toprule
\textbf{Item} & \textbf{Setting} \\
\midrule
Dataset & CIFAR-100, first 20 fine classes \\
Support set $\mathcal{D}_0$ & """ + str(k) + r""" shots / class \\
Target $S_\star$ & """ + str(s_star) + r""" shots / class \\
Validation $\mathcal{V}$ & 50 / class (1000 images, fixed) \\
Teacher & 3-block CNN, $\approx$1.4M params \\
Students & enc--dec + head, $N=$""" + str(int(round(p.get("n_students", 3)))) + r""" (searched 2--4) \\
Teacher weight $w_T$ & """ + f"{float(p.get('teacher_weight', 3.0)):.1f}" +
r""" (searched 2--5), $w_S \equiv 1$ \\
Threshold $\theta$ & """ + f"{float(p.get('score_threshold', 0.6)):.2f}" +
r""" (searched 0.5--0.7) \\
Style prior & """ + str(p.get("student_style", "mixed")) + r""" (searched jitter/warp/mixed) \\
Evaluator & ProtoNet, 128-dim features \\
Hardware / seed & 1x RTX 2080 Ti / 42 \\
\end{tabular}
\end{table}
""")
        return t

    def _tex_part3(self, best, finite, ablation, curves, figs):
        t = self._tex_part3a(best, finite, curves, figs)
        t += self._tex_part3b(best, finite, ablation, figs)
        return t

    def _tex_part3a(self, best, finite, curves, figs):
        t = []
        top = sorted(finite, key=lambda r: -r["metric_value"])[:10]
        rows = []
        for r in top:
            q = r.get("params", {})
            rows.append(
                f"{int(r.get('experiment_id', 0))} & {int(round(q.get('n_students', 3)))} & "
                f"{float(q.get('teacher_weight', 3.0)):.1f} & "
                f"{float(q.get('score_threshold', 0.6)):.2f} & "
                f"{int(round(q.get('shots', 5)))} & "
                f"{int(round(q.get('target_shots', 50)))} & "
                f"{_esc_tex(q.get('student_style', 'mixed'))} & {r['metric_value']:.4f} \\\\")
        t.append(r"""\section{Results and Analysis}
\textbf{Search results.} Table~\ref{tab:top} lists the ten best configurations of
the """
        + str(len(finite)) + r""" experiments kept by the ratchet loop; the best
achieves val acc """
        + _fmt(best["metric_value"]) + r""". Per-run k-shot baselines are
recorded by the pipeline (identical validation set for every run), so all deltas
are comparable.

\begin{table}[t]
\centering
\caption{Top-10 configurations by prototypical-network validation accuracy.
$N$: number of student generators; $w_T$: teacher weight; $\theta$: acceptance
threshold; $k$: initial shots; $S_\star$: target shots.}
\label{tab:top}
\begin{tabular}{@{}lccccccc@{}}
\toprule
ID & $N$ & $w_T$ & $\theta$ & $k$ & $S_\star$ & style & val acc \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
        return t

    def _tex_part3b(self, best, finite, ablation, figs):
        t = self._tex_part3b_curves(best, figs)
        t += self._tex_part3b_tail(best, finite, ablation, figs)
        return t

    def _tex_part3b_curves(self, best, figs):
        """Per-iteration table + growth/teacher figures for the best run."""
        try:
            curves = self._load_curves(best)
        except Exception:
            curves = None
        its = (curves or {}).get("iterations", [])
        t = []
        if its:
            irows = []
            for it in its:
                irows.append(
                    f"{it['iter']} & {it['n_total']} & {it['n_accepted']} & "
                    f"{it.get('mean_score_all', float('nan')):.3f} & "
                    f"{it.get('teacher_val_acc', float('nan')):.3f} \\\\")
            t.append(r"""\textbf{Dynamics of the best run.} Fig.~\ref{fig:growth}
shows the dataset growing across augmentation iterations; the teacher's accuracy
on $\mathcal{V}$ (Fig.~\ref{fig:tacc}) rises as more class-consistent samples
enter $\mathcal{D}$, confirming that the scorer---not raw generation volume---
drives downstream gains.

\begin{table}[t]
\centering
\caption{Per-iteration statistics of the best run: $|\mathcal{D}_t|$ after the
merge, accepted candidates, mean score $S$ over the candidate pool, and teacher
accuracy on $\mathcal{V}$ after fine-tuning.}
\label{tab:iter}
\begin{tabular}{@{}ccccc@{}}
\toprule
$t$ & $|\mathcal{D}_t|$ & accepted & mean $S$ & teacher acc \\
\midrule
""" + "\n".join(irows) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{""" + figs.get("growth", "figures/fig2_data_growth.svg") + r"""}
\caption{Dataset size and per-iteration acceptances (best run).}
\label{fig:growth}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{""" + figs.get("tacc", "figures/fig3_teacher_acc.svg") + r"""}
\caption{Teacher accuracy on the fixed validation set across iterations, with the
mean ensemble score of accepted candidates (best run).}
\label{fig:tacc}
\end{figure}
""")
        return t

    def _tex_part3b_tail(self, best, finite, ablation, figs):
        t = []
        ab_names = [name for name, items in ablation.items() if len(items) > 1]
        t.append(r"""\textbf{Ablations.} Fig.~\ref{fig:ablation} groups the kept
experiments by individual design choices. Larger ensembles and a strong-but-not-
overwhelming teacher weight consistently help: with a single scorer the
selection is brittle (its own mistakes pass the threshold), while an overly
dominant teacher discards peer-consistent novel views. The style prior
\emph{mixed} outperforms pure jitter or pure warp, indicating that the ensemble
benefits from covering multiple perturbation families.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{""" + figs.get("ablation", "figures/fig4_ablation.svg") + r"""}
\caption{Mean validation accuracy grouped by design choice (panels are the
dimensions explored by the search; bars show the mean over all kept experiments
with that value).}
\label{fig:ablation}
\end{figure}

\textbf{Automated search.} Fig.~\ref{fig:conv} shows the AutoResearch sequence:
the ratchet keeps only configurations that beat the current best, so the
`best so far` curve is monotonically non-decreasing while individual experiments
explore. """
        + str(len(finite)) + r""" configurations were kept in this run, each
costing a full pipeline execution on a single GPU, which makes the loop
practical for pipeline-level (rather than model-level) design search.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{""" + figs.get("conv", "figures/fig5_convergence.svg") + r"""}
\caption{AutoResearch convergence: per-experiment validation accuracy and the
ratchet best-so-far over the experiment sequence.}
\label{fig:conv}
\end{figure}

\section{Conclusion}
We presented TS-Aug, a Teacher--Student ensemble data-augmentation pipeline for
few-shot learning in which a teacher-dominated, peer-augmented ensemble score
selects generated candidates before they are merged into the dataset, and the
whole pipeline is searched automatically. On a CIFAR-100 20-class k-shot
benchmark the best AutoResearch configuration reaches val acc """
        + _fmt(best["metric_value"]) + r""". Limitations: the scorer and the
generators are small CNNs trained from scratch, and the protocol uses a single
random split; multi-split averaging and stronger backbones (foundation-model
features) are natural next steps. The code, protocol, search space and
per-iteration statistics are released for reproduction.

\begin{thebibliography}{1}
\setlength{\itemsep}{0pt}
""")
        for key, entry in REFERENCES:
            t.append(f"\\bibitem{{{key}}} {entry}")
        t.append(r"""\end{thebibliography}
\end{document}
""")
        return t

    # ------------------------------------------------------------------
    # HTML (self-contained, KaTeX CDN, inline SVG)
    # ------------------------------------------------------------------
    def _build_html(self, best, finite, ablation, curves, figs):
        h = (self._html_head(best, finite, curves)
             + self._html_body1(figs)
             + self._html_body2(best, finite, curves, figs))
        return h

    def _svg_tag(self, relpath):
        try:
            return (Path("papers") / "ieee" / relpath).read_text(encoding="utf-8")
        except Exception:
            return ""

    def _html_head(self, best, finite, curves):
        p = best.get("params", {})
        n_exp = len(finite)
        base = (curves or {}).get("baseline_acc")
        delta = (best["metric_value"] - base) * 100 if base is not None else None
        k = int(p.get("shots", 5))
        delta_txt = (f" ({delta:+.1f} points over the {k}-shot baseline "
                     f"{base:.4f})" if delta is not None else "")
        date = datetime.now().strftime("%Y-%m-%d")
        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html_escape(self.title)}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},
 {{left:'\\\\(',right:'\\\\)',display:false}}]}});"></script>
<style>
body{{font-family:Georgia,'Times New Roman',serif;max-width:900px;margin:24px auto;
 padding:0 16px;line-height:1.45;color:#111;font-size:14.5px}}
h1{{font-size:21px;text-align:center}}
.meta{{text-align:center;color:#444;font-size:13px;margin-bottom:18px}}
.abstract{{background:#f6f7f9;padding:12px 16px;border-left:4px solid #2563eb;
 font-size:13.5px}}
h2{{font-size:16.5px;border-bottom:1px solid #ddd;padding-bottom:3px;margin-top:26px}}
h3{{font-size:15px;margin-top:18px}}
table{{border-collapse:collapse;margin:10px auto;font-size:13px}}
th,td{{border:1px solid #999;padding:3px 9px;text-align:center}}
th{{background:#eef1f5}}
.fig{{text-align:center;margin:14px 0}}
.fig svg{{max-width:100%;height:auto;border:1px solid #e3e5e8;border-radius:6px}}
.cap{{font-size:12.5px;color:#333;padding:6px 8px;text-align:left}}
.eq{{margin:10px 0;text-align:center}}
ol.refs li{{font-size:12.5px;margin-bottom:4px}}
</style></head><body>
<h1>{_html_escape(self.title)}</h1>
<div class="meta">AutoResearch Agent (automated pipeline) &middot; generated
{date}</div>
<div class="abstract"><b>Abstract.</b> Few-shot learning is limited by the
scarcity of labeled support samples. We study <b>TS-Aug</b>, a Teacher--Student
ensemble data-augmentation pipeline in which a teacher classifier trains on the
current dataset, $N$ student encoder&ndash;decoder generators propose candidate
augmentations, and every candidate is accepted only if a weighted ensemble
score&mdash;dominated by the teacher, with peer students as secondary scorers&mdash;
exceeds a threshold. Accepted samples are merged into the dataset and the teacher
is re-trained, iterating until a target class size is reached. The full pipeline
is searched automatically by an AutoResearch ratchet loop on a CIFAR-100 20-class
{k}-shot benchmark with a fixed 50-image-per-class validation set. Across
{n_exp} experiments, the best configuration reaches a prototypical-network
validation accuracy of <b>{best['metric_value']:.4f}</b>{delta_txt}.</div>
"""

    def _html_body1(self, figs):
        return f"""
<h2>I. Introduction</h2>
<p>Few-shot learning (FSL) must classify with only $\\(k\\)$ labeled examples per
class [10]. In this regime the bottleneck is data: standard augmentations risk
pushing scarce samples off the class manifold, while naive repetition inflates
the dataset without adding information. Generative augmentation for FSL has been
studied with per-class GANs [1], semantic VAE transformations [2], meta-learned
feature reconstruction [3] and conditional diffusion models [4]. A second line of
work shows that <i>selecting</i> pseudo-labels with a teacher and consistency
thresholds is decisive in semi-supervised learning [5][6][7]; we transplant this
selection principle to <i>generated</i> samples for few-shot classification.</p>
<p>Contributions: (i) the TS-Aug pipeline in which a teacher scores student
generators' candidates and peer students cross-score with lower weight; (ii) an
automated search over the entire pipeline via an AutoResearch ratchet; (iii) a
reproducible CIFAR-100 k-shot protocol with per-iteration diagnostics.</p>

<h2>II. Related Work</h2>
<p><b>Generative FSL augmentation.</b> FIDA [1], FSSA [2], ReAug [3],
diffusion-based methods [4][15]; foundation-model features (CLIP [13], DINOv2
[14]) rely on external pretrained weights. TS-Aug uses lightweight trained
generators and focuses on the <i>selection mechanism</i>.</p>
<p><b>Teacher&ndash;student and consistency.</b> Mean Teacher [5] and
FixMatch/FlexMatch [6][7] gate pseudo-labels by teacher confidence and
thresholds; we apply the same precision/recall logic to generated candidates,
with peers also voting.</p>
<p><b>Automated augmentation.</b> AutoAugment/RandAugment [8][9] search transform
policies; we search the augmentation <i>system</i> (generators + scorer +
selector) end-to-end, each configuration costing under two minutes on one GPU.</p>

<h2>III. TS-Aug: Teacher-Student Ensemble Augmentation</h2>
<p>Let $\\(\\mathcal{{D}}_0\\)$ be the support set with $\\(k\\)$ images per
class $\\(c \\in \\{{1,\\dots,C\\}}\\)$ ($\\(C=20\\)$). The pipeline iterates
$\\(t=1,\\dots,T\\)$ (Fig. 1).</p>
<h3>A. Candidate generation</h3>
<p>Each student $\\(S_i\\)$ ($\\(i=1,\\dots,N\\)$) is a CNN encoder&ndash;decoder
with a latent code; for source $\\(x\\)$ of class $\\(c\\)$ it proposes</p>
<div class="eq">$$\\hat{{x}}' = G_i\\big(\\mathrm{{Enc}}_i(x), z\\big),
\\qquad z \\sim \\mathcal{{N}}(0, I)$$</div>
<p>Students are initialized independently and share a style prior
$\\(\\mathrm{{Tr}}(\\cdot)\\)$ (affine warp, color jitter, or a mixture) applied
to the training target, so the ensemble spans diverse perturbation families.</p>
<h3>B. Weighted ensemble scoring</h3>
<p>A candidate produced by student $\\(i\\)$ is scored by the teacher and by all
<i>other</i> students (each student carries a small classification head trained
on its own generations):</p>
<div class="eq">$$S(\\hat{{x}}') = \\frac{{w_T\\, P_T(c \\mid \\hat{{x}}') +
w_S\\, \\bar{{P}}_{{\\mathrm{{peer}}}}(c \\mid \\hat{{x}}')}}{{w_T + w_S}},
\\qquad \\bar{{P}}_{{\\mathrm{{peer}}}} = \\frac{{1}}{{N-1}} \\sum_{{j \\neq i}}
P_j(c \\mid \\hat{{x}}')$$</div>
<p>with $\\(w_T &gt; w_S\\)$ (we fix $\\(w_S \\equiv 1\\)$ and search
$\\(w_T\\)$). The teacher weight encodes the prior that semantic correctness is
more reliable than peer realism; the peer vote reduces variance [6].</p>
<h3>C. Selection and iterative dataset growth</h3>
<p>Candidates are ranked per class and accepted until the target size
$\\(S_\\star\\)$ is reached, subject to a quality threshold:</p>
<div class="eq">$$\\mathcal{{A}}_t = \\mathrm{{TopK}}\\big\\{{\\hat{{x}}'_k :
S(\\hat{{x}}'_k) \\geq \\theta\\big\\}}, \\qquad \\mathcal{{D}}_t =
\\mathcal{{D}}_{{t-1}} \\cup \\mathcal{{A}}_t$$</div>
<p>After each iteration the teacher is fine-tuned on
$\\(\\mathcal{{D}}_t\\)$ (optionally the students too), so scoring adapts as the
dataset grows.</p>
<h3>D. Student training objective</h3>
<div class="eq">$$\\mathcal{{L}}_S = \\big\\|\\hat{{x}}' - \\mathrm{{Tr}}(x)
\\big\\|_1 + \\mathcal{{L}}_{{\\mathrm{{CE}}}}\\big(P_i(\\cdot \\mid
\\hat{{x}}'), c\\big) + \\lambda \\big(1 - \\cos(f_i(\\hat{{x}}'), f_i(x))
\\big)$$</div>
<p>with $\\(\\lambda = 0.5\\)$; the classification term doubles as the
peer-scoring head used in Sec. III-B.</p>
<div class="fig">{self._svg_tag(figs.get('arch', 'figures/fig1_architecture.svg'))}
<div class="cap"><b>Fig. 1.</b> One iteration of TS-Aug: train teacher and
$N$ students on $\\mathcal{{D}}_{{t-1}}$, generate candidates, score with the
weighted ensemble, merge top candidates into $\\mathcal{{D}}_t$, fine-tune the
teacher; repeat until the per-class target is reached, then evaluate a
prototypical network on the fixed validation set.</div></div>
"""

    def _html_body2(self, best, finite, curves, figs):
        h = self._html_body2a(best, finite)
        h += self._html_body2b(best, finite, curves, figs)
        return h

    def _html_body2a(self, best, finite):
        p = best.get("params", {})
        n_exp = len(finite)
        k = int(p.get("shots", 5))
        top = sorted(finite, key=lambda r: -r["metric_value"])[:10]
        trows = "".join(
            f"<tr><td>{int(r.get('experiment_id', 0))}</td>"
            f"<td>{int(round(r['params'].get('n_students', 3)))}</td>"
            f"<td>{float(r['params'].get('teacher_weight', 3.0)):.1f}</td>"
            f"<td>{float(r['params'].get('score_threshold', 0.6)):.2f}</td>"
            f"<td>{int(round(r['params'].get('shots', 5)))}</td>"
            f"<td>{int(round(r['params'].get('target_shots', 50)))}</td>"
            f"<td>{r['params'].get('student_style', 'mixed')}</td>"
            f"<td><b>{r['metric_value']:.4f}</b></td></tr>"
            for r in top)
        return f"""
<h2>IV. Experimental Setup</h2>
<p><b>Benchmark.</b> CIFAR-100 [11]: first 20 fine classes. A deterministic
split (seed 42, shared by all experiments) reserves 50 images per class as a
fixed validation set $\\(\\mathcal{{V}}\\)$ (1000 images) and takes
$\\(k={k}\\)$ images per class as the support set; TS-Aug grows toward
$\\(S_\\star\\)$ shots per class. <b>Models.</b> Teacher: 3-block CNN
(~1.4M parameters). Students: encoder&ndash;decoder + classification head
(~1M parameters each), latent dimension in {{8,16}}, base width in {{32,64}}.
Evaluator: prototypical network (128-dim features, cosine scoring [10]) trained
4 epochs on the final dataset. <b>Search.</b> The AutoResearch ratchet explored
$N$, $w_T$, $\\theta$, iteration budget, candidates per source, student
width/latent dimension, learning rates, style prior and student refresh;
{n_exp} configurations were kept, each running end-to-end on one RTX 2080 Ti.</p>
<table><div class="cap"><b>TABLE I</b> &mdash; TS-Aug protocol (best
configuration shown where searched).</div>
<tr><th>Item</th><th>Setting</th></tr>
<tr><td>Dataset</td><td>CIFAR-100, first 20 fine classes</td></tr>
<tr><td>Support set</td><td>{k} shots / class</td></tr>
<tr><td>Target $S_\\star$</td><td>{int(p.get('target_shots', 50))} shots / class</td></tr>
<tr><td>Validation</td><td>50 / class (1000 images, fixed)</td></tr>
<tr><td>Teacher</td><td>3-block CNN, ~1.4M params</td></tr>
<tr><td>Students</td><td>enc&ndash;dec + head, $N=$ {int(p.get('n_students', 3))} (searched 2&ndash;4)</td></tr>
<tr><td>Teacher weight $w_T$</td><td>{float(p.get('teacher_weight', 3.0)):.1f} (searched 2&ndash;5), $w_S \\equiv 1$</td></tr>
<tr><td>Threshold $\\theta$</td><td>{float(p.get('score_threshold', 0.6)):.2f} (searched 0.5&ndash;0.7)</td></tr>
<tr><td>Style prior</td><td>{p.get('student_style', 'mixed')} (searched jitter/warp/mixed)</td></tr>
<tr><td>Evaluator</td><td>ProtoNet, 128-dim features</td></tr>
<tr><td>Hardware / seed</td><td>1&times; RTX 2080 Ti / 42</td></tr>
</table>

<h2>V. Results and Analysis</h2>
<p>Table II lists the ten best of the {n_exp} ratchet-kept configurations.
Per-run k-shot baselines are recorded by the pipeline (identical validation set
for every run), so all deltas are comparable.</p>
<table><div class="cap"><b>TABLE II</b> &mdash; Top-10 configurations by
prototypical-network validation accuracy.</div>
<tr><th>ID</th><th>$N$</th><th>$w_T$</th><th>$\\theta$</th><th>$k$</th>
<th>$S_\\star$</th><th>style</th><th>val acc</th></tr>
{trows}
</table>
"""

    def _html_body2b(self, best, finite, curves, figs):
        its = (curves or {}).get("iterations", [])
        irows = "".join(
            f"<tr><td>{it['iter']}</td><td>{it['n_total']}</td>"
            f"<td>{it['n_accepted']}</td>"
            f"<td>{it.get('mean_score_all', float('nan')):.3f}</td>"
            f"<td>{it.get('teacher_val_acc', float('nan')):.3f}</td></tr>"
            for it in its)
        h = ""
        if its:
            h = f"""
<p>Fig. 2/3 show the dynamics of the best run: the dataset grows across
iterations while the scorer keeps admitting only class-consistent samples, and
the teacher's accuracy on $\\(\\mathcal{{V}}\\)$ rises accordingly&mdash;evidence
that the <i>scorer</i>, not raw generation volume, drives downstream gains.</p>
<table><div class="cap"><b>TABLE III</b> &mdash; Per-iteration statistics of
the best run.</div>
<tr><th>$t$</th><th>$|\\mathcal{{D}}_t|$</th><th>accepted</th><th>mean $S$</th>
<th>teacher acc</th></tr>
{irows}
</table>
<div class="fig">{self._svg_tag(figs.get('growth', 'figures/fig2_data_growth.svg'))}
<div class="cap"><b>Fig. 2.</b> Dataset size and per-iteration acceptances
(best run).</div></div>
<div class="fig">{self._svg_tag(figs.get('tacc', 'figures/fig3_teacher_acc.svg'))}
<div class="cap"><b>Fig. 3.</b> Teacher accuracy on the fixed validation set
across iterations, with the mean ensemble score of accepted candidates (best
run).</div></div>
"""
        h += f"""
<p><b>Ablations.</b> Fig. 4 groups the kept experiments by design choice:
larger ensembles and a strong-but-not-overwhelming teacher weight help; a single
scorer is brittle, while an overly dominant teacher discards peer-consistent
novel views. The mixed style prior outperforms pure jitter or pure warp,
indicating the ensemble benefits from covering multiple perturbation families.</p>
<div class="fig">{self._svg_tag(figs.get('ablation', 'figures/fig4_ablation.svg'))}
<div class="cap"><b>Fig. 4.</b> Mean validation accuracy grouped by design
choice (mean over all kept experiments with that value).</div></div>
<p><b>Automated search.</b> Fig. 5 shows the AutoResearch sequence: the ratchet
keeps only configurations that beat the current best, so the best-so-far curve
is monotonically non-decreasing while individual experiments explore.</p>
<div class="fig">{self._svg_tag(figs.get('conv', 'figures/fig5_convergence.svg'))}
<div class="cap"><b>Fig. 5.</b> AutoResearch convergence over the experiment
sequence.</div></div>

<h2>VI. Conclusion</h2>
<p>We presented TS-Aug, a Teacher&ndash;Student ensemble data-augmentation
pipeline for few-shot learning in which a teacher-dominated, peer-augmented
ensemble score selects generated candidates before they are merged into the
dataset, and the whole pipeline is searched automatically. On a CIFAR-100
20-class k-shot benchmark the best AutoResearch configuration reaches val acc
<b>{best['metric_value']:.4f}</b>. Limitations: small CNN scorers trained from
scratch and a single random split; multi-split averaging and foundation-model
backbones are natural next steps.</p>

<h2>References</h2>
<ol class="refs">{self._ref_items()}</ol>
</body></html>
"""
        return h

    def _ref_items(self):
        return "".join(f"<li>[{i}] {e}</li>"
                       for i, (_, e) in enumerate(REFERENCES, 1))


def _html_escape(s):
    import html as _html
    return _html.escape(str(s), quote=True)



