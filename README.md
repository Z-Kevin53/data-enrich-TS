# AutoResearch: Data Enrichment (data_enrich)

A second AutoResearch workflow project, sibling of `autoresearch_wf` (the MNIST
digit task). This one does **no digit recognition** — the research question is
about **data enrichment** itself.

## Research question
Given only a small **seed** subset of labeled emails (15-50% of the train
split), which enrichment strategy yields the best downstream classifier?

- Task: binary email classification (spam vs non-spam)
- Data: UCI **Spambase** — 4601 emails, 57 numeric features, public domain.
  Downloaded on first run, cached under `data/spambase/` (git-ignored).
- Enrichment strategies searched: `synthetic` (SMOTE-lite interpolation),
  `oversample` (minority duplication), `noise` (gaussian-augmented copies),
  `hybrid` (synthetic + noise), `none` (baseline). Enrichment targets the
  minority class (non-spam, ~39%).
- Model: small MLP; hyperparameters (lr, batch size, hidden dim, dropout,
  optimizer, weight decay) are co-searched.
- Metric: `val_acc` on a fixed 15% validation split — **direction: higher**
  (this project exercises the framework's "bigger is better" path).

## Server layout
- `/home/zwk/Joey` — sibling folder (pre-existing, untouched)
- `/home/zwk/data_enrich` — this project (synced from this folder)
- Python env: symlinked from `/home/zwk/autoresearch_wf/.venv`

## Usage (from this folder, Windows)
```powershell
python run_remote.py --max-experiments 5        # sync + run + pull
python run_remote.py --sync-only                # only push code
python run_remote.py --pull-only                # only pull results/papers
python run_remote.py --fresh --max-experiments 10  # wipe remote history first
python test_run.py                              # local simulate-mode smoke
```

GPU: pinned to **physical GPU 0** (nvidia-smi UUID), CPU fallback — same
convention as the MNIST project.
