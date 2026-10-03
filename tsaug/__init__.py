"""TS-Aug: Teacher-Student ensemble data augmentation for few-shot learning.

Pipeline (see docs/02_experimental_design.md):
  1. Teacher CNN trained on the current few-shot set D.
  2. N Student encoder-decoder networks generate candidate augmentations.
  3. Candidates are scored by the weighted ensemble
     S(x') = [wT*P_T(c|x') + wS*mean_{j!=i} P_j(c|x')] / (wT + wS).
  4. Top-scoring candidates (S >= theta) are accepted per class;
     the Teacher is fine-tuned on the enlarged D.
  5. Repeat until the per-class target size is reached.
"""
