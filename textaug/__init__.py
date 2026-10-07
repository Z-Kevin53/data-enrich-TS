"""TS (Teacher-Student) ensemble augmentation for few-shot TEXT classification.

Text analog of the tsaug (image) package:
  data.py    - AG News k-shot benchmark, fixed deterministic split
  models.py  - TextEncoder / TeacherText / StudentText / ProtoText
  pipeline.py- training loops, token-level style prior, ensemble scoring, run_ts
"""
