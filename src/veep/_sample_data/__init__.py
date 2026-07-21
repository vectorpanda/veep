"""Bundled sample data for veep.samples.

Files in this package:
  - sample.parquet  — 50 short text snippets, pre-embedded (384-dim, cosine-normalized)
  - queries.json    — 5 pre-embedded query vectors keyed by prompt
  - regenerate.py   — offline regenerator (requires sentence-transformers)

Public access is via ``veep.samples``. This package only exists to mark the
data directory importable so ``importlib.resources`` can locate the files
inside the wheel.
"""
