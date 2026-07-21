"""Bundled real-embeddings sample for the quickstart.

~5,000 popular movies (titles + plot summaries from English Wikipedia,
selected and ranked by TMDb popularity), pre-embedded with the same
``all-MiniLM-L6-v2`` ONNX model that ``samples.encode()`` uses, so query
and corpus vectors live in the same embedding space.

Use ``samples.dataframe()`` to upload the corpus and ``samples.encode(text)``
to embed an arbitrary query string.

Each row has columns:
    id     — opaque film-NNNNN identifier
    title  — film title (e.g. "The Lord of the Rings: The Fellowship of the Ring")
    year   — release year (e.g. 2001)
    genre  — comma-separated genre list (e.g. "fantasy adventure")
    plot   — first ~800 chars of the Wikipedia plot summary
    vector — 384-dim cosine-normalized embedding of the plot text

Plot text is licensed under CC BY-SA 4.0 (see ATTRIBUTION.md in
veep/_sample_data/). Search runs against the plot vectors; results carry
title/year/genre/plot back as metadata.

Regenerate by running ``python -m veep._sample_data.regenerate`` on a
host with the ``samples`` extra installed plus pandas, pyarrow, and the
``datasets`` library.
"""

from __future__ import annotations

from importlib import resources
from typing import Any

from ._encoder import encode

_DATA_PKG = "veep._sample_data"
_CORPUS_FILE = "sample.parquet"
DIMENSION = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

__all__ = ["DIMENSION", "MODEL_NAME", "dataframe", "encode"]


def dataframe() -> Any:
    """Return the bundled corpus as a pandas DataFrame.

    ~5000 rows with columns ``id`` (str), ``title`` (str), ``year`` (int),
    ``genre`` (str), ``plot`` (str), ``vector`` (list[float32], length 384).
    Requires ``pip install veep[pandas]``.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "samples.dataframe() requires pandas. "
            "Install with: pip install veep[pandas]"
        ) from None
    with resources.files(_DATA_PKG).joinpath(_CORPUS_FILE).open("rb") as fh:
        return pd.read_parquet(fh)
