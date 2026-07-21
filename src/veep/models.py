"""Data models returned by the veep SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Result:
    """A single search result.

    Attributes:
        key: The unique identifier for this vector.
        score: Similarity score (higher is more similar for cosine/dot_product).
        metadata: Any metadata fields stored alongside the vector.
        vector: The matched vector itself, populated only when the query was
            issued with ``with_vectors=True`` (server-sqve.2). Lets clients
            do post-hoc work like MMR diversity re-ranking that needs the
            raw embedding back. Default ``None`` keeps the wire compact.
    """

    key: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: list[float] | None = None

    def __repr__(self) -> str:
        meta = f", metadata={self.metadata}" if self.metadata else ""
        return f"Result(key={self.key!r}, score={self.score:.4f}{meta})"


class QueryResults:
    """Container for query results. Iterable, indexable, and length-aware.

    Attributes:
        worker_stats: Internal performance stats from the query engine.
    """

    def __init__(self, results: list[Result], worker_stats: list[dict] | None = None):
        self._results = results
        self.worker_stats = worker_stats or []

    def __iter__(self):
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def __getitem__(self, index):
        return self._results[index]

    def __repr__(self) -> str:
        return f"QueryResults({len(self._results)} results)"

    def __bool__(self) -> bool:
        return len(self._results) > 0


@dataclass
class Collection:
    """A vector collection.

    Attributes:
        name: The collection name.
        tier: Storage tier ('hot', 'warm', or 'paused').
        is_active: Whether the collection is actively serving queries.
        vector_count: Number of vectors stored, or None for partially-created
            collections that have not yet been populated.
        storage_gb: Storage used in gigabytes, or None if not yet reported.
        status: Processing status ('unknown', 'processing', 'ready', 'error',
            'capacity_limited', 'failed', etc.).
        dimension: Vector dimension (if known).
        failure_reason: When the collection is in a failure state on the
            server side (memory pressure, capacity-limited, etc.), this
            carries the human-readable reason. None for healthy collections.
        optimization_state: How far the collection's index optimization has
            progressed: ``'raw'`` -> ``'pca_optimized'`` ->
            ``'index_tuning_in_flight'`` -> ``'index_optimized'`` (fully
            converged). Poll for ``'index_optimized'`` when you need a
            finalized index -- the ``status`` field flips to ``'ready'`` while
            the optimizer is still tuning. None if the server didn't report it.
        target_recall: The collection's recall target (0.50-0.999, default
            0.95). Vector Panda serves the fastest search configuration
            whose measured recall meets this target. Set it at create time
            or change it later with ``vp.collections.update()``.
    """

    name: str
    tier: str = "hot"
    is_active: bool = True
    vector_count: int | None = None
    storage_gb: float | None = None
    status: str = "unknown"
    dimension: int | None = None
    failure_reason: str | None = None
    optimization_state: str | None = None
    target_recall: float | None = None


@dataclass
class FileInfo:
    """Information about an uploaded file.

    Attributes:
        name: The filename.
        size: File size in bytes.
        modified: ISO 8601 timestamp of last modification.
        is_wal: True if this entry is the synthetic ``__wal__`` row that
            represents inline-upserted vectors held in the per-collection
            write-ahead log (server-5pbm.1.5.3). The WAL is one logical
            file from the customer's perspective — the rows that have
            been inline-upserted but not yet folded into an artifact.
            Default False for ordinary uploaded files.
        vector_count: Net upsert count from the WAL (after dedup-aware
            shadowing of deletes). Only set when ``is_wal`` is True; for
            ordinary uploaded files the row count isn't known at the
            file level, so this stays None.
    """

    name: str
    size: int
    modified: str
    is_wal: bool = False
    vector_count: int | None = None


@dataclass
class UploadResult:
    """Result of a file upload or replace operation.

    Attributes:
        status: 'created' for new uploads, 'replaced' for replacements.
        collection: The collection name.
        filename: The uploaded filename.
        size: File size in bytes.
    """

    status: str
    collection: str
    filename: str
    size: int = 0


@dataclass
class FetchResult:
    """Result of fetching a single vector by key.

    Attributes:
        key: The vector's key (as returned by a query).
        found: Whether the vector was found.
        vector: The vector data (list of floats), or None if not found.
        metadata: Metadata fields for the vector, or None.
    """

    key: str
    found: bool
    vector: list[float] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ExportResult:
    """Result of an export operation.

    When ``wait=True`` (default), the export blocks until every part is
    on disk; ``path`` points at the directory holding ``manifest.json``
    plus ``part-NNNN.parquet`` files. When ``wait=False``, only
    ``job_id`` is populated — the customer is expected to poll status
    themselves.

    Attributes:
        job_id: Server-side job identifier.
        path: Destination directory once parts are downloaded. ``None``
            when ``wait=False``.
        parts: Number of parquet parts written. ``0`` when ``wait=False``.
        total_bytes: Total bytes written across all parts. ``0`` when
            ``wait=False``.
        status: Server-reported status. One of ``"rolling"``,
            ``"running"``, ``"complete"``, ``"failed"``.
    """

    job_id: str
    path: Any = None
    parts: int = 0
    total_bytes: int = 0
    status: str = "rolling"


@dataclass
class SchemaInfo:
    """Schema state for a collection.

    Attributes:
        state: Schema state ('analyzing', 'confirmed', etc.).
        id_field: The field used as the vector key.
        vector_field: The field containing embedding vectors.
        format: Data format (e.g., 'parquet').
        dimension: Vector dimension.
        analyzed: Number of samples analyzed so far.
        pending: Number of pending samples.
    """

    state: str
    id_field: str | None = None
    vector_field: str | None = None
    format: str | None = None
    dimension: int | None = None
    analyzed: int = 0
    pending: int = 0
