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
    """

    key: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

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
    """

    name: str
    tier: str = "hot"
    is_active: bool = True
    vector_count: int | None = None
    storage_gb: float | None = None
    status: str = "unknown"
    dimension: int | None = None
    failure_reason: str | None = None


@dataclass
class FileInfo:
    """Information about an uploaded file.

    Attributes:
        name: The filename.
        size: File size in bytes.
        modified: ISO 8601 timestamp of last modification.
    """

    name: str
    size: int
    modified: str


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
