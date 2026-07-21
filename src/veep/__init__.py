"""veep -- Python SDK for Vector Panda vector search.

Module structure and method signatures:

    from veep import VP

    vp = VP(api_key="...", verbose=True)

    # Collections (resource-based)
    vp.collections.create("my_collection", tier="hot")    -> Collection
    vp.collections.get("my_collection")                   -> Collection
    vp.collections.list()                                 -> list[Collection]
    vp.collections.delete("my_collection")                -> None
    vp.collections.status("my_collection")                -> str
    vp.collections.export("my_collection", "out/")        -> ExportResult

    # Vectors / files
    vp.vectors.upsert("col", "data.parquet")              -> UploadResult  (file)
    vp.vectors.upsert("col", vectors=[{...}])             -> UploadResult  (inline)
    vp.vectors.upsert("col", dataframe=df)                -> UploadResult  (DataFrame)
    vp.vectors.replace("col", "data.parquet")             -> UploadResult
    vp.vectors.query("col", [0.1, ...])                   -> QueryResults
    vp.vectors.query("col", [0.1, ...], filter={...})     -> QueryResults  (filtered)
    vp.vectors.query_batch([...])                         -> list[QueryResults]
    vp.vectors.delete("col", "data.parquet")              -> None  (file delete)
    vp.vectors.delete("col", ids=["k1", "k2"])            -> dict  (id delete)
    vp.vectors.list_files("col")                          -> list[FileInfo]

    # Schema
    vp.schema.get("my_collection")                        -> SchemaInfo
    vp.schema.confirm("my_collection", id_field, vec_field) -> dict

    # Health
    vp.ping()                                             -> bool

    # Authentication (device flow + credential persistence)
    vp = VP.login()                                       # interactive OAuth via browser
    vp = VP.from_creds()                                  # load ~/.veep/credentials.json
    vp.save()                                             # persist api_key for later

API endpoints (all through consumer-site .120):
    POST   /api/v1/collections                              create collection
    GET    /api/v1/collections                              list collections
    GET    /api/v1/collections/:name                        get collection detail
    GET    /api/v1/collections/:name/status                 lightweight status
    DELETE /api/v1/collections/:name                        delete collection
    POST   /api/v1/collections/:name/files/:filename        upload file (multipart)
    PUT    /api/v1/collections/:name/files/:filename        replace file (multipart)
    DELETE /api/v1/collections/:name/files/:filename        delete file
    GET    /api/v1/collections/:name/files                  list files
    GET    /api/v1/collections/:name/schema                 get schema
    POST   /api/v1/collections/:name/schema/confirm         confirm schema
    POST   /api/v1/query                                    single query
    POST   /api/v1/query/batch                              batch query
    GET    /api/v1/health                                   health check
"""

from __future__ import annotations

from . import samples
from .client import VP
from .collections import Collections
from .exceptions import (
    AuthError,
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    CollectionNotReadyError,
    CollectionRecentlyDeletedError,
    FileAlreadyExistsError,
    NotFoundError,
    QueryError,
    ServerError,
    TimeoutError,
    UploadError,
    ValidationError,
    VeepError,
)
from .models import (
    Collection,
    ExportResult,
    FetchResult,
    FileInfo,
    QueryResults,
    Result,
    SchemaInfo,
    UploadResult,
)
from .vectors import Vectors

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("veep")
except Exception:  # pragma: no cover — running from a checkout without install
    __version__ = "0.0.0+source"

__all__ = [
    "VP",
    "Collections",
    "Vectors",
    "samples",
    "AuthError",
    "Collection",
    "CollectionAlreadyExistsError",
    "CollectionNotFoundError",
    "CollectionNotReadyError",
    "CollectionRecentlyDeletedError",
    "ExportResult",
    "FetchResult",
    "FileAlreadyExistsError",
    "FileInfo",
    "NotFoundError",
    "QueryError",
    "QueryResults",
    "Result",
    "SchemaInfo",
    "ServerError",
    "TimeoutError",
    "UploadError",
    "UploadResult",
    "ValidationError",
    "VeepError",
    "__version__",
]
