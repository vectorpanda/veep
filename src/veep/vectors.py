"""Vector operations for the veep SDK."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .exceptions import (
    CollectionNotFoundError,
    CollectionNotReadyError,
    FileAlreadyExistsError,
    NotFoundError,
    ServerError,
    TimeoutError,
    UploadError,
    ValidationError,
)
from .models import FetchResult, FileInfo, QueryResults, Result, UploadResult

# server-dl7r: terminal collection states the upsert wait-loop treats as fatal.
_READY_STATUS = "ready"
_TERMINAL_FAILURE_STATES = ("error", "failed", "capacity_limited")


def _unwrap_key(raw: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """server-fxox: translate a wire result's key back to the customer's id.

    Vector Panda stores keys internally as u128 hashes (xxh3_128 of the
    original id when the id wasn't already numeric). Workers return those
    hashes in result.key. The customer's original id is preserved as
    metadata.key_original. The SDK promotes that field back to r.key and
    removes the duplicate from metadata so customers see exactly what they
    uploaded — not a 39-char hash with the real id buried in metadata.
    """
    metadata = raw.get("metadata")
    if isinstance(metadata, dict) and "key_original" in metadata:
        original = metadata.pop("key_original")
        return str(original), (metadata if metadata else None)
    return raw.get("key", ""), metadata

# server-cvms.2: chunked upload defaults. Each chunk is read into RAM as a bytes
# object so requests can retry it on transient failure; peak per-call RSS ≈ one
# chunk. The server (server-y9je) caps at 256 MiB and reports its limit in the
# session-start response — we honour the smaller of the two.
_DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024

# server-gzb.3.2: extension whitelist mirrors the source-side detector in
# vector-panda-source/src/extractors/mod.rs. The SDK only checks the extension
# is on the list and the file is non-empty; real format/schema validation
# happens server-side.
_SUPPORTED_EXTENSIONS = frozenset({
    "parquet",
    "csv",
    "jsonl", "json",
    "arrow", "feather", "ipc",
    "bvecs", "fvecs", "ivecs",
    "fbin", "u8bin", "i8bin",
    "npy", "npz",
    "h5", "hdf5", "hdf",
    "safetensors",
})

if TYPE_CHECKING:
    from .client import VP

logger = logging.getLogger("veep")

VALID_FILENAME = re.compile(r"^[a-zA-Z0-9._-]+$")


def _coerce_vector_columns(table):
    """Convert list<double> columns to FixedSizeList<float32> for the source server."""
    import pyarrow as pa

    for i, field in enumerate(table.schema):
        if not pa.types.is_large_list(field.type) and not pa.types.is_list(field.type):
            continue
        col = table.column(i)
        first_valid = None
        for val in col:
            if val.is_valid:
                first_valid = val.as_py()
                break
        if first_valid is None or not isinstance(first_valid, list):
            continue
        dim = len(first_valid)
        if dim == 0:
            continue
        import numpy as np
        arrays = [v.as_py() if v.is_valid else [0.0] * dim for v in col]
        flat = np.array(arrays, dtype=np.float32).reshape(-1)
        flat_arrow = pa.array(flat, type=pa.float32())
        fixed_col = pa.FixedSizeListArray.from_arrays(flat_arrow, dim)
        table = table.set_column(i, pa.field(field.name, fixed_col.type), fixed_col)
    return table



class Vectors:
    """Upload, query, and manage vectors.

    Access this through ``client.vectors`` -- do not instantiate directly.

    Example::

        vp = VP(api_key="...")
        vp.vectors.upsert("my_collection", "embeddings.parquet")
        results = vp.vectors.query("my_collection", vector=[0.1, 0.2, 0.3])
    """

    def __init__(self, client: VP):
        self._client = client

    def upsert(
        self,
        collection: str,
        file_path: str | Path | None = None,
        *,
        vectors: list[dict[str, Any]] | None = None,
        dataframe: Any | None = None,
        table: Any | None = None,
        wait_seconds: float | None = None,
    ) -> UploadResult:
        """Upload vectors to a collection.

        Four modes:

        1. **File upload** -- pass a file path::

            vp.vectors.upsert("products", "embeddings.parquet")
            vp.vectors.upsert("products", "embeddings.csv")
            vp.vectors.upsert("products", "embeddings.fbin")

           Supported extensions: ``.parquet``, ``.csv``, ``.jsonl``/``.json``,
           ``.arrow``/``.feather``/``.ipc``, ``.bvecs``/``.fvecs``/``.ivecs``,
           ``.fbin``/``.u8bin``/``.i8bin``, ``.npy``/``.npz``,
           ``.h5``/``.hdf5``/``.hdf``, ``.safetensors``.

        2. **Inline vectors** -- pass a list of dicts::

            vp.vectors.upsert("products", vectors=[
                {"id": "abc", "vector": [0.1, 0.2, ...], "metadata": {"color": "red"}},
            ])

        3. **DataFrame** -- pass a pandas DataFrame (requires ``pip install veep[pandas]``)::

            vp.vectors.upsert("products", dataframe=df)

        4. **Arrow Table** -- pass a pyarrow Table directly (requires pyarrow)::

            import pyarrow.parquet as pq
            tbl = pq.read_table("embeddings.parquet")
            vp.vectors.upsert("products", table=tbl)

           Useful when you've already loaded the data with pyarrow (e.g., to slice
           or filter rows) and don't want to round-trip through a temp file.

        Args:
            collection: Collection name.
            file_path: Path to a file on disk (mode 1).
            vectors: List of vector dicts with 'id', 'vector', and optional 'metadata' (mode 2).
            dataframe: A pandas DataFrame with id, vector, and optional metadata columns (mode 3).
            table: A pyarrow.Table with id, vector, and optional metadata columns (mode 4).
            wait_seconds: How long to wait for the collection to become queryable
                after upload (server-sqom.6). Defaults to the client's
                ``upload_timeout``. If the wait expires before the ingest
                pipeline finishes, ``CollectionNotReadyError`` is raised with
                guidance — do NOT retry upsert in that case (it would create
                duplicate state); inspect ``vp.collections.status(name)`` and
                contact support if the collection stays stuck.

        Returns:
            An UploadResult once the data is queryable. ``upsert`` blocks until
            the collection's status is ``ready`` (server-dl7r) so a follow-up
            ``query`` or ``fetch`` always sees the new vectors. Polling overhead
            is small — under a second on typical small collections.
        """
        modes = sum([
            file_path is not None,
            vectors is not None,
            dataframe is not None,
            table is not None,
        ])
        if modes == 0:
            raise ValidationError(
                "Provide one of: file_path, vectors=, dataframe=, or table=."
            )
        if modes > 1:
            raise ValidationError(
                "Provide only one of: file_path, vectors=, dataframe=, or table=."
            )

        if file_path is not None:
            result = self._upsert_file(collection, Path(file_path))
        elif vectors is not None:
            result = self._upsert_vectors(collection, vectors)
        elif dataframe is not None:
            result = self._upsert_dataframe(collection, dataframe)
        else:
            result = self._upsert_arrow_table(collection, table)

        # server-dl7r: block until the data is queryable. Beginners expect
        # upsert == queryable on return; this absorbs the ingest-pipeline
        # round trip so the very next query/fetch in their notebook works.
        self._wait_until_ready(collection, wait_seconds=wait_seconds)
        return result

    def _maybe_raise_not_ready(self, collection: str, exc: Exception) -> None:
        """If the collection exists but isn't ready, raise CollectionNotReadyError.

        Called from query() / query_batch() / fetch() when the wire returns an
        ambiguous 'no workers serving' / 'no current epoch' style failure.
        Caller should re-raise the original exception if this returns normally.
        """
        try:
            cols = {c.name: c for c in self._client.collections.list()}
        except Exception:  # noqa: BLE001
            return  # probe failed; let the original exception surface
        col = cols.get(collection)
        if col is None:
            return
        if col.status != _READY_STATUS:
            raise CollectionNotReadyError(
                collection, status=col.status, suggested_wait_seconds=2.0,
            ) from exc

    def _wait_until_ready(self, collection: str, wait_seconds: float | None = None) -> None:
        """Poll vp.collections.status() until it reports 'ready'.

        Backoff: 100ms -> 200 -> 400 -> 800 -> capped at 1s. Deadline is
        ``wait_seconds`` if passed, else the client's ``upload_timeout``
        (None = no deadline). Raises ServerError on a terminal failure
        state, CollectionNotReadyError on deadline (server-sqom.6 — the
        ingest pipeline didn't complete; retrying upsert would just
        duplicate state).
        """
        timeout = wait_seconds if wait_seconds is not None else self._client.upload_timeout
        deadline = (time.time() + timeout) if timeout is not None else None
        delay = 0.1
        while True:
            try:
                status = self._client.collections.status(collection)
            except (ServerError, TimeoutError):
                # Transient gateway/coordinator hiccup (5xx or timeout) while
                # polling for readiness. The upload already succeeded; readiness
                # is still pending, so a blip on the status poll must NOT abort
                # the wait (server-n7m5). Keep polling until the deadline; if the
                # error is persistent, the deadline check below converts it to
                # the actionable CollectionNotReadyError.
                if deadline is not None and time.time() >= deadline:
                    raise CollectionNotReadyError(
                        name=collection,
                        status="unknown",
                        message=(
                            f"Collection '{collection}' status could not be "
                            f"confirmed within {timeout:.0f}s of upload (the "
                            f"status endpoint kept erroring). Your data was "
                            f"uploaded; do NOT re-run upsert. Check "
                            f"vp.collections.status('{collection}')."
                        ),
                    ) from None
                time.sleep(delay)
                delay = min(delay * 2, 1.0)
                continue
            if status == _READY_STATUS:
                return
            if status in _TERMINAL_FAILURE_STATES:
                # Surface the real reason if available so the customer can act.
                col = self._client.collections.get(collection)
                reason = col.failure_reason or status
                raise ServerError(
                    f"Collection '{collection}' failed during ingest: {reason}"
                )
            if deadline is not None and time.time() >= deadline:
                raise CollectionNotReadyError(
                    name=collection,
                    status=status,
                    message=(
                        f"Collection '{collection}' did not become queryable "
                        f"within {timeout:.0f}s of upload (last status: "
                        f"'{status}'). Your data was uploaded successfully, but "
                        f"the ingest pipeline hasn't finished indexing it yet. "
                        f"Do NOT re-run upsert — that would create duplicate "
                        f"state. Instead:\n"
                        f"  - Call vp.collections.status('{collection}') to "
                        f"check current state.\n"
                        f"  - For unusually large datasets, pass "
                        f"wait_seconds=N to upsert() (or VP(upload_timeout=N)) "
                        f"to extend the wait.\n"
                        f"  - If the status stays '{status}' for more than "
                        f"~30s, contact support."
                    ),
                )
            time.sleep(delay)
            delay = min(delay * 2, 1.0)

    def _confirm_existing_upload(self, collection: str, filename: str) -> FileInfo | None:
        """Check whether `filename` is already committed to `collection`.

        server-t4d9 helper: used to disambiguate "the prior attempt's TCP
        connection died but the file actually committed" from "the user is
        trying to upsert a duplicate." Returns the FileInfo if present,
        else None. Swallows lookup errors so a flaky list endpoint can't
        turn a recoverable retry into a hard failure.
        """
        try:
            for f in self.list_files(collection):
                if f.name == filename:
                    return f
        except Exception:  # noqa: BLE001
            logger.debug("list_files probe failed during 409 soft-success check", exc_info=True)
        return None

    def _chunked_send_file(
        self,
        collection: str,
        path: Path,
        *,
        method: str,
        status_label: str,
    ) -> UploadResult:
        # server-cvms.2: chunked upload protocol. Replaces the legacy single-POST
        # path which buffered the whole file in client RAM (server-qnws) and
        # tripped source's body-size limit on multi-GB shards (server-3qga).
        # server-5pbm.1.2.3: shared by upsert (POST → source POST /files) and
        # replace (PUT → source PUT /files); only the verb recorded in the
        # upload session changes.
        filename = path.name
        file_size = path.stat().st_size

        start_resp = self._client._request(
            "POST",
            f"/api/v1/collections/{collection}/files/{filename}/uploads",
            json={"content_length": file_size, "method": method},
            accept_statuses=(201,),
            timeout=self._client.upload_timeout,
            retries=3,
        )

        start = start_resp.json()
        upload_id = start["upload_id"]
        chunk_size = min(_DEFAULT_CHUNK_SIZE, int(start.get("chunk_size_max") or _DEFAULT_CHUNK_SIZE))
        logger.info("  session=%s chunk_size=%d", upload_id, chunk_size)

        parts: list[dict] = []
        try:
            # PUT each chunk. Each chunk is read into a bytes object so
            # requests can replay it on transient retry without a seek dance.
            # Peak RAM ≈ chunk_size, regardless of file size.
            with open(path, "rb") as f:
                part_number = 0
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    part_number += 1
                    etag = hashlib.sha256(chunk).hexdigest()
                    self._client._request(
                        "PUT",
                        f"/api/v1/uploads/{upload_id}/parts/{part_number}",
                        data=chunk,
                        headers={
                            "Content-Type": "application/octet-stream",
                            "X-Content-Sha256": etag,
                        },
                        accept_statuses=(200,),
                        timeout=self._client.upload_timeout,
                        retries=3,
                    )
                    parts.append({"part_number": part_number, "etag": etag})
                    logger.info("  part %d uploaded (%d bytes)", part_number, len(chunk))

            complete_resp = self._client._request(
                "POST",
                f"/api/v1/uploads/{upload_id}/complete",
                json={"parts": parts},
                accept_statuses=(201,),
                timeout=self._client.upload_timeout,
                retries=3,
            )
            data = complete_resp.json()
            logger.info(
                "Uploaded '%s' (%d bytes, %d parts).",
                filename, data.get("size", file_size), len(parts),
            )
            return UploadResult(
                status=status_label,
                collection=data.get("collection", collection),
                filename=data.get("filename", filename),
                size=data.get("size", file_size),
            )
        except Exception:
            # Best-effort cleanup so a half-uploaded session doesn't camp on
            # consumer-site temp space until the 24h GC sweep.
            try:
                self._client._request(
                    "DELETE",
                    f"/api/v1/uploads/{upload_id}",
                    accept_statuses=(204,),
                )
            except Exception:  # noqa: BLE001
                logger.debug("upload session %s cleanup DELETE failed", upload_id)
            raise

    def _upsert_file(self, collection: str, path: Path) -> UploadResult:
        _validate_file(path)
        filename = path.name
        file_size = path.stat().st_size

        # server-dl7r: idempotency. If the same file (by name + size) is already
        # in the collection, short-circuit to a no-op success. Lets beginners
        # re-run the quickstart cell without a manual delete in between.
        # If the size differs we fall through and let the upload run; the server
        # will surface the conflict honestly.
        existing = self._confirm_existing_upload(collection, filename)
        if existing is not None and existing.size == file_size:
            logger.info(
                "File '%s' already present in '%s' at the same size (%d bytes). "
                "Treating as no-op success — no re-upload.",
                filename, collection, existing.size,
            )
            return UploadResult(
                status="created",
                collection=collection,
                filename=filename,
                size=existing.size,
            )

        logger.info(
            "Starting chunked upload of '%s' (%d bytes) → collection '%s'...",
            filename, file_size, collection,
        )

        # server-t4d9: if a previous attempt's TCP connection was torn down
        # mid-stream after the file had already committed server-side, the
        # retry will hit 409 FileAlreadyExistsError. Treat that as
        # soft-success so callers don't see a "failure" on data that's
        # actually present.
        try:
            return self._chunked_send_file(
                collection, path, method="POST", status_label="created",
            )
        except FileAlreadyExistsError:
            existing = self._confirm_existing_upload(collection, filename)
            if existing is not None:
                logger.info(
                    "Upload retry surfaced 409 for '%s' but the file is present "
                    "on the server (size=%d) — treating prior attempt as success.",
                    filename, existing.size,
                )
                return UploadResult(
                    status="created",
                    collection=collection,
                    filename=filename,
                    size=existing.size,
                )
            raise

    def _upsert_vectors(self, collection: str, vectors: list[dict[str, Any]]) -> UploadResult:
        if not vectors:
            raise ValidationError("vectors list cannot be empty.")

        for i, v in enumerate(vectors):
            if "vector" not in v:
                raise ValidationError(f"Vector at index {i} is missing 'vector' field.")

        # server-2ngs / server-6muc: mirror the row's id into metadata so
        # filter={"id": ...} works on inline-upsert rows the same way it
        # works on file-upload rows (where the source extractors include
        # id in the metadata payload). Without this, filters targeting
        # the id field silently no-op against WAL-resident rows because
        # id isn't in the metadata payload at all. Don't overwrite if
        # the caller already supplied it explicitly.
        normalized = []
        for v in vectors:
            if "id" in v:
                meta = dict(v.get("metadata") or {})
                meta.setdefault("id", v["id"])
                normalized.append({**v, "metadata": meta})
            else:
                normalized.append(v)

        logger.info("Upserting %d vectors to collection '%s'...", len(normalized), collection)

        resp = self._client._request(
            "POST",
            f"/api/v1/collections/{collection}/vectors",
            json={"vectors": normalized},
            accept_statuses=(200, 201, 202),
        )

        data = resp.json()
        logger.info("Upserted %d vectors.", data.get("vector_count", len(vectors)))
        return UploadResult(
            status=data.get("status", "processing"),
            collection=collection,
            filename=data.get("file", ""),
            size=data.get("vector_count", len(vectors)),
        )

    def _upsert_dataframe(self, collection: str, dataframe: Any) -> UploadResult:
        try:
            import pandas as pd
        except ImportError:
            raise ValidationError(
                "pandas is required for DataFrame upsert. "
                "Install it with: pip install veep[pandas]"
            ) from None

        if not isinstance(dataframe, pd.DataFrame):
            raise ValidationError("dataframe must be a pandas DataFrame.")
        if dataframe.empty:
            raise ValidationError("DataFrame is empty.")

        try:
            import pyarrow as pa
        except ImportError:
            raise ValidationError(
                "pyarrow is required for DataFrame upsert. "
                "Install it with: pip install veep[pandas]"
            ) from None

        table = pa.Table.from_pandas(dataframe, preserve_index=False)
        return self._upsert_arrow_table(collection, table, source_label="DataFrame")

    def _upsert_arrow_table(
        self, collection: str, table: Any, *, source_label: str = "Arrow Table",
    ) -> UploadResult:
        # server-a5iq: shared path for DataFrame (via from_pandas) and direct
        # pyarrow.Table inputs. Writes the table to a temp .parquet on local
        # disk, then hands off to the chunked file uploader. The temp file is
        # the simplest way to plug into the chunked-upload protocol without
        # re-implementing it for in-memory inputs; total RAM cost is bounded
        # by the table itself + one chunk during transmit.
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ValidationError(
                "pyarrow is required for Arrow Table upsert. "
                "Install it with: pip install pyarrow"
            ) from None

        if not isinstance(table, pa.Table):
            raise ValidationError("table must be a pyarrow.Table.")
        if table.num_rows == 0:
            raise ValidationError("table is empty.")

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        safe_name = f"_arrow_{int(__import__('time').time() * 1_000_000)}.parquet"
        tmp_path_renamed = tmp_path.parent / safe_name

        try:
            coerced = _coerce_vector_columns(table)
            pq.write_table(coerced, tmp_path)
            logger.info(
                "Serialized %s (%d rows) to temp Parquet for collection '%s'.",
                source_label, table.num_rows, collection,
            )
            tmp_path.rename(tmp_path_renamed)
            return self._upsert_file(collection, tmp_path_renamed)
        finally:
            for p in (tmp_path, tmp_path_renamed):
                try:
                    p.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass

    def replace(
        self,
        collection: str,
        file_path: str | Path,
    ) -> UploadResult:
        """Replace an existing file in a collection.

        Use this when you want to update vectors that were previously uploaded.

        Args:
            collection: Collection name.
            file_path: Path to the replacement file on disk.

        Returns:
            An UploadResult with replacement confirmation details.

        Raises:
            UploadError: If the file doesn't exist or can't be read.
            NotFoundError: If the file doesn't exist in the collection.
            AuthError: If your API key is invalid.
        """
        # server-5pbm.1.2.3: chunked protocol with method='PUT' so
        # consumer-site forwards the assembled body to source as PUT
        # (replace_file) instead of POST (add_file). No idempotency
        # short-circuit — the customer asked for a replace; honor it.
        path = Path(file_path)
        _validate_file(path)
        filename = path.name

        logger.info(
            "Replacing '%s' in '%s' via chunked upload (%d bytes)...",
            filename, collection, path.stat().st_size,
        )

        return self._chunked_send_file(
            collection, path, method="PUT", status_label="replaced",
        )

    def query(
        self,
        collection: str,
        vector: list[float],
        *,
        top_k: int = 10,
        min_score: float = -1.0,
        metric: str = "cosine",
        with_metadata: bool = True,
        with_vectors: bool = False,
        filter: dict[str, Any] | None = None,
        use_index: str | None = None,
        index_params: dict[str, Any] | None = None,
    ) -> QueryResults:
        """Search a collection by vector similarity.

        Args:
            collection: Collection name.
            vector: Query vector (list of floats matching the collection's dimension).
            top_k: Maximum number of results to return. Default 10.
            min_score: Minimum similarity score. Default -1.0 (no filter — return
                all results up to ``top_k`` in score order). Cosine similarity
                runs ``[-1, 1]`` (higher is more similar); raw dot product and
                euclidean run their natural ranges. Set to a positive number
                like 0.7 to keep only highly-similar results.
            metric: 'cosine' (default), 'euclidean', or 'dot_product'.
            with_metadata: Return metadata fields alongside each result. Default
                True. Pass with_metadata=False on hot query paths if you only
                need keys + scores and want a leaner response.
            with_vectors: Return the matched vector itself on each ``Result``.
                Default False. Set to True when you need the raw embeddings
                back — e.g., for client-side Maximal Marginal Relevance
                re-ranking. Adds ``top_k × dim × 4`` bytes to the response
                wire size; otherwise costs nothing. Requires server build
                with the ``include_vectors`` flag (server-sqve.2).
            filter: Metadata filter predicates. Example: ``{"category": "shoes"}``.
            use_index: Index strategy to use (e.g., 'pca', 'hnsw'). None for default.
            index_params: Flat dict of index-specific parameters. Requires ``use_index``.
                Example: ``index_params={"reduced_dimensions": 64}`` with
                ``use_index="pca"``.

        Returns:
            A QueryResults containing Result objects, iterable and indexable.

        Raises:
            ServerError: If Vector Panda is temporarily unavailable (5xx).
            TimeoutError: If the query takes too long.
            ValidationError: If ``index_params`` is set without ``use_index``.
            AuthError: If your API key is invalid.
        """
        if not vector:
            raise ValidationError("Query vector cannot be empty.")
        if metric not in ("cosine", "euclidean", "dot_product"):
            raise ValidationError(
                f"metric '{metric}' is not valid. "
                f"Choose 'cosine', 'euclidean', or 'dot_product'."
            )
        if index_params is not None and use_index is None:
            raise ValidationError(
                "index_params requires use_index. Pass use_index='hnsw' (or another "
                "strategy name) so the SDK knows which index the params target."
            )

        logger.info(
            "Querying collection '%s' (top_k=%d, metric=%s)...",
            collection,
            top_k,
            metric,
        )

        payload: dict[str, Any] = {
            "collection": collection,
            "vector": vector,
            "top_k": top_k,
            "similarity_threshold": min_score,
            "distance_metric": metric,
            "include_metadata": with_metadata,
        }
        if filter is not None:
            payload["filter"] = filter
            if not with_metadata:
                payload["include_metadata"] = True
        if use_index is not None:
            payload["use_index"] = use_index
        if index_params is not None:
            payload["index_params"] = {use_index: index_params}
        if with_vectors:
            payload["include_vectors"] = True

        # server-dl7r: when the collection is mid-ingest, the wire returns
        # 503 ("no current epoch") or 404 ("no workers serving"). Both mean
        # "still loading," not "broken" or "not found." Surface a clearly-
        # named exception so the customer's error is the actionable one.
        try:
            resp = self._client._request("POST", "/api/v1/query", json=payload)
        except (ServerError, CollectionNotFoundError, NotFoundError) as exc:
            self._maybe_raise_not_ready(collection, exc)
            raise
        data = resp.json()

        results = []
        for r in data.get("results", []):
            key, metadata = _unwrap_key(r)
            vector_field = r.get("vector") if with_vectors else None
            results.append(
                Result(
                    key=key,
                    score=r.get("score", 0.0),
                    metadata=metadata or {},
                    vector=vector_field,
                )
            )
        qr = QueryResults(results, worker_stats=data.get("worker_stats"))
        logger.info("Got %d result(s).", len(qr))
        return qr

    def query_batch(
        self,
        queries: list[dict[str, Any]],
    ) -> list[QueryResults]:
        """Execute multiple queries in a single request.

        Each query dict should have the same fields as the query() method:
        'collection', 'vector', and optional 'top_k', 'similarity_threshold', etc.

        Args:
            queries: A list of query dicts, each with at least 'collection' and 'vector'.

        Returns:
            A list of QueryResults, one per input query, in the same order.

        Raises:
            ValidationError: If queries is empty or exceeds 100 items.
            AuthError: If your API key is invalid.
        """
        if not queries:
            raise ValidationError("queries list cannot be empty.")
        if len(queries) > 100:
            raise ValidationError(
                f"Too many queries ({len(queries)}). Maximum is 100 per batch."
            )

        # server-5pbm.1.6 / server-jan0: vp.vectors.query() builds its payload
        # with explicit similarity_threshold=-1.0 (no filter, since cosine scores
        # are raw [-1, 1] not normalized 0-1), distance_metric="cosine",
        # include_metadata=True. The coord-side defaults differ
        # (similarity_threshold=0.7, include_metadata=False), and the
        # consumer-site /query/batch handler forwards each query dict through
        # unchanged. Without the same defaults baked in here, batch silently
        # drops sub-0.7 hits and returns hits without metadata. Apply the
        # single-query defaults to each dict so query_batch behaves identically
        # to N parallel query() calls.
        normalized = [
            {
                "similarity_threshold": -1.0,
                "distance_metric": "cosine",
                "include_metadata": True,
                **q,
            }
            for q in queries
        ]

        logger.info("Running batch of %d queries...", len(normalized))

        resp = self._client._request(
            "POST",
            "/api/v1/query/batch",
            json={"queries": normalized},
        )
        data = resp.json()

        batch_results = []
        for item in data.get("results", []):
            if item.get("status", 200) == 200 and "results" in item:
                results = []
                for r in item["results"]:
                    key, metadata = _unwrap_key(r)
                    results.append(Result(key=key, score=r.get("score", 0.0), metadata=metadata or {}))
                batch_results.append(QueryResults(results))
            else:
                batch_results.append(QueryResults([]))

        logger.info("Batch complete: %d queries processed.", len(batch_results))
        return batch_results

    def fetch(
        self,
        collection: str,
        key: str,
        *,
        with_metadata: bool = True,
    ) -> FetchResult:
        """Fetch a single vector by its key.

        The key is the identifier returned in query results (Result.key).

        Args:
            collection: Collection name.
            key: The vector key (as returned by a query's Result.key).
            with_metadata: Return metadata alongside the vector. Default True.

        Returns:
            A FetchResult with the vector data if found, or found=False.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            AuthError: If your API key is invalid.
        """
        logger.info("Fetching vector '%s' from collection '%s'...", key, collection)
        try:
            resp = self._client._request(
                "GET",
                f"/api/v1/collections/{collection}/vectors/{key}",
                params={"include_metadata": str(with_metadata).lower()},
            )
        except (CollectionNotFoundError, NotFoundError):
            # server-dl7r: a 404 here can mean two different things — the
            # collection genuinely doesn't exist, OR the collection exists
            # but isn't serving yet. Probe the list to decide, and surface
            # a clearly-named exception in either case rather than silently
            # returning an empty FetchResult that confuses callers downstream.
            try:
                cols = {c.name: c for c in self._client.collections.list()}
            except Exception:  # noqa: BLE001
                raise CollectionNotFoundError(collection) from None
            if collection in cols:
                col = cols[collection]
                if col.status != _READY_STATUS:
                    raise CollectionNotReadyError(
                        collection, status=col.status, suggested_wait_seconds=2.0,
                    ) from None
                # Collection is ready — the key really doesn't exist.
                return FetchResult(
                    key=key, found=False, vector=None, metadata=None,
                )
            raise CollectionNotFoundError(collection) from None

        data = resp.json()
        found = data.get("found", False)
        if found:
            logger.info("Vector '%s' found (%d dimensions).", key, len(data.get("vector", [])))
        else:
            logger.info("Vector '%s' not found.", key)
        # server-fxox: lift metadata.key_original back to the customer's key.
        canonical_key, metadata = _unwrap_key(data)
        return FetchResult(
            key=canonical_key or key,
            found=found,
            vector=data.get("vector"),
            metadata=metadata,
        )

    def delete(
        self,
        collection: str,
        filename: str | None = None,
        *,
        ids: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Delete vectors from a collection.

        Two shapes:

        1. **By file** -- ``vp.vectors.delete("col", "data.parquet")``. Removes the
           file and all vectors it contained. Returns None.
        2. **By id** -- ``vp.vectors.delete("col", ids=["k1", "k2"])``. Removes the
           specific vectors and returns a dict with ``deleted_count`` and
           ``files_modified``.

        Args:
            collection: Collection name.
            filename: Name of a file to delete (positional, mode 1).
            ids: List of vector IDs to delete (keyword-only, mode 2).

        Returns:
            None for file delete, dict for id delete.

        Raises:
            ValidationError: If neither or both of filename and ids are provided,
                or if filename has invalid characters, or ids is empty.
            NotFoundError: If the file does not exist (file mode).
            AuthError: If your API key is invalid.
        """
        if filename is None and ids is None:
            raise ValidationError(
                "Pass either filename (positional) or ids= (keyword) to delete."
            )
        if filename is not None and ids is not None:
            raise ValidationError(
                "Pass either filename or ids=, not both."
            )

        if filename is not None:
            if not VALID_FILENAME.match(filename):
                raise ValidationError(
                    f"Filename '{filename}' is invalid. "
                    f"Use only letters, numbers, dots, hyphens, and underscores."
                )
            logger.info("Deleting '%s' from collection '%s'...", filename, collection)
            self._client._request(
                "DELETE",
                f"/api/v1/collections/{collection}/files/{filename}",
            )
            logger.info("Deleted '%s'.", filename)
            return None

        if not ids:
            raise ValidationError("ids list cannot be empty.")

        logger.info("Deleting %d vectors from collection '%s'...", len(ids), collection)
        resp = self._client._request(
            "POST",
            f"/api/v1/collections/{collection}/vectors/delete",
            json={"ids": ids},
            accept_statuses=(200,),
        )
        data = resp.json()
        logger.info("Deleted %d vectors.", data.get("deleted_count", 0))
        return data

    def update(
        self,
        collection: str,
        vectors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Update existing vectors (delete old + write new).

        Args:
            collection: Collection name.
            vectors: List of vector dicts with 'id', 'vector', and optional 'metadata'.

        Returns:
            A dict with 'updated_count' and 'files_modified'.

        Raises:
            ValidationError: If vectors list is empty.
            AuthError: If your API key is invalid.
        """
        if not vectors:
            raise ValidationError("vectors list cannot be empty.")

        for i, v in enumerate(vectors):
            if "id" not in v:
                raise ValidationError(f"Vector at index {i} is missing 'id' field.")
            if "vector" not in v:
                raise ValidationError(f"Vector at index {i} is missing 'vector' field.")

        logger.info("Updating %d vectors in collection '%s'...", len(vectors), collection)

        resp = self._client._request(
            "POST",
            f"/api/v1/collections/{collection}/vectors/update",
            json={"vectors": vectors},
            accept_statuses=(200,),
        )

        data = resp.json()
        logger.info("Updated %d vectors.", data.get("updated_count", 0))
        return data

    def list_files(self, collection: str) -> list[FileInfo]:
        """List all files uploaded to a collection.

        Args:
            collection: Collection name.

        Returns:
            A list of FileInfo objects with name, size, and modification time.

        Raises:
            AuthError: If your API key is invalid.
        """
        logger.info("Listing files in collection '%s'...", collection)
        resp = self._client._request(
            "GET",
            f"/api/v1/collections/{collection}/files",
        )
        data = resp.json()
        files = [
            FileInfo(
                name=f["name"],
                size=f["size"],
                modified=f["modified"],
                # server-5pbm.1.5.3: WAL synthetic entry carries these fields;
                # they're absent on ordinary files and default to False/None.
                is_wal=bool(f.get("is_wal", False)),
                vector_count=f.get("vector_count"),
            )
            for f in data.get("files", [])
        ]
        logger.info("Found %d file(s) in '%s'.", len(files), collection)
        return files


def _validate_file(path: Path) -> None:
    if not path.exists():
        raise UploadError(
            f"File not found: {path}\n"
            f"Check the path and try again."
        )
    if not path.is_file():
        raise UploadError(
            f"'{path}' is not a file.\n"
            f"Pass the path to a file, not a directory."
        )
    if not VALID_FILENAME.match(path.name):
        raise ValidationError(
            f"Filename '{path.name}' contains invalid characters. "
            f"Use only letters, numbers, dots, hyphens, and underscores."
        )
    ext = path.suffix.lower().lstrip(".")
    if ext not in _SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
        raise ValidationError(
            f"Unsupported file extension '.{ext}' for '{path.name}'.\n"
            f"Supported: {supported}."
        )
    if path.stat().st_size == 0:
        raise UploadError(
            f"File '{path.name}' is empty (0 bytes)."
        )
