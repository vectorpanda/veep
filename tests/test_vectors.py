"""Tests for vector operations."""

import pytest
import responses

from veep import VP
from veep.exceptions import (
    CollectionNotReadyError,
    FileAlreadyExistsError,
    NotFoundError,
    QueryError,
    TimeoutError,
    UploadError,
    ValidationError,
)
from veep.models import FetchResult, FileInfo, UploadResult

HOST = "http://localhost:3000"


def make_client():
    return VP(api_key="test_key", host=HOST)


# -- query --


# server-fxox: SDK lifts metadata.key_original back to r.key.


@responses.activate
def test_query_promotes_metadata_key_original_to_result_key():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/query",
        json={
            "results": [
                {
                    "key": "113633879664880909310655119435976674117",
                    "score": 1.0,
                    "metadata": {"key_original": "0704.3024", "title": "Are extrasolar oceans common"},
                },
            ],
            "worker_stats": [],
        },
        status=200,
    )
    c = make_client()
    results = c.vectors.query("arxiv", vector=[0.1] * 384, with_metadata=True)
    assert len(results) == 1
    assert results[0].key == "0704.3024"  # promoted from metadata.key_original
    assert "key_original" not in results[0].metadata  # duplicate dropped
    assert results[0].metadata["title"] == "Are extrasolar oceans common"


@responses.activate
def test_fetch_promotes_metadata_key_original_to_result_key():
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/arxiv/vectors/0704.3024",
        json={
            "key": "0704.3024",
            "found": True,
            "vector": [0.1] * 384,
            "metadata": {"key_original": "0704.3024", "title": "Test"},
        },
        status=200,
    )
    c = make_client()
    result = c.vectors.fetch("arxiv", "0704.3024")
    assert result.found is True
    assert result.key == "0704.3024"
    assert "key_original" not in result.metadata
    assert result.metadata["title"] == "Test"


@responses.activate
def test_query_without_key_original_keeps_wire_key():
    """Backwards-compat: results from older coord versions or collections
    that don't carry key_original metadata still produce r.key from the
    wire's key field."""
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/query",
        json={
            "results": [{"key": "abc", "score": 0.5, "metadata": {"title": "no key_original here"}}],
            "worker_stats": [],
        },
        status=200,
    )
    c = make_client()
    results = c.vectors.query("col", vector=[0.1])
    assert results[0].key == "abc"
    assert results[0].metadata["title"] == "no key_original here"


@responses.activate
def test_query_returns_results():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/query",
        json={
            "results": [
                {"key": "vec_1", "score": 0.95, "metadata": {}},
                {"key": "vec_2", "score": 0.88, "metadata": {"label": "cat"}},
            ],
            "worker_stats": [],
        },
        status=200,
    )
    c = make_client()
    results = c.vectors.query("my_collection", vector=[0.1, 0.2, 0.3])
    assert len(results) == 2
    assert results[0].key == "vec_1"
    assert results[0].score == 0.95
    assert results[1].metadata == {"label": "cat"}


@responses.activate
def test_query_with_options():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/query",
        json={"results": [{"key": "v1", "score": 0.9}], "worker_stats": []},
        status=200,
    )
    c = make_client()
    results = c.vectors.query(
        "col",
        vector=[0.1],
        top_k=5,
        min_score=0.5,
        metric="dot_product",
        with_metadata=True,
        use_index="pca",
        index_params={"reduced_dimensions": 64},
    )
    assert len(results) == 1

    body = responses.calls[0].request.body
    import json
    sent = json.loads(body)
    assert sent["top_k"] == 5
    assert sent["use_index"] == "pca"
    assert sent["include_metadata"] is True   # SDK 'with_metadata' -> wire 'include_metadata'
    assert sent["distance_metric"] == "dot_product"  # SDK 'metric' -> wire 'distance_metric'
    assert sent["similarity_threshold"] == 0.5       # SDK 'min_score'   -> wire 'similarity_threshold'
    assert sent["index_params"] == {"pca": {"reduced_dimensions": 64}}  # SDK flat -> wire wrapped


def test_query_index_params_without_use_index_raises():
    c = make_client()
    with pytest.raises(ValidationError, match="index_params requires use_index"):
        c.vectors.query("col", vector=[0.1], index_params={"reduced_dimensions": 64})


@responses.activate
def test_query_empty_results():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/query",
        json={"results": [], "worker_stats": []},
        status=200,
    )
    c = make_client()
    results = c.vectors.query("col", vector=[0.1])
    assert len(results) == 0
    assert not results


def test_query_empty_vector():
    c = make_client()
    with pytest.raises(ValidationError, match="cannot be empty"):
        c.vectors.query("col", vector=[])


def test_query_invalid_metric():
    c = make_client()
    with pytest.raises(ValidationError, match="not valid"):
        c.vectors.query("col", vector=[0.1], metric="manhattan")


# server-dl7r: not-ready surfacing.


@responses.activate
def test_query_against_not_ready_collection_raises_not_ready():
    """Wire returns 503 'no current epoch' when ingest is in flight.
    SDK probes collections.list, sees status != 'ready', and raises
    CollectionNotReadyError with a human-friendly message."""
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/query",
        json={"error": "Collection test_client/arxiv has no current epoch (still loading or never distributed)"},
        status=503,
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections",
        json={"collections": [
            {"name": "arxiv", "tier": "hot", "is_active": True, "status": "awaiting_artifacts"},
        ]},
        status=200,
    )
    c = make_client()
    with pytest.raises(CollectionNotReadyError) as exc_info:
        c.vectors.query("arxiv", vector=[0.1] * 384)
    assert exc_info.value.collection_name == "arxiv"
    assert exc_info.value.status == "awaiting_artifacts"
    assert "still being prepared" in str(exc_info.value)


@responses.activate
def test_fetch_against_not_ready_collection_raises_not_ready():
    """Wire returns 404 'no workers serving' during ingest. Without the
    not-ready probe this would silently return FetchResult(found=False)
    and a downstream query would error misleadingly."""
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/arxiv/vectors/abc",
        json={"error": "No workers serving collection arxiv"},
        status=404,
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections",
        json={"collections": [
            {"name": "arxiv", "tier": "hot", "is_active": True, "status": "updating"},
        ]},
        status=200,
    )
    c = make_client()
    with pytest.raises(CollectionNotReadyError) as exc_info:
        c.vectors.fetch("arxiv", "abc")
    assert exc_info.value.status == "updating"


@responses.activate
def test_fetch_against_ready_collection_with_missing_key_returns_not_found():
    """If the collection IS ready and the key really doesn't exist, fetch
    returns FetchResult(found=False) so callers can distinguish 'no such
    vector' from 'collection not ready'."""
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/arxiv/vectors/missing",
        json={"error": "Vector not found"},
        status=404,
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections",
        json={"collections": [
            {"name": "arxiv", "tier": "hot", "is_active": True, "status": "ready"},
        ]},
        status=200,
    )
    c = make_client()
    result = c.vectors.fetch("arxiv", "missing")
    assert result.found is False
    assert result.vector is None


@responses.activate
def test_upsert_blocks_until_ready(tmp_path, monkeypatch):
    """upsert polls collections.status until 'ready' before returning,
    so customers get a queryable collection on call return."""
    f = tmp_path / "data.parquet"
    f.write_bytes(b"x" * 32)

    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/c/files/data.parquet/uploads",
        json={"upload_id": "up_99"},
        status=201,
    )
    responses.add(
        responses.PUT,
        f"{HOST}/api/v1/uploads/up_99/parts/1",
        json={"ok": True},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/uploads/up_99/complete",
        json={"collection": "c", "filename": "data.parquet", "size": 32},
        status=201,
    )
    # Two not-ready probes, then ready.
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/c/status",
        json={"status": "awaiting_artifacts"},
        status=200,
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/c/status",
        json={"status": "updating"},
        status=200,
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/c/status",
        json={"status": "ready"},
        status=200,
    )

    monkeypatch.setattr("veep.vectors.time.sleep", lambda *_: None)
    c = make_client()
    result = c.vectors.upsert("c", f)
    assert result.size == 32
    status_calls = [call for call in responses.calls if "/status" in call.request.url]
    assert len(status_calls) == 3


@responses.activate
def test_query_502_raises_query_error():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/query",
        json={"error": "Query service unavailable"},
        status=502,
    )
    c = make_client()
    with pytest.raises(QueryError, match="unavailable"):
        c.vectors.query("col", vector=[0.1])


@responses.activate
def test_query_504_raises_timeout():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/query",
        json={"error": "Query timed out"},
        status=504,
    )
    c = make_client()
    with pytest.raises(TimeoutError, match="too long"):
        c.vectors.query("col", vector=[0.1])


# -- query_batch --


@responses.activate
def test_query_batch():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/query/batch",
        json={
            "results": [
                {"status": 200, "results": [{"key": "a", "score": 0.9}]},
                {"status": 200, "results": [{"key": "b", "score": 0.8}]},
            ]
        },
        status=200,
    )
    c = make_client()
    batch = c.vectors.query_batch([
        {"collection": "col", "vector": [0.1]},
        {"collection": "col", "vector": [0.2]},
    ])
    assert len(batch) == 2
    assert batch[0][0].key == "a"
    assert batch[1][0].key == "b"


@responses.activate
def test_query_batch_with_failure():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/query/batch",
        json={
            "results": [
                {"status": 200, "results": [{"key": "a", "score": 0.9}]},
                {"status": 502, "error": "unavailable"},
            ]
        },
        status=200,
    )
    c = make_client()
    batch = c.vectors.query_batch([
        {"collection": "col", "vector": [0.1]},
        {"collection": "col", "vector": [0.2]},
    ])
    assert len(batch) == 2
    assert len(batch[0]) == 1
    assert len(batch[1]) == 0


def test_query_batch_empty():
    c = make_client()
    with pytest.raises(ValidationError, match="cannot be empty"):
        c.vectors.query_batch([])


def test_query_batch_too_many():
    c = make_client()
    with pytest.raises(ValidationError, match="Maximum is 100"):
        c.vectors.query_batch([{"collection": "c", "vector": [0.1]}] * 101)


# -- upsert (chunked protocol, server-cvms.2) --
# Three endpoints per upload:
#   1. POST   /collections/{col}/files/{file}/uploads     -> 201 {upload_id, chunk_size_max?}
#   2. PUT    /uploads/{upload_id}/parts/{N}              -> 200 (per chunk; X-Content-Sha256 header)
#   3. POST   /uploads/{upload_id}/complete               -> 201 {collection, filename, size}


@responses.activate
def test_upsert(tmp_path):
    f = tmp_path / "data.parquet"
    f.write_bytes(b"fake parquet data")  # 17 bytes — fits in one chunk

    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/products/files/data.parquet/uploads",
        json={"upload_id": "up_123", "chunk_size_max": 67108864},
        status=201,
    )
    responses.add(
        responses.PUT,
        f"{HOST}/api/v1/uploads/up_123/parts/1",
        json={"ok": True},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/uploads/up_123/complete",
        json={"status": "created", "collection": "products", "filename": "data.parquet", "size": 17},
        status=201,
    )
    # server-dl7r: upsert blocks until status='ready'. Mock the status probe.
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/products/status",
        json={"collection": "products", "status": "ready"},
        status=200,
    )

    c = make_client()
    result = c.vectors.upsert("products", f)
    assert isinstance(result, UploadResult)
    assert result.status == "created"
    assert result.filename == "data.parquet"
    assert result.size == 17


@responses.activate
def test_upsert_chunk_carries_sha256_header(tmp_path):
    """The PUT for each chunk must carry X-Content-Sha256 (server-cvms.2)
    so the server can verify each chunk and replay-safe retry."""
    import hashlib
    payload = b"test content"
    expected_etag = hashlib.sha256(payload).hexdigest()

    f = tmp_path / "data.parquet"
    f.write_bytes(payload)

    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/col/files/data.parquet/uploads",
        json={"upload_id": "up_abc"},
        status=201,
    )
    responses.add(
        responses.PUT,
        f"{HOST}/api/v1/uploads/up_abc/parts/1",
        json={"ok": True},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/uploads/up_abc/complete",
        json={"collection": "col", "filename": "data.parquet", "size": len(payload)},
        status=201,
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/col/status",
        json={"collection": "col", "status": "ready"},
        status=200,
    )

    c = make_client()
    c.vectors.upsert("col", f)

    # Find the PUT-parts call by method (its index varies as the upsert flow grows).
    chunk_call = next(call for call in responses.calls if call.request.method == "PUT")
    assert "/uploads/up_abc/parts/1" in chunk_call.request.url
    assert chunk_call.request.headers["X-Content-Sha256"] == expected_etag
    assert chunk_call.request.headers["Content-Type"] == "application/octet-stream"
    assert chunk_call.request.body == payload


def test_upsert_file_not_found(tmp_path):
    c = make_client()
    with pytest.raises(UploadError, match="not found"):
        c.vectors.upsert("col", tmp_path / "nonexistent.parquet")


def test_upsert_directory(tmp_path):
    c = make_client()
    with pytest.raises(UploadError, match="not a file"):
        c.vectors.upsert("col", tmp_path)


@responses.activate
def test_upsert_idempotent_when_same_file_already_present(tmp_path):
    """server-dl7r: re-running upsert for a file that already exists at
    the same size short-circuits to a no-op success — no re-upload, no
    409, no 'use a different filename or DELETE first' confusion. Lets
    beginners re-run the quickstart cell without any manual cleanup."""
    f = tmp_path / "data.parquet"
    f.write_bytes(b"data" * 4)  # 16 bytes

    # list_files probe says the file is already there at the same size.
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/col/files",
        json={
            "collection": "col",
            "files": [
                {"name": "data.parquet", "size": 16, "modified": "2026-01-01T00:00:00Z"},
            ],
        },
        status=200,
    )
    # Status probe for the wait-until-ready loop.
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/col/status",
        json={"status": "ready"},
        status=200,
    )

    c = make_client()
    result = c.vectors.upsert("col", f)
    assert result.status == "created"
    assert result.size == 16

    # No POST /uploads (start session) should have happened.
    starts = [call for call in responses.calls if "/uploads" in call.request.url and call.request.method == "POST"]
    assert starts == [], "upsert should short-circuit; no upload-start call expected"


@responses.activate
def test_upsert_file_already_exists_hard_fail(tmp_path):
    """Start-session 409 + the file is NOT in list_files: raise."""
    f = tmp_path / "data.parquet"
    f.write_bytes(b"data")

    # Start returns 409 — file already exists per the server.
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/col/files/data.parquet/uploads",
        json={"error": "File already exists. Use PUT to replace."},
        status=409,
    )
    # SDK probes list_files to confirm — file is NOT actually present, so re-raise.
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/col/files",
        json={"collection": "col", "files": []},
        status=200,
    )

    c = make_client()
    with pytest.raises(FileAlreadyExistsError, match="already exists"):
        c.vectors.upsert("col", f)


@responses.activate
def test_upsert_file_already_exists_soft_success(tmp_path):
    """Start-session 409 + the file IS already present: soft-success
    (server-t4d9). Covers the case where a previous attempt's TCP
    connection died after the file had committed server-side."""
    f = tmp_path / "data.parquet"
    f.write_bytes(b"data")

    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/col/files/data.parquet/uploads",
        json={"error": "File already exists. Use PUT to replace."},
        status=409,
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/col/files",
        json={
            "collection": "col",
            "files": [
                {"name": "data.parquet", "size": 4, "modified": "2026-01-01T00:00:00Z"},
            ],
        },
        status=200,
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/col/status",
        json={"collection": "col", "status": "ready"},
        status=200,
    )

    c = make_client()
    result = c.vectors.upsert("col", f)
    assert result.status == "created"
    assert result.filename == "data.parquet"
    assert result.size == 4


# -- replace --


@responses.activate
def test_replace(tmp_path):
    f = tmp_path / "data.parquet"
    f.write_bytes(b"updated content")

    responses.add(
        responses.PUT,
        f"{HOST}/api/v1/collections/col/files/data.parquet",
        json={"status": "replaced", "collection": "col", "filename": "data.parquet", "size": 15},
        status=200,
    )
    c = make_client()
    result = c.vectors.replace("col", f)
    assert result.status == "replaced"


@responses.activate
def test_replace_not_found(tmp_path):
    f = tmp_path / "data.parquet"
    f.write_bytes(b"data")

    responses.add(
        responses.PUT,
        f"{HOST}/api/v1/collections/col/files/data.parquet",
        json={"error": "File not found"},
        status=404,
    )
    c = make_client()
    with pytest.raises(NotFoundError):
        c.vectors.replace("col", f)


# -- delete --


@responses.activate
def test_delete_file():
    responses.add(
        responses.DELETE,
        f"{HOST}/api/v1/collections/col/files/data.parquet",
        json={"status": "deleted", "collection": "col", "filename": "data.parquet"},
        status=200,
    )
    c = make_client()
    c.vectors.delete("col", "data.parquet")


@responses.activate
def test_delete_file_not_found():
    responses.add(
        responses.DELETE,
        f"{HOST}/api/v1/collections/col/files/missing.parquet",
        json={"error": "File not found"},
        status=404,
    )
    c = make_client()
    with pytest.raises(NotFoundError):
        c.vectors.delete("col", "missing.parquet")


def test_delete_invalid_filename():
    c = make_client()
    with pytest.raises(ValidationError, match="invalid"):
        c.vectors.delete("col", "bad file name!")


@responses.activate
def test_delete_by_ids():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/col/vectors/delete",
        json={"deleted_count": 2, "files_modified": ["data.parquet"]},
        status=200,
    )
    c = make_client()
    result = c.vectors.delete("col", ids=["k1", "k2"])
    assert result["deleted_count"] == 2
    assert result["files_modified"] == ["data.parquet"]


def test_delete_requires_filename_or_ids():
    c = make_client()
    with pytest.raises(ValidationError, match="filename .* or ids"):
        c.vectors.delete("col")


def test_delete_rejects_both_filename_and_ids():
    c = make_client()
    with pytest.raises(ValidationError, match="not both"):
        c.vectors.delete("col", "data.parquet", ids=["k1"])


def test_delete_empty_ids_list():
    c = make_client()
    with pytest.raises(ValidationError, match="ids list cannot be empty"):
        c.vectors.delete("col", ids=[])


# -- list_files --


@responses.activate
def test_list_files():
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/col/files",
        json={
            "collection": "col",
            "files": [
                {"name": "data.parquet", "size": 1024, "modified": "2026-01-01T00:00:00Z"},
                {"name": "more.parquet", "size": 2048, "modified": "2026-01-02T00:00:00Z"},
            ],
        },
        status=200,
    )
    c = make_client()
    files = c.vectors.list_files("col")
    assert len(files) == 2
    assert files[0].name == "data.parquet"
    assert files[0].size == 1024
    assert isinstance(files[0], FileInfo)


# -- fetch --


@responses.activate
def test_fetch_found():
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/col/vectors/12345",
        json={
            "key": "12345",
            "found": True,
            "vector": [0.1, 0.2, 0.3],
            "metadata": {"label": "cat"},
        },
        status=200,
    )
    c = make_client()
    result = c.vectors.fetch("col", "12345")
    assert isinstance(result, FetchResult)
    assert result.found is True
    assert result.vector == [0.1, 0.2, 0.3]
    assert result.metadata == {"label": "cat"}


@responses.activate
def test_fetch_not_found():
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/col/vectors/99999",
        json={"key": "99999", "found": False},
        status=200,
    )
    c = make_client()
    result = c.vectors.fetch("col", "99999")
    assert result.found is False
    assert result.vector is None


@responses.activate
def test_fetch_without_metadata():
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/col/vectors/12345",
        json={"key": "12345", "found": True, "vector": [0.1, 0.2]},
        status=200,
    )
    c = make_client()
    result = c.vectors.fetch("col", "12345", with_metadata=False)
    assert result.found is True
    assert "include_metadata=false" in responses.calls[0].request.url


@responses.activate
def test_list_files_empty():
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/col/files",
        json={"collection": "col", "files": []},
        status=200,
    )
    c = make_client()
    files = c.vectors.list_files("col")
    assert files == []
