"""Tests for collection management."""

import json
from pathlib import Path

import pytest
import responses

from veep import VP, Collection, ExportResult
from veep.exceptions import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    CollectionRecentlyDeletedError,
    ServerError,
    ValidationError,
)

HOST = "http://localhost:3000"


def make_client():
    return VP(api_key="test_key", host=HOST)


@responses.activate
def test_create_collection():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections",
        json={"status": "created", "client_id": "c1", "collection": "products", "tier": "hot"},
        status=201,
    )
    c = make_client()
    col = c.collections.create("products")
    assert isinstance(col, Collection)
    assert col.name == "products"
    assert col.tier == "hot"


@responses.activate
def test_create_collection_warm_tier():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections",
        json={"status": "created", "client_id": "c1", "collection": "archive", "tier": "warm"},
        status=201,
    )
    c = make_client()
    col = c.collections.create("archive", tier="warm")
    assert col.tier == "warm"


def test_create_collection_invalid_name():
    c = make_client()
    with pytest.raises(ValidationError, match="invalid"):
        c.collections.create("bad name!")


def test_create_collection_invalid_tier():
    c = make_client()
    with pytest.raises(ValidationError, match="not valid"):
        c.collections.create("ok_name", tier="blazing")


@responses.activate
def test_create_collection_with_target_recall():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections",
        json={"status": "created", "client_id": "c1", "collection": "products", "tier": "hot"},
        status=201,
    )
    c = make_client()
    c.collections.create("products", target_recall=0.99)
    body = json.loads(responses.calls[0].request.body)
    assert body["target_recall"] == 0.99


def test_create_collection_target_recall_out_of_range():
    c = make_client()
    with pytest.raises(ValidationError, match="target_recall"):
        c.collections.create("ok_name", target_recall=0.3)
    with pytest.raises(ValidationError, match="target_recall"):
        c.collections.create("ok_name", target_recall=1.0)


@responses.activate
def test_update_collection_target_recall():
    responses.add(
        responses.PATCH,
        f"{HOST}/api/v1/collections/products",
        json={"status": "updated", "collection": "products", "target_recall": 0.9},
        status=200,
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/products",
        json={"name": "products", "tier": "hot", "status": "ready", "target_recall": 0.9},
        status=200,
    )
    c = make_client()
    col = c.collections.update("products", target_recall=0.9)
    body = json.loads(responses.calls[0].request.body)
    assert body == {"target_recall": 0.9}
    assert col.target_recall == 0.9


def test_update_collection_target_recall_out_of_range():
    c = make_client()
    with pytest.raises(ValidationError, match="target_recall"):
        c.collections.update("products", target_recall=0.2)


@responses.activate
def test_create_collection_with_schema():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections",
        json={
            "status": "created",
            "client_id": "c1",
            "collection": "products",
            "tier": "hot",
            "schema_confirmed": True,
        },
        status=201,
    )
    c = make_client()
    col = c.collections.create(
        "products",
        id_field="product_id",
        vector_field="embedding",
        dimension=384,
    )
    assert col.name == "products"
    assert col.dimension == 384

    import json
    body = json.loads(responses.calls[0].request.body)
    assert body["id_field"] == "product_id"
    assert body["vector_field"] == "embedding"
    assert body["dimension"] == 384


@responses.activate
def test_create_collection_with_schema_and_format():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections",
        json={"status": "created", "client_id": "c1", "collection": "docs", "tier": "warm", "schema_confirmed": True},
        status=201,
    )
    c = make_client()
    col = c.collections.create(
        "docs",
        tier="warm",
        id_field="doc_id",
        vector_field="emb",
        format="parquet",
    )
    assert col.tier == "warm"

    import json
    body = json.loads(responses.calls[0].request.body)
    assert body["format"] == "parquet"


def test_create_collection_schema_requires_both_fields():
    c = make_client()
    with pytest.raises(ValidationError, match="both"):
        c.collections.create("test", id_field="id")
    with pytest.raises(ValidationError, match="both"):
        c.collections.create("test", vector_field="emb")


@responses.activate
def test_create_collection_already_exists_returns_existing():
    """Default if_exists='ignore': create() catches 409 from POST and falls
    through to self.get(name), returning the existing Collection. This is
    the friendly default — re-running a setup script doesn't fail."""
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections",
        json={"error": "Collection already exists"},
        status=409,
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/products",
        json={
            "name": "products",
            "tier": "hot",
            "is_active": True,
            "vector_count": 5000,
            "storage_gb": 1.2,
            "status": "ready",
            "dimension": 384,
        },
        status=200,
    )
    c = make_client()
    col = c.collections.create("products")
    assert isinstance(col, Collection)
    assert col.name == "products"
    assert col.vector_count == 5000
    assert col.status == "ready"


@responses.activate
def test_create_collection_already_exists_if_exists_error_raises():
    """if_exists='error' is the strict opt-in: a 409 from the server
    surfaces as CollectionAlreadyExistsError instead of being swallowed."""
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections",
        json={"error": "Collection already exists"},
        status=409,
    )
    c = make_client()
    with pytest.raises(CollectionAlreadyExistsError):
        c.collections.create("products", if_exists="error")


@responses.activate
def test_create_collection_if_exists_replace_sends_force_destroy():
    """if_exists='replace' sends force_destroy=true in the create body. Coord
    UPSERTs the row and clears any post-delete cooldown in one call — the SDK
    no longer needs a separate delete + create round-trip (server-or56)."""
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections",
        json={"status": "created", "client_id": "c1", "collection": "products", "tier": "hot"},
        status=201,
    )
    c = make_client()
    col = c.collections.create("products", if_exists="replace")
    assert isinstance(col, Collection)
    assert col.name == "products"
    assert col.tier == "hot"
    # Single POST with force_destroy in the body — no DELETE step
    assert len(responses.calls) == 1
    assert responses.calls[0].request.method == "POST"
    body = json.loads(responses.calls[0].request.body)
    assert body.get("force_destroy") is True
    assert body.get("collection") == "products"


@responses.activate
def test_create_collection_recently_deleted_default_raises():
    """Default if_exists='ignore' raises CollectionRecentlyDeletedError on a
    post-delete cooldown — there's no existing collection to return, so the
    only sensible defaults are wait-and-retry or use if_exists='replace'."""
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections",
        json={
            "error": "Collection 'products' was recently deleted",
            "code": "recently_deleted",
            "collection": "products",
            "retry_after_secs": 42,
        },
        status=409,
    )
    c = make_client()
    with pytest.raises(CollectionRecentlyDeletedError) as exc_info:
        c.collections.create("products")
    assert exc_info.value.collection_name == "products"
    assert exc_info.value.retry_after_secs == 42


@responses.activate
def test_create_collection_recently_deleted_if_exists_error_raises():
    """if_exists='error' also raises CollectionRecentlyDeletedError — strict
    mode surfaces both gates (already_exists AND recently_deleted)."""
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections",
        json={
            "error": "Collection 'products' was recently deleted",
            "code": "recently_deleted",
            "collection": "products",
            "retry_after_secs": 30,
        },
        status=409,
    )
    c = make_client()
    with pytest.raises(CollectionRecentlyDeletedError):
        c.collections.create("products", if_exists="error")


@responses.activate
def test_create_collection_recently_deleted_if_exists_replace_bypasses():
    """if_exists='replace' sends force_destroy=true; coord clears the cooldown
    and creates immediately. The SDK never sees the recently_deleted 409."""
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections",
        json={"status": "created", "client_id": "c1", "collection": "products", "tier": "hot"},
        status=201,
    )
    c = make_client()
    col = c.collections.create("products", if_exists="replace")
    assert col.name == "products"
    assert len(responses.calls) == 1
    body = json.loads(responses.calls[0].request.body)
    assert body.get("force_destroy") is True


def test_create_collection_invalid_if_exists_value():
    """if_exists must be one of the three allowed values."""
    c = make_client()
    with pytest.raises(ValidationError):
        c.collections.create("products", if_exists="overwrite")


@responses.activate
def test_list_collections():
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections",
        json={
            "collections": [
                {
                    "name": "products",
                    "tier": "hot",
                    "is_active": True,
                    "vector_count": 5000,
                    "storage_gb": 1.2,
                },
                {
                    "name": "docs",
                    "tier": "warm",
                    "is_active": False,
                    "vector_count": 100,
                    "storage_gb": 0.1,
                },
            ]
        },
        status=200,
    )
    c = make_client()
    cols = c.collections.list()
    assert len(cols) == 2
    assert cols[0].name == "products"
    assert cols[0].vector_count == 5000
    assert cols[1].name == "docs"
    assert cols[1].is_active is False


@responses.activate
def test_list_collections_empty():
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections",
        json={"collections": []},
        status=200,
    )
    c = make_client()
    cols = c.collections.list()
    assert cols == []


@responses.activate
def test_get_collection():
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/products",
        json={
            "name": "products",
            "tier": "hot",
            "is_active": True,
            "vector_count": 5000,
            "storage_gb": 1.2,
            "status": "ready",
            "dimension": 384,
        },
        status=200,
    )
    c = make_client()
    col = c.collections.get("products")
    assert col.name == "products"
    assert col.dimension == 384
    assert col.status == "ready"


@responses.activate
def test_get_collection_not_found():
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/nope",
        json={"error": "Collection not found"},
        status=404,
    )
    c = make_client()
    with pytest.raises(CollectionNotFoundError, match="nope"):
        c.collections.get("nope")


@responses.activate
def test_delete_collection():
    responses.add(
        responses.DELETE,
        f"{HOST}/api/v1/collections/products",
        json={"status": "deleted", "client_id": "c1", "collection": "products"},
        status=200,
    )
    c = make_client()
    c.collections.delete("products")


@responses.activate
def test_status_collection():
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/products/status",
        json={"collection": "products", "status": "ready"},
        status=200,
    )
    c = make_client()
    s = c.collections.status("products")
    assert s == "ready"


@responses.activate
def test_status_collection_not_found():
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/nope/status",
        json={"error": "Collection not found"},
        status=404,
    )
    c = make_client()
    with pytest.raises(CollectionNotFoundError):
        c.collections.status("nope")


# --- export -----------------------------------------------------------


@responses.activate
def test_export_nowait_returns_job_id():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/products/export-jobs",
        json={"job_id": "job-abc"},
        status=202,
    )
    c = make_client()
    result = c.collections.export("products", "/tmp/ignored", wait=False)
    assert isinstance(result, ExportResult)
    assert result.job_id == "job-abc"
    assert result.path is None
    assert result.parts == 0
    body = json.loads(responses.calls[0].request.body)
    assert body == {}


@responses.activate
def test_export_send_email_true_passes_through():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/products/export-jobs",
        json={"job_id": "j1"},
        status=202,
    )
    c = make_client()
    c.collections.export("products", "/tmp/ignored", wait=False, send_email=True)
    body = json.loads(responses.calls[0].request.body)
    assert body == {"send_email": True}


@responses.activate
def test_export_send_email_address_passes_through():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/products/export-jobs",
        json={"job_id": "j1"},
        status=202,
    )
    c = make_client()
    c.collections.export(
        "products", "/tmp/ignored", wait=False, send_email="user@example.com"
    )
    body = json.loads(responses.calls[0].request.body)
    assert body == {"send_email": "user@example.com"}


@responses.activate
def test_export_input_dim_passes_through():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/products/export-jobs",
        json={"job_id": "j1"},
        status=202,
    )
    c = make_client()
    c.collections.export("products", "/tmp/ignored", wait=False, input_dim=384)
    body = json.loads(responses.calls[0].request.body)
    assert body == {"input_dim": 384}


def test_export_invalid_name():
    c = make_client()
    with pytest.raises(ValidationError, match="invalid"):
        c.collections.export("bad name!", "/tmp/x", wait=False)


def test_export_invalid_send_email():
    c = make_client()
    with pytest.raises(ValidationError, match="send_email"):
        c.collections.export("ok", "/tmp/x", wait=False, send_email=123)
    with pytest.raises(ValidationError, match="send_email"):
        c.collections.export("ok", "/tmp/x", wait=False, send_email="")


def test_export_invalid_input_dim():
    c = make_client()
    with pytest.raises(ValidationError, match="input_dim"):
        c.collections.export("ok", "/tmp/x", wait=False, input_dim=0)
    with pytest.raises(ValidationError, match="input_dim"):
        c.collections.export("ok", "/tmp/x", wait=False, input_dim=-5)


def test_export_invalid_poll_interval():
    c = make_client()
    with pytest.raises(ValidationError, match="poll_interval"):
        c.collections.export("ok", "/tmp/x", wait=False, poll_interval_s=0)


@responses.activate
def test_export_failed_status_raises_server_error():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/products/export-jobs",
        json={"job_id": "j2"},
        status=202,
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/products/export-jobs/j2",
        json={
            "job_id": "j2",
            "client_id": "c1",
            "collection_name": "products",
            "status": "failed",
            "error": "ran out of disk",
            "parts_written": 0,
            "total_bytes": 0,
        },
        status=200,
    )
    c = make_client()
    with pytest.raises(ServerError, match="ran out of disk"):
        c.collections.export("products", "/tmp/ignored", poll_interval_s=0.01)


@responses.activate
def test_export_wait_downloads_parts(tmp_path: Path):
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/products/export-jobs",
        json={"job_id": "jx"},
        status=202,
    )
    # First poll: running. Second poll: complete.
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/products/export-jobs/jx",
        json={
            "job_id": "jx", "client_id": "c1", "collection_name": "products",
            "status": "running", "parts_written": 0, "total_bytes": 0,
        },
        status=200,
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/products/export-jobs/jx",
        json={
            "job_id": "jx", "client_id": "c1", "collection_name": "products",
            "status": "complete", "parts_written": 2, "total_bytes": 22,
        },
        status=200,
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/products/export-jobs/jx/manifest",
        json={
            "job_id": "jx",
            "client_id": "c1",
            "collection_name": "products",
            "vector_count": 10,
            "total_bytes": 22,
            "parts": [
                {"filename": "part-0001.parquet", "bytes": 12, "rows": 7},
                {"filename": "part-0002.parquet", "bytes": 10, "rows": 3},
            ],
            "snapshot_high_water_seq": 42,
            "input_dim": 384,
            "started_at_unix_ms": 1,
            "completed_at_unix_ms": 2,
        },
        status=200,
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/products/export-jobs/jx/download",
        body=b"part-1-bytes",
        status=200,
        content_type="application/octet-stream",
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/products/export-jobs/jx/download",
        body=b"part-2-bx",
        status=200,
        content_type="application/octet-stream",
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/products/export-jobs/jx/download",
        body=b"# Export Summary: products\n\nhello readme\n",
        status=200,
        content_type="text/markdown; charset=utf-8",
    )

    c = make_client()
    out_dir = tmp_path / "export"
    result = c.collections.export(
        "products", out_dir, poll_interval_s=0.01,
    )

    assert result.job_id == "jx"
    assert result.path == out_dir
    assert result.parts == 2
    assert result.status == "complete"
    # server-z78b: local sidecars are underscore-prefixed so parquet readers skip them.
    assert (out_dir / "_manifest.json").exists()
    assert (out_dir / "part-0001.parquet").read_bytes() == b"part-1-bytes"
    assert (out_dir / "part-0002.parquet").read_bytes() == b"part-2-bx"
    assert (out_dir / "_EXPORT_README.md").read_text().startswith("# Export Summary")
    manifest = json.loads((out_dir / "_manifest.json").read_text())
    assert manifest["parts"][0]["filename"] == "part-0001.parquet"


@responses.activate
def test_export_wait_handles_missing_readme(tmp_path: Path):
    """Older servers without the EXPORT_README.md writer return 503; the
    SDK swallows it and the export still succeeds."""
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/products/export-jobs",
        json={"job_id": "jr"},
        status=202,
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/products/export-jobs/jr",
        json={
            "job_id": "jr", "client_id": "c1", "collection_name": "products",
            "status": "complete", "parts_written": 1, "total_bytes": 5,
        },
        status=200,
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/products/export-jobs/jr/manifest",
        json={
            "job_id": "jr", "client_id": "c1", "collection_name": "products",
            "vector_count": 1, "total_bytes": 5,
            "parts": [{"filename": "part-0001.parquet", "bytes": 5, "rows": 1}],
            "snapshot_high_water_seq": 0, "input_dim": 4,
            "started_at_unix_ms": 1, "completed_at_unix_ms": 2,
        },
        status=200,
    )
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/products/export-jobs/jr/download",
        body=b"hello",
        status=200,
        content_type="application/octet-stream",
    )
    # Older server: no EXPORT_README.md writer. 503 from the sidecar route.
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/products/export-jobs/jr/download",
        json={"error": "EXPORT_README.md missing on disk"},
        status=503,
    )

    c = make_client()
    out_dir = tmp_path / "export"
    result = c.collections.export("products", out_dir, poll_interval_s=0.01)

    assert result.status == "complete"
    assert (out_dir / "part-0001.parquet").read_bytes() == b"hello"
    assert not (out_dir / "_EXPORT_README.md").exists()
