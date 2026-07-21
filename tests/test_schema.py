"""Tests for schema management."""

import pytest
import responses

from veep import VP
from veep.exceptions import ValidationError
from veep.models import SchemaInfo

HOST = "http://localhost:3000"


def make_client():
    return VP(api_key="test_key", host=HOST)


@responses.activate
def test_get_schema():
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/products/schema",
        json={
            "state": "confirmed",
            "id_field": "product_id",
            "vector_field": "embedding",
            "format": "parquet",
            "dimension": 384,
            "samples_analyzed": 100,
            "pending_count": 0,
        },
        status=200,
    )
    c = make_client()
    s = c.schema.get("products")
    assert isinstance(s, SchemaInfo)
    assert s.state == "confirmed"
    assert s.id_field == "product_id"
    assert s.vector_field == "embedding"
    assert s.dimension == 384


@responses.activate
def test_get_schema_analyzing():
    responses.add(
        responses.GET,
        f"{HOST}/api/v1/collections/new_col/schema",
        json={
            "state": "analyzing",
            "samples_analyzed": 5,
            "pending_count": 95,
        },
        status=200,
    )
    c = make_client()
    s = c.schema.get("new_col")
    assert s.state == "analyzing"
    assert s.id_field is None
    assert s.analyzed == 5


@responses.activate
def test_confirm_schema():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/products/schema/confirm",
        json={"status": "confirmed"},
        status=200,
    )
    c = make_client()
    result = c.schema.confirm(
        "products",
        id_field="product_id",
        vector_field="embedding",
    )
    assert result["status"] == "confirmed"

    import json
    body = json.loads(responses.calls[0].request.body)
    assert body["idField"] == "product_id"
    assert body["vectorField"] == "embedding"


@responses.activate
def test_confirm_schema_with_options():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/products/schema/confirm",
        json={"status": "confirmed"},
        status=200,
    )
    c = make_client()
    c.schema.confirm(
        "products",
        id_field="id",
        vector_field="emb",
        format="parquet",
        dimension=768,
    )
    import json
    body = json.loads(responses.calls[0].request.body)
    assert body["format"] == "parquet"
    assert body["dimension"] == 768


def test_confirm_schema_missing_id_field():
    c = make_client()
    with pytest.raises(ValidationError, match="id_field"):
        c.schema.confirm("col", id_field="", vector_field="emb")


def test_confirm_schema_missing_vector_field():
    c = make_client()
    with pytest.raises(ValidationError, match="vector_field"):
        c.schema.confirm("col", id_field="id", vector_field="")


# -- schema.update (server-5pbm.1.2.4) --


@responses.activate
def test_update_schema_success():
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/products/schema/update",
        json={
            "success": True,
            "message": "Schema updated and 3 source file(s) rewritten.",
            "files_rewritten": 3,
        },
        status=200,
    )
    c = make_client()
    result = c.schema.update("products", id_field="new_id", reprocess=True)
    assert result["success"] is True
    assert result.get("files_rewritten") == 3


@responses.activate
def test_update_schema_no_op_emits_warning(caplog):
    """server-5pbm.1.2.4: server returns success+warning when the requested
    fields match the current confirmed schema. SDK should surface the warning
    at WARNING log level so the customer notices the no-op."""
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/products/schema/update",
        json={
            "success": True,
            "message": "No-op: schema unchanged.",
            "warning": "Schema update was a no-op — pass DIFFERENT field names AND reprocess=True.",
        },
        status=200,
    )
    c = make_client()
    import logging
    with caplog.at_level(logging.WARNING, logger="veep"):
        result = c.schema.update("products", id_field="id", vector_field="emb")
    assert result["success"] is True
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("no-op" in r.message.lower() for r in warnings)


@responses.activate
def test_update_schema_reprocess_required():
    """409 when reprocess=False but the collection has source files."""
    responses.add(
        responses.POST,
        f"{HOST}/api/v1/collections/products/schema/update",
        json={
            "error": "Schema change requires reprocessing",
            "detail": "Collection 'products' has 3 source file(s) ...",
        },
        status=409,
    )
    c = make_client()
    result = c.schema.update("products", id_field="new_id", reprocess=False)
    assert "error" in result or "detail" in result


@responses.activate
def test_update_schema_validation_no_fields():
    c = make_client()
    with pytest.raises(ValidationError, match="at least one"):
        c.schema.update("col")
