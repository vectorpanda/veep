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
