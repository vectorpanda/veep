"""Tests for collection management."""

import pytest
import responses

from veep import VP, Collection
from veep.exceptions import (
    CollectionNotFoundError,
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
    """create() catches 409 from POST /collections and falls through to
    self.get(name), returning the existing Collection. This makes
    create() idempotent — callers don't need to special-case 'already
    exists' for partially-completed setups."""
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
