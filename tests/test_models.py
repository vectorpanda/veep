"""Tests for veep data models."""

from veep import Collection, QueryResults, Result
from veep.models import FileInfo, SchemaInfo, UploadResult


def test_result_repr():
    r = Result(key="vec_001", score=0.9512)
    assert "vec_001" in repr(r)
    assert "0.9512" in repr(r)


def test_result_repr_with_metadata():
    r = Result(key="vec_001", score=0.95, metadata={"title": "hello"})
    assert "metadata" in repr(r)


def test_query_results_iterable():
    items = [Result(key=f"k{i}", score=0.9 - i * 0.1) for i in range(3)]
    qr = QueryResults(items)
    assert len(qr) == 3
    assert list(qr) == items
    assert qr[0].key == "k0"


def test_query_results_empty():
    qr = QueryResults([])
    assert len(qr) == 0
    assert list(qr) == []
    assert not qr


def test_query_results_truthy():
    qr = QueryResults([Result(key="a", score=0.9)])
    assert qr


def test_collection_dataclass():
    c = Collection(name="test", tier="hot", is_active=True, vector_count=1000)
    assert c.name == "test"
    assert c.storage_gb is None
    assert c.status == "unknown"
    assert c.dimension is None


def test_collection_defaults():
    # server-0k60: vector_count / storage_gb default to None to honor the
    # "unknown / partially-created" state the server may report.
    c = Collection(name="minimal")
    assert c.tier == "hot"
    assert c.is_active is True
    assert c.vector_count is None
    assert c.storage_gb is None


def test_file_info():
    f = FileInfo(name="data.parquet", size=1024, modified="2026-01-01T00:00:00Z")
    assert f.name == "data.parquet"
    assert f.size == 1024


def test_upload_result():
    u = UploadResult(status="created", collection="col", filename="data.parquet", size=512)
    assert u.status == "created"
    assert u.size == 512


def test_schema_info():
    s = SchemaInfo(state="confirmed", id_field="id", vector_field="emb", dimension=384)
    assert s.state == "confirmed"
    assert s.dimension == 384
    assert s.format is None
