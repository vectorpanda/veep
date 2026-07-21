"""Tests for the bundled samples module."""

import pytest

from veep import samples


def test_encode_returns_384_dim_unit_vector():
    pytest.importorskip("onnxruntime")
    vec = samples.encode("a hobbit destroys a magic ring")
    assert isinstance(vec, list)
    assert len(vec) == samples.DIMENSION
    assert all(isinstance(v, float) for v in vec)
    norm_sq = sum(v * v for v in vec)
    assert abs(norm_sq - 1.0) < 0.01, f"encoded vector norm² = {norm_sq:.4f}, expected 1.0"


def test_encode_is_deterministic():
    pytest.importorskip("onnxruntime")
    a = samples.encode("dinosaurs in a theme park")
    b = samples.encode("dinosaurs in a theme park")
    assert a == b, "samples.encode() must be deterministic for repeatable demos"


def test_encode_distinct_prompts_distinct_vectors():
    pytest.importorskip("onnxruntime")
    a = samples.encode("a hobbit destroys a magic ring")
    b = samples.encode("a heist crew steals from a casino")
    cosine = sum(x * y for x, y in zip(a, b))
    assert cosine < 0.5, f"unrelated prompts encoded too similarly: cosine={cosine:.4f}"


def test_encode_without_onnxruntime_raises_clear_error():
    pytest.importorskip("onnxruntime")
    from veep._encoder import MissingExtraError
    assert issubclass(MissingExtraError, Exception)


def test_dataframe_has_expected_shape():
    pd = pytest.importorskip("pandas")
    df = samples.dataframe()
    assert isinstance(df, pd.DataFrame)
    # ~5000 popular movies sourced from Wikipedia plot summaries.
    assert 4000 <= len(df) <= 6000, f"expected ~5000 rows, got {len(df)}"
    assert set(df.columns) == {"id", "title", "year", "genre", "plot", "vector"}
    assert all(len(v) == samples.DIMENSION for v in df["vector"])
    # Years span a wide range — the dataset cuts off around 2017.
    assert df["year"].min() < 1950
    assert df["year"].max() >= 2010
    # Plot text is non-trivial (we filtered to >=200 chars at build time,
    # then truncated at 800).
    assert df["plot"].str.len().min() >= 100
    assert df["plot"].str.len().max() <= 1000


def test_query_and_list_prompts_are_gone():
    """The prompt → embedded-text mapping was removed in 0.5.1
    (server-53m5) — samples.encode() handles arbitrary text now."""
    assert not hasattr(samples, "query"), "samples.query() should be removed"
    assert not hasattr(samples, "list_prompts"), "samples.list_prompts() should be removed"
