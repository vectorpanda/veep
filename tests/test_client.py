"""Tests for VP client."""

import pytest
import requests
import responses

from veep import VP
from veep.exceptions import AuthError, ServerError


def test_requires_api_key():
    with pytest.raises(AuthError, match="API key required"):
        VP()


def test_accepts_api_key():
    c = VP(api_key="test_key", host="http://localhost:3000")
    assert c.api_key == "test_key"
    assert c.host == "http://localhost:3000"


def test_strips_trailing_slash():
    c = VP(api_key="k", host="http://localhost:3000/")
    assert c.host == "http://localhost:3000"


def test_env_var_api_key(monkeypatch):
    monkeypatch.setenv("VEEP_API_KEY", "env_key")
    c = VP(host="http://localhost:3000")
    assert c.api_key == "env_key"


def test_env_var_host(monkeypatch):
    monkeypatch.setenv("VEEP_HOST", "http://env-host:3000")
    c = VP(api_key="k")
    assert c.host == "http://env-host:3000"


def test_explicit_key_overrides_env(monkeypatch):
    monkeypatch.setenv("VEEP_API_KEY", "env_key")
    c = VP(api_key="explicit_key", host="http://localhost:3000")
    assert c.api_key == "explicit_key"


@responses.activate
def test_ping_returns_true():
    responses.add(responses.GET, "http://localhost:3000/api/v1/health", body="OK", status=200)
    c = VP(api_key="k", host="http://localhost:3000")
    assert c.ping() is True


@responses.activate
def test_ping_returns_false_on_error():
    responses.add(responses.GET, "http://localhost:3000/api/v1/health", status=500)
    c = VP(api_key="k", host="http://localhost:3000")
    assert c.ping() is False


def test_ping_returns_false_on_connection_error():
    c = VP(api_key="k", host="http://192.0.2.1:1", timeout=0.1)
    assert c.ping() is False


def test_verbose_mode_no_crash(capsys):
    c = VP(api_key="k", host="http://localhost:3000", verbose=True)
    assert c.verbose is True
    captured = capsys.readouterr()
    assert "Connected" in captured.err


def test_has_sub_resources():
    c = VP(api_key="k", host="http://localhost:3000")
    assert hasattr(c, "collections")
    assert hasattr(c, "vectors")
    assert hasattr(c, "schema")


@responses.activate
def test_auth_error_on_401():
    responses.add(
        responses.GET,
        "http://localhost:3000/api/v1/collections",
        json={"error": "Invalid API key"},
        status=401,
    )
    c = VP(api_key="bad", host="http://localhost:3000")
    with pytest.raises(AuthError, match="rejected"):
        c.collections.list()


@responses.activate
def test_server_error_preserves_status_code():
    responses.add(
        responses.GET,
        "http://localhost:3000/api/v1/collections",
        json={"error": "internal"},
        status=500,
    )
    c = VP(api_key="k", host="http://localhost:3000")
    with pytest.raises(ServerError) as exc_info:
        c.collections.list()
    assert exc_info.value.status_code == 500


# ---------------------------------------------------------------
# server-t4d9 + server-4i6b: bounded retry on transient errors
# ---------------------------------------------------------------


@responses.activate
def test_retries_on_503_then_succeeds(monkeypatch):
    monkeypatch.setattr("veep.client.time.sleep", lambda *_: None)
    url = "http://localhost:3000/api/v1/collections"
    responses.add(responses.GET, url, status=503)
    responses.add(responses.GET, url, json={"collections": []}, status=200)

    c = VP(api_key="k", host="http://localhost:3000")
    resp = c._request("GET", "/api/v1/collections", retries=2)

    assert resp.status_code == 200
    assert len(responses.calls) == 2


@responses.activate
def test_no_retry_by_default():
    responses.add(
        responses.GET,
        "http://localhost:3000/api/v1/collections",
        status=503,
    )
    c = VP(api_key="k", host="http://localhost:3000")
    with pytest.raises(ServerError):
        c._request("GET", "/api/v1/collections")
    assert len(responses.calls) == 1


def test_reset_session_replaces_session(monkeypatch):
    c = VP(api_key="k", host="http://localhost:3000")
    original = c._session
    c._reset_session()
    assert c._session is not original
    assert c._session.headers["Authorization"] == "Bearer k"


def test_connection_error_resets_session_and_retries(monkeypatch):
    monkeypatch.setattr("veep.client.time.sleep", lambda *_: None)
    c = VP(api_key="k", host="http://localhost:3000")

    sessions_used: list[int] = []
    call_count = {"n": 0}

    def fake_request(self, *args, **kwargs):  # noqa: ANN001
        sessions_used.append(id(self))
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise requests.exceptions.ConnectionError("simulated")
        resp = requests.Response()
        resp.status_code = 200
        resp._content = b'{"ok": true}'
        return resp

    monkeypatch.setattr(requests.Session, "request", fake_request)
    resp = c._request("GET", "/api/v1/collections", retries=1)
    assert resp.status_code == 200
    assert call_count["n"] == 2
    assert sessions_used[0] != sessions_used[1]  # session was replaced
