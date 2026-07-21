"""Tests for VP client."""

import io
from contextlib import redirect_stdout

import pytest
import requests
import responses

from veep import VP
from veep.auth import device_login
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


# server-sqom.1: detect README-style placeholder api_keys before the first
# network call so brand-new customers get an actionable error.

@pytest.mark.parametrize("placeholder", [
    "veep_live_REPLACE_ME",
    "veep_test_REPLACE_ME",
    "REPLACE_ME",
    "your_api_key",
    "YOUR_API_KEY",
    "your-api-key",
    "<your-api-key>",
    "PASTE_YOUR_KEY_HERE",
])
def test_rejects_placeholder_api_key(placeholder):
    with pytest.raises(AuthError, match="placeholder"):
        VP(api_key=placeholder, host="http://localhost:3000")


def test_real_looking_keys_are_not_treated_as_placeholders():
    # Realistic keys must not trip the placeholder check.
    c = VP(api_key="veep_test_abc123", host="http://localhost:3000")
    assert c.api_key == "veep_test_abc123"


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


# ---------------------------------------------------------------
# server-sqom: on_device_code callback for programmatic login flows
# ---------------------------------------------------------------


@responses.activate
def test_device_login_fires_callback_with_code_and_url():
    """device_login fires on_device_code(device_code, verification_url) before polling."""
    host = "https://api.example.test"
    responses.post(
        f"{host}/api/v1/auth/device",
        json={
            "device_code": "dc_test_123",
            "user_code": "ABCD12",
            "verification_url": f"{host}/auth/device?code=ABCD12",
            "expires_in": 600,
            "interval": 1,
        },
    )
    responses.post(
        f"{host}/api/v1/auth/device/token",
        json={"api_key": "veep_test_test_xyz", "client_id": "client_test_1"},
    )

    captured = []

    def callback(device_code: str, verification_url: str) -> None:
        captured.append((device_code, verification_url))

    result = device_login(host=host, open_browser=False, timeout_s=5, on_device_code=callback)

    assert captured == [("dc_test_123", f"{host}/auth/device?code=ABCD12")]
    assert result["api_key"] == "veep_test_test_xyz"
    assert result["client_id"] == "client_test_1"


@responses.activate
def test_device_login_does_not_open_browser_when_callback_set(monkeypatch):
    """server-sqom.9: when on_device_code is set, webbrowser.open is suppressed.

    A caller who provides on_device_code is opting into programmatic handling
    of the verification URL. Also auto-opening a browser dispatches the URL
    through two channels (callback + browser tab), which is a surprise.
    """
    host = "https://api.example.test"
    responses.post(
        f"{host}/api/v1/auth/device",
        json={
            "device_code": "dc_nobrowse",
            "user_code": "NOBR01",
            "verification_url": f"{host}/auth/device?code=NOBR01",
            "expires_in": 600,
            "interval": 1,
        },
    )
    responses.post(
        f"{host}/api/v1/auth/device/token",
        json={"api_key": "veep_test_nobrowse", "client_id": "client_nobrowse"},
    )

    opens: list[str] = []

    import veep.auth as auth_mod
    monkeypatch.setattr(
        auth_mod.webbrowser, "open", lambda url: opens.append(url) or True
    )

    device_login(
        host=host,
        open_browser=True,  # explicitly request browser open
        timeout_s=5,
        on_device_code=lambda dc, _url: None,  # ...but provide callback
    )

    assert opens == [], f"webbrowser.open was called despite callback being set: {opens!r}"


@responses.activate
def test_device_login_opens_browser_when_no_callback(monkeypatch):
    """The default path (no callback) still opens the browser as before."""
    host = "https://api.example.test"
    responses.post(
        f"{host}/api/v1/auth/device",
        json={
            "device_code": "dc_browse",
            "user_code": "BROW01",
            "verification_url": f"{host}/auth/device?code=BROW01",
            "expires_in": 600,
            "interval": 1,
        },
    )
    responses.post(
        f"{host}/api/v1/auth/device/token",
        json={"api_key": "veep_test_browse", "client_id": "client_browse"},
    )

    opens: list[str] = []
    import veep.auth as auth_mod
    monkeypatch.setattr(
        auth_mod.webbrowser, "open", lambda url: opens.append(url) or True
    )

    device_login(host=host, open_browser=True, timeout_s=5)

    assert opens == [f"{host}/auth/device?code=BROW01"], (
        f"expected browser to open the URL once, got: {opens!r}"
    )


@responses.activate
def test_device_login_suppresses_stdout_when_callback_set():
    """When on_device_code is provided, the URL is not printed to stdout."""
    host = "https://api.example.test"
    responses.post(
        f"{host}/api/v1/auth/device",
        json={
            "device_code": "dc_quiet_42",
            "user_code": "QUIET1",
            "verification_url": f"{host}/auth/device?code=QUIET1",
            "expires_in": 600,
            "interval": 1,
        },
    )
    responses.post(
        f"{host}/api/v1/auth/device/token",
        json={"api_key": "veep_test_quiet", "client_id": "client_quiet"},
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        device_login(
            host=host,
            open_browser=False,
            timeout_s=5,
            on_device_code=lambda dc, _url: None,
        )

    output = buf.getvalue()
    assert "QUIET1" not in output, f"URL leaked to stdout: {output!r}"
    assert "https://" not in output, f"URL leaked to stdout: {output!r}"


@responses.activate
def test_device_login_prints_url_when_no_callback():
    """When on_device_code is omitted, existing stdout printing is preserved."""
    host = "https://api.example.test"
    responses.post(
        f"{host}/api/v1/auth/device",
        json={
            "device_code": "dc_loud_99",
            "user_code": "LOUD99",
            "verification_url": f"{host}/auth/device?code=LOUD99",
            "expires_in": 600,
            "interval": 1,
        },
    )
    responses.post(
        f"{host}/api/v1/auth/device/token",
        json={"api_key": "veep_test_loud", "client_id": "client_loud"},
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        device_login(host=host, open_browser=False, timeout_s=5)

    output = buf.getvalue()
    assert "LOUD99" in output, f"URL not printed: {output!r}"


@responses.activate
def test_vp_login_plumbs_callback_and_uses_600s_default():
    """VP.login forwards on_device_code to device_login and defaults timeout_s=600."""
    host = "https://api.example.test"
    responses.post(
        f"{host}/api/v1/auth/device",
        json={
            "device_code": "dc_plumb",
            "user_code": "PLUMB1",
            "verification_url": f"{host}/auth/device?code=PLUMB1",
            "expires_in": 600,
            "interval": 1,
        },
    )
    responses.post(
        f"{host}/api/v1/auth/device/token",
        json={"api_key": "veep_test_plumb", "client_id": "client_plumb"},
    )

    captured = []
    vp = VP.login(
        host=host,
        open_browser=False,
        on_device_code=lambda dc, url: captured.append((dc, url)),
    )

    assert captured == [("dc_plumb", f"{host}/auth/device?code=PLUMB1")]
    assert vp.api_key == "veep_test_plumb"


def test_vp_login_default_timeout_is_600():
    """VP.login() default timeout_s is 600s (was 300s pre-server-sqom.3)."""
    import inspect

    sig = inspect.signature(VP.login)
    assert sig.parameters["timeout_s"].default == 600
