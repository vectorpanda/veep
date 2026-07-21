"""Shared pytest fixtures for the veep test suite.

server-mubw: redirect veep.auth.CREDENTIALS_FILE to a tmp_path for every test
so the suite cannot clobber the developer's real ~/.veep/credentials.json. The
previous failure mode: a test that called VP.login() with mocked HTTP responses
would land veep_test_plumb / api.example.test in the real credentials file, and
the next VP.from_creds() (or the operator's own veep usage) would silently
hit api.example.test until the file was restored.
"""

from __future__ import annotations

import pytest

import veep.auth


@pytest.fixture(autouse=True)
def _isolate_credentials_file(tmp_path, monkeypatch):
    """Point veep.auth at a per-test tmp credentials file."""
    creds_dir = tmp_path / ".veep"
    creds_file = creds_dir / "credentials.json"
    monkeypatch.setattr(veep.auth, "CREDENTIALS_DIR", creds_dir)
    monkeypatch.setattr(veep.auth, "CREDENTIALS_FILE", creds_file)
    yield
