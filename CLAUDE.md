# veep — Python SDK for Vector Panda

## Project Overview

Python client library for the Vector Panda vector search API. Published to PyPI as `veep`.

This lives under `/home/mike/server/veep/` in the monorepo.

## Architecture

- **Thin HTTP wrapper** — all state lives server-side. The SDK makes REST calls and returns typed results.
- **Single host** — all traffic routes through consumer-site (.120) at `/api/v1/*` endpoints.
- **Resource-based API** — `client.collections.*`, `client.vectors.*`, `client.schema.*` sub-resources.

## Public API

```python
from veep import VP

# Authentication (device flow — opens browser for OAuth)
vp = VP.login()              # interactive, saves to ~/.veep/credentials.json
vp = VP.from_creds()                  # reuse saved credentials
vp = VP(api_key="...")       # explicit key
vp.save()                             # persist for later

# Collections
vp.collections.create("name", tier="hot")
vp.collections.get("name")
vp.collections.list()
vp.collections.delete("name")
vp.collections.status("name")

# Vectors
vp.vectors.upsert("collection", "file.parquet")
vp.vectors.replace("collection", "file.parquet")
vp.vectors.query("collection", vector=[...])
vp.vectors.query_batch([...])
vp.vectors.delete("collection", "filename")
vp.vectors.list_files("collection")

# Schema
vp.schema.get("collection")
vp.schema.confirm("collection", id_field="id", vector_field="emb")

# Health
vp.ping()
```

## API Endpoints Used

All traffic goes through consumer-site (.120):

| SDK Method | HTTP | Endpoint |
|------------|------|----------|
| `collections.create()` | POST | `/api/v1/collections` |
| `collections.list()` | GET | `/api/v1/collections` |
| `collections.get()` | GET | `/api/v1/collections/{name}` |
| `collections.status()` | GET | `/api/v1/collections/{name}/status` |
| `collections.delete()` | DELETE | `/api/v1/collections/{name}` |
| `vectors.upsert()` | POST | `/api/v1/collections/{col}/files/{file}` |
| `vectors.replace()` | PUT | `/api/v1/collections/{col}/files/{file}` |
| `vectors.query()` | POST | `/api/v1/query` |
| `vectors.query_batch()` | POST | `/api/v1/query/batch` |
| `vectors.delete()` | DELETE | `/api/v1/collections/{col}/files/{file}` |
| `vectors.list_files()` | GET | `/api/v1/collections/{col}/files` |
| `schema.get()` | GET | `/api/v1/collections/{col}/schema` |
| `schema.confirm()` | POST | `/api/v1/collections/{col}/schema/confirm` |
| `health()` | GET | `/api/v1/health` |
| `VP.login()` | POST | `/api/v1/auth/device` + `/api/v1/auth/device/token` |

Auth: Bearer token in Authorization header for all endpoints (except health and device auth).
Credentials persist to `~/.veep/credentials.json` (chmod 600).

## Package Structure

```
veep/
├── src/veep/
│   ├── __init__.py       # Public API exports
│   ├── client.py         # VP class, HTTP engine, login()/from_creds()
│   ├── auth.py           # Device auth flow, credential persistence
│   ├── collections.py    # Collections sub-resource
│   ├── vectors.py        # Vectors sub-resource
│   ├── schema.py         # Schema sub-resource
│   ├── models.py         # Result, Collection, FileInfo, etc.
│   └── exceptions.py     # Full exception hierarchy
├── tests/
│   ├── test_client.py
│   ├── test_collections.py
│   ├── test_vectors.py
│   ├── test_schema.py
│   └── test_models.py
├── .github/workflows/publish.yml
├── pyproject.toml
├── README.md
├── API_QUESTIONS.md
└── CLAUDE.md
```

## Coding Rules

- **Python 3.9+** minimum.
- **Minimal dependencies**: `requests` for HTTP. No pyarrow dependency (removed — files uploaded as-is).
- **Type hints** on all public methods. Use `from __future__ import annotations`.
- **Docstrings** on all public classes and methods (Google style).
- **Exception hierarchy**: `VeepError` base, with specific subclasses for every failure mode.
- **Verbose mode**: `verbose=True` logs in plain English via Python logging.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Build & Publish

GitHub Actions publishes to PyPI on version tag push (`v*`).

Manual:
```bash
pip install build twine
python -m build
twine upload dist/*
```

## Commit Rules

- Prefix: `veep:` for commits touching this directory.
