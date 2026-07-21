# veep — Python SDK for Vector Panda

Search your vectors in five minutes.

## Install

```bash
pip install veep[pandas]
```

Requires Python 3.9+. `pip install veep` works on its own (only `requests` is mandatory). The quickstart below uses two extras: `[samples]` for the bundled text encoder (~22MB ONNX model + onnxruntime), and `[pandas]` for the parquet upload helper. NumPy, pandas, and PyArrow are otherwise optional — install them only if you use those upload modes.

## Quickstart

Five steps. Copy-paste the whole block — it runs end-to-end against real embeddings, no `sentence-transformers` install required.

```python
# pip install veep[samples,pandas]

from veep import VP, samples

# 1. Sign in. Opens your browser for a one-time OAuth handshake (works in
#    terminals, Jupyter, and SSH) and saves credentials to
#    ~/.veep/credentials.json so subsequent runs reuse them — see the
#    Authentication section below for the API-key alternative.
#    During private beta you'll need an invite — request access at https://vectorpanda.com.
vp = VP.login()

# 2. Create a collection.
vp.collections.create("quickstart", tier="hot")

# 3. Upload the bundled corpus: ~5,000 popular movies (titles + genre +
#    plot summary, sourced from English Wikipedia), pre-embedded with
#    sentence-transformers/all-MiniLM-L6-v2 (384-dim, cosine-normalized).
#    upsert() blocks until the data is queryable — typically a few
#    seconds on a fast connection. Returns an UploadResult with the
#    server-side filename, byte size, and status='created'.
vp.vectors.upsert("quickstart", dataframe=samples.dataframe())

# 4. Encode an arbitrary sentence with the bundled model and search by
#    meaning. The corpus is plot text; titles/years/genres come back as
#    metadata. Try any prompt — "a heist crew steals from a casino",
#    "an astronaut stranded on mars", "boxer comes back from injury".
results = vp.vectors.query(
    "quickstart",
    vector=samples.encode("a hobbit destroys a magic ring"),
    top_k=5,
)

# 5. Print results — top 5 should all be Lord of the Rings adaptations.
for r in results:
    title = r.metadata.get("title")
    year  = r.metadata.get("year")
    print(f"  {r.score:.4f}  {title} ({year})")
```

That's it. You're searching real embeddings against real movie plots. `upsert` blocks until the collection is queryable, so step 4 always sees the data from step 3 — no manual polling. `samples.encode(text)` uses a bundled INT8-quantized ONNX export of the same model the corpus was built with, so build-time and runtime agree on the embedding space.

The plot text is sourced from English Wikipedia under CC BY-SA 4.0 — see `veep/_sample_data/ATTRIBUTION.md` in the package for the full notice.

## Upload Modes

Pick whichever shape your data already has:

```python
# pandas DataFrame with id + vector + optional metadata columns
import pandas as pd
df = pd.DataFrame({"id": ids, "vector": list(embeddings), "category": tags})
vp.vectors.upsert("col", dataframe=df)

# Parquet / CSV / JSONL file on disk
vp.vectors.upsert("col", "embeddings.parquet")

# pyarrow Table — useful when you've already loaded with pyarrow
import pyarrow.parquet as pq
tbl = pq.read_table("embeddings.parquet")
vp.vectors.upsert("col", table=tbl)
```

All three serialize through the same chunked-upload pipeline — pandas and pyarrow modes write a temp parquet under the hood, so RAM cost is bounded by the dataset itself plus one chunk in flight.

```python
# Inline list of dicts — for one-off small batches (under ~1000 vectors).
# Goes straight to the WAL rather than the artifact pipeline; latency is
# sub-second but the collection won't materialize as a query target until
# at least one of the modes above has run.
vp.vectors.upsert("col", vectors=[
    {"id": "abc", "vector": [0.1, 0.2, ...], "metadata": {"color": "red"}},
])
```

## Authentication

Four ways to connect — pick whichever fits your workflow:

```python
# Option 1: Interactive device-flow login (the quickstart form — terminals,
# Jupyter, SSH). Opens your browser for Google or GitHub sign-in. Saves
# credentials to ~/.veep/credentials.json so future runs skip the browser step.
vp = VP.login()

# Option 2: Reuse saved credentials from a prior login()
vp = VP.from_creds()

# Option 3: Explicit API key — paste from the dashboard
vp = VP(api_key="veep_live_...")

# Option 4: Environment variable
# export VEEP_API_KEY=veep_live_...
vp = VP()
```

`login()` uses the same device authorization pattern as `gh auth login` — it works in terminals, Jupyter notebooks, and remote SSH sessions. The verification URL is clickable in notebooks. Use it when copy-pasting an API key isn't convenient (CI runners that read from a vault, transient containers, etc.).

For programmatic flows (CLIs that want to email the verification link via
a different transport, embedded contexts that handle the URL with custom
UI, automated tests), `VP.login()` accepts an `on_device_code` callback
that fires once with `(device_code, verification_url)` after the SDK
gets them from the server and before polling starts:

```python
def show_url(device_code: str, url: str) -> None:
    # send the URL via your preferred channel (Slack, email, custom UI, etc.)
    print(f"Visit {url} to authorize this session.")

vp = VP.login(host="https://api.vectorpanda.com", on_device_code=show_url)
```

When `on_device_code` is set, the SDK suppresses its default stdout
output of the URL — the callback is the canonical surface for that
information.

```python
# Full options
vp = VP(
    api_key="your_key",        # or set VEEP_API_KEY env var
    host="https://...",         # optional, defaults to Vector Panda cloud
    timeout=120,                # request timeout in seconds
    verbose=True,               # log what the client is doing in plain English
)

# Save credentials for later
vp.save()                       # writes to ~/.veep/credentials.json
```

## Collections

```python
# Create a collection (with schema for instant processing)
col = vp.collections.create(
    "products",
    tier="hot",
    id_field="product_id",
    vector_field="embedding",
)

# Or create without schema (auto-detected from first upload)
col = vp.collections.create("products", tier="hot")

# List all collections
for col in vp.collections.list():
    count = col.vector_count if col.vector_count is not None else "—"
    size = f"{col.storage_gb:.1f} GB" if col.storage_gb is not None else "—"
    print(f"{col.name}: {count} vectors, {size}")

# Get details about one collection
col = vp.collections.get("products")
print(col.dimension, col.status)

# Check processing status
status = vp.collections.status("products")  # "ready", "processing", "unknown", "error"

# Delete a collection (permanent)
vp.collections.delete("products")
```

## Querying

```python
results = vp.vectors.query(
    "products",
    vector=[0.1, 0.2, ...],          # your query vector
    top_k=10,                          # max results (default: 10)
    min_score=0.7,                     # only return results with score >= this (cosine 0-1)
    metric="cosine",                   # "cosine", "euclidean", "dot_product"
    with_metadata=True,                # return metadata fields
)

for r in results:
    print(f"{r.key}: {r.score:.4f} — {r.metadata}")

# Batch queries (up to 100 at once)
batch = vp.vectors.query_batch([
    {"collection": "products", "vector": query_vec_1, "top_k": 5},
    {"collection": "products", "vector": query_vec_2, "top_k": 5},
])
for query_results in batch:
    print(f"Got {len(query_results)} results")

# Fetch a single vector by key (the key from a query result)
result = vp.vectors.fetch("products", "12345")
if result.found:
    print(f"Vector: {result.vector[:5]}...")
    print(f"Metadata: {result.metadata}")
```

## File Management

```python
# Replace an existing file (idempotent: same content = no-op)
result = vp.vectors.replace("products", "product_embeddings.parquet")

# List uploaded files
for f in vp.vectors.list_files("products"):
    print(f"{f.name}: {f.size} bytes, modified {f.modified}")

# Delete an uploaded file
vp.vectors.delete("products", "old_embeddings.parquet")
```

## Schema

After uploading files, Vector Panda auto-detects which columns hold your vector keys and embeddings. You can inspect and confirm the schema:

```python
schema = vp.schema.get("products")
print(schema.state)         # "analyzing" or "confirmed"
print(schema.vector_field)  # e.g., "embedding"
print(schema.id_field)      # e.g., "product_id"

# Confirm or override the detected schema
vp.schema.confirm("products", id_field="product_id", vector_field="embedding")
```

## Index Parameters

For advanced use, pass index-specific parameters to queries:

```python
results = vp.vectors.query(
    "products",
    vector=query_vec,
    use_index="pca",
    index_params={"pca": {"reduced_dimensions": 64, "candidate_multiplier": 10}},
)
```

## Health Check

```python
if vp.ping():
    print("Vector Panda is up")
```

## Verbose Mode

Turn on `verbose=True` to see what the client is doing:

```python
vp = VP(api_key="...", verbose=True)
vp.collections.list()
# veep: Connected to https://api.vectorpanda.com
# veep: Listing collections...
# veep: Found 3 collection(s).
```

## Error Handling

Every error tells you what happened and what to do about it:

```python
from veep import VP
from veep.exceptions import (
    CollectionNotFoundError,
    CollectionAlreadyExistsError,
    CollectionNotReadyError,
    AuthError,
    ValidationError,
)

try:
    vp.collections.get("nonexistent")
except CollectionNotFoundError as e:
    print(e)
    # Collection 'nonexistent' not found.
    # Use vp.collections.list() to see available collections.
```

| Exception | When |
|-----------|------|
| `AuthError` | Invalid or missing API key |
| `ValidationError` | Bad parameter (name, vector, etc.) |
| `CollectionNotFoundError` | Collection doesn't exist |
| `CollectionAlreadyExistsError` | Collection already exists |
| `CollectionNotReadyError` | Collection is still ingesting. From `query`/`fetch`: retry shortly. From `upsert`: ingest pipeline didn't finish before `wait_seconds` / `upload_timeout`; do **not** retry upsert (creates duplicate state) — call `vp.collections.status(name)` and contact support if stuck |
| `UploadError` | File not found or unreadable |
| `FileAlreadyExistsError` | File exists (use `replace`) |
| `TimeoutError` | Request timed out (e.g., HTTP 504) |
| `ServerError` | Server-side failure (HTTP 5xx). Includes `status_code` |
| `QueryError` | (Deprecated, retained for backwards compat — server errors now raise `ServerError`) |

## Configuration

| Environment Variable | Description |
|---------------------|-------------|
| `VEEP_API_KEY` | Default API key |
| `VEEP_HOST` | Default API host |

## Beta Status

Vector Panda is in private beta. `pip install veep` works for everyone, but creating an account currently requires an invite — request one at [vectorpanda.com](https://vectorpanda.com). If you already have an API key, everything in this README is live.

## License

MIT
