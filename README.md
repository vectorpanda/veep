# veep — Python SDK for Vector Panda

Search your vectors in five minutes.

## Install

```bash
pip install veep
```

Requires Python 3.9+. The only mandatory dependency is `requests`. NumPy, pandas, and PyArrow are optional — install them only if you use those upload modes.

## Quickstart

Five steps. Copy-paste the whole block — it runs end-to-end.

```python
import numpy as np
from veep import VP

# 1. Sign in. Opens your browser for Google or GitHub OAuth.
#    Credentials save to ~/.veep/credentials.json so future runs skip this step.
vp = VP.login()

# 2. Create a collection.
vp.collections.create("quickstart", tier="hot")

# 3. Upload 100 random 64-dim vectors.
rng = np.random.default_rng(42)
vectors = [
    {"id": f"item_{i}", "vector": rng.standard_normal(64).tolist()}
    for i in range(100)
]
vp.vectors.upsert("quickstart", vectors=vectors)

# 4. Query the 5 nearest vectors to a random target.
query_vec = rng.standard_normal(64).tolist()
results = vp.vectors.query("quickstart", vector=query_vec, top_k=5)

# 5. Print results.
for r in results:
    print(f"{r.key}  score={r.score:.4f}")
```

That's it. You're searching vectors. `upsert` blocks until the collection is queryable, so step 4 always sees the data from step 3 — no manual polling.

## Upload Modes

Pick whichever shape your data already has. All four take the same `upsert()` call:

```python
# Inline list of dicts (the quickstart shape — good for small batches)
vp.vectors.upsert("col", vectors=[
    {"id": "abc", "vector": [0.1, 0.2, ...], "metadata": {"color": "red"}},
])

# Parquet / CSV / JSONL file on disk
vp.vectors.upsert("col", "embeddings.parquet")

# pandas DataFrame with id + vector + optional metadata columns
import pandas as pd
df = pd.DataFrame({"id": ids, "vector": list(embeddings), "category": tags})
vp.vectors.upsert("col", dataframe=df)

# pyarrow Table — useful when you've already loaded with pyarrow
import pyarrow.parquet as pq
tbl = pq.read_table("embeddings.parquet")
vp.vectors.upsert("col", table=tbl)
```

The file path is the bulk-upload path: it streams your file through a chunked protocol with no client-side RAM ceiling. The other three are convenience wrappers that serialize to Arrow and use the same backend.

## Authentication

Three ways to connect — pick whichever fits your workflow:

```python
# Option 1: Interactive login (recommended for first use)
# Opens your browser for Google or GitHub sign-in.
# Saves credentials to ~/.veep/credentials.json automatically.
vp = VP.login()

# Option 2: Reuse saved credentials (no browser, no key needed)
# Works after a previous login() call.
vp = VP.from_creds()

# Option 3: Explicit API key (CI/CD, headless environments)
vp = VP(api_key="sk_live_...")

# Option 4: Environment variable
# export VEEP_API_KEY=sk_live_...
vp = VP()
```

`login()` uses the same device authorization pattern as `gh auth login` — it works in terminals, Jupyter notebooks, and remote SSH sessions. The verification URL is clickable in notebooks.

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
| `CollectionNotReadyError` | Collection is still ingesting; retry shortly |
| `UploadError` | File not found or unreadable |
| `FileAlreadyExistsError` | File exists (use `replace`) |
| `QueryError` | Query service unavailable |
| `TimeoutError` | Request timed out |
| `ServerError` | Unexpected server error |

## Configuration

| Environment Variable | Description |
|---------------------|-------------|
| `VEEP_API_KEY` | Default API key |
| `VEEP_HOST` | Default API host |

## Beta Status

Vector Panda is in private beta. `pip install veep` works for everyone, but creating an account currently requires an invite — request one at [vectorpanda.com](https://vectorpanda.com). If you already have an API key, everything in this README is live.

## License

MIT
