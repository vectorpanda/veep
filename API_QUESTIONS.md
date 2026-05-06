# API Questions & Gaps

Issues discovered while building the veep Python SDK.

## Resolved

### 1. Individual vector fetch — RESOLVED
`GET /api/v1/collections/:collection/vectors/:key` implemented across the full stack:
worker key scan, coordinator fan-out/aggregation, consumer-site proxy, SDK `vectors.fetch()`.
Metadata included by default via artifact server lookup.

### 3. Multipart vs raw binary upload — RESOLVED
Fixed in SDK v0.2.0 to use proper multipart/form-data.

### 4. Schema auto-detection timing — RESOLVED
`collections.create()` now accepts `id_field` and `vector_field`. When provided, schema is
pre-confirmed at creation time — no polling, no manual confirmation step. Auto-detection
only triggers when fields are omitted or don't match the uploaded data.

## Open

### 2. Individual vector delete
There is no endpoint to delete individual vectors by key. Deletion operates at the file level
only (`DELETE /api/v1/collections/:collection/files/:filename`). If a user wants to remove one
vector from a 10,000-vector file, they must delete the entire file and re-upload without it.

**Impact**: The SDK exposes `vectors.delete(collection, filename)` which deletes files, not
vectors. This is semantically confusing for users who think in terms of vectors, not files.
This is part of the larger vector-addressable abstraction work (server-v28q).

### 5. Collection describe response shape
`GET /api/v1/collections/:collection` proxies raw coordinator response without normalizing
field names. The SDK does fragile multi-field parsing (`collection_name` vs `name` vs
`collection`). Should normalize on the consumer-site side (server-rf2u).
