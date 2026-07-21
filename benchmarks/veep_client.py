"""VectorDBBench VectorDB client for Vector Panda, wrapping the veep SDK.

Zilliz VectorDBBench (https://github.com/zilliztech/vectordbbench) ships
native clients for Milvus/Pinecone/Qdrant/pgvector/etc. but NOT for Vector
Panda, and has no generic REST client. This module implements the VDBBench
``VectorDB`` ABC on top of the veep SDK so VDBBench can drive Vector Panda
like any first-class client and emit standardized recall@k / QPS / p99.

Metric-correct by construction: VDBBench cases carry a distance metric
(L2 / COSINE / IP) and compute ground truth under THAT metric. We map each
to the matching veep query metric (euclidean / cosine / dot_product), so an
L2 case measures L2 recall against an L2 index (or exact brute-force
fallback), not cosine-against-L2 nonsense.

See README.md in this directory for how to run the benchmark against your
own Vector Panda account, and https://www.vectorpanda.com/benchmarks for
our published results and raw output.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

from pydantic import BaseModel, SecretStr
from vectordb_bench.backend.clients.api import (
    DBCaseConfig,
    DBConfig,
    MetricType,
    VectorDB,
)

log = logging.getLogger(__name__)

# VDBBench MetricType -> veep query metric string. veep validates the metric
# kwarg against exactly this set (vectors.query: cosine/euclidean/dot_product).
_METRIC_MAP: dict[MetricType, str] = {
    MetricType.L2: "euclidean",
    MetricType.COSINE: "cosine",
    MetricType.IP: "dot_product",
}

# veep collection serving-status strings. Collections also report an
# optimization_state; optimize() below prefers that signal ('index_optimized')
# and falls back to a settled serving status when it isn't available.
_SERVING_READY = {"ready", "active"}
_TERMINAL_FAILURE = {"failed", "error", "capacity_limited"}


class VeepConfig(DBConfig):
    """Connection config: veep API key + API host.

    db_label/version/note come from DBConfig.
    """

    api_key: SecretStr
    host: str = "https://api.vectorpanda.com"

    def to_dict(self) -> dict:
        return {
            "api_key": self.api_key.get_secret_value(),
            "host": self.host,
        }


class VeepCaseConfig(BaseModel, DBCaseConfig):
    """Case config: carries the dataset metric + collection tier.

    Vector Panda has no customer-facing index/HNSW knobs -- the coordinator's
    auto-optimizer owns index strategy + hyperparameters (the seeder IS the
    benchmark). So index_param()/search_param() expose no tuning; search_param
    only threads the query metric through.
    """

    metric_type: MetricType | None = None
    tier: str = "hot"
    # Column names to lock the collection schema to. Defaults suit inline
    # upserts ({"id","vector"}); the bulk-file benchmark points vector_field at
    # the dataset's own column (e.g. "emb" for Cohere) so its parquet files
    # ingest directly with no rewrite.
    id_field: str = "id"
    vector_field: str = "vector"
    # Whether this dataset is large enough that the coordinator's auto-optimizer
    # will actually run (it only tunes collections above ~50k vectors, on a
    # ~10-min cadence). True for real benchmark datasets (SIFT/GIST/Cohere ~1M+):
    # optimize() then waits for optimization_state == 'index_optimized' (100%).
    # False for tiny collections that will never leave 'raw' (brute-force is
    # already final): optimize() waits only for a settled serving status.
    expect_optimization: bool = True
    # optimize() convergence poll.
    optimize_timeout_s: float = 3600.0
    optimize_poll_s: float = 15.0
    # number of consecutive stable vector_count polls to treat as settled
    # (fallback / sub-threshold path only).
    optimize_stable_polls: int = 3

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {"metric": _METRIC_MAP.get(self.metric_type, "cosine")}


class VeepVDB(VectorDB):
    """VDBBench VectorDB implementation backed by the veep SDK."""

    name = "VeepVDB"
    # The veep client wraps a requests.Session; do not share it across the
    # MultiProcessingSearchRunner's worker processes. thread_safe=False makes
    # VDBBench deep-copy the instance and call init() per process/thread.
    thread_safe = False

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig | None,
        collection_name: str,
        drop_old: bool = False,
        **kwargs,
    ) -> None:
        self.dim = dim
        self.api_key = db_config["api_key"]
        self.host = db_config["host"]
        self.collection_name = collection_name
        self.case_config = db_case_config

        self.metric = "cosine"
        self.tier = "hot"
        self.id_field = "id"
        self.vector_field = "vector"
        self.expect_optimization = True
        self.optimize_timeout_s = 3600.0
        self.optimize_poll_s = 15.0
        self.optimize_stable_polls = 3
        if db_case_config is not None:
            self.metric = db_case_config.search_param().get("metric", "cosine")
            self.tier = getattr(db_case_config, "tier", "hot")
            self.id_field = getattr(db_case_config, "id_field", "id")
            self.vector_field = getattr(db_case_config, "vector_field", "vector")
            self.expect_optimization = getattr(db_case_config, "expect_optimization", True)
            self.optimize_timeout_s = getattr(db_case_config, "optimize_timeout_s", 3600.0)
            self.optimize_poll_s = getattr(db_case_config, "optimize_poll_s", 15.0)
            self.optimize_stable_polls = getattr(db_case_config, "optimize_stable_polls", 3)

        # Transient per-process client; (re)built in init() and lazily on use.
        self._client = None

        client = self._new_client()

        # Lock the schema explicitly (id_field/vector_field/dimension) so ingest
        # never hits the auto-detect + confirm path. Inline JSON upserts and the
        # column-mapped file path both use the "id"/"vector" names.
        # drop_old -> if_exists="replace": force_destroy bypasses BOTH the
        # existing-row gate and the post-delete cooldown atomically. A separate
        # delete()+create() would trip the cooldown (CollectionRecentlyDeleted).
        client.collections.create(
            self.collection_name,
            tier=self.tier,
            id_field=self.id_field,
            vector_field=self.vector_field,
            dimension=self.dim,
            if_exists="replace" if drop_old else "ignore",
        )
        log.info(
            "VeepVDB ready: collection=%s dim=%d metric=%s tier=%s host=%s",
            self.collection_name, self.dim, self.metric, self.tier, self.host,
        )

    # -- client lifecycle -------------------------------------------------

    def __getstate__(self):
        # MultiProcessingSearchRunner pickles this instance into worker
        # processes; never carry a live requests.Session across the fork.
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._client = None

    def _new_client(self):
        # Imported lazily so this module imports even where veep isn't
        # installed (e.g. the VDBBench config-discovery pass).
        from veep import VP

        return VP(api_key=self.api_key, host=self.host)

    def _client_or_new(self):
        if self._client is None:
            self._client = self._new_client()
        return self._client

    @contextmanager
    def init(self):
        """Create/destroy the per-process veep client (VDBBench contract)."""
        self._client = self._new_client()
        try:
            yield
        finally:
            self._client = None

    def need_normalize_cosine(self) -> bool:
        # veep computes cosine natively (the worker normalizes internally), so
        # VDBBench does NOT need to L2-normalize inputs for COSINE datasets.
        return False

    # -- data plane -------------------------------------------------------

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        assert len(embeddings) == len(metadata)
        client = self._client_or_new()
        rows = [
            {"id": str(metadata[i]), "vector": list(embeddings[i])}
            for i in range(len(embeddings))
        ]

        def _count():
            try:
                return client.collections.get(self.collection_name).vector_count or 0
            except Exception:
                return None

        # Retrying an upsert is only safe if we know the prior attempt did NOT
        # land -- upsert's post-write status poll can time out on a busy gateway
        # (server-n7m5) after the write already succeeded, and a blind retry
        # then double-writes. So verify by vector_count before retrying.
        attempts = 3
        last_exc: Exception | None = None
        before = _count()
        for attempt in range(1, attempts + 1):
            try:
                client.vectors.upsert(self.collection_name, vectors=rows)
                return len(embeddings), None
            except Exception as e:
                last_exc = e
                after = _count()
                if before is not None and after is not None and after >= before + len(rows):
                    log.info(
                        "insert_embeddings: upsert raised (%s) but rows landed "
                        "(count %s -> %s); treating as success", e, before, after,
                    )
                    return len(embeddings), None
                log.warning(
                    "insert_embeddings attempt %d/%d failed on %d rows: %s",
                    attempt, attempts, len(embeddings), e,
                )
                if attempt < attempts:
                    time.sleep(2.0 * attempt)
        return 0, last_exc

    def search_embedding(self, query: list[float], k: int = 100, **kwargs) -> list[int]:
        # server-0uq0: absorb transient server stalls here with real backoff.
        # VDBBench's serial_runner retries 5x with no delay — during a
        # control-plane stall (server-x0wa: 30s cancel + ~1s 502 window) all
        # five land inside the outage and the whole multi-hour run dies. A
        # stalled query should degrade that sample's stats, not abort the
        # campaign. Non-transient errors still raise immediately.
        client = self._client_or_new()
        # Coerce array-like queries (e.g. a numpy row from the MP runner's
        # float32 test_data) to native Python floats — np.float32 scalars are not
        # JSON-serializable, so list(np_row) would 500 every query. .tolist()
        # yields native floats; a plain list passes through unchanged.
        qvec = query.tolist() if hasattr(query, "tolist") else list(query)
        backoffs = [1.0, 2.0, 4.0, 8.0, 15.0, 30.0]
        for i, delay in enumerate([*backoffs, None]):
            try:
                results = client.vectors.query(
                    self.collection_name,
                    vector=qvec,
                    top_k=k,
                    metric=self.metric,
                    with_metadata=False,
                )
                break
            except Exception as e:
                msg = str(e)
                transient = (
                    "502" in msg
                    or "temporarily unavailable" in msg
                    or "took too long and was cancelled" in msg
                )
                if not transient or delay is None:
                    raise
                log.warning(
                    "search_embedding transient error (attempt %d, sleeping %.0fs): %s",
                    i + 1, delay, msg,
                )
                time.sleep(delay)
        # VDBBench metadata ids are ints; veep keys round-trip as strings.
        return [int(r.key) for r in results]

    def optimize(self, data_size: int | None = None) -> None:
        """Block until the index is FINAL before the timed query phase.

        Querying mid-optimization measures an interim-winner index and makes
        recall/QPS invalid (the server-vjnf trap).

        For a real benchmark dataset (``expect_optimization=True``) the
        auto-optimizer will run, so we wait for the real convergence signal
        (server-rg2e): ``optimization_state == 'index_optimized'`` (100% tuned).
        A server that predates rg2e reports no signal (field stays None); we
        then fall back to a BEST-EFFORT settled-serving-status wait and log it.

        For a sub-threshold collection (``expect_optimization=False``) the
        optimizer never runs -- the collection sits at ``'raw'`` and serves via
        exact brute-force, which is already final -- so we just wait for a
        settled serving status.
        """
        client = self._client_or_new()
        deadline = time.monotonic() + self.optimize_timeout_s
        last_count = -1
        stable = 0
        last_state = None
        signal_seen = False

        while time.monotonic() < deadline:
            try:
                col = client.collections.get(self.collection_name)
            except Exception as e:
                # Transient gateway/coord hiccup (5xx, timeout) -- a poll loop is
                # inherently retryable, so don't abort a long run; try next tick.
                log.info("optimize(): transient poll error, retrying: %s", e)
                time.sleep(self.optimize_poll_s)
                continue
            status = getattr(col, "status", None)
            count = getattr(col, "vector_count", None)
            opt_state = getattr(col, "optimization_state", None)

            if (status, opt_state) != last_state:
                log.info(
                    "optimize(): status=%s optimization_state=%s vector_count=%s",
                    status, opt_state, count,
                )
                last_state = (status, opt_state)

            if status in _TERMINAL_FAILURE:
                raise RuntimeError(
                    f"collection {self.collection_name} entered terminal status "
                    f"{status!r} (failure_reason={getattr(col, 'failure_reason', None)!r})"
                )

            if opt_state is not None:
                signal_seen = True

            # Preferred path: the real converged signal for datasets that tune.
            if self.expect_optimization and signal_seen:
                if opt_state == "index_optimized":
                    log.info("optimize(): converged (optimization_state=index_optimized)")
                    return
            # Settled-serving path: sub-threshold collections, or a pre-rg2e
            # server that never surfaces the signal.
            elif status in _SERVING_READY:
                if count == last_count:
                    stable += 1
                    if stable >= self.optimize_stable_polls:
                        why = (
                            "sub-threshold (optimizer will not run)"
                            if not self.expect_optimization
                            else "BEST-EFFORT -- server predates server-rg2e"
                        )
                        log.info(
                            "optimize(): settled at status=%s opt_state=%s vector_count=%s (%s)",
                            status, opt_state, count, why,
                        )
                        return
                else:
                    stable = 0
                    last_count = count
            else:
                stable = 0

            time.sleep(self.optimize_poll_s)

        raise TimeoutError(
            f"optimize(): collection {self.collection_name} did not converge within "
            f"{self.optimize_timeout_s}s (last status={last_state})"
        )
