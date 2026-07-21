"""Run a standard VectorDBBench dataset against your Vector Panda account.

This drives Zilliz's open-source VectorDBBench harness
(https://github.com/zilliztech/vectordbbench) end to end using the VeepVDB
client in this directory: download the dataset (VDBBench's own loader),
load it into a collection, wait for the auto-optimizer to converge, then
measure recall@k with VDBBench's serial runner and QPS/latency with its
multiprocess concurrency runner. Output is a JSON you can compare directly
against results/ in this directory and https://www.vectorpanda.com/benchmarks.

Setup (Python 3.11):

    pip install vectordb-bench veep
    export VEEP_API_KEY=...   # from your dashboard

Run:

    python run_benchmark.py --dataset cohere-1m
    python run_benchmark.py --dataset cohere-1m --skip-load   # reuse the collection

Notes:
  - The collection this creates bills like any other (Cohere 1M is ~2.9 GB,
    about $17/month on the hot tier at current list prices). Delete it when
    you're done.
  - The optimizer convergence wait is real work on a 1M corpus — budget a
    few hours for a from-scratch run. --skip-load + a converged collection
    re-measures in ~15 minutes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.filter import NonFilter
from vectordb_bench.backend.runner.mp_runner import MultiProcessingSearchRunner
from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner
from veep_client import VeepCaseConfig, VeepVDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_benchmark")


def get_manager(dataset: str):
    """Return (DatasetManager, human label) for the requested dataset."""
    from vectordb_bench.backend.cases import CaseType
    from vectordb_bench.backend.dataset import DatasetWithSizeType

    if dataset == "cohere-small":
        return DatasetWithSizeType.CohereSmall.get_manager(), "Cohere-768 100k (cosine)"
    if dataset == "cohere-1m":
        return CaseType.Performance768D1M.case_cls().dataset, "Cohere-768 1M (cosine)"
    raise SystemExit(f"unknown --dataset {dataset!r}")


def _flatten(x):
    if isinstance(x, (tuple, list)):
        for i in x:
            yield from _flatten(i)
    else:
        yield x


def _jsonable(x):
    if isinstance(x, (tuple, list)):
        return [_jsonable(i) for i in x]
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cohere-1m", choices=["cohere-small", "cohere-1m"])
    ap.add_argument("--host", default="https://api.vectorpanda.com")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--concurrencies", default="1,10,20,50,100,150,200")
    ap.add_argument("--duration", type=int, default=30, help="seconds per concurrency step")
    ap.add_argument("--skip-load", action="store_true", help="reuse an existing collection")
    ap.add_argument("--keep", action="store_true", help="don't delete the collection after")
    ap.add_argument("--optimize-timeout", type=float, default=7200.0)
    args = ap.parse_args()

    api_key = os.environ.get("VEEP_API_KEY")
    if not api_key:
        raise SystemExit("set VEEP_API_KEY (dashboard -> API keys)")

    mgr, label = get_manager(args.dataset)
    data = mgr.data
    coll = f"vdbbench-{args.dataset}"
    log.info("%s -> collection=%s dim=%d size=%d (%s)",
             label, coll, data.dim, data.size, args.host)

    log.info("Preparing dataset (download on first run) ...")
    mgr.prepare(DatasetSource.S3, filters=NonFilter())

    cfg = VeepCaseConfig(
        metric_type=data.metric_type,
        tier="hot",
        id_field=data.train_id_field,
        vector_field=data.train_vector_field,
        expect_optimization=True,
        optimize_timeout_s=args.optimize_timeout,
        optimize_poll_s=20.0,
    )
    db = VeepVDB(
        dim=data.dim,
        db_config={"api_key": api_key, "host": args.host},
        db_case_config=cfg,
        collection_name=coll,
        drop_old=not args.skip_load,
    )

    from veep import VP
    vp = VP(api_key=api_key, host=args.host)

    result: dict = {
        "dataset": label, "collection": coll, "dim": data.dim, "size": data.size,
        "k": args.k, "host": args.host,
    }

    if not args.skip_load:
        t0 = time.monotonic()
        for i, fname in enumerate(mgr.train_files):
            log.info("Uploading train file %d/%d: %s", i + 1, len(mgr.train_files), fname)
            vp.vectors.upsert(coll, file_path=str(mgr.data_dir / fname), wait_seconds=3600.0)
        result["insert_duration_s"] = round(time.monotonic() - t0, 2)
        log.info("Bulk load complete in %ss", result["insert_duration_s"])

    log.info("Waiting for the optimizer to converge ...")
    t0 = time.monotonic()
    with db.init():
        db.optimize(data_size=data.size)
    result["optimize_duration_s"] = round(time.monotonic() - t0, 2)

    test_emb = np.asarray(mgr.test_data, dtype=np.float32)

    log.info("Serial search: recall@%d over %d queries", args.k, len(test_emb))
    serial = SerialSearchRunner(db=db, test_data=test_emb, ground_truth=mgr.gt_data, k=args.k)
    serial_metrics = serial.run()
    # Runner return shapes vary across vectordb-bench versions, so record the
    # raw value and pull the headline numbers defensively. Convention:
    # serial -> (recall, ndcg, p99, ...); mp -> (max_qps, conc, qps, p99, ...).
    result["serial_search_raw"] = _jsonable(serial_metrics)
    nums = [v for v in _flatten(serial_metrics) if isinstance(v, (int, float))]
    if nums:
        result["recall"] = round(nums[0], 4)
    if len(nums) >= 3:
        result["serial_latency_p99_s"] = round(nums[2], 6)

    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    log.info("Concurrency sweep %s, %ss each", concurrencies, args.duration)
    mp = MultiProcessingSearchRunner(
        db=db, test_data=test_emb, k=args.k,
        concurrencies=concurrencies, duration=args.duration,
    )
    mp_metrics = mp.run()
    result["concurrent_search_raw"] = _jsonable(mp_metrics)
    nums = [v for v in _flatten(mp_metrics) if isinstance(v, (int, float))]
    if nums:
        result["max_qps"] = round(nums[0], 2)
    try:
        mp.stop()
    except Exception:
        pass

    out = Path(f"vdbbench-{args.dataset}-result.json")
    out.write_text(json.dumps(result, indent=2))
    log.info("Saved %s", out)
    print(json.dumps(result, indent=2))

    if not args.keep and not args.skip_load:
        vp.collections.delete(coll)
        log.info("Deleted %s (pass --keep to retain it)", coll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
