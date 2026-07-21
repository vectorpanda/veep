# Benchmark Vector Panda with VectorDBBench

This directory contains a [VectorDBBench](https://github.com/zilliztech/vectordbbench)
`VectorDB` client for Vector Panda plus a runner, so you can measure
recall@k / QPS / latency on a standard public dataset against your own
account — the same way we produced the numbers on
[vectorpanda.com/benchmarks](https://www.vectorpanda.com/benchmarks).

VectorDBBench is the open-source benchmark harness maintained by Zilliz.
It downloads the dataset and its ground truth itself; nothing here is
synthetic or hand-picked. `results/` holds the raw harness output for our
published runs, including the one our own rate limiter spoiled (kept
because it's a good lesson in benchmark skepticism).

## Run it

Python 3.11:

```bash
pip install vectordb-bench veep
export VEEP_API_KEY=...          # dashboard -> API keys

python run_benchmark.py --dataset cohere-1m
```

The runner creates a collection named `vdbbench-cohere-1m`, bulk-loads the
dataset's parquet files, waits for the automatic optimizer to converge
(there are no index knobs to tune — that's the product), then runs
VectorDBBench's serial recall pass and multiprocess concurrency sweep.
Output lands in `vdbbench-cohere-1m-result.json`.

Fair warnings:

- **It bills like real usage.** Cohere 1M is ~2.9 GB — about $17/month on
  the hot tier at current list prices, queries included. The runner deletes
  the collection at the end unless you pass `--keep`.
- **A from-scratch run takes hours**, most of it waiting for index
  optimization to converge on 1M vectors. Re-measuring an already-converged
  collection (`--skip-load --keep`) takes ~15 minutes.
- **Client hardware matters at high concurrency.** The sweep spawns one
  process per concurrent client; a laptop will bottleneck before the
  server does. Our published runs used a 32-core client host.

## Files

- `veep_client.py` — the VectorDBBench `VectorDB` implementation (wraps the
  `veep` SDK's create/upsert/query/status calls; ~350 lines, no magic).
- `run_benchmark.py` — dataset download, load, converge, measure, JSON out.
- `results/` — raw harness output for the published runs.

If your numbers disagree materially with ours, we want to know:
https://www.vectorpanda.com/support
