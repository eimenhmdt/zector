# Zector: Fastest Open-Source Vector Index — Design

Date: 2026-06-10

## Goal

Beat hnswlib, usearch, and FAISS on **single-thread QPS at ≥95% recall@10**,
measured on full **nytimes-256-angular** and **glove-100-angular** datasets,
with fully reproducible scripts and a benchmark-led README.

## Why this metric

Single-thread QPS at fixed recall is how ann-benchmarks.com ranks engines.
It is the most credible, most checkable "fastest" claim for an open-source
launch. Multi-thread totals and build speed are secondary (reported, not
headlined).

## Baselines

All in-process, pip-installable libraries — apples-to-apples, no servers:

- **hnswlib** — the reference HNSW implementation
- **usearch** — the current fast newcomer
- **FAISS** (HNSW flat) — the Meta-backed gold standard

## Architecture of the work

### 1. Benchmark harness (`bench/`)

- One runner per engine (`bench_zector.py`, `bench_hnswlib.py`,
  `bench_usearch.py`, `bench_faiss.py`) sharing a common module for dataset
  loading, ground-truth recall, latency stats, and JSON result output.
- `run_all.sh` runs engines **sequentially, never in parallel**.
- Machine guardrails: 4 threads max for builds, 1 thread for queries,
  load-average check before each run, abort on memory pressure.
- Output: per-engine JSON + a combined recall-vs-QPS markdown table and plot.
- Datasets: nytimes-256-angular (~290K) and glove-100-angular (~1.18M),
  standard ann-benchmarks HDF5 format, full train sets, full query sets.

### 2. Baseline measurement

Run all four engines, sweep ef (or equivalent) to trace each engine's
recall/QPS curve. Identify Zector's standing at the 90/95/99% recall
operating points.

### 3. Optimization loop (iterate until #1 at 95% recall)

Profile the biggest gap and fix it; re-run Zector only; repeat. Candidate
areas already identified in the codebase:

- SQ8/int8 distance kernels (NEON on this machine)
- searchLayer memory layout and prefetch tuning
- ef-adaptive turbo cascade thresholds
- visited-set and candidate-heap overhead in the hot path

Constraint: `zig build test` stays green; recall at each ef must not regress
(recall regression check in the harness).

### 4. Ship

- README rewrite: headline table (QPS at 95% recall vs the three baselines),
  methodology section, one-command repro (`bench/run_all.sh`), honest
  hardware disclosure.
- All benchmark code committed.

## Success criteria

Zector has the highest QPS at ≥95% recall@10 on both datasets,
single-thread queries, on this machine — reproducible by anyone via the
committed scripts.

## Non-goals

- Server-mode benchmarks (Qdrant/Milvus/etc.)
- L2/euclidean datasets (SIFT) — possible follow-up
- Distributed/multi-node anything
