# Fastest Vector DB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Zector beat hnswlib, usearch, and FAISS on single-thread QPS at ≥95% recall@10 on nytimes-256-angular and glove-100-angular, with reproducible benchmarks and a viral-ready README.

**Architecture:** A `bench/` suite (shared `common.py` + one runner per engine + report generator + sequential `run_all.sh` with machine guardrails) establishes honest baselines; then an iterative measure→profile→optimize loop on `src/main.zig` closes gaps until Zector leads; finally README rewrite with the headline table.

**Tech Stack:** Zig (engine), Python 3 + h5py/numpy/hnswlib/usearch/faiss-cpu (benchmarks), ann-benchmarks HDF5 datasets.

**Machine guardrails (apply to every benchmark run):**
- Engines run sequentially, never in parallel. One benchmark process at a time.
- Build: ≤4 threads. Query: 1 thread.
- Before each run: abort if 1-min load average > 8.
- Datasets stay on disk in the original repo dir; symlinked, not copied.

---

### Task 1: Environment + datasets

**Files:**
- Create: `bench/requirements.txt`
- Create: `bench/datasets/` (symlink nytimes, download glove)

- [ ] **Step 1: Create venv and install deps**

```bash
python3 -m venv bench/.venv
bench/.venv/bin/pip install numpy h5py hnswlib usearch faiss-cpu
```

`bench/requirements.txt`:
```
numpy
h5py
hnswlib
usearch
faiss-cpu
```

- [ ] **Step 2: Link/download datasets**

```bash
mkdir -p bench/datasets
ln -sf /Users/eimen/Documents/random_code/zig/zector/nytimes-256-angular.hdf5 bench/datasets/
curl -L -o bench/datasets/glove-100-angular.hdf5 https://ann-benchmarks.com/glove-100-angular.hdf5
```

- [ ] **Step 3: Verify both open in h5py; commit (datasets gitignored)**

### Task 2: bench/common.py — shared loader/metrics

**Files:**
- Create: `bench/common.py`
- Create: `bench/.gitignore` (datasets, results, .venv)

- [ ] **Step 1: Implement common module**

`bench/common.py` provides:
- `load_dataset(path, limit=0, queries=0)` → `(train f32, test f32, neighbors u64)`
- `guard_machine()` → raise if `os.getloadavg()[0] > 8`
- `run_sweep(name, dataset, search_fn, set_ef_fn, efs, test, neighbors, k)` → list of result dicts `{ef, recall, qps, avg_us, p95_us, p99_us}`; full-dataset ground truth (first k of provided neighbors); one warmup query per ef.
- `save_results(engine, dataset, build_time, results)` → `bench/results/{engine}-{dataset}.json`

- [ ] **Step 2: Smoke test on nytimes with a brute-force sanity check (recall of ground truth vs itself = 100%)**

- [ ] **Step 3: Commit**

### Task 3: Engine runners

**Files:**
- Create: `bench/bench_zector.py` (port of `real_bench.py` onto common.py)
- Create: `bench/bench_hnswlib.py` (port of `hnswlib_bench.py`)
- Create: `bench/bench_usearch.py` (usearch `Index(ndim, metric='cos')`, `expansion_search` per ef)
- Create: `bench/bench_faiss.py` (`IndexHNSWFlat` + inner product on normalized vectors, `efSearch` per ef; `faiss.omp_set_num_threads(1)` for queries)

All runners: `--dataset --limit --queries --k --m --ef-construction --efs --build-threads` CLI; build with ≤4 threads; query single-thread; call `guard_machine()` first; emit JSON via `save_results`.

- [ ] **Step 1: Write all four runners**
- [ ] **Step 2: Smoke-run each on nytimes `--limit 20000 --queries 100` sequentially; verify sane recall (>80% at high ef)**
- [ ] **Step 3: Commit**

### Task 4: Report + orchestrator

**Files:**
- Create: `bench/report.py` — merges `bench/results/*.json`, prints per-dataset markdown table of recall/QPS curves, and computes interpolated **QPS at 95% recall** ranking.
- Create: `bench/run_all.sh` — sequential runs, load-average gate between engines, params: nytimes full (290K) and glove full (1.18M), efs sweep per engine.

- [ ] **Step 1: Write report.py and run_all.sh**
- [ ] **Step 2: Commit**

### Task 5: Full baselines

- [ ] **Step 1: Run full nytimes-256 sweep for all 4 engines (sequential)**
- [ ] **Step 2: Run full glove-100 sweep for all 4 engines (sequential)**
- [ ] **Step 3: Generate report; record standings at 90/95/99% recall in `bench/BASELINE.md`; commit**

### Task 6: Optimization loop (repeat until Zector leads at 95% recall on both datasets)

Protocol per iteration:
1. Identify the largest deficit (dataset × operating point) from the report.
2. Profile that exact configuration (`zig build -Doptimize=ReleaseFast` + `xctrace`/`samply`/manual counters as appropriate).
3. Implement ONE targeted optimization in `src/main.zig`. Candidate backlog (ordered by expected impact, from code review + memory):
   - SQ8/int8 NEON distance kernels and cascade thresholds in turbo path
   - searchLayer visited-set & candidate-heap overhead; memory layout of node metadata
   - ef-adaptive turbo probe counts; binary Hamming filter gating (effective on real embeddings per memory: threshold < 0.6)
   - prefetch distances tuning for M3 Max
4. `zig build test` must pass.
5. Re-run **Zector only** on the affected dataset; recall at each ef must not drop >0.5pp.
6. Commit with measured before/after numbers in the message.

- [ ] Iterate until success criterion met; keep `bench/BASELINE.md` updated.

### Task 7: Recall regression guard

**Files:**
- Create: `bench/check_recall.py` — fails (exit 1) if Zector recall at ef=128 on nytimes-50K subset < stored floor.

- [ ] **Step 1: Write check + stored floor; run; commit**

### Task 8: README + ship

**Files:**
- Modify: `README.md` — headline recall-vs-QPS table, methodology (hardware, thread policy, dataset, full repro via `bench/run_all.sh`), honest caveats.

- [ ] **Step 1: Rewrite README with final numbers**
- [ ] **Step 2: `zig build test` green; final commit**
