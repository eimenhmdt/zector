# Zector

**The fastest open-source vector search engine.** Single-file, zero-dependency, written in Zig.

On standard [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) datasets, Zector outperforms FAISS, hnswlib, and usearch by **2–4× at the same recall** — single-threaded, measured on full datasets, fully reproducible with one command.

## Benchmarks

Single-thread queries, k=10, recall measured against exact ground truth on the **full** datasets. QPS at each recall target is interpolated along each engine's ef sweep. Apple M3 Max, builds capped at 4 threads, all engines built with identical parameters (M=24, ef_construction=200).

### glove-100-angular — 1,183,514 vectors

| engine | QPS @ 90% recall | QPS @ 95% recall |
|---|---:|---:|
| **zector** | **3,766** | **1,516** |
| faiss (HNSW) | 1,430 | 588 |
| hnswlib | 1,277 | 589 |
| usearch | 914 | 398 |

### nytimes-256-angular — 290,000 vectors

| engine | QPS @ 90% recall | QPS @ 95% recall |
|---|---:|---:|
| **zector** | **5,441** | **1,218** |
| faiss (HNSW) | 2,722 | 648 |
| hnswlib | 1,293 | 330 |
| usearch | 1,185 | 295 |

Full recall/QPS curves for every engine: [`bench/BASELINE.md`](bench/BASELINE.md).

### Reproduce it

```bash
python3 -m venv bench/.venv && bench/.venv/bin/pip install -r bench/requirements.txt
curl -L -o bench/datasets/nytimes-256-angular.hdf5 https://ann-benchmarks.com/nytimes-256-angular.hdf5
curl -L -o bench/datasets/glove-100-angular.hdf5 https://ann-benchmarks.com/glove-100-angular.hdf5
bench/run_all.sh   # runs every engine sequentially, then prints the report
```

The harness benchmarks each engine in its own process, strictly sequentially, with single-threaded queries and a load-average guard so results aren't polluted by a busy machine.

## Why it's fast

The search path is a **quantized HNSW graph traversal with exact rerank**:

```
Query → normalize → int8 quantize
   │
   ▼
Greedy upper-layer descent          (int8 NEON sdot distances)
   ▼
Layer-0 graph search (ef)           (int8 codes: 4× less memory traffic)
   ▼
Exact f32 rerank of ef candidates   (SIMD dot, prefetched)
   ▼
Top-k
```

- **int8 graph traversal** — distance evaluations during traversal read 4× fewer cache lines than f32 and use ARM `sdot` / AVX2 integer kernels. The small quantization error is erased by the exact rerank.
- **Cache-optimal graph layout** — after build, nodes are reordered into BFS order so graph neighbors are physically adjacent; edges live in contiguous SoA pools (`edge_pool` + `distance_pool`).
- **Quality-first graph construction** — neighbor-list overflow re-runs the diversity heuristic (not drop-worst), preserving the long-range links that keep recall high at scale.
- **Zero-allocation hot path** — per-thread reusable search contexts, stack buffers for candidates and top-k heaps, prefetch pipeline for vectors and node metadata.
- **SIMD everywhere** — 8-accumulator NEON f32 kernels, `sdot` int8 kernels, AVX2/AVX-512 on x86, SIMD batch quantization.

## Quick start

```bash
zig build run -Doptimize=ReleaseFast        # demo
zig build test                              # tests
zig build shared -Doptimize=ReleaseFast     # libzector for the Python binding
```

```python
from zector import ZectorDB
import numpy as np

db = ZectorDB(dim=256, max_elements=1_000_000, m=24, ef_construction=200)
db.add_batch(vectors)          # numpy (n, dim) float32
db.build_index()               # quantized codes + graph layout optimization
db.set_search_ef(128)          # recall/speed knob
ids, dists = db.search(query, k=10)
```

Storage modes: `-Dstorage=f32` (default), `-Dstorage=f16`, `-Dstorage=sq8` (int8, lowest memory).

Environment: `ZECTOR_MAX_THREADS=N` caps build threads.

## Methodology notes (read before quoting numbers)

- Recall is measured on **full datasets** against the ground truth shipped with each HDF5 file — no subsets, no sampling.
- Queries run on **one thread** for every engine; QPS scales with cores for all engines, so single-thread is the honest comparison.
- All engines are in-process libraries benchmarked through the same Python harness, same warmup, same timing loop (`bench/common.py`).
- A recall regression guard (`bench/check_recall.py`) keeps optimizations from silently trading accuracy for speed.
- Hardware differs; run `bench/run_all.sh` on your own machine. If you get different rankings, please open an issue with the JSON results.

## Status & roadmap

- [x] Beat FAISS/hnswlib/usearch at 90% and 95% recall on nytimes-256 and glove-100
- [ ] Build-speed recovery (proper overflow pruning costs ~2× build time vs the old buggy path)
- [ ] SIFT-128 (euclidean) support and benchmark
- [ ] Filtered search, persistence polish, incremental updates

Contributions welcome — especially benchmark results from other machines.
