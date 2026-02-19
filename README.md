# Zector

High-performance vector search library/database in Zig, designed around cache locality, SIMD, and low-allocation query paths.

## What is optimized

- SIMD dot/cosine kernels (AVX2, AVX-512, 8-wide NEON on Apple Silicon).
- Min-heap/max-heap HNSW search with early termination and prefetch-driven graph traversal.
- 3-level candidate filtering: binary Hamming → int8 approximate → f32 exact (each level is 10-50x cheaper).
- Fused IVF scan + rerank with zero intermediate allocation (all filtering in one pass).
- Data-oriented HNSW graph layout using contiguous `edge_pool` + `distance_pool` (SoA-style).
- Prefetch pipeline: quantized vectors, storage vectors, and graph node metadata prefetched ahead of use.
- Reusable per-thread search contexts (`visited`/candidate buffers) to avoid heap churn in hot query loops.
- Slab-based vector storage to avoid repeated giant-buffer reallocations.
- Parallel batch ingest and parallel batch search.
- Turbo path: IVF coarse filtering + binary/int8 quantization + HNSW assist + exact rerank.
- SIMD batch quantization for query vectors (f32 → i8 conversion).
- Binary quantization (1-bit sign encoding) with adaptive Hamming threshold for real embedding workloads.
- Sampled K-means for fast IVF construction + fused assignment/quantization/binary encoding in single pass.
- File-backed mode with mmap support for vectors and IVF postings.

## Quick start

```bash
zig build run -Doptimize=ReleaseFast
zig build benchmark
zig build test
```

## Local benchmark snapshot

Environment:

- Date (UTC): `2026-02-18`
- Zig: `0.14.0`
- Target: `aarch64-macos` (Apple M3 Max, 16 cores)

### Demo run (`zig build run -Doptimize=ReleaseFast`)

Workload: random `100,000` vectors, dimension `768`, `100` queries, `k=10`.

| Metric | Result |
|---|---:|
| Build time (HNSW + IVF) | `11,039 ms` |
| Ingest throughput | `9,058 vectors/sec` |
| Single-thread search | `1,315 QPS` |
| Multi-thread batch search | `10,000 QPS` |

### Benchmark harness (`zig build benchmark`)

High-recall mode (M=48, ef_construction=600, search_ef=512):

| Configuration | Index time | Throughput | Avg query time | QPS | Recall@10 |
|---|---:|---:|---:|---:|---:|
| Small (1K, 128d) | `113 ms` | `8,850 v/s` | `113 us` | `8,726` | `100%` |
| Medium (10K, 384d) | `530 ms` | `18,868 v/s` | `1,569 us` | `637` | `100%` |
| Large (50K, 768d) | `59,674 ms` | `838 v/s` | `16,585 us` | `60` | `95.7%` |

## Online comparisons (public benchmark references)

These are useful reference points, but not apples-to-apples with the local run above because datasets, hardware, recall targets, and deployment models differ.

### 1) Qdrant public benchmark (single-node ANN test)

For `dbpedia-openai-1M` (1536-dim) at precision target `0.99`, Qdrant publishes:

- `Qdrant`: `1238 RPS`, `p95 4.95 ms`
- `Milvus`: `1126 RPS`, `p95 5.76 ms`
- `Redis`: `344 RPS`, `p95 18.8 ms`

Source: [Qdrant benchmark](https://qdrant.tech/benchmarks/)

### 2) Pinecone published latency figures

Pinecone publishes latency benchmarks for a `10M` record workload (`1024` dimensions), including:

- `p50`: `7.8 ms`
- `p90`: `44 ms`
- `p99`: `82 ms`

Source: [Pinecone performance benchmarks](https://www.pinecone.io/benchmarks/)

### 3) VectorDBBench ecosystem leaderboard

VectorDBBench (Zilliz-maintained benchmark suite) publishes cross-system leaderboard snapshots. Recent sample values shown on the public benchmark page include:

- `milvus-s-t3`: `1110.6` (throughput value in table)
- `qdrant-cloud`: `911.72`
- `pinecone-serverless`: `696.95`

Sources:

- [VectorDBBench repository](https://github.com/zilliztech/VectorDBBench)
- [Public benchmark page](https://zilliz.com/benchmark)

## How to reproduce and compare fairly

Use the same:

- Dataset and embedding model
- `k`, recall target, and distance metric
- Hardware (CPU/RAM/storage class)
- Concurrency level and warmup policy

Then compare:

- QPS/throughput at fixed recall
- p50/p95/p99 latency
- Memory usage per vector
- Build time and update performance

## Architecture

```
Query → Normalize → Centroid Scan → IVF Probe Selection
                                         │
              ┌──────────────────────────┤
              ▼                          ▼
        Binary Hamming            Int8 Approximate
        Pre-filter (1-bit)        Distance (8-bit)
        ~12 XOR+popcount          ~48 SIMD ops
              │                          │
              └──────────┬───────────────┘
                         ▼
                  F32 Exact Rerank
                  (only survivors)
                         │
                         ▼
                  Max-Heap Top-K
```

Each filtering level is 10-50x cheaper than the next, creating a cascade that eliminates bad candidates early.

## Status

Tuned for low-overhead ANN search in Zig. Optimization passes applied:

- Heap-based HNSW search with early termination (was flat candidate list)
- 8-wide NEON accumulator kernel (was 4-wide) + 8-accumulator AVX2 for 768d
- SIMD int8 dot product with 2x unrolled accumulators
- Fused IVF scan + rerank with 3-level filtering cascade (zero intermediate allocation)
- Binary quantization (1-bit) with adaptive Hamming threshold for real embeddings
- SIMD batch query quantization (f32 → i8)
- Sampled K-means for IVF construction + fused assignment/quantization/binary pass
- Adaptive probe scaling: fast mode (low ef) vs recall mode (high ef + HNSW assist)
- Prefetch pipeline for graph nodes, vectors, and quantized data
- Parallel IVF assignment with thread-local counters

Next optimization passes could focus on:

- Product Quantization (PQ) with asymmetric distance computation for sub-microsecond scan
- Graph reordering for cache-optimal traversal patterns
- HNSW layer-aware batched insertion
