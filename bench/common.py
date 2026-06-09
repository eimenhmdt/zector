"""Shared helpers for Zector benchmark suite.

All benchmarks follow the same protocol:
- full ann-benchmarks HDF5 dataset (train/test/neighbors)
- build with a small thread count, query with ONE thread
- recall@k against the dataset's provided ground truth
- sweep the engine's quality knob (ef) to trace a recall/QPS curve
"""

import json
import os
import time

import h5py
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MAX_LOAD_AVG = 8.0


def guard_machine() -> None:
    """Refuse to start a benchmark if the machine is already under load."""
    load1 = os.getloadavg()[0]
    if load1 > MAX_LOAD_AVG:
        raise SystemExit(f"Machine busy (load avg {load1:.1f} > {MAX_LOAD_AVG}); refusing to benchmark.")


def load_dataset(path: str, limit: int = 0, queries: int = 0):
    """Load an ann-benchmarks HDF5 file. Returns (train, test, neighbors)."""
    with h5py.File(path, "r") as f:
        train_ds, test_ds, neighbors_ds = f["train"], f["test"], f["neighbors"]
        n_train = train_ds.shape[0] if limit == 0 else min(limit, train_ds.shape[0])
        n_test = test_ds.shape[0] if queries == 0 else min(queries, test_ds.shape[0])
        train = np.asarray(train_ds[:n_train], dtype=np.float32)
        test = np.asarray(test_ds[:n_test], dtype=np.float32)
        neighbors = np.asarray(neighbors_ds[:n_test], dtype=np.int64)
    return train, test, neighbors


def subset_ground_truth(neighbors: np.ndarray, n_train: int, k: int):
    """Ground truth filtered to the loaded subset (only valid when limit != 0)."""
    truth = []
    for row in neighbors:
        filtered = [int(v) for v in row if int(v) < n_train][:k]
        truth.append(set(filtered))
    return truth


def full_ground_truth(neighbors: np.ndarray, k: int):
    return [set(int(v) for v in row[:k]) for row in neighbors]


def run_sweep(search_fn, set_ef_fn, efs, test, truth, k):
    """Sweep ef values; search_fn(query, k) -> iterable of ids.

    Returns list of dicts with recall/QPS/latency stats per ef.
    """
    results = []
    n = test.shape[0]
    for ef in efs:
        set_ef_fn(ef)
        search_fn(test[0], k)  # warmup (may trigger lazy rebuilds)

        latencies = np.empty(n, dtype=np.float64)
        total_recall = 0.0
        counted = 0
        t_total0 = time.perf_counter()
        for i in range(n):
            t0 = time.perf_counter()
            ids = search_fn(test[i], k)
            latencies[i] = time.perf_counter() - t0
            gt = truth[i]
            if not gt:
                continue
            found = len(set(int(v) for v in ids) & gt)
            total_recall += found / len(gt)
            counted += 1
        elapsed = time.perf_counter() - t_total0

        recall = total_recall / max(counted, 1)
        results.append({
            "ef": int(ef),
            "recall": round(recall, 5),
            "qps": round(n / elapsed, 1),
            "avg_us": round(float(np.mean(latencies)) * 1e6, 1),
            "p95_us": round(float(np.percentile(latencies, 95)) * 1e6, 1),
            "p99_us": round(float(np.percentile(latencies, 99)) * 1e6, 1),
        })
        print(f"  ef={ef:>4}  recall@{k}={recall*100:6.2f}%  qps={results[-1]['qps']:>10,.0f}  "
              f"avg={results[-1]['avg_us']:>8,.0f}us  p99={results[-1]['p99_us']:>8,.0f}us", flush=True)
    return results


def save_results(engine: str, dataset_path: str, build_seconds: float, n_train: int,
                 k: int, params: dict, results: list) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset = os.path.splitext(os.path.basename(dataset_path))[0]
    out = {
        "engine": engine,
        "dataset": dataset,
        "n_train": int(n_train),
        "k": int(k),
        "build_seconds": round(build_seconds, 2),
        "params": params,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"{engine}-{dataset}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {path}")
    return path


def parse_efs(raw: str):
    return [int(p.strip()) for p in raw.split(",") if p.strip()]


def add_common_args(parser):
    parser.add_argument("--dataset", default="bench/datasets/nytimes-256-angular.hdf5")
    parser.add_argument("--limit", type=int, default=0, help="Train vectors (0 = all)")
    parser.add_argument("--queries", type=int, default=0, help="Queries (0 = all)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, default=24)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--efs", default="16,24,32,48,64,96,128,192,256")
    parser.add_argument("--build-threads", type=int, default=4)


def prepare(args):
    """Common preamble: guard machine, load dataset, build ground truth."""
    guard_machine()
    print(f"Loading {args.dataset} (limit={args.limit}, queries={args.queries})")
    train, test, neighbors = load_dataset(args.dataset, args.limit, args.queries)
    print(f"Loaded {train.shape[0]:,} train vectors, {test.shape[0]:,} queries, dim={train.shape[1]}")
    if args.limit == 0:
        truth = full_ground_truth(neighbors, args.k)
    else:
        truth = subset_ground_truth(neighbors, train.shape[0], args.k)
    return train, test, truth
