"""FAISS HNSW baseline runner. Build: N threads, query: 1 thread.

Uses IndexHNSWFlat with inner-product metric on L2-normalized vectors,
which is equivalent to cosine similarity (matches the angular datasets).
"""

import argparse
import os
import sys
import time

import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import add_common_args, parse_efs, prepare, run_sweep, save_results


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS HNSW recall/QPS benchmark")
    add_common_args(parser)
    args = parser.parse_args()

    train, test, truth = prepare(args)
    n, dim = train.shape
    train_n = normalize(train)
    test_n = normalize(test)

    index = faiss.IndexHNSWFlat(dim, args.m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = args.ef_construction

    print(f"Building (threads={args.build_threads})...")
    faiss.omp_set_num_threads(args.build_threads)
    t0 = time.perf_counter()
    index.add(train_n)
    build_s = time.perf_counter() - t0
    print(f"Built in {build_s:.2f}s ({n / build_s:,.0f} vec/s)")

    faiss.omp_set_num_threads(1)  # single-thread queries

    def set_ef(ef):
        index.hnsw.efSearch = ef

    def search_fn(q, k):
        _, ids = index.search(q.reshape(1, -1), k)
        return ids[0]

    # query vectors must be normalized too; wrap test set
    results = run_sweep(search_fn, set_ef, parse_efs(args.efs), test_n, truth, args.k)
    save_results("faiss", args.dataset, build_s, n, args.k,
                 {"m": args.m, "ef_construction": args.ef_construction,
                  "build_threads": args.build_threads},
                 results)


if __name__ == "__main__":
    main()
