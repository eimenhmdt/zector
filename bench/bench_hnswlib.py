"""hnswlib baseline runner. Build: N threads, query: 1 thread."""

import argparse
import os
import sys
import time

import hnswlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import add_common_args, parse_efs, prepare, run_sweep, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="hnswlib recall/QPS benchmark")
    add_common_args(parser)
    args = parser.parse_args()

    train, test, truth = prepare(args)
    n, dim = train.shape

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, ef_construction=args.ef_construction, M=args.m)

    print(f"Building (threads={args.build_threads})...")
    index.set_num_threads(args.build_threads)
    t0 = time.perf_counter()
    index.add_items(train, np.arange(n, dtype=np.uint64))
    build_s = time.perf_counter() - t0
    print(f"Built in {build_s:.2f}s ({n / build_s:,.0f} vec/s)")

    index.set_num_threads(1)  # single-thread queries

    def search_fn(q, k):
        labels, _ = index.knn_query(q, k=k)
        return labels[0]

    results = run_sweep(search_fn, index.set_ef, parse_efs(args.efs), test, truth, args.k)
    save_results("hnswlib", args.dataset, build_s, n, args.k,
                 {"m": args.m, "ef_construction": args.ef_construction,
                  "build_threads": args.build_threads},
                 results)


if __name__ == "__main__":
    main()
