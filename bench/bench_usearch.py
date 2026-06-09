"""usearch baseline runner. Build: N threads, query: 1 thread."""

import argparse
import os
import sys
import time

import numpy as np
from usearch.index import Index

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import add_common_args, parse_efs, prepare, run_sweep, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="usearch recall/QPS benchmark")
    add_common_args(parser)
    parser.add_argument("--dtype", default="f32", help="usearch internal dtype: f32, f16, i8, b1")
    args = parser.parse_args()

    train, test, truth = prepare(args)
    n, dim = train.shape

    index = Index(ndim=dim, metric="cos", dtype=args.dtype,
                  connectivity=args.m, expansion_add=args.ef_construction)

    print(f"Building (threads={args.build_threads}, dtype={args.dtype})...")
    t0 = time.perf_counter()
    index.add(np.arange(n, dtype=np.uint64), train, threads=args.build_threads)
    build_s = time.perf_counter() - t0
    print(f"Built in {build_s:.2f}s ({n / build_s:,.0f} vec/s)")

    def set_ef(ef):
        index.expansion_search = ef

    def search_fn(q, k):
        matches = index.search(q, k, threads=1)
        return matches.keys

    results = run_sweep(search_fn, set_ef, parse_efs(args.efs), test, truth, args.k)
    save_results("usearch", args.dataset, build_s, n, args.k,
                 {"m": args.m, "ef_construction": args.ef_construction,
                  "dtype": args.dtype, "build_threads": args.build_threads},
                 results)


if __name__ == "__main__":
    main()
