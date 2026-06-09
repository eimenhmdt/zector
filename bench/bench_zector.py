"""Zector benchmark runner. Build: ZECTOR_MAX_THREADS, query: single Python thread."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from common import add_common_args, parse_efs, prepare, run_sweep, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Zector recall/QPS benchmark")
    add_common_args(parser)
    parser.add_argument("--disable-turbo", action="store_true")
    args = parser.parse_args()

    os.environ["ZECTOR_MAX_THREADS"] = str(args.build_threads)
    from zector import ZectorDB

    train, test, truth = prepare(args)
    n, dim = train.shape

    db = ZectorDB(dim=dim, max_elements=n + 16, m=args.m, ef_construction=args.ef_construction)
    db.set_turbo_enabled(not args.disable_turbo)

    print(f"Building (threads={args.build_threads})...")
    t0 = time.perf_counter()
    db.add_batch(train)
    if not args.disable_turbo:
        db.build_index()
    build_s = time.perf_counter() - t0
    print(f"Built in {build_s:.2f}s ({n / build_s:,.0f} vec/s)")

    def search_fn(q, k):
        ids, _ = db.search(q, k=k)
        return ids

    results = run_sweep(search_fn, db.set_search_ef, parse_efs(args.efs), test, truth, args.k)
    save_results("zector", args.dataset, build_s, n, args.k,
                 {"m": args.m, "ef_construction": args.ef_construction,
                  "turbo": not args.disable_turbo, "build_threads": args.build_threads},
                 results)


if __name__ == "__main__":
    main()
