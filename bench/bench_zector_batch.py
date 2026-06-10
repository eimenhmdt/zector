"""Multi-core batch-search throughput for Zector.

Same recall protocol as the single-thread benches, but queries are submitted
through batch_search, which fans out across all cores.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from common import add_common_args, guard_machine, parse_efs, prepare, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Zector multi-core batch throughput")
    add_common_args(parser)
    parser.add_argument("--search-threads", type=int, default=0, help="0 = all cores")
    args = parser.parse_args()

    os.environ["ZECTOR_MAX_THREADS"] = str(args.build_threads)
    from zector import ZectorDB

    train, test, truth = prepare(args)
    n, dim = train.shape

    db = ZectorDB(dim=dim, max_elements=n + 16, m=args.m, ef_construction=args.ef_construction)
    print(f"Building (threads={args.build_threads})...")
    t0 = time.perf_counter()
    db.add_batch(train)
    db.build_index()
    build_s = time.perf_counter() - t0
    print(f"Built in {build_s:.2f}s")

    if args.search_threads:
        os.environ["ZECTOR_MAX_THREADS"] = str(args.search_threads)

    k = args.k
    nq = test.shape[0]
    results = []
    for ef in parse_efs(args.efs):
        db.set_search_ef(ef)
        db.batch_search(test[:64], k=k)  # warmup

        guard_machine()
        t0 = time.perf_counter()
        ids, _ = db.batch_search(test, k=k)
        elapsed = time.perf_counter() - t0

        total = 0.0
        counted = 0
        for i in range(nq):
            gt = truth[i]
            if not gt:
                continue
            total += len(set(int(v) for v in ids[i]) & gt) / len(gt)
            counted += 1
        recall = total / max(counted, 1)
        qps = nq / elapsed
        results.append({"ef": int(ef), "recall": round(recall, 5), "qps": round(qps, 1),
                        "avg_us": round(elapsed / nq * 1e6, 1), "p95_us": 0.0, "p99_us": 0.0})
        print(f"  ef={ef:>4}  recall@{k}={recall*100:6.2f}%  batch QPS={qps:>12,.0f}", flush=True)

    save_results("zector-batch", args.dataset, build_s, n, k,
                 {"m": args.m, "ef_construction": args.ef_construction,
                  "mode": "batch-multicore"}, results)


if __name__ == "__main__":
    main()
