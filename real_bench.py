import argparse
import os
import time

import h5py
import numpy as np


def parse_efs(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall/QPS benchmark for Zector on ANN HDF5 datasets.")
    parser.add_argument("--dataset", default="nytimes-256-angular.hdf5")
    parser.add_argument("--limit", type=int, default=50_000, help="Train vectors to load. Use 0 for full train set.")
    parser.add_argument("--queries", type=int, default=200, help="Queries to run. Use 0 for all queries.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, default=24)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--efs", default="32,64,128,256")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--disable-turbo", action="store_true", help="Search pure HNSW instead of IVF turbo.")
    args = parser.parse_args()

    os.environ["ZECTOR_MAX_THREADS"] = str(args.threads)

    from zector import ZectorDB

    print(f"Loading dataset: {args.dataset}")
    with h5py.File(args.dataset, "r") as f:
        train_ds = f["train"]
        test_ds = f["test"]
        neighbors_ds = f["neighbors"]

        train_count = train_ds.shape[0] if args.limit == 0 else min(args.limit, train_ds.shape[0])
        query_count = test_ds.shape[0] if args.queries == 0 else min(args.queries, test_ds.shape[0])

        train_data = np.asarray(train_ds[:train_count], dtype=np.float32)
        test_data = np.asarray(test_ds[:query_count], dtype=np.float32)
        neighbors = np.asarray(neighbors_ds[:query_count], dtype=np.uint64)

    dim = train_data.shape[1]
    print(f"Loaded: {train_count:,} train vectors, {query_count:,} queries, {dim} dimensions")
    if args.limit != 0:
        print("Recall note: subset mode filters ground truth to loaded vector IDs.")

    db = ZectorDB(dim=dim, max_elements=train_count + 16, m=args.m, ef_construction=args.ef_construction)
    db.set_turbo_enabled(not args.disable_turbo)

    print(f"\nIngesting {train_count:,} vectors...")
    t0 = time.perf_counter()
    db.add_batch(train_data)
    ingest_time = time.perf_counter() - t0
    print(f"Ingested in {ingest_time:.2f}s ({train_count / ingest_time:,.0f} vec/sec)")

    if args.disable_turbo:
        print("Turbo index disabled.")
    else:
        print("Building turbo index...")
        t0 = time.perf_counter()
        db.build_index()
        index_time = time.perf_counter() - t0
        print(f"Index built in {index_time:.2f}s")

    k = args.k
    loaded_ids = set(range(train_count))

    print("\nResults")
    print("| ef | Recall@{} | QPS | avg us | p95 us | p99 us |".format(k))
    print("|---:|---:|---:|---:|---:|---:|")

    for search_ef in parse_efs(args.efs):
        db.set_search_ef(search_ef)
        db.search(test_data[0], k=k)

        total_recall = 0.0
        recall_queries = 0
        latencies = np.empty(query_count, dtype=np.float64)

        start_total = time.perf_counter()
        for i in range(query_count):
            if args.limit == 0:
                ground_truth = set(int(v) for v in neighbors[i][:k])
                denom = k
            else:
                filtered = [int(v) for v in neighbors[i] if int(v) in loaded_ids]
                ground_truth = set(filtered[:k])
                denom = len(ground_truth)
                if denom == 0:
                    latencies[i] = 0.0
                    continue

            s = time.perf_counter()
            ids, _ = db.search(test_data[i], k=k)
            latencies[i] = time.perf_counter() - s

            found = len(set(int(v) for v in ids.tolist()) & ground_truth)
            total_recall += found / denom
            recall_queries += 1

        elapsed = time.perf_counter() - start_total
        used_latencies = latencies[:query_count]
        if args.limit != 0:
            used_latencies = used_latencies[used_latencies > 0.0]

        recall = (total_recall / max(recall_queries, 1)) * 100.0
        qps = query_count / elapsed
        avg_us = float(np.mean(used_latencies) * 1e6)
        p95_us = float(np.percentile(used_latencies, 95) * 1e6)
        p99_us = float(np.percentile(used_latencies, 99) * 1e6)
        print(f"| {search_ef} | {recall:5.1f}% | {qps:,.0f} | {avg_us:,.0f} | {p95_us:,.0f} | {p99_us:,.0f} |")


if __name__ == "__main__":
    main()
