import argparse
import time

import h5py
import hnswlib
import numpy as np


def parse_efs(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="hnswlib baseline for ANN HDF5 datasets.")
    parser.add_argument("--dataset", default="nytimes-256-angular.hdf5")
    parser.add_argument("--limit", type=int, default=50_000, help="Train vectors to load. Use 0 for full train set.")
    parser.add_argument("--queries", type=int, default=200, help="Queries to run. Use 0 for all queries.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=400)
    parser.add_argument("--efs", default="128,256,512")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

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

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=train_count, ef_construction=args.ef_construction, M=args.m)
    index.set_num_threads(args.threads)

    print(f"\nIngesting {train_count:,} vectors...")
    ids = np.arange(train_count, dtype=np.uint64)
    t0 = time.perf_counter()
    index.add_items(train_data, ids)
    build_time = time.perf_counter() - t0
    print(f"Built in {build_time:.2f}s ({train_count / build_time:,.0f} vec/sec)")

    k = args.k
    loaded_ids = set(range(train_count))

    print("\nResults")
    print("| ef | Recall@{} | QPS | avg us | p95 us | p99 us |".format(k))
    print("|---:|---:|---:|---:|---:|---:|")

    for ef in parse_efs(args.efs):
        index.set_ef(ef)
        index.knn_query(test_data[0], k=k)

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
            labels, _ = index.knn_query(test_data[i], k=k)
            latencies[i] = time.perf_counter() - s

            found = len(set(int(v) for v in labels[0].tolist()) & ground_truth)
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
        print(f"| {ef} | {recall:5.1f}% | {qps:,.0f} | {avg_us:,.0f} | {p95_us:,.0f} | {p99_us:,.0f} |")


if __name__ == "__main__":
    main()
