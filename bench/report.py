"""Merge bench/results/*.json into per-dataset tables and a QPS@recall ranking."""

import glob
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RECALL_TARGETS = [0.90, 0.95, 0.99]


def qps_at_recall(results, target):
    """Linear interpolation of QPS at a recall target along the engine's curve.

    Returns None if the engine never reaches the target.
    """
    pts = sorted(results, key=lambda r: r["recall"])
    below = [p for p in pts if p["recall"] < target]
    above = [p for p in pts if p["recall"] >= target]
    if not above:
        return None
    hi = min(above, key=lambda p: p["recall"])
    if not below:
        return hi["qps"]
    lo = max(below, key=lambda p: p["recall"])
    span = hi["recall"] - lo["recall"]
    if span <= 0:
        return hi["qps"]
    frac = (target - lo["recall"]) / span
    return lo["qps"] + frac * (hi["qps"] - lo["qps"])


def main() -> None:
    runs = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        with open(path) as f:
            runs.append(json.load(f))
    if not runs:
        raise SystemExit("No results in bench/results/")

    datasets = sorted({r["dataset"] for r in runs})
    for ds in datasets:
        ds_runs = [r for r in runs if r["dataset"] == ds]
        n_train = max(r["n_train"] for r in ds_runs)
        k = ds_runs[0]["k"]
        print(f"\n## {ds}  ({n_train:,} vectors, k={k}, single-thread queries)\n")

        print("| engine | build (s) | " + " | ".join(f"QPS @ {int(t*100)}% recall" for t in RECALL_TARGETS) + " |")
        print("|---|---:|" + "---:|" * len(RECALL_TARGETS))
        rows = []
        for r in sorted(ds_runs, key=lambda r: -(qps_at_recall(r["results"], 0.95) or 0)):
            cells = []
            for t in RECALL_TARGETS:
                q = qps_at_recall(r["results"], t)
                cells.append(f"{q:,.0f}" if q else "—")
            rows.append((r["engine"], r["build_seconds"], cells))
            print(f"| {r['engine']} | {r['build_seconds']:,.1f} | " + " | ".join(cells) + " |")

        print(f"\n### Full curves — {ds}\n")
        for r in sorted(ds_runs, key=lambda r: r["engine"]):
            print(f"**{r['engine']}** (params: {r['params']})")
            print("| ef | recall | QPS | avg us | p99 us |")
            print("|---:|---:|---:|---:|---:|")
            for p in r["results"]:
                print(f"| {p['ef']} | {p['recall']*100:.2f}% | {p['qps']:,.0f} | {p['avg_us']:,.0f} | {p['p99_us']:,.0f} |")
            print()


if __name__ == "__main__":
    main()
