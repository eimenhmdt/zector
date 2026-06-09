"""Recall regression guard for Zector.

Runs a quick 50K-subset nytimes benchmark and fails (exit 1) if recall at any
checked ef drops below the floors recorded in bench/recall_floors.json.

Update the floors deliberately (after verified recall improvements), never to
make a failing run pass.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FLOORS_PATH = os.path.join(HERE, "recall_floors.json")
RESULT_PATH = os.path.join(HERE, "results", "zector-nytimes-256-angular.json")


def main() -> None:
    with open(FLOORS_PATH) as f:
        spec = json.load(f)
    efs = ",".join(str(e) for e in sorted(int(k) for k in spec["floors"]))

    cmd = [
        os.path.join(HERE, ".venv", "bin", "python"),
        os.path.join(HERE, "bench_zector.py"),
        "--dataset", os.path.join(HERE, "datasets", "nytimes-256-angular.hdf5"),
        "--limit", str(spec["limit"]),
        "--queries", str(spec["queries"]),
        "--efs", efs,
    ]
    subprocess.run(cmd, check=True)

    with open(RESULT_PATH) as f:
        results = {r["ef"]: r["recall"] for r in json.load(f)["results"]}

    failed = False
    for ef_str, floor in spec["floors"].items():
        got = results.get(int(ef_str))
        status = "OK" if got is not None and got >= floor else "FAIL"
        if status == "FAIL":
            failed = True
        print(f"ef={ef_str}: recall={got} floor={floor} {status}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
