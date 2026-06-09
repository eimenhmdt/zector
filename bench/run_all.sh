#!/bin/zsh
# Run the full benchmark suite: 4 engines x 2 datasets, strictly sequential.
# Guardrails: load-average gate before each run (also enforced inside each runner).
set -e
cd "$(dirname "$0")/.."

PY=bench/.venv/bin/python
DATASETS=(bench/datasets/nytimes-256-angular.hdf5 bench/datasets/glove-100-angular.hdf5)
ENGINES=(zector hnswlib usearch faiss)
EFS="${EFS:-16,24,32,48,64,96,128,192,256}"
QUERIES="${QUERIES:-0}"

wait_for_quiet() {
  while true; do
    load=$(sysctl -n vm.loadavg | awk '{print $2}')
    ok=$(echo "$load < 8.0" | bc)
    [ "$ok" = "1" ] && break
    echo "Load average $load too high; waiting 30s..."
    sleep 30
  done
}

# Ensure the zector shared lib is current
zig build shared -Doptimize=ReleaseFast

for ds in $DATASETS; do
  for eng in $ENGINES; do
    wait_for_quiet
    echo "\n=== $eng on $(basename $ds) ==="
    $PY bench/bench_$eng.py --dataset "$ds" --queries "$QUERIES" --efs "$EFS"
  done
done

echo "\n=== REPORT ==="
$PY bench/report.py
