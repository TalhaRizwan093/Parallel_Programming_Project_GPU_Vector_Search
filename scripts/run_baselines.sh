#!/usr/bin/env bash
set -e
export PATH="$HOME/.local/bin:/usr/local/cuda-12.4/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"

ROOT="/mnt/d/University Work/Period 4/Parallel Programming/Project/Implementation"
cd "$ROOT"
mkdir -p reports

echo "==> 1. CPU baseline (numpy BLAS, skipping the python-loop one — takes too long)"
python3 tools/cpu_baseline.py --db data/wiki.bin --n-queries 100 \
    --skip-pure-python --out reports/cpu_baseline.csv

echo
echo "==> 2. FAISS GPU baseline (IndexFlatL2 + IndexIVFPQ)"
python3 tools/faiss_baseline.py --in data/wiki.bin --queries 100 \
    --csv reports/faiss_baseline.csv

echo
echo "==> 3. Show all results"
echo "-- CPU --"
cat reports/cpu_baseline.csv
echo
echo "-- FAISS --"
cat reports/faiss_baseline.csv
