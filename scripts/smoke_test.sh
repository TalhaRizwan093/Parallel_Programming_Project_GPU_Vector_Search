#!/usr/bin/env bash

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
cd "$ROOT"

export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}

if [ -d .venv ]; then
    source .venv/bin/activate
fi

echo "==> Step 1/6: build CUDA engine"
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

echo "==> Step 2/6: generate synthetic dataset"
mkdir -p data reports
if [ ! -f data/syn.bin ]; then
    python3 tools/make_synthetic.py --n 50000 --out data/syn.bin
fi

echo "==> Step 3/6: train PQ codebook"
pip install -q scikit-learn==1.5.1
if [ ! -f data/syn_pq.bin ]; then
    python3 tools/prepare_dataset.py --pq --in data/syn.bin --out data/syn_pq.bin
fi

echo "==> Step 4/6: compute CPU reference"
python3 tools/cpu_reference.py --db data/syn.bin --n-queries 100 \
    --out reports/cpu_reference.bin

echo "==> Step 5/6: run GPU kernels"
for v in v1 v2; do
    echo "    -- $v --"
    ./build/search --db data/syn.bin --version "$v" --n-queries 100 \
        --warmup 3 --runs 5 --bench --out "reports/${v}_out.bin"
    python3 tools/compare_topk.py reports/cpu_reference.bin "reports/${v}_out.bin" \
        || echo "    [warn] recall below 99% for $v"
done

echo "    -- v3 (PQ) --"
./build/search --db data/syn.bin --pq data/syn_pq.bin --version v3 --n-queries 100 \
    --warmup 3 --runs 5 --bench --out reports/v3_out.bin
python3 tools/compare_topk.py reports/cpu_reference.bin reports/v3_out.bin \
    || echo "    [info] PQ recall < 99% is expected (we target ≥95%)"

echo "==> Step 6/6: done."
echo
echo "Reports written to ./reports/"
ls -la reports/
