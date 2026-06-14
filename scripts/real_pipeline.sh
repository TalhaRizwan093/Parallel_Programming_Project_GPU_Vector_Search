#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/cuda-12.4/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
cd "$ROOT"
mkdir -p reports

echo "==> 1. Train PQ codebook on real Wikipedia data"
if [ ! -f data/wiki_pq.bin ]; then
    python3 tools/prepare_dataset.py --pq --in data/wiki.bin --out data/wiki_pq.bin
else
    echo "    (already exists: $(stat -c%s data/wiki_pq.bin) bytes)"
fi

echo
echo "==> 2. Compute CPU reference top-10 (100 queries vs 1M db)"
if [ ! -f reports/cpu_reference_wiki.bin ]; then
    python3 tools/cpu_reference.py --db data/wiki.bin --n-queries 100 \
        --out reports/cpu_reference_wiki.bin
else
    echo "    (already exists)"
fi

echo
echo "==> 3. Run V1 (naive)"
./build/search --db data/wiki.bin --version v1 --n-queries 100 \
    --warmup 2 --runs 5 --bench --out reports/v1_wiki.bin
python3 tools/compare_topk.py reports/cpu_reference_wiki.bin reports/v1_wiki.bin || true

echo
echo "==> 4. Run V2 (tiled)"
./build/search --db data/wiki.bin --version v2 --n-queries 100 \
    --warmup 2 --runs 5 --bench --out reports/v2_wiki.bin
python3 tools/compare_topk.py reports/cpu_reference_wiki.bin reports/v2_wiki.bin || true

echo
echo "==> 5. Run V3 (PQ)"
./build/search --db data/wiki.bin --pq data/wiki_pq.bin --version v3 \
    --n-queries 100 --warmup 2 --runs 5 --bench --out reports/v3_wiki.bin
python3 tools/compare_topk.py reports/cpu_reference_wiki.bin reports/v3_wiki.bin || true

echo
echo "==> done."
