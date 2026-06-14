#!/usr/bin/env bash
set -u    # do NOT use -e: we tolerate grep returning 1 on no match
export PATH="$HOME/.local/bin:/usr/local/cuda-12.4/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
cd "$ROOT"
mkdir -p reports build/sweep

OUT=reports/chunk_sweep.csv
echo "chunk,threads,median_ms,qps,recall_strict_10,recall_soft_10_vs_top100" > "$OUT"

NVCC="/usr/local/cuda-12.4/bin/nvcc -O3 -arch=sm_86 -std=c++17 -lineinfo"
CHUNK_VALUES=(2048 4096 8192 16384)
THREADS=256

for CHUNK in "${CHUNK_VALUES[@]}"; do
    BIN="build/sweep/search_c${CHUNK}"
    if [ ! -x "$BIN" ]; then
        echo "==> compiling CHUNK=$CHUNK"
        $NVCC \
            -DSWEEP_CHUNK=$CHUNK -DSWEEP_THREADS=$THREADS \
            -I engine/include \
            engine/main.cu \
            engine/kernels/v1_naive.cu \
            engine/kernels/v2_tiled.cu \
            engine/kernels/v3_pq_chunk_sweep.cu \
            -o "$BIN" || { echo "[fail] compile $CHUNK"; continue; }
    else
        echo "==> reusing $BIN"
    fi

    echo "==> benching CHUNK=$CHUNK"
    OUTBIN="reports/v3_c${CHUNK}.bin"
    BENCH_OUT=$("$BIN" --db data/wiki.bin --pq data/wiki_pq.bin --version v3 \
                     --n-queries 100 --warmup 2 --runs 5 --bench --out "$OUTBIN" 2>&1)
    MED=$(echo "$BENCH_OUT" | grep "median" | head -1 | awk '{print $3}')
    QPS=$(echo "$BENCH_OUT" | grep "QPS" | head -1 | awk '{print $3}')
    [ -z "$MED" ] && MED="NA"
    [ -z "$QPS" ] && QPS="NA"

    CMP=$(python3 tools/compare_topk.py reports/cpu_reference_wiki.bin "$OUTBIN" 2>&1)
    R10=$(echo "$CMP" | grep "top-10 vs top-10" | awk -F'= ' '{print $2}' | head -1 | tr -d ' ')
    RSOFT=$(echo "$CMP" | grep "ref-top-100" | awk -F'= ' '{print $2}' | awk '{print $1}' | head -1)
    [ -z "$R10" ] && R10="NA"
    [ -z "$RSOFT" ] && RSOFT="NA"

    echo "$CHUNK,$THREADS,$MED,$QPS,$R10,$RSOFT" | tee -a "$OUT"
done

echo
echo "==> sweep complete:"
cat "$OUT"
