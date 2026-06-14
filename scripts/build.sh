#!/usr/bin/env bash
set -e
export PATH="/usr/local/cuda-12.4/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
cd "$ROOT"

echo "==> cmake configure"
cmake -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -30

echo
echo "==> cmake build"
cmake --build build -j 2>&1 | tail -40

echo
echo "==> sanity test"
./build/search --hello
