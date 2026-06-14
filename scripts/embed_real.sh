#!/usr/bin/env bash
set -e
export PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
cd "$ROOT"
mkdir -p data reports

N="${1:-1000000}"
echo "[run] embedding $N Wikipedia paragraphs on GPU"
python3 tools/embed_wikipedia.py --n "$N" --out data/wiki.bin --batch 128 --chunk 20000
