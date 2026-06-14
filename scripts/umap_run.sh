#!/usr/bin/env bash
set -e
export PATH="$HOME/.local/bin:/usr/bin:/bin"
cd "/mnt/d/University Work/Period 4/Parallel Programming/Project/Implementation"

if ! python3 -c "import umap" 2>/dev/null; then
    pip install --user --quiet umap-learn==0.5.6
fi

python3 tools/umap_precompute.py --in data/wiki.bin --out reports/umap.npz --n-sample 100000
