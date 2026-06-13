from __future__ import annotations

import argparse
import struct
import time
import sys
from pathlib import Path

import numpy as np

def load_fp32(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        arr = np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)
    return arr

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("reports/umap.npz"))
    p.add_argument("--n-sample", type=int, default=100_000)
    p.add_argument("--n-clusters", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"[umap] loading {args.in_path}")
    db = load_fp32(args.in_path)
    print(f"[umap] db shape = {db.shape}")

    rng = np.random.default_rng(args.seed)
    n_sample = min(args.n_sample, db.shape[0])
    sample_idx = rng.choice(db.shape[0], size=n_sample, replace=False)
    sample = db[sample_idx]
    print(f"[umap] sampled {n_sample:,} vectors")

    print("[umap] k-means clustering for colors...")
    from sklearn.cluster import MiniBatchKMeans
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=args.seed,
                         batch_size=2048, max_iter=100, n_init=3)
    cluster = km.fit_predict(sample).astype(np.int32)
    print(f"[umap]   done in {time.time()-t0:.1f}s")

    print("[umap] running UMAP (this is the long step, ~10-15 min)...")
    import umap
    t0 = time.time()
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2,
        metric="cosine", random_state=args.seed,
        verbose=True,
    )
    xy = reducer.fit_transform(sample).astype(np.float32)
    print(f"[umap]   done in {time.time()-t0:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        xy=xy,
        cluster=cluster,
        sample_idx=sample_idx.astype(np.int32),
    )
    print(f"[umap] saved {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
