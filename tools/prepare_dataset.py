from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

PQ_SUB = 32
PQ_DIM = 12
N_CENTROIDS = 256

def load_fp32(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        arr = np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)
    return arr

def train_pq(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    from sklearn.cluster import KMeans

    assert vectors.shape[1] == PQ_SUB * PQ_DIM, vectors.shape
    n = vectors.shape[0]

    rng = np.random.default_rng(seed=42)
    train_n = min(200_000, n)
    train_idx = rng.choice(n, size=train_n, replace=False)

    codebook = np.zeros((PQ_SUB, N_CENTROIDS, PQ_DIM), dtype=np.float32)
    codes = np.zeros((n, PQ_SUB), dtype=np.uint8)

    for s in range(PQ_SUB):
        sub_train = vectors[train_idx, s * PQ_DIM:(s + 1) * PQ_DIM]
        sub_all   = vectors[:,         s * PQ_DIM:(s + 1) * PQ_DIM]
        print(f"  PQ sub-quantiser {s+1:2d}/{PQ_SUB} "
              f"(train_n={train_n:,}, dim={PQ_DIM})")
        km = KMeans(
            n_clusters=N_CENTROIDS,
            init="k-means++",
            n_init=4,
            max_iter=50,
            tol=1e-5,
            algorithm="lloyd",
            random_state=42,
        )
        km.fit(sub_train)
        codebook[s] = km.cluster_centers_.astype(np.float32)
        codes[:, s] = km.predict(sub_all).astype(np.uint8)
    return codebook, codes

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", dest="out_path", type=Path, required=True)
    p.add_argument("--pq", action="store_true",
                   help="Produce PQ-compressed output (currently the only mode).")
    args = p.parse_args()

    if not args.pq:
        print("error: only --pq mode is implemented.", file=sys.stderr)
        return 2

    print(f"[prep] loading {args.in_path}")
    vectors = load_fp32(args.in_path)
    print(f"[prep] shape = {vectors.shape}, {vectors.nbytes/1e9:.2f} GB")

    print("[prep] training PQ codebook...")
    codebook, codes = train_pq(vectors)
    print(f"[prep] codebook shape = {codebook.shape}, codes shape = {codes.shape}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "wb") as f:
        f.write(struct.pack("<iiii", vectors.shape[0], PQ_SUB, PQ_DIM, N_CENTROIDS))
        codebook.tofile(f)
        codes.tofile(f)
    print(f"[prep] wrote {args.out_path}  ({args.out_path.stat().st_size/1e6:.1f} MB)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
