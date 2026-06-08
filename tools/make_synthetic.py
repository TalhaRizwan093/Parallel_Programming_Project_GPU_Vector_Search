from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50_000)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--k", type=int, default=64, help="Number of latent clusters.")
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"[syn] generating {args.n:,} × {args.dim}-d vectors around {args.k} clusters")
    centroids = rng.standard_normal((args.k, args.dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    assign = rng.integers(0, args.k, size=args.n)
    noise = rng.standard_normal((args.n, args.dim)).astype(np.float32) * args.noise
    vectors = centroids[assign] + noise

    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    vectors = vectors.astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<ii", vectors.shape[0], vectors.shape[1]))
        vectors.tofile(f)
    print(f"[syn] wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
