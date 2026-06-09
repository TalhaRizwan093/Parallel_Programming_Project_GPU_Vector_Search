from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

K = 100
DIM = 384

def load_fp32(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        arr = np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)
    return arr

def write_topk(path: Path, ids: np.ndarray, dists: np.ndarray) -> None:
    n_q, k = ids.shape
    assert ids.shape == dists.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", n_q, k))

        for i in range(n_q):
            f.write(ids[i].astype(np.int32).tobytes())
            f.write(dists[i].astype(np.float32).tobytes())

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--n-queries", type=int, default=100)
    p.add_argument("--queries", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    db = load_fp32(args.db)
    print(f"[cpu] db: {db.shape}, {db.nbytes / 1e6:.1f} MB")

    if args.queries is not None:
        queries = load_fp32(args.queries)
    else:
        step = max(1, db.shape[0] // args.n_queries)
        idx = np.array([(i * step) % db.shape[0] for i in range(args.n_queries)])
        queries = db[idx].copy()
    print(f"[cpu] queries: {queries.shape}")

    print("[cpu] computing exact top-10 (this is the reference)")

    q_sq = (queries * queries).sum(axis=1, keepdims=True)
    db_sq = (db * db).sum(axis=1)[None, :]
    dot = queries @ db.T
    dists = q_sq + db_sq - 2 * dot
    dists = np.maximum(dists, 0)

    part = np.argpartition(dists, K, axis=1)[:, :K]
    rows = np.arange(dists.shape[0])[:, None]
    part_dists = dists[rows, part]
    order = np.argsort(part_dists, axis=1)
    ids = part[rows, order]
    sorted_dists = part_dists[rows, order]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_topk(args.out, ids.astype(np.int32), sorted_dists.astype(np.float32))
    print(f"[cpu] wrote {args.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
