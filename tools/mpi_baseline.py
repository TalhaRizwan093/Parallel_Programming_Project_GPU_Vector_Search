from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

def load_db(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        return np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)

def merge_topk(a: list[tuple[float, int]], b: list[tuple[float, int]], k: int) -> list[tuple[float, int]]:
    return sorted(a + b, key=lambda x: x[0])[:k]

def main() -> int:
    from mpi4py import MPI

    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=Path("data/wiki.bin"))
    p.add_argument("--queries", type=int, default=10)
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        full_db = load_db(args.db)
        n, d = full_db.shape
        chunk_size = n // size
        chunks = [full_db[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]
    else:
        chunks = None
        d = 384

    local_db = comm.scatter(chunks, root=0)
    print(f"[rank {rank}] holds {local_db.shape[0]} db vectors", flush=True)

    for q_idx in range(args.queries):
        if rank == 0:
            query = local_db[q_idx % local_db.shape[0]].copy()
        else:
            query = np.empty(d, dtype=np.float32)

        comm.Bcast(query, root=0)

        t0 = MPI.Wtime()
        diffs = local_db - query
        dists = np.einsum("ij,ij->i", diffs, diffs)

        local_top = list(zip(dists[:args.k * 4].tolist(),
                             range(args.k * 4)))
        local_top = sorted(local_top, key=lambda x: x[0])[:args.k]

        global_top = comm.reduce(local_top, op=lambda a, b: merge_topk(a, b, args.k), root=0)
        elapsed = MPI.Wtime() - t0
        if rank == 0:
            print(f"[query {q_idx}] {elapsed*1000:.2f} ms  top-1 dist={global_top[0][0]:.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
