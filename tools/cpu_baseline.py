from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
import time
from pathlib import Path

K = 10
DIM = 384

def load_fp32(path: Path):
    import numpy as np
    with open(path, "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        return np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)

def time_method(name: str, fn, n_queries: int, n_runs: int = 3) -> dict:
    times = []

    fn()
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    median = times[len(times) // 2]
    return {
        "method": name,
        "n_queries": n_queries,
        "total_ms": median,
        "per_query_ms": median / n_queries,
        "qps": 1000.0 * n_queries / median,
    }

def run_single_thread_python(db, queries):

    import numpy as np
    n_q = queries.shape[0]
    out = np.zeros((n_q, K), dtype=np.int32)
    n_db = db.shape[0]
    for qi in range(n_q):
        q = queries[qi]

        diffs = db - q

        d = np.einsum("ij,ij->i", diffs, diffs)
        idx = np.argpartition(d, K)[:K]
        out[qi] = idx[np.argsort(d[idx])]
    return out

def run_numpy_matmul(db, queries):

    import numpy as np

    q2 = (queries * queries).sum(axis=1, keepdims=True)
    db2 = (db * db).sum(axis=1)[None, :]
    dot = queries @ db.T
    d = q2 + db2 - 2 * dot
    np.maximum(d, 0, out=d)
    idx = np.argpartition(d, K, axis=1)[:, :K]
    rows = np.arange(d.shape[0])[:, None]
    sorted_idx = idx[rows, np.argsort(d[rows, idx], axis=1)]
    return sorted_idx

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--n-queries", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("reports/cpu_baseline.csv"))
    p.add_argument("--skip-pure-python", action="store_true",
                   help="Skip the slowest pure-Python baseline (saves ~5 min on 1M db)")
    args = p.parse_args()

    import numpy as np

    db = load_fp32(args.db).astype(np.float32, copy=False)
    print(f"[cpu] db: {db.shape}, {db.nbytes / 1e9:.2f} GB")
    rng = np.random.default_rng(seed=42)
    step = max(1, db.shape[0] // args.n_queries)
    q_idx = [(i * step) % db.shape[0] for i in range(args.n_queries)]
    queries = db[q_idx].copy()
    print(f"[cpu] queries: {queries.shape}")

    rows = []

    if not args.skip_pure_python:
        print(f"[cpu] running pure-Python loop (this will be slow)...")
        r = time_method(
            "python-loop", lambda: run_single_thread_python(db, queries),
            n_queries=args.n_queries, n_runs=1)
        rows.append(r)
        print(f"     {r['method']:<22} {r['per_query_ms']:>10.2f} ms/q  {r['qps']:>10.1f} QPS")

    cpu_count = os.cpu_count() or 1
    print(f"[cpu] running numpy BLAS on {cpu_count} cores...")
    r = time_method(
        f"numpy-blas-{cpu_count}c", lambda: run_numpy_matmul(db, queries),
        n_queries=args.n_queries, n_runs=3)
    rows.append(r)
    print(f"     {r['method']:<22} {r['per_query_ms']:>10.2f} ms/q  {r['qps']:>10.1f} QPS")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n_queries", "total_ms", "per_query_ms", "qps"])
        w.writeheader()
        w.writerows(rows)
    print(f"[cpu] wrote {args.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
