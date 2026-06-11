from __future__ import annotations

import argparse
import csv
import struct
import sys
import time
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
    p.add_argument("--queries", type=int, default=100)
    p.add_argument("--csv", type=Path, default=Path("reports/faiss_baseline.csv"))
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    import faiss

    print("[faiss] loading database...")
    vectors = load_fp32(args.in_path)
    n, d = vectors.shape
    print(f"[faiss] n={n:,}, d={d}")

    rng = np.random.default_rng(seed=42)
    q_idx = rng.choice(n, size=args.queries, replace=False)
    queries = vectors[q_idx].copy()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print("[faiss] IndexFlatL2 ...")
    res = faiss.StandardGpuResources()
    cpu_idx = faiss.IndexFlatL2(d)
    gpu_idx = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
    gpu_idx.add(vectors)

    gpu_idx.search(queries[:10], args.k)
    t0 = time.perf_counter()
    D, I = gpu_idx.search(queries, args.k)
    elapsed = time.perf_counter() - t0
    qps = len(queries) / elapsed
    rows.append({"index": "FlatL2", "qps": qps, "lat_ms": elapsed * 1000 / len(queries)})
    print(f"  FlatL2: QPS={qps:.1f}  latency={elapsed * 1000 / len(queries):.2f} ms")

    print("[faiss] IndexIVFPQ training...")
    quantiser = faiss.IndexFlatL2(d)
    cpu_idx = faiss.IndexIVFPQ(quantiser, d, 1024, 32, 8)
    cpu_idx.train(vectors)
    cpu_idx.add(vectors)
    gpu_idx = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
    gpu_idx.nprobe = 16
    gpu_idx.search(queries[:10], args.k)
    t0 = time.perf_counter()
    D2, I2 = gpu_idx.search(queries, args.k)
    elapsed = time.perf_counter() - t0
    qps = len(queries) / elapsed

    recall = float(np.mean([
        len(set(I[i]) & set(I2[i])) / args.k for i in range(len(queries))
    ]))
    rows.append({"index": "IVFPQ", "qps": qps, "lat_ms": elapsed * 1000 / len(queries), "recall_at_k": recall})
    print(f"  IVFPQ:  QPS={qps:.1f}  latency={elapsed*1000/len(queries):.2f} ms  recall@{args.k}={recall:.3f}")

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "qps", "lat_ms", "recall_at_k"])
        w.writeheader()
        w.writerows(rows)
    print(f"[faiss] wrote {args.csv}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
