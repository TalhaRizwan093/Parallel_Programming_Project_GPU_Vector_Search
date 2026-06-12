from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.resolve()
ENGINE = ROOT / "build" / "search"
WIKI = ROOT / "data" / "wiki.bin"
PQ = ROOT / "data" / "wiki_pq.bin"
SYN = ROOT / "data" / "syn.bin"
SYN_PQ = ROOT / "data" / "syn_pq.bin"

K = 10

def load_db(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        return np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)

def cpu_reference_topk(db: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:

    q2 = (queries * queries).sum(axis=1, keepdims=True)
    db2 = (db * db).sum(axis=1)[None, :]
    d = q2 + db2 - 2 * (queries @ db.T)
    np.maximum(d, 0, out=d)
    part = np.argpartition(d, k, axis=1)[:, :k]
    rows = np.arange(d.shape[0])[:, None]
    return part[rows, np.argsort(d[rows, part], axis=1)]

def write_queries(path: Path, queries: np.ndarray) -> None:
    n, dim = queries.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", n, dim))
        queries.astype(np.float32).tofile(f)

def read_engine_out(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        raw = f.read()
    n_q = int.from_bytes(raw[0:4], "little", signed=True)
    k = int.from_bytes(raw[4:8], "little", signed=True)
    per_q = k * 4 + k * 4
    ids = np.zeros((n_q, k), dtype=np.int32)
    dists = np.zeros((n_q, k), dtype=np.float32)
    payload = raw[8:]
    for i in range(n_q):
        off = i * per_q
        ids[i] = np.frombuffer(payload[off:off + k * 4], dtype=np.int32)
        dists[i] = np.frombuffer(payload[off + k * 4:off + per_q], dtype=np.float32)
    return ids, dists

def run_kernel(version: str, db_path: Path, queries: np.ndarray,
               pq_path: Path | None = None) -> np.ndarray:

    if not ENGINE.exists():
        raise FileNotFoundError(f"engine binary missing: {ENGINE} (run scripts/build.sh)")
    tmp = Path(tempfile.gettempdir())
    qbin = tmp / f"trec_q_{os.getpid()}.bin"
    out = tmp / f"trec_out_{os.getpid()}.bin"
    write_queries(qbin, queries)
    cmd = [str(ENGINE),
           "--db", str(db_path),
           "--version", version,
           "--queries", str(qbin),
           "--out", str(out),
           "--warmup", "0", "--runs", "1", "--bench"]
    if version == "v3":
        if pq_path is None:
            raise ValueError("V3 needs --pq")
        cmd.extend(["--pq", str(pq_path)])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"engine failed for {version}: {proc.stderr}")
    ids, _ = read_engine_out(out)
    qbin.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    return ids

def recall_at_k(reference: np.ndarray, ours: np.ndarray, k: int | None = None) -> float:
    n = reference.shape[0]
    k_ref = reference.shape[1] if k is None else min(k, reference.shape[1])
    k_ours = ours.shape[1]
    hits = 0
    for i in range(n):
        ref_set = set(int(x) for x in reference[i, :k_ref])
        cand_set = set(int(x) for x in ours[i, :k_ours])
        hits += len(ref_set & cand_set)
    return hits / (n * k_ours)

def _pick_db_and_pq() -> tuple[Path, Path]:

    if WIKI.exists() and PQ.exists():
        return WIKI, PQ
    if SYN.exists() and SYN_PQ.exists():
        return SYN, SYN_PQ
    raise FileNotFoundError(
        "No dataset found. Run scripts/embed_real.sh OR "
        "python tools/make_synthetic.py + tools/prepare_dataset.py")

def _make_query_set(db: np.ndarray, n: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(db.shape[0], size=n, replace=False)
    return db[idx].astype(np.float32, copy=True)

def _shared_setup(n_queries: int = 50):
    db_path, pq_path = _pick_db_and_pq()
    db = load_db(db_path)
    queries = _make_query_set(db, n=n_queries)
    ref = cpu_reference_topk(db, queries, k=K)
    return db_path, pq_path, queries, ref

def test_v1_exact_recall():

    db_path, _, queries, ref = _shared_setup()
    ours = run_kernel("v1", db_path, queries)
    r = recall_at_k(ref, ours, k=K)
    assert r >= 0.99, f"V1 recall@10 = {r:.4f}, expected ≥0.99"

def test_v2_exact_recall():

    db_path, _, queries, ref = _shared_setup()
    ours = run_kernel("v2", db_path, queries)
    r = recall_at_k(ref, ours, k=K)
    assert r >= 0.99, f"V2 recall@10 = {r:.4f}, expected ≥0.99"

def test_v3_pq_soft_recall():

    db_path, pq_path, queries, _ = _shared_setup()
    db = load_db(db_path)
    ref_100 = cpu_reference_topk(db, queries, k=100)
    ours = run_kernel("v3", db_path, queries, pq_path=pq_path)

    n_q = ours.shape[0]
    hits = 0
    for i in range(n_q):
        ref_set = set(int(x) for x in ref_100[i])
        for x in ours[i]:
            if int(x) in ref_set:
                hits += 1
    soft = hits / (n_q * ours.shape[1])
    assert soft >= 0.90, f"V3.1 soft recall@10 = {soft:.4f}, expected ≥0.90"

def test_v3_top1_match():

    db_path, pq_path, queries, ref = _shared_setup()
    ours = run_kernel("v3", db_path, queries, pq_path=pq_path)
    top1_match = int(np.sum(ref[:, 0] == ours[:, 0]))
    rate = top1_match / ours.shape[0]
    assert rate >= 0.95, f"V3.1 top-1 match = {top1_match}/{ours.shape[0]} = {rate:.4f}"

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="all", choices=["v1", "v2", "v3", "all"])
    p.add_argument("--n-queries", type=int, default=20,
                   help="Smaller for --quick smoke testing")
    p.add_argument("--quick", action="store_true",
                   help="Use 20 queries instead of full suite's 50")
    args = p.parse_args()

    n_q = 20 if args.quick else args.n_queries
    print(f"[test] running with {n_q} queries on engine={ENGINE}")
    db_path, pq_path, queries, ref = _shared_setup(n_queries=n_q)
    print(f"[test] db={db_path.name}, pq={pq_path.name}, ref.shape={ref.shape}")

    versions = [args.version] if args.version != "all" else ["v1", "v2", "v3"]
    failed = 0
    for v in versions:
        try:
            ours = run_kernel(v, db_path, queries, pq_path=pq_path if v == "v3" else None)
            r = recall_at_k(ref, ours, k=K)
            print(f"[test] {v}: recall@10 = {r:.4f}")
            if v in ("v1", "v2"):
                if r < 0.99: print(f"        FAIL: expected ≥0.99"); failed += 1
                else: print(f"        PASS")
            else:
                if r < 0.50: print(f"        FAIL: PQ recall suspiciously low"); failed += 1
                else: print(f"        PASS (PQ recall is approximation-bounded)")
        except Exception as e:
            print(f"[test] {v}: ERROR {e}")
            failed += 1
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
