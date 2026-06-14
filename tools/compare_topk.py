from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

def load_topk(path: Path) -> tuple[np.ndarray, np.ndarray]:

    with open(path, "rb") as f:
        raw = f.read()
    n_q = int.from_bytes(raw[0:4], "little", signed=True)
    K = int.from_bytes(raw[4:8], "little", signed=True)
    payload = raw[8:]
    per_q = K * 4 + K * 4
    if len(payload) != n_q * per_q:
        raise AssertionError(f"expected {n_q*per_q} bytes, got {len(payload)}")
    ids = np.zeros((n_q, K), dtype=np.int32)
    dists = np.zeros((n_q, K), dtype=np.float32)
    for i in range(n_q):
        off = i * per_q
        ids[i] = np.frombuffer(payload[off:off + K * 4], dtype=np.int32)
        dists[i] = np.frombuffer(payload[off + K * 4:off + per_q], dtype=np.float32)
    return ids, dists

def recall_at_k(ref_ids: np.ndarray, cand_ids: np.ndarray, k: int) -> float:
    n_q = ref_ids.shape[0]
    hits = 0
    for i in range(n_q):
        ref_set = set(int(x) for x in ref_ids[i][:k])
        cand_set = set(int(x) for x in cand_ids[i][:k])
        hits += len(ref_set & cand_set)
    return hits / (n_q * k)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    args = p.parse_args()

    ref_ids, ref_d = load_topk(args.reference)
    cand_ids, cand_d = load_topk(args.candidate)
    if ref_ids.shape[0] != cand_ids.shape[0]:
        print(f"[cmp] n_q mismatch: ref {ref_ids.shape[0]}, cand {cand_ids.shape[0]}",
              file=sys.stderr)
        return 1

    n_q = ref_ids.shape[0]

    K_ref = ref_ids.shape[1]
    K_cand = cand_ids.shape[1]
    K_common = min(K_ref, K_cand)

    print(f"[cmp] n_q={n_q}  K_ref={K_ref}  K_cand={K_cand}")
    for k in (1, 5, 10, 25, 50, 100):
        if k > K_common:
            break
        r = recall_at_k(ref_ids, cand_ids, k)
        print(f"[cmp] recall@{k:<3d} (top-{k} vs top-{k}) = {r:.4f}")

    if K_ref > K_cand:
        hits_soft = 0
        for i in range(n_q):
            ref_set = set(int(x) for x in ref_ids[i])
            for x in cand_ids[i]:
                if int(x) in ref_set:
                    hits_soft += 1
        soft = hits_soft / (n_q * K_cand)
        print(f"[cmp] recall@{K_cand:<3d} (cand-top vs ref-top-{K_ref}) = {soft:.4f}  <- soft / near-top")

    top1_match = int(np.sum(ref_ids[:, 0] == cand_ids[:, 0]))
    print(f"[cmp] top-1 exact-id match: {top1_match}/{n_q}")

    min_K = min(K_ref, K_cand)
    diff = float(np.abs(ref_d[:, :min_K] - cand_d[:, :min_K]).max())
    print(f"[cmp] max distance abs diff (first {min_K}): {diff:.4f}")

    main_recall = recall_at_k(ref_ids, cand_ids, K_common)
    return 0 if main_recall >= 0.99 else 2

if __name__ == "__main__":
    sys.exit(main())
