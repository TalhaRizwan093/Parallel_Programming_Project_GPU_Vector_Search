from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

def stream_paragraphs(
    snapshot: str,
    target_n: int,
    min_chars: int,
    max_chars: int,
):

    from datasets import load_dataset

    ds = load_dataset(
        "wikimedia/wikipedia", snapshot, split="train", streaming=True,
    )
    yielded = 0
    for article in ds:
        for para in article["text"].split("\n\n"):
            para = para.strip().replace("\n", " ")
            if min_chars <= len(para) <= max_chars:
                yield para
                yielded += 1
                if yielded >= target_n:
                    return

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--chunk", type=int, default=10_000,
                   help="Embed and write this many paragraphs at a time (memory bound).")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--snapshot", default="20231101.en")
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=2000)
    p.add_argument("--fp16", action="store_true", default=True,
                   help="Run the embedder in fp16 (default, much faster).")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    from sentence_transformers import SentenceTransformer
    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[embed] device={device}  model={args.model}")
    model = SentenceTransformer(args.model, device=device)

    dim = model.get_sentence_embedding_dimension()
    assert dim == 384, f"expected 384-dim, got {dim}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_path = args.out
    txt_path = args.out.with_suffix(".txt")

    with open(out_path, "wb") as f_bin, open(txt_path, "w", encoding="utf-8") as f_txt:
        f_bin.write(struct.pack("<ii", args.n, dim))

        buf: list[str] = []
        total_done = 0
        t0 = time.time()
        for para in stream_paragraphs(
            args.snapshot, args.n, args.min_chars, args.max_chars
        ):
            buf.append(para)
            if len(buf) >= args.chunk:
                emb = model.encode(
                    buf, batch_size=args.batch, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False,
                ).astype(np.float32)
                emb.tofile(f_bin)
                for s in buf:
                    f_txt.write(s + "\n")
                total_done += len(buf)
                buf.clear()
                rate = total_done / (time.time() - t0)
                eta = (args.n - total_done) / max(rate, 1e-6)
                print(f"[embed] {total_done:>8,} / {args.n:,}  "
                      f"({rate:.0f}/s, ETA {eta/60:.1f} min)",
                      flush=True)

        if buf:
            emb = model.encode(
                buf, batch_size=args.batch, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            ).astype(np.float32)
            emb.tofile(f_bin)
            for s in buf:
                f_txt.write(s + "\n")
            total_done += len(buf)

    with open(out_path, "r+b") as f:
        f.seek(0)
        f.write(struct.pack("<ii", total_done, dim))

    print(f"[embed] done.  {total_done:,} paragraphs written")
    print(f"        binary: {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")
    print(f"        text:   {txt_path}  ({txt_path.stat().st_size / 1e6:.1f} MB)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
