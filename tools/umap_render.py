from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=Path, default=Path("reports/umap.npz"))
    p.add_argument("--out", type=Path, default=Path("reports/plot_umap.png"))
    args = p.parse_args()

    data = np.load(args.in_path)
    xy = data["xy"]
    cluster = data["cluster"]
    print(f"[umap-render] {xy.shape[0]:,} points, {cluster.max()+1} clusters")

    fig, ax = plt.subplots(figsize=(10, 9))
    cmap = plt.cm.get_cmap("tab20", cluster.max() + 1)
    ax.scatter(xy[:, 0], xy[:, 1], c=cluster, cmap=cmap,
               s=1.5, alpha=0.55, linewidths=0)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Embedding Space of 100,000 Wikipedia Paragraphs\n"
                 "(UMAP of all-MiniLM-L6-v2, coloured by k-means cluster)",
                 fontsize=12)
    ax.grid(linestyle=":", alpha=0.3)
    ax.set_aspect("equal")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[umap-render] saved {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
