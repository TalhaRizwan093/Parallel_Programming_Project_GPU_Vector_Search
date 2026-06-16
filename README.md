# GPU Vector Search Engine

A from scratch CUDA k nearest neighbour search engine over 1 million Wikipedia paragraph embeddings. Three hand written kernels (naive, coalesced, product quantised) benchmarked against FAISS. The full write up is delivered separately as the project report.

## Prerequisites

- NVIDIA GPU with compute capability 8.6 (RTX 30 series). Other architectures work by passing `-DCMAKE_CUDA_ARCHITECTURES=<arch>` to cmake.
- NVIDIA driver 535 or newer.
- Linux or Windows with WSL2 (Ubuntu 22.04).
- About 30 GB free disk for the dataset and embeddings.

## Setup (one time)

```bash
bash setup_wsl.sh                  # system packages + CUDA Toolkit 12.4 (needs sudo)
bash scripts/install_deps.sh       # python deps (torch, sentence-transformers, faiss, etc)
bash scripts/build.sh              # compile the CUDA engine
```

## Quick smoke test (2 minutes, no dataset download)

Generates a 50k synthetic dataset, builds the engine, runs all three kernels, and checks recall against a CPU reference.

```bash
bash scripts/smoke_test.sh
```

## Full pipeline on real Wikipedia (about 60 minutes)

```bash
bash scripts/embed_real.sh 1000000   # embed 1M Wikipedia paragraphs (about 45 min, one time)
bash scripts/real_pipeline.sh        # train PQ codebook, build CPU reference, benchmark V1/V2/V3
bash scripts/run_baselines.sh        # CPU 16 core baseline and FAISS GPU SOTA comparison
bash scripts/chunk_sweep.sh          # optional, reproduces the V3.1 CHUNK sweep figure
```

## Run a single benchmark

```bash
./build/search --db data/wiki.bin --version v1 --n-queries 100 --bench
./build/search --db data/wiki.bin --version v2 --n-queries 100 --bench
./build/search --db data/wiki.bin --pq data/wiki_pq.bin --version v3 --n-queries 100 --bench
```

## Tests

```bash
python3 tests/test_recall.py --quick     # 20 query smoke test
pytest tests/test_recall.py              # full suite
```

## Live demo

```bash
python3 demo/ui.py                       # then open http://localhost:5000
```

## Docker

```bash
docker build -t gpu-search .
docker run --rm -it --gpus all -v $(pwd):/workspace gpu-search bash
```

## Layout

```
engine/      CUDA kernels (v1 naive, v2 coalesced, v3 product quantised) and the driver
tools/       data pipeline (embed, quantise, CPU reference, FAISS baseline, UMAP)
scripts/     setup and reproduction scripts
tests/       recall verification against a CPU reference
demo/        Flask search UI
```
