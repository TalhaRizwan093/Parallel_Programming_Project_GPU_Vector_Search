#!/usr/bin/env bash
set -e
export PATH="$HOME/.local/bin:$PATH"

pip install --user --quiet --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0

pip install --user --quiet \
    sentence-transformers==3.0.1 \
    transformers==4.44.2 \
    tokenizers==0.19.1 \
    huggingface-hub==0.24.6 \
    datasets==2.21.0 \
    pyarrow==17.0.0 \
    faiss-gpu-cu12==1.8.0 \
    numpy==1.26.4 \
    scikit-learn==1.5.1 \
    matplotlib==3.9.2 \
    umap-learn==0.5.6 \
    flask==3.0.3 \
    python-docx==1.1.2 \
    pytest==8.3.2 \
    tqdm

python3 - <<'PY'
import torch
print("torch", torch.__version__, "CUDA", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
import sentence_transformers, datasets, faiss, sklearn, matplotlib, docx
print("all imports OK")
PY
