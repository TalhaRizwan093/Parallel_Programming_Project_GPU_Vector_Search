#!/usr/bin/env bash

set -euo pipefail

echo "==> WSL environment setup starting"
echo "    user: $(whoami)"
echo "    pwd:  $(pwd)"

echo "==> Installing build tools (apt)..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    gnupg \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev

if ! command -v nvcc >/dev/null 2>&1; then
    echo "==> Installing CUDA Toolkit 12.4 for WSL-Ubuntu..."
    TMPDIR=$(mktemp -d)
    cd "$TMPDIR"
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -y
    sudo apt-get install -y cuda-toolkit-12-4
    cd - >/dev/null
    rm -rf "$TMPDIR"
else
    echo "==> nvcc already installed: $(nvcc --version | grep release)"
fi

if ! grep -q "cuda-12.4/bin" "$HOME/.bashrc"; then
    cat >> "$HOME/.bashrc" <<'EOF'

export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}
EOF
fi
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}

if [ ! -d .venv ]; then
    echo "==> Creating Python venv in .venv/"
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip wheel

pip install --no-cache-dir numpy==1.26.4 matplotlib==3.9.2 pytest==8.3.2 ruff==0.5.7

echo
echo "==> Setup complete."
echo "    nvcc: $(nvcc --version | grep release || echo MISSING)"
echo "    gcc:  $(gcc --version | head -1)"
echo "    cmake: $(cmake --version | head -1)"
echo "    python: $(python3 --version)"
echo
echo "Next: from this same shell, run:"
echo "    cmake -B build -DCMAKE_BUILD_TYPE=Release"
echo "    cmake --build build -j"
echo "    ./build/search --hello"
