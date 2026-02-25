#!/bin/bash
set -e

echo "Installing Ray and vLLM..."

pip install -U pip wheel

pip install vllm>=0.4.0
pip install ray>=2.10.0

pip install transformers>=4.40.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install tqdm jsonlines

echo "Installation completed!"
