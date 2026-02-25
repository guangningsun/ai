#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="/data/benchmark"
RESULTS_DIR="$BENCHMARK_DIR/results"
MODEL_PATH="/data/shared_model/Qwen2.5-14B-Instruct"
NUM_PROMPTS=100
MAX_TOKENS=256

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Ray + vLLM Benchmark Test"
echo "Model: Qwen/Qwen2.5-14B-Instruct"
echo "=========================================="

PYTHON="/root/miniconda3/envs/tf/bin/python"

echo ""
echo "[1/1] Testing 2-GPU Tensor Parallel (current host)..."
$PYTHON "$SCRIPT_DIR/2_dual_gpu_ray.py" \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size 2 \
    --num-prompts $NUM_PROMPTS \
    --max-tokens $MAX_TOKENS \
    --output "$RESULTS_DIR/tp_2_gpu.json"

echo ""
echo "=========================================="
echo "Benchmark completed!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="
