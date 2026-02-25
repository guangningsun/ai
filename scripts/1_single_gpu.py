#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from vllm import LLM, SamplingParams


def load_prompts(prompt_file: str = None) -> List[str]:
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    return [
        "Explain the concept of machine learning in simple terms.",
        "What are the main differences between supervised and unsupervised learning?",
        "Describe how a neural network processes information.",
        "What is the purpose of backpropagation in deep learning?",
        "Explain the vanishing gradient problem and how to address it.",
        "What are the advantages of using transformers over RNNs?",
        "How does attention mechanism work in sequence-to-sequence models?",
        "What is transfer learning and why is it important?",
        "Explain the difference between precision and recall.",
        "What is overfitting and how can it be prevented?",
    ] * 10


def run_single_gpu_benchmark(
    model_path: str,
    gpu_id: int,
    num_prompts: int = 100,
    max_tokens: int = 256,
    warmup: int = 3,
    output_file: str = None
) -> Dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"[Single-GPU {gpu_id}] Loading model from {model_path}...")
    start_load = time.time()
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    
    load_time = time.time() - start_load
    print(f"[Single-GPU {gpu_id}] Model loaded in {load_time:.2f}s")
    
    prompts = load_prompts()[:num_prompts]
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=0.95,
    )
    
    print(f"[Single-GPU {gpu_id}] Running warmup ({warmup} iterations)...")
    for i in range(warmup):
        _ = llm.generate(prompts[:2], sampling_params)
    
    print(f"[Single-GPU {gpu_id}] Running benchmark with {len(prompts)} prompts...")
    
    latencies = []
    total_tokens = 0
    start_time = time.time()
    
    for i in range(0, len(prompts), 10):
        batch = prompts[i:i+10]
        batch_start = time.time()
        outputs = llm.generate(batch, sampling_params)
        batch_time = time.time() - batch_start
        
        for output in outputs:
            latencies.append(batch_time / len(batch))
            total_tokens += len(output.outputs[0].token_ids)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    throughput = len(prompts) / total_time
    tokens_per_sec = total_tokens / total_time
    avg_latency = sum(latencies) / len(latencies)
    
    result = {
        "mode": "single_gpu",
        "gpu_id": gpu_id,
        "num_prompts": len(prompts),
        "max_tokens": max_tokens,
        "total_time": total_time,
        "throughput": throughput,
        "tokens_per_second": tokens_per_sec,
        "avg_latency": avg_latency,
        "load_time": load_time,
    }
    
    print(f"\n[Single-GPU {gpu_id}] Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.2f} prompts/s")
    print(f"  Tokens/sec: {tokens_per_sec:.2f}")
    print(f"  Avg latency: {avg_latency:.3f}s")
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Results saved to {output_file}")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data/shared_model")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"/data/benchmark/results/single_gpu_gpu{args.gpu_id}.json"
    
    run_single_gpu_benchmark(
        model_path=args.model_path,
        gpu_id=args.gpu_id,
        num_prompts=args.num_prompts,
        max_tokens=args.max_tokens,
        output_file=args.output
    )
