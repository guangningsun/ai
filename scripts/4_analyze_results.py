#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    results = []
    results_path = Path(results_dir)
    
    for json_file in results_path.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            data['file'] = json_file.name
            results.append(data)
    
    return results


def calculate_speedup(base: float, value: float) -> float:
    return (value - base) / base * 100


def print_benchmark_report(results: List[Dict[str, Any]]):
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS COMPARISON")
    print("=" * 70)
    
    single_gpu_0 = None
    single_gpu_1 = None
    dual_gpu = None
    
    for r in results:
        if r.get('mode') == 'single_gpu' and r.get('gpu_id') == 0:
            single_gpu_0 = r
        elif r.get('mode') == 'single_gpu' and r.get('gpu_id') == 1:
            single_gpu_1 = r
        elif r.get('mode') == 'dual_gpu_ray':
            dual_gpu = r
    
    if not single_gpu_0:
        print("ERROR: Single GPU 0 results not found!")
        return
    
    single_avg = {
        'throughput': (single_gpu_0['throughput'] + (single_gpu_1['throughput'] if single_gpu_1 else 0)) / 2,
        'tokens_per_second': (single_gpu_0['tokens_per_second'] + (single_gpu_1['tokens_per_second'] if single_gpu_1 else 0)) / 2,
        'avg_latency': (single_gpu_0['avg_latency'] + (single_gpu_1['avg_latency'] if single_gpu_1 else 0)) / 2,
    }
    
    print(f"\n{'Metric':<30} {'Single GPU':<15} {'Dual GPU Ray':<15} {'Speedup':<15}")
    print("-" * 75)
    
    throughput_speedup = calculate_speedup(single_avg['throughput'], dual_gpu['throughput'])
    print(f"{'Throughput (prompts/s)':<30} {single_avg['throughput']:<15.2f} {dual_gpu['throughput']:<15.2f} {throughput_speedup:>+10.1f}%")
    
    tps_speedup = calculate_speedup(single_avg['tokens_per_second'], dual_gpu['tokens_per_second'])
    print(f"{'Tokens/Second':<30} {single_avg['tokens_per_second']:<15.2f} {dual_gpu['tokens_per_second']:<15.2f} {tps_speedup:>+10.1f}%")
    
    latency_speedup = calculate_speedup(single_avg['avg_latency'], dual_gpu['avg_latency'])
    print(f"{'Avg Latency (s)':<30} {single_avg['avg_latency']:<15.3f} {dual_gpu['avg_latency']:<15.3f} {latency_speedup:>+10.1f}%")
    
    print(f"\n{'Load Time (s)':<30} {single_gpu_0['load_time']:<15.2f} {dual_gpu['load_time']:<15.2f}")
    print(f"{'Total Time (s)':<30} {single_gpu_0['total_time']:<15.2f} {dual_gpu['total_time']:<15.2f}")
    
    print("\n" + "-" * 75)
    print("DETAILED RESULTS:")
    print("-" * 75)
    
    for r in sorted(results, key=lambda x: x.get('mode', '')):
        print(f"\n{r['file']}:")
        print(f"  Mode: {r.get('mode')}, GPUs: {r.get('num_gpus', r.get('gpu_id', 'N/A'))}")
        print(f"  Prompts: {r.get('num_prompts')}, Max Tokens: {r.get('max_tokens')}")
        print(f"  Throughput: {r['throughput']:.2f} prompts/s")
        print(f"  Tokens/sec: {r['tokens_per_second']:.2f}")
        print(f"  Latency: {r['avg_latency']:.3f}s")
    
    print("\n" + "=" * 70)
    print("ANALYSIS:")
    print("=" * 70)
    
    if throughput_speedup > 0:
        print(f"✓ Dual GPU with Ray achieves {throughput_speedup:.1f}% higher throughput")
    else:
        print(f"✗ Dual GPU shows {abs(throughput_speedup):.1f}% lower throughput (possible overhead)")
    
    if tps_speedup > 0:
        print(f"✓ Token generation speed improved by {tps_speedup:.1f}%")
    else:
        print(f"✗ Token generation speed decreased by {abs(tps_speedup):.1f}%")
    
    if dual_gpu['load_time'] > single_gpu_0['load_time']:
        print(f"  Note: Dual GPU has longer load time ({dual_gpu['load_time']:.1f}s vs {single_gpu_0['load_time']:.1f}s)")
        print(f"         due to initializing 2 vLLM instances")
    
    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python 4_analyze_results.py <results_dir>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not os.path.isdir(results_dir):
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)
    
    results = load_results(results_dir)
    
    if not results:
        print("Error: No result files found!")
        sys.exit(1)
    
    print_benchmark_report(results)
    
    output_file = os.path.join(results_dir, "benchmark_report.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    main()
