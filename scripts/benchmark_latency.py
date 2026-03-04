"""
延迟基准测试

测量不同 FastV 配置下的推理延迟和显存使用

用法:
    bash run.sh python scripts/benchmark_latency.py
    bash run.sh python scripts/benchmark_latency.py --model-path models/llava-v1.5-7b
"""

import argparse
import sys
import os
import time
import gc
import json

import torch
import numpy as np
from PIL import Image

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastv.core import load_model, prepare_input, run_vanilla, run_fastv


def benchmark_config(model, tokenizer, input_ids, image_tensor, device,
                     fastv_r=0.0, fastv_k=2, num_runs=5, max_new_tokens=128):
    """测试一个配置的延迟"""
    times = []

    for i in range(num_runs + 2):  # 前2次预热
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.time()

        with torch.no_grad():
            if fastv_r == 0.0:
                _ = run_vanilla(model, tokenizer, input_ids, image_tensor, device, max_new_tokens)
            else:
                _ = run_fastv(model, tokenizer, input_ids, image_tensor, device,
                              fastv_k=fastv_k, fastv_r=fastv_r, max_new_tokens=max_new_tokens)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        if i >= 2:  # 跳过预热
            times.append(elapsed)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    return {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'peak_mem_gb': float(peak_mem),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--num-runs', type=int, default=5, help='每个配置测试次数')
    args = parser.parse_args()

    device = 'cuda'
    print(f"加载模型: {args.model_path}")
    tokenizer, model, image_processor, context_len = load_model(args.model_path, device)

    # 创建测试图片 (灰色, 用于标准化测试)
    image = Image.new('RGB', (336, 336), color=(128, 128, 128))
    prompt = "Describe this image in detail."

    from fastv.core import prepare_input as prep
    input_ids, image_tensor = prep(tokenizer, image_processor, image, prompt, device)

    configs = [
        {'name': 'Vanilla (no FastV)', 'fastv_r': 0.0, 'fastv_k': 2},
        {'name': 'FastV K=2, R=25%', 'fastv_r': 0.25, 'fastv_k': 2},
        {'name': 'FastV K=2, R=50%', 'fastv_r': 0.50, 'fastv_k': 2},
        {'name': 'FastV K=2, R=75%', 'fastv_r': 0.75, 'fastv_k': 2},
        {'name': 'FastV K=3, R=50%', 'fastv_r': 0.50, 'fastv_k': 3},
        {'name': 'FastV K=3, R=75%', 'fastv_r': 0.75, 'fastv_k': 3},
    ]

    print(f"\n{'='*80}")
    print(f"{'Config':<25} | {'Mean(s)':>8} | {'Std(s)':>8} | {'Min(s)':>8} | {'Mem(GB)':>8} | {'Speedup':>8}")
    print(f"{'='*80}")

    baseline_time = None
    results = []

    for cfg in configs:
        gc.collect()
        torch.cuda.empty_cache()

        result = benchmark_config(
            model, tokenizer, input_ids, image_tensor, device,
            fastv_r=cfg['fastv_r'],
            fastv_k=cfg['fastv_k'],
            num_runs=args.num_runs,
        )

        if baseline_time is None:
            baseline_time = result['mean']

        speedup = baseline_time / result['mean'] if result['mean'] > 0 else 0

        print(f"{cfg['name']:<25} | {result['mean']:>8.3f} | {result['std']:>8.3f} | "
              f"{result['min']:>8.3f} | {result['peak_mem_gb']:>8.1f} | {speedup:>7.2f}x")

        results.append({**cfg, **result, 'speedup': float(speedup)})

    print(f"{'='*80}")

    os.makedirs('results', exist_ok=True)
    with open('results/latency_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到 results/latency_benchmark.json")


if __name__ == '__main__':
    main()
