"""
延迟基准测试

测量不同 FastV 配置下的推理延迟和显存使用

用法:
    python scripts/benchmark_latency.py
    python scripts/benchmark_latency.py --model-path models/llava-v1.5-7b
    python scripts/benchmark_latency.py --model-path liuhaotian/llava-v1.5-13b
"""

import argparse
import time
import gc
import torch
import numpy as np
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

from fastv.fastv_llama import compute_image_token_importance, select_important_tokens


def benchmark_config(model, processor, image, prompt, device, fastv_r=0.0, fastv_k=2, num_runs=5):
    """
    测试一个特定配置的延迟

    fastv_r=0.0 表示不剪枝 (vanilla)
    """
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    times = []

    for i in range(num_runs + 2):  # 前2次预热
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.time()

        with torch.no_grad():
            if fastv_r == 0.0:
                # Vanilla
                output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            else:
                # FastV
                outputs = model(**inputs, output_attentions=True, return_dict=True, use_cache=True)
                seq_len = outputs.logits.shape[1]

                attn_k = outputs.attentions[fastv_k - 1]
                image_length = 576
                image_start = max(0, seq_len - image_length - (inputs['input_ids'].shape[1] - 1))

                importance = compute_image_token_importance(attn_k, image_start, image_length)
                num_keep = int(image_length * (1 - fastv_r))
                keep_indices = select_important_tokens(importance, num_keep)

                # Build pruned KV cache
                prefix_idx = torch.arange(image_start, device=device)
                selected_img_idx = keep_indices[0] + image_start
                suffix_idx = torch.arange(image_start + image_length, seq_len, device=device)
                keep_all = torch.cat([prefix_idx, selected_img_idx, suffix_idx])

                pruned_past = tuple(
                    (k[:, :, keep_all, :], v[:, :, keep_all, :])
                    for k, v in outputs.past_key_values
                )

                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = [next_token]
                attn_mask = torch.ones((1, keep_all.shape[0] + 1), dtype=torch.long, device=device)
                eos = getattr(model.config, 'eos_token_id', 2)

                for _ in range(127):
                    out = model.language_model(
                        input_ids=next_token, attention_mask=attn_mask,
                        past_key_values=pruned_past, use_cache=True, return_dict=True,
                    )
                    pruned_past = out.past_key_values
                    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    if next_token.item() == eos:
                        break
                    generated_ids.append(next_token)
                    attn_mask = torch.cat([attn_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        if i >= 2:  # 跳过前2次预热
            times.append(elapsed)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'peak_mem_gb': peak_mem,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--num-runs', type=int, default=5, help='每个配置测试次数')
    args = parser.parse_args()

    device = 'cuda'
    print(f"加载模型: {args.model_path}")

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    # 创建测试图片 (纯色，用于标准化测试)
    image = Image.new('RGB', (336, 336), color=(128, 128, 128))
    prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

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
            model, processor, image, prompt, device,
            fastv_r=cfg['fastv_r'],
            fastv_k=cfg['fastv_k'],
            num_runs=args.num_runs,
        )

        if baseline_time is None:
            baseline_time = result['mean']

        speedup = baseline_time / result['mean'] if result['mean'] > 0 else 0

        print(f"{cfg['name']:<25} | {result['mean']:>8.3f} | {result['std']:>8.3f} | "
              f"{result['min']:>8.3f} | {result['peak_mem_gb']:>8.1f} | {speedup:>7.2f}x")

        results.append({**cfg, **result, 'speedup': speedup})

    print(f"{'='*80}")

    # 保存结果
    import json
    import os
    os.makedirs('results', exist_ok=True)
    with open('results/latency_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到 results/latency_benchmark.json")


if __name__ == '__main__':
    main()
