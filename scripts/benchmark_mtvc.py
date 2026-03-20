"""
Vanilla vs FastV vs MTVC 多轮延迟对比

测量 5 轮对话场景下三种方法的总延迟和 token 用量

用法:
    bash run.sh python scripts/benchmark_mtvc.py
    bash run.sh python scripts/benchmark_mtvc.py --model-path models/llava-v1.5-7b --rounds 5
"""

import argparse
import gc
import json
import os
import sys
import time

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastv.core import load_model, prepare_input, run_vanilla, run_fastv
from mtvc import MTVCConfig, MultiTurnEngine


QUESTIONS = [
    "Describe this image in detail.",
    "What objects are in the foreground?",
    "What colors are prominent?",
    "Is there any text visible?",
    "What is the mood of this image?",
]


def benchmark_vanilla(model, tokenizer, image_processor, image, device, rounds, max_new_tokens=128):
    """Vanilla: each round processes all 576 image tokens independently."""
    total_time = 0
    total_tokens = 0

    for i in range(rounds):
        input_ids, image_tensor = prepare_input(
            tokenizer, image_processor, image, QUESTIONS[i], device
        )
        torch.cuda.synchronize()
        start = time.time()
        _ = run_vanilla(model, tokenizer, input_ids, image_tensor, device, max_new_tokens)
        torch.cuda.synchronize()
        total_time += time.time() - start
        total_tokens += 576

    return total_time, total_tokens


def benchmark_fastv(model, tokenizer, image_processor, image, device, rounds,
                    fastv_k=2, fastv_r=0.75, max_new_tokens=128):
    """FastV: each round prunes to (1-R)*576 tokens independently."""
    total_time = 0
    kept = int(576 * (1 - fastv_r))
    total_tokens = 0

    for i in range(rounds):
        input_ids, image_tensor = prepare_input(
            tokenizer, image_processor, image, QUESTIONS[i], device
        )
        torch.cuda.synchronize()
        start = time.time()
        _ = run_fastv(model, tokenizer, input_ids, image_tensor, device,
                      fastv_k=fastv_k, fastv_r=fastv_r, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        total_time += time.time() - start
        total_tokens += kept

    return total_time, total_tokens


def benchmark_mtvc(model, tokenizer, image_processor, image, device, rounds,
                   config=None, max_new_tokens=128):
    """MTVC: round 1 full + round 2+ retrieval."""
    config = config or MTVCConfig(max_new_tokens=max_new_tokens)
    engine = MultiTurnEngine(model, tokenizer, image_processor, config, device)
    engine.start_session(image)

    torch.cuda.synchronize()
    total_start = time.time()

    for i in range(rounds):
        engine.chat(QUESTIONS[i])

    torch.cuda.synchronize()
    total_time = time.time() - total_start
    summary = engine.get_summary()
    total_tokens = summary['total_image_tokens']

    return total_time, total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--num-trials', type=int, default=3, help='重复试验次数')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    args = parser.parse_args()

    device = 'cuda'
    rounds = min(args.rounds, len(QUESTIONS))

    print(f"加载模型: {args.model_path}")
    tokenizer, model, image_processor, _ = load_model(args.model_path, device)

    image = Image.new('RGB', (336, 336), color=(128, 128, 128))

    # Warmup
    print("预热...")
    input_ids, image_tensor = prepare_input(tokenizer, image_processor, image, "warmup", device)
    with torch.no_grad():
        _ = model(input_ids, images=image_tensor, use_cache=False)
    torch.cuda.synchronize()

    methods = [
        ("Vanilla", lambda: benchmark_vanilla(
            model, tokenizer, image_processor, image, device, rounds, args.max_new_tokens)),
        ("FastV R=75%", lambda: benchmark_fastv(
            model, tokenizer, image_processor, image, device, rounds,
            max_new_tokens=args.max_new_tokens)),
        ("MTVC", lambda: benchmark_mtvc(
            model, tokenizer, image_processor, image, device, rounds,
            max_new_tokens=args.max_new_tokens)),
    ]

    print(f"\n{'=' * 72}")
    print(f"  {rounds}-round multi-turn benchmark ({args.num_trials} trials)")
    print(f"{'=' * 72}")
    print(f"  {'Method':<16}{'Time (s)':<14}{'Img Tokens':<14}{'Speedup':<10}{'Savings':<10}")
    print(f"  {'-' * 60}")

    results = []
    baseline_time = None

    for name, fn in methods:
        gc.collect()
        torch.cuda.empty_cache()

        times = []
        tokens = None
        for _ in range(args.num_trials):
            t, tok = fn()
            times.append(t)
            tokens = tok

        avg_time = np.mean(times)
        if baseline_time is None:
            baseline_time = avg_time

        speedup = baseline_time / avg_time if avg_time > 0 else 0
        vanilla_tokens = 576 * rounds
        savings = (1 - tokens / vanilla_tokens) * 100

        print(f"  {name:<16}{avg_time:<14.3f}{tokens:<14}{speedup:<10.2f}x{savings:<10.1f}%")

        results.append({
            'method': name,
            'rounds': rounds,
            'avg_time': float(avg_time),
            'std_time': float(np.std(times)),
            'img_tokens': tokens,
            'speedup': float(speedup),
            'savings_pct': float(savings),
        })

    print(f"{'=' * 72}")

    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  峰值显存: {mem_gb:.1f} GB")

    os.makedirs('results', exist_ok=True)
    with open('results/mtvc_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到 results/mtvc_benchmark.json")


if __name__ == '__main__':
    main()
