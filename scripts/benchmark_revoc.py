"""
Comprehensive benchmark: Vanilla vs FastV vs ReVoC (cosine) vs ReVoC (learned)

Measures across N-round conversations:
  - Total latency
  - Image token count
  - Per-round breakdown

Usage:
    bash run.sh python scripts/benchmark_revoc.py
    bash run.sh python scripts/benchmark_revoc.py --rounds 5 --num-trials 3
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
from revoc import RevoCConfig, RevoCEngine, LLaVAAdapter


QUESTIONS = [
    "Describe this image in detail.",
    "What objects are in the foreground?",
    "What colors are prominent?",
    "Is there any text visible?",
    "What is the mood of this image?",
]


def benchmark_vanilla(model, tokenizer, image_processor, image, device, rounds, max_tok=128):
    total_time = 0
    total_tokens = 0
    for i in range(rounds):
        ids, tensor = prepare_input(tokenizer, image_processor, image, QUESTIONS[i], device)
        torch.cuda.synchronize()
        t0 = time.time()
        _ = run_vanilla(model, tokenizer, ids, tensor, device, max_tok)
        torch.cuda.synchronize()
        total_time += time.time() - t0
        total_tokens += 576
    return total_time, total_tokens


def benchmark_fastv(model, tokenizer, image_processor, image, device, rounds,
                    fastv_k=2, fastv_r=0.75, max_tok=128):
    total_time = 0
    kept = int(576 * (1 - fastv_r))
    total_tokens = 0
    for i in range(rounds):
        ids, tensor = prepare_input(tokenizer, image_processor, image, QUESTIONS[i], device)
        torch.cuda.synchronize()
        t0 = time.time()
        _ = run_fastv(model, tokenizer, ids, tensor, device,
                      fastv_k=fastv_k, fastv_r=fastv_r, max_new_tokens=max_tok)
        torch.cuda.synchronize()
        total_time += time.time() - t0
        total_tokens += kept
    return total_time, total_tokens


def benchmark_revoc(adapter, image, device, rounds, config, max_tok=128):
    config.max_new_tokens = max_tok
    engine = RevoCEngine(adapter, config, device)
    engine.start_session(image)

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(rounds):
        engine.chat(QUESTIONS[i])
    torch.cuda.synchronize()
    total_time = time.time() - t0

    summary = engine.get_summary()
    return total_time, summary['total_image_tokens']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--num-trials', type=int, default=3)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--retriever-weights', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda'
    rounds = min(args.rounds, len(QUESTIONS))

    # Load model
    print(f"Loading model: {args.model_path}")
    tokenizer, model, image_processor, _ = load_model(args.model_path, device)

    # Also load via adapter for ReVoC
    adapter = LLaVAAdapter()
    adapter.model = model
    adapter.tokenizer = tokenizer
    adapter.image_processor = image_processor
    adapter.device = device

    image = Image.new('RGB', (336, 336), color=(128, 128, 128))

    # Warmup
    print("Warmup...")
    ids, tensor = prepare_input(tokenizer, image_processor, image, "warmup", device)
    with torch.no_grad():
        _ = model(ids, images=tensor, use_cache=False)
    torch.cuda.synchronize()

    # Methods
    revoc_cosine_config = RevoCConfig(
        n_clusters=64, n_retrieve_clusters=12, retriever_type="cosine",
    )
    revoc_cosine_config.validate()

    methods = [
        ("Vanilla", lambda: benchmark_vanilla(
            model, tokenizer, image_processor, image, device, rounds, args.max_new_tokens)),
        ("FastV R=75%", lambda: benchmark_fastv(
            model, tokenizer, image_processor, image, device, rounds,
            max_tok=args.max_new_tokens)),
        ("ReVoC (cosine)", lambda: benchmark_revoc(
            adapter, image, device, rounds, revoc_cosine_config, args.max_new_tokens)),
    ]

    # Add learned retriever if weights available
    if args.retriever_weights and os.path.exists(args.retriever_weights):
        revoc_learned_config = RevoCConfig(
            n_clusters=64, n_retrieve_clusters=12, retriever_type="cross_attention",
        )
        revoc_learned_config.validate()
        methods.append(("ReVoC (learned)", lambda: benchmark_revoc(
            adapter, image, device, rounds, revoc_learned_config, args.max_new_tokens)))

    # Run benchmark
    print(f"\n{'=' * 76}")
    print(f"  {rounds}-round multi-turn benchmark ({args.num_trials} trials)")
    print(f"{'=' * 76}")
    print(f"  {'Method':<20}{'Time (s)':<14}{'Img Tokens':<14}{'Speedup':<10}{'Savings':<10}")
    print(f"  {'-' * 66}")

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

        print(f"  {name:<20}{avg_time:<14.3f}{tokens:<14}{speedup:<10.2f}x{savings:<10.1f}%")

        results.append({
            'method': name, 'rounds': rounds,
            'avg_time': float(avg_time), 'std_time': float(np.std(times)),
            'img_tokens': tokens, 'speedup': float(speedup),
            'savings_pct': float(savings),
        })

    print(f"{'=' * 76}")

    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak GPU memory: {mem_gb:.1f} GB")

    os.makedirs('results', exist_ok=True)
    with open('results/revoc_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/revoc_benchmark.json")


if __name__ == '__main__':
    main()
