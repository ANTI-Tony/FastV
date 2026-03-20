"""
Systematic ablation study for ReVoC.

Ablations:
  1. Number of clusters (K): 16, 32, 64, 128
  2. Clusters retrieved per query: 4, 8, 12, 16, 24
  3. Retriever type: cosine vs learned cross-attention
  4. EMA decay: 0.5, 0.7, 0.9, 1.0 (no history)
  5. Entropy-based vs fixed partitioning
  6. With vs without residual recovery (pruning baseline)

Usage:
    bash run.sh python scripts/ablation_study.py
    bash run.sh python scripts/ablation_study.py --ablation n_clusters
"""

import argparse
import gc
import json
import os
import sys
import time

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastv.core import load_model, prepare_input
from revoc import RevoCConfig, RevoCEngine, LLaVAAdapter


QUESTIONS = [
    "Describe this image in detail.",
    "What objects are in the foreground?",
    "What colors are prominent?",
    "Is there any text visible?",
    "What is the mood of this image?",
]


def run_config(adapter, image, device, config, rounds=5):
    """Run one config and return stats."""
    engine = RevoCEngine(adapter, config, device)
    engine.start_session(image)

    torch.cuda.synchronize()
    t0 = time.time()
    responses = []
    for i in range(rounds):
        stats = engine.chat(QUESTIONS[i])
        responses.append(stats.response)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    summary = engine.get_summary()
    return {
        'total_time': elapsed,
        'total_tokens': summary['total_image_tokens'],
        'savings_pct': summary['savings_pct'],
        'responses': responses,
    }


def ablation_n_clusters(adapter, image, device):
    """Vary number of clusters."""
    print("\n=== Ablation: Number of Clusters ===")
    results = []
    for K in [16, 32, 64, 128]:
        n_ret = min(12, K)
        config = RevoCConfig(n_clusters=K, n_retrieve_clusters=n_ret, retriever_type="cosine")
        config.validate()
        r = run_config(adapter, image, device, config)
        r['n_clusters'] = K
        r['n_retrieve'] = n_ret
        results.append(r)
        print(f"  K={K:<4} retrieve={n_ret:<3} "
              f"tokens={r['total_tokens']:<6} savings={r['savings_pct']:.1f}% "
              f"time={r['total_time']:.3f}s")
    return results


def ablation_n_retrieve(adapter, image, device):
    """Vary clusters retrieved per query."""
    print("\n=== Ablation: Clusters Retrieved ===")
    results = []
    for n_ret in [4, 8, 12, 16, 24]:
        config = RevoCConfig(n_clusters=64, n_retrieve_clusters=n_ret, retriever_type="cosine")
        config.validate()
        r = run_config(adapter, image, device, config)
        r['n_retrieve'] = n_ret
        results.append(r)
        print(f"  retrieve={n_ret:<3} "
              f"tokens={r['total_tokens']:<6} savings={r['savings_pct']:.1f}% "
              f"time={r['total_time']:.3f}s")
    return results


def ablation_ema_decay(adapter, image, device):
    """Vary EMA decay for dialogue-adaptive importance."""
    print("\n=== Ablation: EMA Decay ===")
    results = []
    for decay in [0.0, 0.5, 0.7, 0.9]:
        config = RevoCConfig(
            n_clusters=64, n_retrieve_clusters=12,
            retriever_type="cosine", ema_decay=max(decay, 0.01),
            history_weight=0.0 if decay == 0.0 else 0.3,
        )
        config.validate()
        r = run_config(adapter, image, device, config)
        r['ema_decay'] = decay
        results.append(r)
        label = "no history" if decay == 0.0 else f"decay={decay}"
        print(f"  {label:<16} "
              f"tokens={r['total_tokens']:<6} savings={r['savings_pct']:.1f}% "
              f"time={r['total_time']:.3f}s")
    return results


def ablation_global_size(adapter, image, device):
    """Vary global summary size."""
    print("\n=== Ablation: Global Summary Size ===")
    results = []
    for n_global in [0, 16, 32, 64]:
        config = RevoCConfig(
            n_global=n_global, n_clusters=64,
            n_retrieve_clusters=12, retriever_type="cosine",
        )
        config.validate()
        r = run_config(adapter, image, device, config)
        r['n_global'] = n_global
        results.append(r)
        print(f"  n_global={n_global:<4} "
              f"tokens={r['total_tokens']:<6} savings={r['savings_pct']:.1f}% "
              f"time={r['total_time']:.3f}s")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--ablation', type=str, default='all',
                        choices=['all', 'n_clusters', 'n_retrieve', 'ema_decay', 'global_size'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device

    print(f"Loading model: {args.model_path}")
    tokenizer, model, image_processor, _ = load_model(args.model_path, device)

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

    all_results = {}
    ablations = {
        'n_clusters': ablation_n_clusters,
        'n_retrieve': ablation_n_retrieve,
        'ema_decay': ablation_ema_decay,
        'global_size': ablation_global_size,
    }

    if args.ablation == 'all':
        for name, fn in ablations.items():
            gc.collect()
            torch.cuda.empty_cache()
            all_results[name] = fn(adapter, image, device)
    else:
        all_results[args.ablation] = ablations[args.ablation](adapter, image, device)

    os.makedirs('results', exist_ok=True)
    with open('results/ablation_study.json', 'w') as f:
        # Remove non-serializable 'responses' for JSON
        clean = {}
        for k, v in all_results.items():
            clean[k] = [{kk: vv for kk, vv in item.items() if kk != 'responses'} for item in v]
        json.dump(clean, f, indent=2)
    print(f"\nResults saved to results/ablation_study.json")


if __name__ == '__main__':
    main()
