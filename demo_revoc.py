"""
ReVoC Multi-Turn Demo — Recoverable Visual Compression

Demonstrates:
  1. Round 1: full 576 tokens + compressed cache construction
  2. Round 2~5: query-guided cluster recovery (~140 tokens)
  3. Theoretical compression bounds (pruning vs ReVoC)
  4. Entropy distribution visualization

Usage:
    bash run.sh python demo_revoc.py --model-path models/llava-v1.5-7b
    bash run.sh python demo_revoc.py --retriever cosine --rounds 5
"""

import argparse
import torch

from fastv.core import load_image
from revoc import (
    RevoCConfig, RevoCEngine, LLaVAAdapter,
    compute_compression_bounds, print_session_summary, print_compression_bounds,
)
from revoc.utils import compare_methods_table, visualize_entropy_distribution


DEFAULT_QUESTIONS = [
    "Describe this image in detail.",
    "What objects are in the foreground of the image?",
    "What colors are prominent in this image?",
    "Is there any text visible in the image? If so, what does it say?",
    "What is the overall mood or atmosphere of this image?",
]


def main():
    parser = argparse.ArgumentParser(description='ReVoC Multi-Turn Demo')
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--image-url', type=str,
                        default='https://llava-vl.github.io/static/images/view.jpg')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--retriever', type=str, default='cosine',
                        choices=['cosine', 'cross_attention'])
    parser.add_argument('--retriever-weights', type=str, default=None,
                        help='Path to trained retriever checkpoint')
    parser.add_argument('--n-clusters', type=int, default=64)
    parser.add_argument('--n-retrieve', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    num_rounds = min(args.rounds, len(DEFAULT_QUESTIONS))

    # Config
    config = RevoCConfig(
        n_clusters=args.n_clusters,
        n_retrieve_clusters=args.n_retrieve,
        retriever_type=args.retriever,
    )
    config.validate()

    # Load model via adapter
    print(f"Loading model: {args.model_path}")
    adapter = LLaVAAdapter()
    adapter.load(args.model_path, device)

    # Load image
    print(f"Loading image: {args.image_url}")
    image = load_image(args.image_url)

    # Create engine
    engine = RevoCEngine(adapter, config, device)
    engine.start_session(image)

    # Load trained retriever if provided
    if args.retriever_weights and args.retriever == 'cross_attention':
        engine.retriever.load_retriever_weights(args.retriever_weights)
        print(f"Loaded retriever weights from {args.retriever_weights}")

    # GPU warmup
    print("GPU warmup...")
    input_ids, image_tensor = adapter.prepare_input(image, "warmup")
    with torch.no_grad():
        _ = adapter.model(input_ids, images=image_tensor, use_cache=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ===== Multi-turn conversation =====
    print(f"\nStarting {num_rounds}-round conversation\n")

    for i in range(num_rounds):
        query = DEFAULT_QUESTIONS[i]
        print("=" * 70)
        print(f"  Round {i + 1}: {query}")
        print("=" * 70)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        stats = engine.chat(query, verbose=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print(f"  Image tokens:      {stats.image_tokens_used}")
        if stats.clusters_unpacked > 0:
            print(f"  Clusters unpacked: {stats.clusters_unpacked}")
            print(f"  Tokens recovered:  {stats.tokens_recovered}")
        print(f"  Seq length:        {stats.total_seq_len}")
        print(f"  Time:              {stats.elapsed:.3f}s")
        print(f"  Response: {stats.response[:400]}")
        print()

    # ===== Session summary =====
    summary = engine.get_summary()
    print_session_summary(summary)

    # ===== Theoretical bounds =====
    if engine.session.cache is not None:
        cache = engine.session.cache
        # Filter to only clustered tokens (exclude global tokens marked as -1)
        clustered_mask = cache.cluster_assignments >= 0
        clustered_features = cache.image_features[clustered_mask]
        clustered_assignments = cache.cluster_assignments[clustered_mask]
        bounds = compute_compression_bounds(
            clustered_features,
            clustered_assignments,
            cache.cluster_centers,
            n_unpacked=config.n_retrieve_clusters,
        )
        print_compression_bounds(bounds)

    # ===== Entropy visualization =====
    if engine.session.cache is not None:
        try:
            visualize_entropy_distribution(
                engine.session.cache.entropy,
                engine.session.cache.importance,
                save_path="results/entropy_dist.png",
            )
        except Exception as e:
            print(f"  Visualization skipped: {e}")

    # ===== Method comparison =====
    vanilla_tok = 576 * num_rounds
    fastv_tok = 144 * num_rounds
    revoc_tok = summary['total_image_tokens']
    compare_methods_table({
        'Vanilla': {'tokens': vanilla_tok, 'time': 0},
        'FastV R=75%': {'tokens': fastv_tok, 'time': 0},
        'ReVoC': {'tokens': revoc_tok, 'time': summary['total_time']},
    }, num_rounds)

    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n  Peak GPU memory: {mem_gb:.1f} GB")

    print("\nDemo complete!")


if __name__ == '__main__':
    main()
