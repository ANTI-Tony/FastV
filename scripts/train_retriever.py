"""
Train the cross-attention retriever via distillation.

The retriever learns to select visual token clusters such that the
model's output with recovered tokens matches the full-token output.

Only ~4M params are trained. VLM backbone is frozen.

Usage:
    bash run.sh python scripts/train_retriever.py
    bash run.sh python scripts/train_retriever.py --data-path data/textvqa/TextVQA_0.5.1_val.json
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revoc import RevoCConfig, LLaVAAdapter
from revoc.distill import train_retriever


def main():
    parser = argparse.ArgumentParser(description='Train ReVoC Retriever')
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--data-path', type=str, default='data/textvqa/TextVQA_0.5.1_val.json')
    parser.add_argument('--image-dir', type=str, default='data/textvqa/train_images')
    parser.add_argument('--output', type=str, default='checkpoints/retriever.pt')
    parser.add_argument('--max-samples', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n-clusters', type=int, default=64)
    parser.add_argument('--n-retrieve', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    config = RevoCConfig(
        n_clusters=args.n_clusters,
        n_retrieve_clusters=args.n_retrieve,
        retriever_type="cross_attention",
        distill_lr=args.lr,
        distill_epochs=args.epochs,
    )

    print(f"Loading model: {args.model_path}")
    adapter = LLaVAAdapter()
    adapter.load(args.model_path, args.device)

    print(f"Training retriever ({config.retriever_heads}-head cross-attention)")
    print(f"  Clusters: {config.n_clusters}, Retrieve: {config.n_retrieve_clusters}")
    print(f"  LR: {config.distill_lr}, Epochs: {config.distill_epochs}")
    print(f"  Max samples: {args.max_samples}")

    train_retriever(
        adapter=adapter,
        config=config,
        data_path=args.data_path,
        image_dir=args.image_dir,
        output_path=args.output,
        max_samples=args.max_samples,
        device=args.device,
    )


if __name__ == '__main__':
    main()
