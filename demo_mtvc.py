"""
MTVC 多轮对话 Demo

演示 Multi-Granularity Visual Token Cache 在多轮对话中的效果：
  - Round 1: 完整 576 image tokens + 建缓存
  - Round 2~5: 查询引导检索 ~150 tokens

用法:
    bash run.sh python demo_mtvc.py --model-path models/llava-v1.5-7b
    bash run.sh python demo_mtvc.py --image-url path/to/image.jpg --rounds 5
"""

import argparse
import torch

from fastv.core import load_model, load_image
from mtvc import MTVCConfig, MultiTurnEngine
from mtvc.utils import print_session_summary, compare_methods_table


DEFAULT_QUESTIONS = [
    "Describe this image in detail.",
    "What objects are in the foreground of the image?",
    "What colors are prominent in this image?",
    "Is there any text visible in the image? If so, what does it say?",
    "What is the overall mood or atmosphere of this image?",
]


def main():
    parser = argparse.ArgumentParser(description='MTVC Multi-Turn Demo')
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--image-url', type=str,
                        default='https://llava-vl.github.io/static/images/view.jpg')
    parser.add_argument('--rounds', type=int, default=5, help='对话轮数')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--k2', type=int, default=64, help='L2 检索数')
    parser.add_argument('--k3', type=int, default=54, help='L3 检索数')
    args = parser.parse_args()

    device = args.device
    num_rounds = min(args.rounds, len(DEFAULT_QUESTIONS))

    # Config
    config = MTVCConfig(default_k2=args.k2, default_k3=args.k3)
    config.validate()

    # Load model
    print(f"加载模型: {args.model_path}")
    tokenizer, model, image_processor, context_len = load_model(args.model_path, device)

    # Load image
    print(f"加载图片: {args.image_url}")
    image = load_image(args.image_url)

    # Create engine and start session
    engine = MultiTurnEngine(model, tokenizer, image_processor, config, device)
    engine.start_session(image)

    # GPU warmup
    print("GPU 预热...")
    from fastv.core import prepare_input
    input_ids, image_tensor = prepare_input(
        tokenizer, image_processor, image, "warmup", device
    )
    with torch.no_grad():
        _ = model(input_ids, images=image_tensor, use_cache=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Multi-turn conversation
    print(f"\n开始 {num_rounds} 轮对话\n")

    for i in range(num_rounds):
        query = DEFAULT_QUESTIONS[i]
        print("=" * 60)
        print(f"  Round {i + 1}: {query}")
        print("=" * 60)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        stats = engine.chat(query, verbose=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print(f"  Image tokens: {stats.image_tokens_used}")
        print(f"  Seq length:   {stats.total_seq_len}")
        print(f"  耗时:         {stats.elapsed:.3f}s")
        print(f"  输出: {stats.response[:500]}")
        print()

    # Summary
    summary = engine.get_summary()
    print_session_summary(summary)

    # Comparison table
    vanilla_total = 576 * num_rounds
    fastv_total = 144 * num_rounds  # R=75%
    mtvc_total = summary['total_image_tokens']
    compare_methods_table(vanilla_total, fastv_total, mtvc_total, num_rounds)

    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n  峰值显存: {mem_gb:.1f} GB")

    print("\nDemo 完成!")


if __name__ == '__main__':
    main()
