"""
RPA Demo — Reasoning-aware Progressive Visual Abstraction

Shows visual tokens decreasing as the model reasons through a problem.

Usage:
    bash run.sh python demo_rpa.py --model-path models/llava-v1.5-7b
    bash run.sh python demo_rpa.py --prompt "Describe this image step by step."
    bash run.sh python demo_rpa.py --compress-ratio 0.6 --check-interval 16
"""

import argparse
import time
import torch

from fastv.core import load_model, load_image, prepare_input, get_multimodal_embeds
from rpa import RPAConfig, rpa_generate
from rpa.utils import print_generation_result, print_abstraction_curve, compare_methods_table
from rpa.generator import RPAGenerationResult


def run_vanilla_cot(model, tokenizer, input_ids, image_tensor, device, max_tokens=512):
    """Vanilla CoT: full 576 tokens throughout."""
    from transformers import LlamaForCausalLM

    t0 = time.time()
    with torch.no_grad():
        full_embeds, img_start, n_img = get_multimodal_embeds(model, input_ids, image_tensor)
        seq_len = full_embeds.shape[1]
        mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
        output_ids = LlamaForCausalLM.generate(
            model, inputs_embeds=full_embeds, attention_mask=mask,
            do_sample=False, max_new_tokens=max_tokens,
        )
    elapsed = time.time() - t0
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    n_gen = output_ids.shape[1]

    return RPAGenerationResult(
        response=response, total_generated=n_gen,
        initial_visual_tokens=576, final_visual_tokens=576,
        compressions=0, abstraction_curve=[(0, 576)], elapsed=elapsed,
    )


def run_fastv_cot(model, tokenizer, input_ids, image_tensor, device, max_tokens=512):
    """FastV CoT: prune to 144 tokens at prefill, then full CoT."""
    from fastv.core import run_fastv

    t0 = time.time()
    response = run_fastv(model, tokenizer, input_ids, image_tensor, device,
                         fastv_k=2, fastv_r=0.75, max_new_tokens=max_tokens)
    elapsed = time.time() - t0

    return RPAGenerationResult(
        response=response, total_generated=len(tokenizer.encode(response)),
        initial_visual_tokens=144, final_visual_tokens=144,
        compressions=0, abstraction_curve=[(0, 144)], elapsed=elapsed,
    )


def main():
    parser = argparse.ArgumentParser(description='RPA Demo')
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--image-url', type=str,
                        default='https://llava-vl.github.io/static/images/view.jpg')
    parser.add_argument('--prompt', type=str,
                        default='Describe this image in great detail. Think about what you see step by step.')
    parser.add_argument('--check-interval', type=int, default=32)
    parser.add_argument('--compress-ratio', type=float, default=0.7)
    parser.add_argument('--min-tokens', type=int, default=64)
    parser.add_argument('--method', type=str, default='evict',
                        choices=['evict', 'merge', 'hybrid'])
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device

    config = RPAConfig(
        check_interval=args.check_interval,
        compress_ratio=args.compress_ratio,
        min_visual_tokens=args.min_tokens,
        method=args.method,
        max_new_tokens=args.max_new_tokens,
    )
    config.validate()

    # Load model
    print(f"Loading model: {args.model_path}")
    tokenizer, model, image_processor, _ = load_model(args.model_path, device)

    # Load image
    print(f"Loading image: {args.image_url}")
    image = load_image(args.image_url)

    # Prepare input
    input_ids, image_tensor = prepare_input(
        tokenizer, image_processor, image, args.prompt, device
    )

    # GPU warmup
    print("GPU warmup...")
    with torch.no_grad():
        _ = model(input_ids, images=image_tensor, use_cache=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ========== 1. Vanilla CoT ==========
    print("\n" + "=" * 60)
    print("  Vanilla CoT (576 tokens throughout)")
    print("=" * 60)
    vanilla = run_vanilla_cot(model, tokenizer, input_ids, image_tensor, device, args.max_new_tokens)
    print_generation_result(vanilla, "Vanilla")

    # ========== 2. FastV + CoT ==========
    print("\n" + "=" * 60)
    print("  FastV R=75% + CoT (144 tokens throughout)")
    print("=" * 60)
    fastv = run_fastv_cot(model, tokenizer, input_ids, image_tensor, device, args.max_new_tokens)
    print_generation_result(fastv, "FastV R=75%")

    # ========== 3. RPA ==========
    print("\n" + "=" * 60)
    print(f"  RPA (576 → progressive compression, ratio={config.compress_ratio})")
    print("=" * 60)

    with torch.no_grad():
        full_embeds, img_start, n_img = get_multimodal_embeds(model, input_ids, image_tensor)

    rpa_result = rpa_generate(
        model, tokenizer, full_embeds,
        visual_start=img_start,
        visual_end=img_start + n_img,
        config=config,
    )
    print_generation_result(rpa_result, "RPA")
    print_abstraction_curve(rpa_result)

    # ========== Comparison ==========
    print("\n" + "=" * 60)
    print("  Comparison")
    print("=" * 60)
    compare_methods_table(vanilla, fastv, rpa_result)

    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n  Peak GPU memory: {mem_gb:.1f} GB")

    print("\nDemo complete!")


if __name__ == '__main__':
    main()
