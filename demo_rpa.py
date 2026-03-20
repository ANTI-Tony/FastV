"""
RPA Demo — Reasoning-aware Progressive Visual Abstraction

Two modes:
  --mode evict: KV cache eviction (original, may have EOS issues)
  --mode mask:  Attention masking (stable, recommended)

Usage:
    bash run.sh python demo_rpa.py --model-path models/llava-v1.5-7b
    bash run.sh python demo_rpa.py --mode mask --max-new-tokens 256
"""

import argparse
import time
import torch

from fastv.core import load_model, load_image, prepare_input, get_multimodal_embeds, run_vanilla, run_fastv
from rpa import RPAConfig
from rpa.generator import rpa_generate, RPAGenerationResult
from rpa.mask_generator import rpa_masked_generate, MaskGenerationResult


def run_vanilla_cot(model, tokenizer, input_ids, image_tensor, device, max_tokens=256):
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
    return response, elapsed


def run_fastv_cot(model, tokenizer, input_ids, image_tensor, device, max_tokens=256):
    t0 = time.time()
    response = run_fastv(model, tokenizer, input_ids, image_tensor, device,
                         fastv_k=2, fastv_r=0.75, max_new_tokens=max_tokens)
    elapsed = time.time() - t0
    return response, elapsed


def main():
    parser = argparse.ArgumentParser(description='RPA Demo')
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--image-url', type=str,
                        default='https://llava-vl.github.io/static/images/view.jpg')
    parser.add_argument('--prompt', type=str,
                        default='Describe this image in great detail. Think about what you see step by step.')
    parser.add_argument('--mode', type=str, default='mask', choices=['evict', 'mask'])
    parser.add_argument('--check-interval', type=int, default=32)
    parser.add_argument('--compress-ratio', type=float, default=0.75)
    parser.add_argument('--min-tokens', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=64)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    config = RPAConfig(
        check_interval=args.check_interval,
        compress_ratio=args.compress_ratio,
        min_visual_tokens=args.min_tokens,
        warmup_tokens=args.warmup,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"Loading model: {args.model_path}")
    tokenizer, model, image_processor, _ = load_model(args.model_path, device)

    print(f"Loading image: {args.image_url}")
    image = load_image(args.image_url)

    input_ids, image_tensor = prepare_input(
        tokenizer, image_processor, image, args.prompt, device
    )

    # Warmup
    print("GPU warmup...")
    with torch.no_grad():
        _ = model(input_ids, images=image_tensor, use_cache=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ========== Vanilla ==========
    print(f"\n{'='*60}")
    print(f"  Vanilla (576 tokens throughout)")
    print(f"{'='*60}")
    v_resp, v_time = run_vanilla_cot(model, tokenizer, input_ids, image_tensor, device, args.max_new_tokens)
    print(f"  Time:     {v_time:.2f}s")
    print(f"  Tokens:   576 (fixed)")
    print(f"  Response: {v_resp[:300]}")

    # ========== FastV ==========
    print(f"\n{'='*60}")
    print(f"  FastV R=75% (144 tokens throughout)")
    print(f"{'='*60}")
    f_resp, f_time = run_fastv_cot(model, tokenizer, input_ids, image_tensor, device, args.max_new_tokens)
    print(f"  Time:     {f_time:.2f}s")
    print(f"  Tokens:   144 (fixed)")
    print(f"  Response: {f_resp[:300]}")

    # ========== RPA ==========
    print(f"\n{'='*60}")
    print(f"  RPA ({args.mode} mode, ratio={config.compress_ratio})")
    print(f"{'='*60}")

    with torch.no_grad():
        full_embeds, img_start, n_img = get_multimodal_embeds(model, input_ids, image_tensor)

    if args.mode == 'mask':
        result = rpa_masked_generate(
            model, tokenizer, full_embeds,
            visual_start=img_start, visual_end=img_start + n_img,
            config=config,
        )
        print(f"  Time:       {result.elapsed:.2f}s")
        print(f"  Generated:  {result.total_generated} tokens")
        print(f"  Visual:     {result.initial_visual_tokens} → {result.final_effective_tokens} effective "
              f"({(1 - result.final_effective_tokens / result.initial_visual_tokens) * 100:.0f}% masked)")
        print(f"  Mask steps: {result.mask_steps}")
        print(f"  Response:   {result.response[:300]}")

        if result.mask_curve:
            print(f"\n  Masking Curve:")
            max_n = result.mask_curve[0][1]
            for step, n in result.mask_curve:
                bar = '█' * int(n / max_n * 40)
                print(f"    Step {step:<6} {n:<6} {bar}")
    else:
        result = rpa_generate(
            model, tokenizer, full_embeds,
            visual_start=img_start, visual_end=img_start + n_img,
            config=config,
        )
        print(f"  Time:       {result.elapsed:.2f}s")
        print(f"  Generated:  {result.total_generated} tokens")
        print(f"  Visual:     {result.initial_visual_tokens} → {result.final_visual_tokens}")
        print(f"  Response:   {result.response[:300]}")

    # ========== Comparison ==========
    print(f"\n{'='*60}")
    print(f"  Comparison")
    print(f"{'='*60}")
    rpa_final = result.final_effective_tokens if args.mode == 'mask' else result.final_visual_tokens
    print(f"  {'Method':<16}{'Visual Tokens':<20}{'Time(s)':<10}")
    print(f"  {'─'*44}")
    print(f"  {'Vanilla':<16}{'576 (fixed)':<20}{v_time:<10.2f}")
    print(f"  {'FastV R=75%':<16}{'144 (fixed)':<20}{f_time:<10.2f}")
    print(f"  {'RPA':<16}{f'576→{rpa_final}':<20}{result.elapsed:<10.2f}")

    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n  Peak GPU memory: {mem_gb:.1f} GB")

    print("\nDemo complete!")


if __name__ == '__main__':
    main()
