"""
FastV 推理 Demo

用法:
    python demo_fastv.py
    python demo_fastv.py --model-path liuhaotian/llava-v1.5-7b --image-url <URL>
    python demo_fastv.py --fastv-k 2 --fastv-r 0.75
"""

import argparse
import time
import torch
from PIL import Image
import requests
from io import BytesIO


def load_image(image_source: str) -> Image.Image:
    """从 URL 或本地路径加载图片"""
    if image_source.startswith(('http://', 'https://')):
        response = requests.get(image_source)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_source)
    return image.convert('RGB')


def run_vanilla(model, processor, image, prompt, device):
    """标准推理（无 FastV）"""
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    torch.cuda.synchronize()
    elapsed = time.time() - start

    response = processor.decode(output[0], skip_special_tokens=True)
    return response, elapsed


def run_with_fastv(model, processor, image, prompt, device, fastv_k=2, fastv_r=0.75):
    """
    带 FastV 的推理

    策略：用 attention mask 方式实现，兼容性更好
    """
    from fastv.fastv_config import FastVConfig
    from fastv.fastv_llama import (
        compute_image_token_importance,
        select_important_tokens,
    )

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']

    # Step 1: Prefill with attention output
    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
            use_cache=True,
        )

    # Step 2: 获取第 K 层的注意力，计算重要性
    attn_k = outputs.attentions[fastv_k - 1]  # (batch, heads, seq, seq)
    seq_len = outputs.logits.shape[1]

    # 自动检测 image token 范围
    # LLaVA 的 image token 通常在前面的 system prompt 之后
    image_token_length = 576
    # 简单启发式：image tokens 占据了序列中最大的连续 token 块
    image_start = seq_len - image_token_length - (input_ids.shape[1] - 1)
    image_start = max(0, image_start)

    # 如果序列太短，跳过 FastV
    if seq_len < image_start + image_token_length:
        print("序列太短，跳过 FastV 剪枝")
        # fallback to greedy decoding
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        response = processor.decode(generated[0], skip_special_tokens=True)
        return response, elapsed, None

    importance = compute_image_token_importance(attn_k, image_start, image_token_length)
    num_keep = int(image_token_length * (1 - fastv_r))
    keep_indices = select_important_tokens(importance, num_keep)

    num_pruned = image_token_length - num_keep
    print(f"  FastV: 保留 {num_keep}/{image_token_length} image tokens, 剪掉 {num_pruned} ({fastv_r*100:.0f}%)")

    # Step 3: 构建 pruned KV cache
    # 建立保留的全局索引
    prefix_idx = torch.arange(image_start, device=device)
    selected_img_idx = keep_indices[0] + image_start
    suffix_idx = torch.arange(image_start + image_token_length, seq_len, device=device)
    keep_all = torch.cat([prefix_idx, selected_img_idx, suffix_idx])

    past_key_values = outputs.past_key_values
    pruned_past = []
    for k, v in past_key_values:
        pruned_past.append((
            k[:, :, keep_all, :],
            v[:, :, keep_all, :],
        ))
    pruned_past = tuple(pruned_past)

    # Step 4: 自回归生成
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = [next_token]
    attn_mask = torch.ones((1, keep_all.shape[0] + 1), dtype=torch.long, device=device)

    eos_token_id = getattr(model.config, 'eos_token_id', 2)

    for step in range(255):
        with torch.no_grad():
            out = model.language_model(
                input_ids=next_token,
                attention_mask=attn_mask,
                past_key_values=pruned_past,
                use_cache=True,
                return_dict=True,
            )
        pruned_past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        if next_token.item() == eos_token_id:
            break

        generated_ids.append(next_token)
        attn_mask = torch.cat([attn_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    all_ids = torch.cat(generated_ids, dim=1)
    response = processor.decode(all_ids[0], skip_special_tokens=True)

    return response, elapsed, importance


def main():
    parser = argparse.ArgumentParser(description='FastV Demo')
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b',
                        help='LLaVA 模型路径')
    parser.add_argument('--image-url', type=str,
                        default='https://llava-vl.github.io/static/images/view.jpg',
                        help='测试图片 URL')
    parser.add_argument('--prompt', type=str,
                        default='USER: <image>\nDescribe this image in detail.\nASSISTANT:',
                        help='输入 prompt')
    parser.add_argument('--fastv-k', type=int, default=2, help='FastV K 参数')
    parser.add_argument('--fastv-r', type=float, default=0.75, help='FastV R 参数')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    args = parser.parse_args()

    device = args.device
    print(f"加载模型: {args.model_path}")

    # 加载模型
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # 加载图片
    print(f"加载图片: {args.image_url}")
    image = load_image(args.image_url)

    # GPU 预热
    print("GPU 预热...")
    warmup_inputs = processor(text=args.prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**warmup_inputs)
    torch.cuda.synchronize()

    print("\n" + "=" * 60)
    print("  Vanilla 推理 (无 FastV)")
    print("=" * 60)
    vanilla_response, vanilla_time = run_vanilla(model, processor, image, args.prompt, device)
    print(f"  耗时: {vanilla_time:.3f}s")
    print(f"  输出: {vanilla_response[-200:]}")

    print("\n" + "=" * 60)
    print(f"  FastV 推理 (K={args.fastv_k}, R={args.fastv_r})")
    print("=" * 60)
    fastv_response, fastv_time, importance = run_with_fastv(
        model, processor, image, args.prompt, device,
        fastv_k=args.fastv_k, fastv_r=args.fastv_r,
    )
    print(f"  耗时: {fastv_time:.3f}s")
    print(f"  输出: {fastv_response[-200:]}")

    # 对比
    speedup = vanilla_time / fastv_time if fastv_time > 0 else 0
    print("\n" + "=" * 60)
    print("  对比结果")
    print("=" * 60)
    print(f"  Vanilla 耗时:  {vanilla_time:.3f}s")
    print(f"  FastV 耗时:    {fastv_time:.3f}s")
    print(f"  加速比:        {speedup:.2f}x")

    # 显存使用
    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  峰值显存:      {mem_gb:.1f} GB")

    # 可视化注意力
    if importance is not None:
        print("\n生成注意力可视化...")
        try:
            from fastv.attention_viz import visualize_attention
            # 需要完整的注意力权重来画可视化，这里先跳过
            print("  (完整可视化需要 output_attentions=True 的所有层输出)")
            print("  使用 scripts/visualize_attention.py 生成完整可视化")
        except ImportError:
            pass

    print("\nDemo 完成!")


if __name__ == '__main__':
    main()
