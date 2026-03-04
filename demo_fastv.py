"""
FastV 推理 Demo (LLaVA 原生)

用法:
    bash run.sh python demo_fastv.py --model-path models/llava-v1.5-7b
    bash run.sh python demo_fastv.py --fastv-k 2 --fastv-r 0.75
"""

import argparse
import time
import torch
from PIL import Image
import requests
from io import BytesIO

from fastv.fastv_llama import compute_image_token_importance, select_important_tokens


def load_image(image_source: str) -> Image.Image:
    if image_source.startswith(('http://', 'https://')):
        response = requests.get(image_source)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_source)
    return image.convert('RGB')


def load_model(model_path, device):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return tokenizer, model, image_processor, context_len


def prepare_input(tokenizer, image_processor, image, prompt, device):
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images

    conv = conv_templates["v1"].copy()
    if DEFAULT_IMAGE_TOKEN not in prompt:
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)
    image_tensor = process_images([image], image_processor, None).to(device, dtype=torch.float16)

    return input_ids, image_tensor


def run_vanilla(model, tokenizer, input_ids, image_tensor, device):
    """标准推理（无 FastV）"""
    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=256,
        )

    torch.cuda.synchronize()
    elapsed = time.time() - start

    new_tokens = output_ids[:, input_ids.shape[1]:]
    response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return response, elapsed


def run_fastv(model, tokenizer, input_ids, image_tensor, device, fastv_k=2, fastv_r=0.75):
    """带 FastV 的推理"""

    torch.cuda.synchronize()
    start = time.time()

    # Step 1: 完整 prefill，获取注意力权重
    # 用顶层 model() 而不是 model.model()，因为 LLaVA 在顶层处理 images
    with torch.no_grad():
        outputs = model(
            input_ids,
            images=image_tensor,
            output_attentions=True,
            return_dict=True,
            use_cache=True,
        )

    logits = outputs.logits
    past_key_values = outputs.past_key_values
    seq_len = logits.shape[1]

    # Step 2: 从第 K 层注意力计算 image token 重要性
    attn_k = outputs.attentions[fastv_k - 1]  # (batch, heads, seq, seq)

    # 检测 image token 范围
    # LLaVA 展开 IMAGE_TOKEN_INDEX(-200) 为 576 个 image tokens
    image_token_length = 576
    text_len = input_ids.shape[1]

    # 找 IMAGE_TOKEN_INDEX 在 input_ids 中的位置
    img_placeholder_pos = (input_ids[0] == -200).nonzero(as_tuple=True)[0]
    if len(img_placeholder_pos) > 0:
        image_start = img_placeholder_pos[0].item()
    else:
        image_start = 35  # fallback

    # 确保范围合理
    if image_start + image_token_length > seq_len:
        print(f"  警告: image 范围超出序列 ({image_start}+{image_token_length} > {seq_len})，回退 vanilla")
        output_ids = model.generate(input_ids, images=image_tensor, do_sample=False, max_new_tokens=256)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        new_tokens = output_ids[:, input_ids.shape[1]:]
        return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip(), elapsed, None

    importance = compute_image_token_importance(attn_k, image_start, image_token_length)
    num_keep = int(image_token_length * (1 - fastv_r))
    keep_indices = select_important_tokens(importance, num_keep)

    num_pruned = image_token_length - num_keep
    print(f"  FastV: 保留 {num_keep}/{image_token_length} image tokens, 剪掉 {num_pruned} ({fastv_r*100:.0f}%)")
    print(f"  Image token 范围: [{image_start}, {image_start + image_token_length}), 序列长度: {seq_len}")

    # Step 3: 构建 pruned KV cache
    prefix_idx = torch.arange(image_start, device=device)
    selected_img_idx = keep_indices[0] + image_start
    suffix_idx = torch.arange(image_start + image_token_length, seq_len, device=device)
    keep_all = torch.cat([prefix_idx, selected_img_idx, suffix_idx])

    pruned_past = []
    for layer_kv in past_key_values:
        k, v = layer_kv[0], layer_kv[1]
        pruned_past.append((
            k[:, :, keep_all, :],
            v[:, :, keep_all, :],
        ))
    pruned_past = tuple(pruned_past)

    # Step 4: 自回归生成
    # 后续 decode 步骤不需要传 images（已经在 KV cache 里了）
    # LLaVA 对 input_ids.shape[1]==1 会跳过多模态处理
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = [next_token]
    attn_mask = torch.ones((1, keep_all.shape[0] + 1), dtype=torch.long, device=device)

    eos_token_id = tokenizer.eos_token_id or 2

    for _ in range(255):
        with torch.no_grad():
            out = model(
                next_token,
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
        attn_mask = torch.cat([
            attn_mask,
            torch.ones((1, 1), dtype=torch.long, device=device)
        ], dim=1)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    all_ids = torch.cat(generated_ids, dim=1)
    response = tokenizer.batch_decode(all_ids, skip_special_tokens=True)[0].strip()

    return response, elapsed, importance


def main():
    parser = argparse.ArgumentParser(description='FastV Demo')
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--image-url', type=str,
                        default='https://llava-vl.github.io/static/images/view.jpg')
    parser.add_argument('--prompt', type=str, default='Describe this image in detail.')
    parser.add_argument('--fastv-k', type=int, default=2, help='FastV K 参数')
    parser.add_argument('--fastv-r', type=float, default=0.75, help='FastV R 参数')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device

    # 加载模型
    print(f"加载模型: {args.model_path}")
    tokenizer, model, image_processor, context_len = load_model(args.model_path, device)
    print(f"  模型加载完成, context_len={context_len}")

    # 加载图片
    print(f"加载图片: {args.image_url}")
    image = load_image(args.image_url)

    # 准备输入
    input_ids, image_tensor = prepare_input(
        tokenizer, image_processor, image, args.prompt, device
    )
    print(f"  input_ids shape: {input_ids.shape}, image_tensor shape: {image_tensor.shape}")

    # GPU 预热
    print("GPU 预热...")
    with torch.no_grad():
        _ = model(input_ids, images=image_tensor, use_cache=False)
    torch.cuda.synchronize()

    # ========== Vanilla ==========
    print("\n" + "=" * 60)
    print("  Vanilla 推理 (无 FastV)")
    print("=" * 60)
    vanilla_resp, vanilla_time = run_vanilla(model, tokenizer, input_ids, image_tensor, device)
    print(f"  耗时: {vanilla_time:.3f}s")
    print(f"  输出: {vanilla_resp[:300]}")

    # ========== FastV ==========
    print("\n" + "=" * 60)
    print(f"  FastV 推理 (K={args.fastv_k}, R={args.fastv_r})")
    print("=" * 60)
    fastv_resp, fastv_time, importance = run_fastv(
        model, tokenizer, input_ids, image_tensor, device,
        fastv_k=args.fastv_k, fastv_r=args.fastv_r,
    )
    print(f"  耗时: {fastv_time:.3f}s")
    print(f"  输出: {fastv_resp[:300]}")

    # ========== 对比 ==========
    speedup = vanilla_time / fastv_time if fastv_time > 0 else 0
    print("\n" + "=" * 60)
    print("  对比结果")
    print("=" * 60)
    print(f"  Vanilla 耗时:  {vanilla_time:.3f}s")
    print(f"  FastV 耗时:    {fastv_time:.3f}s")
    print(f"  加速比:        {speedup:.2f}x")

    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  峰值显存:      {mem_gb:.1f} GB")

    print("\nDemo 完成!")


if __name__ == '__main__':
    main()
