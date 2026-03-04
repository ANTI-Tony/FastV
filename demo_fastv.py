"""
FastV 推理 Demo

方式: 两步法
  Step 1: 完整 forward 拿第 K 层注意力分数，计算 image token 重要性
  Step 2: 只保留重要 image tokens，重新构建 inputs_embeds 跑 generate()

用法:
    bash run.sh python demo_fastv.py --model-path models/llava-v1.5-7b
    bash run.sh python demo_fastv.py --fastv-k 2 --fastv-r 0.5
"""

import argparse
import time
import torch

from fastv.core import load_image, load_model, prepare_input, run_vanilla, run_fastv


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

    # 加载图片 & 准备输入
    print(f"加载图片: {args.image_url}")
    image = load_image(args.image_url)
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
    print("  Vanilla 推理 (576 image tokens)")
    print("=" * 60)
    torch.cuda.synchronize()
    start = time.time()
    vanilla_resp = run_vanilla(model, tokenizer, input_ids, image_tensor, device)
    torch.cuda.synchronize()
    vanilla_time = time.time() - start
    print(f"  耗时: {vanilla_time:.3f}s")
    print(f"  输出: {vanilla_resp[:500]}")

    # ========== FastV ==========
    print("\n" + "=" * 60)
    print(f"  FastV 推理 (K={args.fastv_k}, R={args.fastv_r})")
    print("=" * 60)
    torch.cuda.synchronize()
    start = time.time()
    fastv_resp = run_fastv(
        model, tokenizer, input_ids, image_tensor, device,
        fastv_k=args.fastv_k, fastv_r=args.fastv_r, verbose=True,
    )
    torch.cuda.synchronize()
    fastv_time = time.time() - start
    print(f"  耗时: {fastv_time:.3f}s")
    print(f"  输出: {fastv_resp[:500]}")

    # ========== 对比 ==========
    speedup = vanilla_time / fastv_time if fastv_time > 0 else 0
    print("\n" + "=" * 60)
    print("  对比结果")
    print("=" * 60)
    print(f"  Vanilla 耗时:  {vanilla_time:.3f}s")
    print(f"  FastV 耗时:    {fastv_time:.3f}s")
    print(f"  加速比:        {speedup:.2f}x")
    print(f"  Token 节省:    {576 - int(576*(1-args.fastv_r))}/{576} ({args.fastv_r*100:.0f}%)")
    print(f"  理论 FLOPs:    ~{(1-args.fastv_r)*100:.0f}% of vanilla (layers {args.fastv_k}~32)")

    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  峰值显存:      {mem_gb:.1f} GB")

    print("\nDemo 完成!")


if __name__ == '__main__':
    main()
