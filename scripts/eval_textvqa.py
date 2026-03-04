"""
TextVQA 评测脚本

评测 FastV 在 TextVQA 上的准确率

用法:
    python scripts/eval_textvqa.py
    python scripts/eval_textvqa.py --fastv-r 0.75
    python scripts/eval_textvqa.py --fastv-r 0.0  # Vanilla baseline
"""

import argparse
import json
import os
import time
from tqdm import tqdm

import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

from fastv.fastv_llama import compute_image_token_importance, select_important_tokens


def generate_with_fastv(model, processor, image, prompt, device, fastv_k, fastv_r, max_new_tokens=128):
    """带 FastV 的推理"""
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    if fastv_r == 0.0:
        # Vanilla
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return processor.decode(output[0], skip_special_tokens=True)

    # FastV
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True, use_cache=True)

    seq_len = outputs.logits.shape[1]
    attn_k = outputs.attentions[fastv_k - 1]
    image_length = 576
    image_start = max(0, seq_len - image_length - (inputs['input_ids'].shape[1] - 1))

    importance = compute_image_token_importance(attn_k, image_start, image_length)
    num_keep = int(image_length * (1 - fastv_r))
    keep_indices = select_important_tokens(importance, num_keep)

    prefix_idx = torch.arange(image_start, device=device)
    selected_img_idx = keep_indices[0] + image_start
    suffix_idx = torch.arange(image_start + image_length, seq_len, device=device)
    keep_all = torch.cat([prefix_idx, selected_img_idx, suffix_idx])

    pruned_past = tuple(
        (k[:, :, keep_all, :], v[:, :, keep_all, :])
        for k, v in outputs.past_key_values
    )

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = [next_token]
    attn_mask = torch.ones((1, keep_all.shape[0] + 1), dtype=torch.long, device=device)
    eos = getattr(model.config, 'eos_token_id', 2)

    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            out = model.language_model(
                input_ids=next_token, attention_mask=attn_mask,
                past_key_values=pruned_past, use_cache=True, return_dict=True,
            )
        pruned_past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if next_token.item() == eos:
            break
        generated_ids.append(next_token)
        attn_mask = torch.cat([attn_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)

    all_ids = torch.cat(generated_ids, dim=1)
    return processor.decode(all_ids[0], skip_special_tokens=True)


def textvqa_accuracy(pred: str, answers: list) -> float:
    """TextVQA 准确率计算 (min(#humans who gave ans / 3, 1))"""
    pred = pred.strip().lower()
    answer_counts = {}
    for ans in answers:
        ans = ans.strip().lower()
        answer_counts[ans] = answer_counts.get(ans, 0) + 1

    if pred in answer_counts:
        return min(answer_counts[pred] / 3.0, 1.0)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--data-path', type=str, default='data/textvqa/TextVQA_0.5.1_val.json')
    parser.add_argument('--image-dir', type=str, default='data/textvqa/train_images')
    parser.add_argument('--fastv-k', type=int, default=2)
    parser.add_argument('--fastv-r', type=float, default=0.0, help='0.0=vanilla, 0.5=50%, 0.75=75%')
    parser.add_argument('--max-samples', type=int, default=-1, help='最大评测样本数, -1=全部')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    device = 'cuda'
    config_name = f"K{args.fastv_k}_R{int(args.fastv_r*100)}" if args.fastv_r > 0 else "vanilla"
    print(f"配置: {config_name}")

    # 加载模型
    print(f"加载模型: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    # 加载数据
    print(f"加载数据: {args.data_path}")
    with open(args.data_path) as f:
        data = json.load(f)['data']

    if args.max_samples > 0:
        data = data[:args.max_samples]

    print(f"评测样本数: {len(data)}")

    # 评测
    correct = 0
    total = 0
    results = []

    for item in tqdm(data, desc=f"TextVQA [{config_name}]"):
        image_id = item['image_id']
        question = item['question']
        answers = item.get('answers', [])

        image_path = os.path.join(args.image_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            continue

        try:
            image = Image.open(image_path).convert('RGB')
            prompt = f"USER: <image>\n{question}\nASSISTANT:"

            pred = generate_with_fastv(
                model, processor, image, prompt, device,
                args.fastv_k, args.fastv_r,
            )

            acc = textvqa_accuracy(pred, answers)
            correct += acc
            total += 1

            results.append({
                'image_id': image_id,
                'question': question,
                'prediction': pred,
                'answers': answers,
                'accuracy': acc,
            })

        except Exception as e:
            print(f"跳过 {image_id}: {e}")
            continue

    # 输出结果
    final_acc = correct / total * 100 if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"TextVQA 结果 [{config_name}]")
    print(f"  准确率: {final_acc:.2f}% ({correct:.1f}/{total})")
    print(f"{'='*50}")

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"textvqa_{config_name}.json")
    with open(output_file, 'w') as f:
        json.dump({
            'config': config_name,
            'fastv_k': args.fastv_k,
            'fastv_r': args.fastv_r,
            'accuracy': final_acc,
            'total': total,
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
