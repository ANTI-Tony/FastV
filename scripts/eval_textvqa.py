"""
TextVQA 评测脚本

评测 FastV 在 TextVQA 上的准确率

用法:
    bash run.sh python scripts/eval_textvqa.py
    bash run.sh python scripts/eval_textvqa.py --fastv-r 0.75
    bash run.sh python scripts/eval_textvqa.py --fastv-r 0.0  # Vanilla baseline
"""

import argparse
import json
import os
import sys
import time
from tqdm import tqdm

import torch
from PIL import Image

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastv.core import load_model, prepare_input, run_vanilla, run_fastv


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
    tokenizer, model, image_processor, context_len = load_model(args.model_path, device)

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
    total_time = 0

    for item in tqdm(data, desc=f"TextVQA [{config_name}]"):
        image_id = item['image_id']
        question = item['question']
        answers = item.get('answers', [])

        image_path = os.path.join(args.image_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            continue

        try:
            image = Image.open(image_path).convert('RGB')
            input_ids, image_tensor = prepare_input(
                tokenizer, image_processor, image, question, device
            )

            start = time.time()
            if args.fastv_r == 0.0:
                pred = run_vanilla(model, tokenizer, input_ids, image_tensor, device, max_new_tokens=128)
            else:
                pred = run_fastv(model, tokenizer, input_ids, image_tensor, device,
                                 fastv_k=args.fastv_k, fastv_r=args.fastv_r, max_new_tokens=128)
            total_time += time.time() - start

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
    avg_time = total_time / total if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"TextVQA 结果 [{config_name}]")
    print(f"  准确率: {final_acc:.2f}% ({correct:.1f}/{total})")
    print(f"  平均推理时间: {avg_time:.3f}s/sample")
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
            'avg_time': avg_time,
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
