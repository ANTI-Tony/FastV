"""
多轮 TextVQA 评测

用同一张图的多个问题模拟多轮对话，对比 Vanilla / FastV / MTVC 准确率。

策略: 将 TextVQA 数据按 image_id 分组，每组取前 N 个问题作为多轮对话。

用法:
    bash run.sh python scripts/eval_multiturn.py
    bash run.sh python scripts/eval_multiturn.py --max-images 100 --rounds-per-image 3
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from tqdm import tqdm

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastv.core import load_model, prepare_input, run_vanilla, run_fastv
from mtvc import MTVCConfig, MultiTurnEngine


def textvqa_accuracy(pred: str, answers: list) -> float:
    pred = pred.strip().lower()
    answer_counts = {}
    for ans in answers:
        ans = ans.strip().lower()
        answer_counts[ans] = answer_counts.get(ans, 0) + 1
    if pred in answer_counts:
        return min(answer_counts[pred] / 3.0, 1.0)
    return 0.0


def group_by_image(data):
    """Group TextVQA samples by image_id."""
    groups = defaultdict(list)
    for item in data:
        groups[item['image_id']].append(item)
    return groups


def eval_vanilla_multiturn(model, tokenizer, image_processor, image, questions_data, device):
    """Vanilla: each question is independent."""
    results = []
    total_tokens = 0
    for item in questions_data:
        q = item['question'] + "\nAnswer the question using a single word or phrase."
        input_ids, image_tensor = prepare_input(
            tokenizer, image_processor, image, q, device
        )
        pred = run_vanilla(model, tokenizer, input_ids, image_tensor, device, max_new_tokens=128)
        acc = textvqa_accuracy(pred, item.get('answers', []))
        results.append(acc)
        total_tokens += 576
    return results, total_tokens


def eval_fastv_multiturn(model, tokenizer, image_processor, image, questions_data, device,
                         fastv_k=2, fastv_r=0.75):
    """FastV: each question is independent, with pruning."""
    results = []
    kept = int(576 * (1 - fastv_r))
    total_tokens = 0
    for item in questions_data:
        q = item['question'] + "\nAnswer the question using a single word or phrase."
        input_ids, image_tensor = prepare_input(
            tokenizer, image_processor, image, q, device
        )
        pred = run_fastv(model, tokenizer, input_ids, image_tensor, device,
                         fastv_k=fastv_k, fastv_r=fastv_r, max_new_tokens=128)
        acc = textvqa_accuracy(pred, item.get('answers', []))
        results.append(acc)
        total_tokens += kept
    return results, total_tokens


def eval_mtvc_multiturn(model, tokenizer, image_processor, image, questions_data, device,
                        config=None):
    """MTVC: true multi-turn with cache."""
    config = config or MTVCConfig(max_new_tokens=128)
    engine = MultiTurnEngine(model, tokenizer, image_processor, config, device)
    engine.start_session(image)

    results = []
    for item in questions_data:
        q = item['question'] + "\nAnswer the question using a single word or phrase."
        stats = engine.chat(q)
        acc = textvqa_accuracy(stats.response, item.get('answers', []))
        results.append(acc)

    summary = engine.get_summary()
    return results, summary['total_image_tokens']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--data-path', type=str, default='data/textvqa/TextVQA_0.5.1_val.json')
    parser.add_argument('--image-dir', type=str, default='data/textvqa/train_images')
    parser.add_argument('--max-images', type=int, default=100, help='最大图片数')
    parser.add_argument('--rounds-per-image', type=int, default=3, help='每张图最多几轮')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    device = 'cuda'

    print(f"加载模型: {args.model_path}")
    tokenizer, model, image_processor, _ = load_model(args.model_path, device)

    print(f"加载数据: {args.data_path}")
    with open(args.data_path) as f:
        data = json.load(f)['data']

    # Group by image, filter to images with enough questions
    groups = group_by_image(data)
    multi_images = {
        img_id: items
        for img_id, items in groups.items()
        if len(items) >= args.rounds_per_image
    }

    image_ids = list(multi_images.keys())[:args.max_images]
    print(f"评测: {len(image_ids)} images x {args.rounds_per_image} rounds")

    # Accumulators
    all_results = {'vanilla': [], 'fastv': [], 'mtvc': []}
    total_tokens = {'vanilla': 0, 'fastv': 0, 'mtvc': 0}

    for img_id in tqdm(image_ids, desc="Multi-turn eval"):
        image_path = os.path.join(args.image_dir, f"{img_id}.jpg")
        if not os.path.exists(image_path):
            continue

        try:
            image = Image.open(image_path).convert('RGB')
            questions_data = multi_images[img_id][:args.rounds_per_image]

            # Vanilla
            v_acc, v_tok = eval_vanilla_multiturn(
                model, tokenizer, image_processor, image, questions_data, device)
            all_results['vanilla'].extend(v_acc)
            total_tokens['vanilla'] += v_tok

            # FastV
            f_acc, f_tok = eval_fastv_multiturn(
                model, tokenizer, image_processor, image, questions_data, device)
            all_results['fastv'].extend(f_acc)
            total_tokens['fastv'] += f_tok

            # MTVC
            m_acc, m_tok = eval_mtvc_multiturn(
                model, tokenizer, image_processor, image, questions_data, device)
            all_results['mtvc'].extend(m_acc)
            total_tokens['mtvc'] += m_tok

        except Exception as e:
            print(f"跳过 {img_id}: {e}")
            continue

    # Results
    print(f"\n{'=' * 60}")
    print(f"  多轮 TextVQA 评测结果 ({len(image_ids)} images, {args.rounds_per_image} rounds)")
    print(f"{'=' * 60}")
    print(f"  {'Method':<12}{'Accuracy':<14}{'Total Tokens':<16}{'Savings':<10}")
    print(f"  {'-' * 50}")

    for method in ['vanilla', 'fastv', 'mtvc']:
        acc_list = all_results[method]
        if not acc_list:
            continue
        acc = sum(acc_list) / len(acc_list) * 100
        tok = total_tokens[method]
        savings = (1 - tok / total_tokens['vanilla']) * 100 if total_tokens['vanilla'] > 0 else 0
        print(f"  {method:<12}{acc:<14.2f}%{tok:<16}{savings:<10.1f}%")

    print(f"{'=' * 60}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        'num_images': len(image_ids),
        'rounds_per_image': args.rounds_per_image,
        'methods': {},
    }
    for method in ['vanilla', 'fastv', 'mtvc']:
        acc_list = all_results[method]
        output['methods'][method] = {
            'accuracy': sum(acc_list) / len(acc_list) * 100 if acc_list else 0,
            'total_tokens': total_tokens[method],
            'num_samples': len(acc_list),
        }

    output_file = os.path.join(args.output_dir, 'multiturn_eval.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
