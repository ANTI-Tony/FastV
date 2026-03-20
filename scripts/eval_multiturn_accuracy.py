"""
多轮准确率评测 — Vanilla vs FastV vs ReVoC

使用 TextVQA 同图多问题构造多轮对话，对比三种方法的准确率。
这个脚本是 Table 1 的主要数据来源。

用法:
    bash run.sh python scripts/eval_multiturn_accuracy.py --num-images 100
    bash run.sh python scripts/eval_multiturn_accuracy.py --rounds 5 --num-images 200
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
from revoc import RevoCConfig, RevoCEngine, LLaVAAdapter


def textvqa_accuracy(pred: str, answers: list) -> float:
    pred = pred.strip().lower()
    counts = {}
    for a in answers:
        a = a.strip().lower()
        counts[a] = counts.get(a, 0) + 1
    return min(counts.get(pred, 0) / 3.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description='Multi-Turn Accuracy Evaluation')
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--data-path', type=str, default='data/textvqa/TextVQA_0.5.1_val.json')
    parser.add_argument('--image-dir', type=str, default='data/textvqa/train_images')
    parser.add_argument('--num-images', type=int, default=100)
    parser.add_argument('--rounds', type=int, default=3, help='Questions per image')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device

    print(f"Loading model: {args.model_path}")
    tokenizer, model, image_processor, _ = load_model(args.model_path, device)

    # ReVoC adapter
    adapter = LLaVAAdapter()
    adapter.model = model
    adapter.tokenizer = tokenizer
    adapter.image_processor = image_processor
    adapter.device = device

    revoc_config = RevoCConfig(retriever_type="cosine", max_new_tokens=128)

    # Load and group data by image
    print(f"Loading data: {args.data_path}")
    with open(args.data_path) as f:
        data = json.load(f)['data']

    groups = defaultdict(list)
    for item in data:
        groups[item['image_id']].append(item)

    # Filter to images with enough questions
    multi = {k: v for k, v in groups.items()
             if len(v) >= args.rounds}
    image_ids = list(multi.keys())[:args.num_images]
    print(f"Evaluating: {len(image_ids)} images × {args.rounds} rounds")

    # Results accumulators
    results = {
        'vanilla': {'per_round': defaultdict(list), 'total_tokens': 0, 'total_time': 0},
        'fastv': {'per_round': defaultdict(list), 'total_tokens': 0, 'total_time': 0},
        'revoc': {'per_round': defaultdict(list), 'total_tokens': 0, 'total_time': 0},
    }

    for img_id in tqdm(image_ids, desc="Multi-turn eval"):
        # Find image file
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            p = os.path.join(args.image_dir, str(img_id) + ext)
            if os.path.exists(p):
                img_path = p
                break
        if not img_path:
            continue

        try:
            image = Image.open(img_path).convert('RGB')
            qdata = multi[img_id][:args.rounds]

            # ---- Vanilla: each round independent ----
            t0 = time.time()
            for r_idx, item in enumerate(qdata):
                q = item['question'] + "\nAnswer the question using a single word or phrase."
                ids, tensor = prepare_input(tokenizer, image_processor, image, q, device)
                pred = run_vanilla(model, tokenizer, ids, tensor, device, 128)
                acc = textvqa_accuracy(pred, item.get('answers', []))
                results['vanilla']['per_round'][r_idx].append(acc)
                results['vanilla']['total_tokens'] += 576
            results['vanilla']['total_time'] += time.time() - t0

            # ---- FastV R=75%: each round independent ----
            t0 = time.time()
            for r_idx, item in enumerate(qdata):
                q = item['question'] + "\nAnswer the question using a single word or phrase."
                ids, tensor = prepare_input(tokenizer, image_processor, image, q, device)
                pred = run_fastv(model, tokenizer, ids, tensor, device,
                                 fastv_k=2, fastv_r=0.75, max_new_tokens=128)
                acc = textvqa_accuracy(pred, item.get('answers', []))
                results['fastv']['per_round'][r_idx].append(acc)
                results['fastv']['total_tokens'] += 144
            results['fastv']['total_time'] += time.time() - t0

            # ---- ReVoC: true multi-turn ----
            t0 = time.time()
            engine = RevoCEngine(adapter, revoc_config, device)
            engine.start_session(image)
            for r_idx, item in enumerate(qdata):
                q = item['question'] + "\nAnswer the question using a single word or phrase."
                stats = engine.chat(q)
                acc = textvqa_accuracy(stats.response, item.get('answers', []))
                results['revoc']['per_round'][r_idx].append(acc)
                results['revoc']['total_tokens'] += stats.image_tokens_used
            results['revoc']['total_time'] += time.time() - t0

        except Exception as e:
            print(f"  Skip {img_id}: {e}")
            continue

    # ============================================================
    # 汇总
    # ============================================================
    n_images = len(image_ids)
    print(f"\n{'='*72}")
    print(f"  Multi-Turn Accuracy — {n_images} images × {args.rounds} rounds")
    print(f"{'='*72}")

    # Per-round accuracy
    print(f"\n  Per-Round Accuracy (%):")
    print(f"  {'Method':<12}", end='')
    for r in range(args.rounds):
        print(f"{'R' + str(r+1):<10}", end='')
    print(f"{'Avg':<10}{'Tokens':<12}{'Time(s)':<10}")
    print(f"  {'-'*70}")

    for method in ['vanilla', 'fastv', 'revoc']:
        line = f"  {method:<12}"
        round_accs = []
        for r in range(args.rounds):
            accs = results[method]['per_round'].get(r, [])
            avg = sum(accs) / len(accs) * 100 if accs else 0
            round_accs.append(avg)
            line += f"{avg:<10.1f}"
        overall = sum(round_accs) / len(round_accs) if round_accs else 0
        line += f"{overall:<10.1f}"
        line += f"{results[method]['total_tokens']:<12}"
        line += f"{results[method]['total_time']:<10.1f}"
        print(line)

    print(f"{'='*72}")

    # Accuracy drop from vanilla
    print(f"\n  Accuracy Drop from Vanilla:")
    for method in ['fastv', 'revoc']:
        drops = []
        for r in range(args.rounds):
            v_accs = results['vanilla']['per_round'].get(r, [])
            m_accs = results[method]['per_round'].get(r, [])
            v_avg = sum(v_accs) / len(v_accs) * 100 if v_accs else 0
            m_avg = sum(m_accs) / len(m_accs) * 100 if m_accs else 0
            drops.append(v_avg - m_avg)
        avg_drop = sum(drops) / len(drops) if drops else 0
        print(f"  {method:<12} avg drop: {avg_drop:+.2f}%  "
              f"per-round: {', '.join(f'{d:+.1f}' for d in drops)}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        'num_images': n_images,
        'rounds': args.rounds,
        'methods': {},
    }
    for method in ['vanilla', 'fastv', 'revoc']:
        round_data = {}
        for r in range(args.rounds):
            accs = results[method]['per_round'].get(r, [])
            round_data[f'round_{r+1}'] = sum(accs) / len(accs) * 100 if accs else 0
        output['methods'][method] = {
            'per_round': round_data,
            'avg_accuracy': sum(round_data.values()) / len(round_data) if round_data else 0,
            'total_tokens': results[method]['total_tokens'],
            'total_time': results[method]['total_time'],
        }

    out_path = os.path.join(args.output_dir, 'multiturn_accuracy.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
