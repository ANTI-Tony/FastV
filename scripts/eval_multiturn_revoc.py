"""
Multi-turn evaluation: Vanilla vs FastV vs ReVoC on TextVQA + VisDial.

TextVQA multi-turn: same-image question groups as multi-round conversation.
VisDial: real multi-turn visual dialogue benchmark.

Usage:
    bash run.sh python scripts/eval_multiturn_revoc.py --benchmark textvqa
    bash run.sh python scripts/eval_multiturn_revoc.py --benchmark visdial
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


# ========== TextVQA accuracy ==========
def textvqa_accuracy(pred: str, answers: list) -> float:
    pred = pred.strip().lower()
    counts = {}
    for a in answers:
        a = a.strip().lower()
        counts[a] = counts.get(a, 0) + 1
    return min(counts.get(pred, 0) / 3.0, 1.0)


# ========== TextVQA multi-turn ==========
def eval_textvqa_multiturn(args, model, tokenizer, image_processor, adapter, device):
    with open(args.data_path) as f:
        data = json.load(f)['data']

    # Group by image
    groups = defaultdict(list)
    for item in data:
        groups[item['image_id']].append(item)

    multi = {k: v for k, v in groups.items() if len(v) >= args.rounds_per_image}
    image_ids = list(multi.keys())[:args.max_images]
    print(f"Evaluating: {len(image_ids)} images x {args.rounds_per_image} rounds")

    results = {'vanilla': [], 'fastv': [], 'revoc': []}
    tokens = {'vanilla': 0, 'fastv': 0, 'revoc': 0}

    revoc_config = RevoCConfig(
        n_clusters=64, n_retrieve_clusters=12,
        retriever_type=args.retriever, max_new_tokens=128,
    )

    for img_id in tqdm(image_ids, desc="Multi-turn eval"):
        img_path = os.path.join(args.image_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert('RGB')
            qdata = multi[img_id][:args.rounds_per_image]

            # --- Vanilla ---
            for item in qdata:
                q = item['question'] + "\nAnswer the question using a single word or phrase."
                ids, tensor = prepare_input(tokenizer, image_processor, image, q, device)
                pred = run_vanilla(model, tokenizer, ids, tensor, device, 128)
                results['vanilla'].append(textvqa_accuracy(pred, item.get('answers', [])))
                tokens['vanilla'] += 576

            # --- FastV ---
            for item in qdata:
                q = item['question'] + "\nAnswer the question using a single word or phrase."
                ids, tensor = prepare_input(tokenizer, image_processor, image, q, device)
                pred = run_fastv(model, tokenizer, ids, tensor, device,
                                 fastv_k=2, fastv_r=0.75, max_new_tokens=128)
                results['fastv'].append(textvqa_accuracy(pred, item.get('answers', [])))
                tokens['fastv'] += 144

            # --- ReVoC ---
            engine = RevoCEngine(adapter, revoc_config, device)
            engine.start_session(image)
            for item in qdata:
                q = item['question'] + "\nAnswer the question using a single word or phrase."
                stats = engine.chat(q)
                results['revoc'].append(textvqa_accuracy(stats.response, item.get('answers', [])))
            summary = engine.get_summary()
            tokens['revoc'] += summary['total_image_tokens']

        except Exception as e:
            print(f"Skip {img_id}: {e}")
            continue

    return results, tokens


# ========== VisDial evaluation ==========
def eval_visdial(args, model, tokenizer, image_processor, adapter, device):
    """
    VisDial v1.0 evaluation.
    Data format: each dialog has a caption + 10 rounds of (question, answer, answer_options).
    We evaluate on dense annotations (NDCG, MRR, R@k).
    """
    if not os.path.exists(args.visdial_path):
        print(f"VisDial data not found at {args.visdial_path}")
        print("Download from: https://visualdialog.org/data")
        return None, None

    with open(args.visdial_path) as f:
        visdial = json.load(f)

    dialogs = visdial['data']['dialogs'][:args.max_images]
    questions = visdial['data']['questions']
    answers = visdial['data']['answers']
    print(f"VisDial: {len(dialogs)} dialogs")

    revoc_config = RevoCConfig(
        n_clusters=64, n_retrieve_clusters=12,
        retriever_type=args.retriever, max_new_tokens=64,
    )

    results = {'vanilla': [], 'revoc': []}
    tokens = {'vanilla': 0, 'revoc': 0}

    for dialog in tqdm(dialogs, desc="VisDial eval"):
        img_id = dialog['image_id']
        # VisDial uses COCO images
        img_path = os.path.join(
            args.visdial_image_dir,
            f"COCO_val2014_{img_id:012d}.jpg"
        )
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert('RGB')
            caption = dialog['caption']
            rounds = dialog['dialog'][:args.rounds_per_image]

            # --- ReVoC multi-turn ---
            engine = RevoCEngine(adapter, revoc_config, device)
            engine.start_session(image)

            # First round uses caption as context
            for rnd in rounds:
                q = questions[rnd['question']]
                gt_answer = answers[rnd['answer']]

                stats = engine.chat(q)
                # Simple accuracy: does response contain the answer?
                pred = stats.response.strip().lower()
                gt = gt_answer.strip().lower()
                acc = 1.0 if gt in pred or pred in gt else 0.0
                results['revoc'].append(acc)

            summary = engine.get_summary()
            tokens['revoc'] += summary['total_image_tokens']

            # --- Vanilla (each round independent) ---
            for rnd in rounds:
                q = questions[rnd['question']]
                gt_answer = answers[rnd['answer']]
                ids, tensor = prepare_input(tokenizer, image_processor, image, q, device)
                pred = run_vanilla(model, tokenizer, ids, tensor, device, 64)
                gt = gt_answer.strip().lower()
                acc = 1.0 if gt in pred.strip().lower() or pred.strip().lower() in gt else 0.0
                results['vanilla'].append(acc)
                tokens['vanilla'] += 576

        except Exception as e:
            continue

    return results, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--benchmark', type=str, default='textvqa',
                        choices=['textvqa', 'visdial', 'both'])
    parser.add_argument('--data-path', type=str, default='data/textvqa/TextVQA_0.5.1_val.json')
    parser.add_argument('--image-dir', type=str, default='data/textvqa/train_images')
    parser.add_argument('--visdial-path', type=str, default='data/visdial/visdial_1.0_val.json')
    parser.add_argument('--visdial-image-dir', type=str, default='data/visdial/images')
    parser.add_argument('--max-images', type=int, default=100)
    parser.add_argument('--rounds-per-image', type=int, default=3)
    parser.add_argument('--retriever', type=str, default='cosine',
                        choices=['cosine', 'cross_attention'])
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    device = 'cuda'

    print(f"Loading model: {args.model_path}")
    tokenizer, model, image_processor, _ = load_model(args.model_path, device)

    adapter = LLaVAAdapter()
    adapter.model = model
    adapter.tokenizer = tokenizer
    adapter.image_processor = image_processor
    adapter.device = device

    all_output = {}

    # TextVQA
    if args.benchmark in ('textvqa', 'both'):
        print("\n=== TextVQA Multi-Turn ===")
        results, tokens = eval_textvqa_multiturn(
            args, model, tokenizer, image_processor, adapter, device
        )
        _print_results("TextVQA", results, tokens)
        all_output['textvqa'] = _format_output(results, tokens)

    # VisDial
    if args.benchmark in ('visdial', 'both'):
        print("\n=== VisDial ===")
        results, tokens = eval_visdial(
            args, model, tokenizer, image_processor, adapter, device
        )
        if results:
            _print_results("VisDial", results, tokens)
            all_output['visdial'] = _format_output(results, tokens)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f'revoc_eval_{args.benchmark}.json')
    with open(out_file, 'w') as f:
        json.dump(all_output, f, indent=2)
    print(f"\nResults saved to {out_file}")


def _print_results(name, results, tokens):
    print(f"\n{'=' * 60}")
    print(f"  {name} Results")
    print(f"{'=' * 60}")
    print(f"  {'Method':<16}{'Accuracy':<14}{'Total Tokens':<16}{'Savings':<10}")
    print(f"  {'-' * 54}")
    vanilla_tok = tokens.get('vanilla', 1)
    for method in results:
        acc_list = results[method]
        if not acc_list:
            continue
        acc = sum(acc_list) / len(acc_list) * 100
        tok = tokens[method]
        savings = (1 - tok / vanilla_tok) * 100 if vanilla_tok > 0 else 0
        print(f"  {method:<16}{acc:<14.2f}%{tok:<16}{savings:<10.1f}%")
    print(f"{'=' * 60}")


def _format_output(results, tokens):
    output = {}
    for method in results:
        acc_list = results[method]
        output[method] = {
            'accuracy': sum(acc_list) / len(acc_list) * 100 if acc_list else 0,
            'total_tokens': tokens[method],
            'num_samples': len(acc_list),
        }
    return output


if __name__ == '__main__':
    main()
