"""
对抗恢复基准 — ReVoC 的核心卖点实验

设计: 同一张图片，5轮对话，每轮故意问不同空间区域
  Round 1: 描述整张图 (baseline)
  Round 2: 问图片某个特定区域 A
  Round 3: 问另一个区域 B (远离 A)
  Round 4: 回到区域 A 的细节 ← 关键轮次! 剪枝方法可能已丢失 A 的信息
  Round 5: 全局氛围/总结

对比三种方法:
  1. Vanilla: 每轮独立，全量 576 tokens (质量上界)
  2. FastV: 每轮独立，剪枝后 144 tokens
  3. ReVoC: Round 1 建缓存，后续从缓存检索 ~96 tokens

评测方式: 由于没有 ground truth，用 Vanilla 的回答作为参考
  - 计算 ReVoC/FastV 的回答与 Vanilla 回答的语义相似度
  - 使用模型自身的 embed_tokens 做 sentence embedding 比较

用法:
    bash run.sh python scripts/eval_adversarial_recovery.py
    bash run.sh python scripts/eval_adversarial_recovery.py --num-images 100
"""

import argparse
import json
import os
import sys
import time
import random

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastv.core import load_model, load_image, prepare_input, run_vanilla, run_fastv
from revoc import RevoCConfig, RevoCEngine, LLaVAAdapter


# ============================================================
# 对抗问题模板 — 每轮问不同空间区域
# ============================================================
ADVERSARIAL_TEMPLATES = [
    # 模板1: 前景→背景→前景细节
    [
        "Describe this image briefly.",
        "What objects are in the foreground of this image?",
        "What can you see in the background of this image?",
        "Going back to the foreground, what color are the main objects there?",
        "What is the overall mood of this image?",
    ],
    # 模板2: 左→右→左细节
    [
        "What is shown in this image?",
        "What is on the left side of the image?",
        "What is on the right side of the image?",
        "Describe the left side again - are there any small details you notice?",
        "What colors dominate the entire image?",
    ],
    # 模板3: 中心→边缘→中心细节
    [
        "Describe what you see in this image.",
        "What is at the center of the image?",
        "What is around the edges of the image?",
        "Focus on the center again - what is the main object's shape and color?",
        "Does this image look like it was taken indoors or outdoors?",
    ],
    # 模板4: 上→下→上细节
    [
        "What does this image show?",
        "What can you see in the upper part of the image?",
        "What is in the lower part of the image?",
        "Go back to the upper part - describe any text or signs you see.",
        "What time of day does this image appear to be?",
    ],
]


def compute_response_similarity(model, tokenizer, resp1: str, resp2: str, device: str) -> float:
    """
    Compute cosine similarity between two responses using embed_tokens.
    Proxy for semantic similarity without requiring an external model.
    """
    embed_fn = model.get_model().embed_tokens

    with torch.no_grad():
        ids1 = tokenizer(resp1, return_tensors="pt", add_special_tokens=False,
                         max_length=128, truncation=True).input_ids.to(device)
        ids2 = tokenizer(resp2, return_tensors="pt", add_special_tokens=False,
                         max_length=128, truncation=True).input_ids.to(device)

        emb1 = embed_fn(ids1).mean(dim=1)  # (1, D)
        emb2 = embed_fn(ids2).mean(dim=1)  # (1, D)

        sim = F.cosine_similarity(emb1, emb2, dim=-1).item()

    return sim


def run_vanilla_multiturn(model, tokenizer, image_processor, image, questions, device):
    """Run each question independently with vanilla (quality upper bound)."""
    responses = []
    for q in questions:
        input_ids, image_tensor = prepare_input(
            tokenizer, image_processor, image, q, device
        )
        resp = run_vanilla(model, tokenizer, input_ids, image_tensor, device, max_new_tokens=128)
        responses.append(resp)
    return responses


def run_fastv_multiturn(model, tokenizer, image_processor, image, questions, device,
                        fastv_k=2, fastv_r=0.75):
    """Run each question independently with FastV pruning."""
    responses = []
    for q in questions:
        input_ids, image_tensor = prepare_input(
            tokenizer, image_processor, image, q, device
        )
        resp = run_fastv(model, tokenizer, input_ids, image_tensor, device,
                         fastv_k=fastv_k, fastv_r=fastv_r, max_new_tokens=128)
        responses.append(resp)
    return responses


def run_revoc_multiturn(adapter, image, questions, device, config):
    """Run true multi-turn with ReVoC cache."""
    engine = RevoCEngine(adapter, config, device)
    engine.start_session(image)
    responses = []
    for q in questions:
        stats = engine.chat(q)
        responses.append(stats.response)
    return responses


def main():
    parser = argparse.ArgumentParser(description='Adversarial Recovery Benchmark')
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--image-dir', type=str, default='data/textvqa/train_images',
                        help='Directory with test images')
    parser.add_argument('--num-images', type=int, default=50,
                        help='Number of images to test')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device

    # Load model
    print(f"Loading model: {args.model_path}")
    tokenizer, model, image_processor, _ = load_model(args.model_path, device)

    # Setup ReVoC adapter
    adapter = LLaVAAdapter()
    adapter.model = model
    adapter.tokenizer = tokenizer
    adapter.image_processor = image_processor
    adapter.device = device

    revoc_config = RevoCConfig(retriever_type="cosine", max_new_tokens=128)

    # Collect test images
    image_files = []
    if os.path.isdir(args.image_dir):
        for f in os.listdir(args.image_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_files.append(os.path.join(args.image_dir, f))

    if not image_files:
        # Fallback: use default demo image
        print("No local images found, using demo image")
        image_files = ['https://llava-vl.github.io/static/images/view.jpg']

    random.shuffle(image_files)
    image_files = image_files[:args.num_images]
    print(f"Testing {len(image_files)} images with {len(ADVERSARIAL_TEMPLATES)} templates")

    # ============================================================
    # Main evaluation loop
    # ============================================================
    all_results = []
    per_round_sims = {
        'fastv': {i: [] for i in range(5)},
        'revoc': {i: [] for i in range(5)},
    }

    for img_idx, img_path in enumerate(tqdm(image_files, desc="Adversarial eval")):
        try:
            image = load_image(img_path)
        except Exception:
            continue

        # Pick a template
        template = ADVERSARIAL_TEMPLATES[img_idx % len(ADVERSARIAL_TEMPLATES)]

        try:
            # 1) Vanilla (reference)
            vanilla_resps = run_vanilla_multiturn(
                model, tokenizer, image_processor, image, template, device)

            # 2) FastV
            fastv_resps = run_fastv_multiturn(
                model, tokenizer, image_processor, image, template, device)

            # 3) ReVoC
            revoc_resps = run_revoc_multiturn(
                adapter, image, template, device, revoc_config)

            # Compute per-round similarity to vanilla
            img_result = {'image': img_path, 'rounds': []}
            for r in range(5):
                fastv_sim = compute_response_similarity(
                    model, tokenizer, vanilla_resps[r], fastv_resps[r], device)
                revoc_sim = compute_response_similarity(
                    model, tokenizer, vanilla_resps[r], revoc_resps[r], device)

                per_round_sims['fastv'][r].append(fastv_sim)
                per_round_sims['revoc'][r].append(revoc_sim)

                img_result['rounds'].append({
                    'round': r + 1,
                    'question': template[r],
                    'vanilla': vanilla_resps[r][:200],
                    'fastv': fastv_resps[r][:200],
                    'revoc': revoc_resps[r][:200],
                    'fastv_sim': fastv_sim,
                    'revoc_sim': revoc_sim,
                })

            all_results.append(img_result)

        except Exception as e:
            print(f"  Skip {img_path}: {e}")
            continue

    # ============================================================
    # 汇总报告
    # ============================================================
    print(f"\n{'='*72}")
    print(f"  Adversarial Recovery Benchmark — {len(all_results)} images")
    print(f"{'='*72}")
    print(f"  {'Round':<8}{'Question Type':<35}{'FastV sim':<14}{'ReVoC sim':<14}{'Winner':<8}")
    print(f"  {'-'*75}")

    round_labels = ['Initial description', 'Region A', 'Region B',
                    'RECOVERY (back to A)', 'Global summary']

    for r in range(5):
        f_sims = per_round_sims['fastv'][r]
        r_sims = per_round_sims['revoc'][r]
        f_avg = sum(f_sims) / len(f_sims) if f_sims else 0
        r_avg = sum(r_sims) / len(r_sims) if r_sims else 0
        winner = "ReVoC" if r_avg > f_avg else "FastV"
        marker = "  ←←←" if r == 3 else ""
        print(f"  {r+1:<8}{round_labels[r]:<35}{f_avg:<14.4f}{r_avg:<14.4f}{winner:<8}{marker}")

    print(f"{'='*72}")

    # Overall averages
    all_fastv = [s for r in per_round_sims['fastv'].values() for s in r]
    all_revoc = [s for r in per_round_sims['revoc'].values() for s in r]
    f_overall = sum(all_fastv) / len(all_fastv) if all_fastv else 0
    r_overall = sum(all_revoc) / len(all_revoc) if all_revoc else 0

    # Round 4 specific (recovery round)
    f_r4 = per_round_sims['fastv'][3]
    r_r4 = per_round_sims['revoc'][3]
    f_r4_avg = sum(f_r4) / len(f_r4) if f_r4 else 0
    r_r4_avg = sum(r_r4) / len(r_r4) if r_r4 else 0

    print(f"\n  Overall similarity to vanilla:")
    print(f"    FastV:  {f_overall:.4f}")
    print(f"    ReVoC:  {r_overall:.4f}")
    print(f"\n  Round 4 (RECOVERY) similarity to vanilla:")
    print(f"    FastV:  {f_r4_avg:.4f}")
    print(f"    ReVoC:  {r_r4_avg:.4f}")
    print(f"    Gap:    {r_r4_avg - f_r4_avg:+.4f} {'(ReVoC wins)' if r_r4_avg > f_r4_avg else '(FastV wins)'}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        'num_images': len(all_results),
        'per_round_avg': {
            'fastv': {str(r): sum(s)/len(s) if s else 0
                      for r, s in per_round_sims['fastv'].items()},
            'revoc': {str(r): sum(s)/len(s) if s else 0
                      for r, s in per_round_sims['revoc'].items()},
        },
        'overall': {'fastv': f_overall, 'revoc': r_overall},
        'round4_recovery': {'fastv': f_r4_avg, 'revoc': r_r4_avg},
        'details': all_results,
    }
    out_path = os.path.join(args.output_dir, 'adversarial_recovery.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
