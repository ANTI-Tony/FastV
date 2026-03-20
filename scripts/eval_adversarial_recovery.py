"""
对抗恢复基准 v2 — ReVoC 的核心卖点实验

四种方法对比:
  1. Vanilla: 每轮独立，576 tokens，5次ViT编码 (质量上界)
  2. FastV-indep: 每轮独立编码+剪枝，144 tokens，5次ViT编码 (FastV原始用法)
  3. FastV-cached: Round 1 剪枝，后续复用相同的144 tokens，1次ViT编码
     → 这才是"多轮缓存"场景下的真实对手，Round 4 会翻车
  4. ReVoC: Round 1 建缓存，后续恢复 ~140 tokens，1次ViT编码
     → 可恢复压缩，Round 4 不翻车

论文论点:
  - FastV-indep 质量好但每轮重编码(贵)
  - FastV-cached 省了重编码但 Round 4 翻车(不可逆)
  - ReVoC 既省重编码又不翻车(可恢复)

用法:
    bash run.sh python scripts/eval_adversarial_recovery.py --num-images 50
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
from fastv.core import (
    load_model, load_image, prepare_input,
    get_multimodal_embeds, run_vanilla, run_fastv,
)
from revoc import RevoCConfig, RevoCEngine, LLaVAAdapter


# ============================================================
# 对抗问题模板
# ============================================================
ADVERSARIAL_TEMPLATES = [
    [
        "Describe this image briefly.",
        "What objects are in the foreground of this image?",
        "What can you see in the background of this image?",
        "Going back to the foreground, what color are the main objects there?",
        "What is the overall mood of this image?",
    ],
    [
        "What is shown in this image?",
        "What is on the left side of the image?",
        "What is on the right side of the image?",
        "Describe the left side again - are there any small details you notice?",
        "What colors dominate the entire image?",
    ],
    [
        "Describe what you see in this image.",
        "What is at the center of the image?",
        "What is around the edges of the image?",
        "Focus on the center again - what is the main object's shape and color?",
        "Does this image look like it was taken indoors or outdoors?",
    ],
    [
        "What does this image show?",
        "What can you see in the upper part of the image?",
        "What is in the lower part of the image?",
        "Go back to the upper part - describe any text or signs you see there.",
        "What time of day does this image appear to be?",
    ],
]


def compute_response_similarity(model, tokenizer, resp1, resp2, device):
    """Cosine similarity between responses via embed_tokens mean pooling."""
    embed_fn = model.get_model().embed_tokens
    with torch.no_grad():
        ids1 = tokenizer(resp1, return_tensors="pt", add_special_tokens=False,
                         max_length=128, truncation=True).input_ids.to(device)
        ids2 = tokenizer(resp2, return_tensors="pt", add_special_tokens=False,
                         max_length=128, truncation=True).input_ids.to(device)
        emb1 = embed_fn(ids1).mean(dim=1)
        emb2 = embed_fn(ids2).mean(dim=1)
        sim = F.cosine_similarity(emb1, emb2, dim=-1).item()
    return sim


# ============================================================
# 方法1: Vanilla — 每轮独立全量
# ============================================================
def run_vanilla_multiturn(model, tokenizer, image_processor, image, questions, device):
    responses = []
    for q in questions:
        input_ids, image_tensor = prepare_input(
            tokenizer, image_processor, image, q, device)
        resp = run_vanilla(model, tokenizer, input_ids, image_tensor, device, 128)
        responses.append(resp)
    return responses


# ============================================================
# 方法2: FastV-indep — 每轮独立编码+剪枝
# ============================================================
def run_fastv_indep(model, tokenizer, image_processor, image, questions, device):
    responses = []
    for q in questions:
        input_ids, image_tensor = prepare_input(
            tokenizer, image_processor, image, q, device)
        resp = run_fastv(model, tokenizer, input_ids, image_tensor, device,
                         fastv_k=2, fastv_r=0.75, max_new_tokens=128)
        responses.append(resp)
    return responses


# ============================================================
# 方法3: FastV-cached — Round 1 剪枝，后续复用相同 pruned tokens
#   这模拟了"想省 ViT 编码"的多轮缓存场景
#   Round 1: 完整 forward → attention → 选 top-25% tokens → 缓存
#   Round 2+: 用 Round 1 缓存的 pruned embeds + 新对话模板 → 生成
# ============================================================
def run_fastv_cached(model, tokenizer, image_processor, image, questions, device):
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images
    from transformers import LlamaForCausalLM

    image_tensor = process_images(
        [image], image_processor, None).to(device, dtype=torch.float16)

    responses = []

    # ---- Round 1: 完整 forward + 剪枝 + 缓存 pruned embeds ----
    input_ids_r1, _ = prepare_input(
        tokenizer, image_processor, image, questions[0], device)

    with torch.no_grad():
        full_embeds, image_start, num_img = get_multimodal_embeds(
            model, input_ids_r1, image_tensor)

    if image_start < 0:
        # No image tokens, fallback
        for q in questions:
            ids, tensor = prepare_input(tokenizer, image_processor, image, q, device)
            responses.append(run_vanilla(model, tokenizer, ids, tensor, device, 128))
        return responses

    # Attention hook to get importance
    attn_captured = {}
    llm = model.get_model()

    def hook_fn(module, args, output):
        if len(output) > 1 and output[1] is not None:
            attn_captured['weights'] = output[1].detach()

    hook = llm.layers[1].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(inputs_embeds=full_embeds, output_attentions=True,
                  use_cache=False, return_dict=True)
    hook.remove()

    # Compute importance and select top-25% tokens
    attn_w = attn_captured['weights']
    img_attn = attn_w[:, :, -1, image_start:image_start + num_img]
    importance = img_attn.mean(dim=1)
    num_keep = int(num_img * 0.25)  # R=0.75
    _, top_indices = importance.topk(num_keep, dim=-1)
    top_indices = top_indices.sort(dim=-1).values

    # Cache the selected image embeds (these are FROZEN for all rounds)
    cached_img_embeds = full_embeds[:, image_start + top_indices[0], :]  # (1, 144, D)

    # Generate Round 1 with full embeds
    seq_len = full_embeds.shape[1]
    attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    with torch.no_grad():
        out_ids = LlamaForCausalLM.generate(
            model, inputs_embeds=full_embeds, attention_mask=attn_mask,
            do_sample=False, max_new_tokens=128)
    r1_resp = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    responses.append(r1_resp)

    # ---- Round 2+: 复用 Round 1 的 cached_img_embeds ----
    embed_tokens = model.get_model().embed_tokens

    for q_idx in range(1, len(questions)):
        # Build multi-turn conversation template
        conv = conv_templates["v1"].copy()
        first_q = questions[0]
        if DEFAULT_IMAGE_TOKEN not in first_q:
            first_q = DEFAULT_IMAGE_TOKEN + "\n" + first_q
        conv.append_message(conv.roles[0], first_q)
        conv.append_message(conv.roles[1], responses[0])

        for prev_idx in range(1, q_idx):
            conv.append_message(conv.roles[0], questions[prev_idx])
            conv.append_message(conv.roles[1], responses[prev_idx])

        conv.append_message(conv.roles[0], questions[q_idx])
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(device)

        # Replace image placeholder with CACHED pruned embeds
        safe_ids = input_ids.clone()
        image_mask = safe_ids[0] == IMAGE_TOKEN_INDEX
        safe_ids[0][image_mask] = 0
        input_embeds = embed_tokens(safe_ids)

        image_pos = image_mask.nonzero(as_tuple=True)[0]
        if len(image_pos) > 0:
            img_start = image_pos[0].item()
            prefix = input_embeds[:, :img_start, :]
            suffix = input_embeds[:, img_start + 1:, :]
            round_embeds = torch.cat([prefix, cached_img_embeds, suffix], dim=1)
        else:
            round_embeds = input_embeds

        r_len = round_embeds.shape[1]
        r_mask = torch.ones((1, r_len), dtype=torch.long, device=device)
        with torch.no_grad():
            out_ids = LlamaForCausalLM.generate(
                model, inputs_embeds=round_embeds, attention_mask=r_mask,
                do_sample=False, max_new_tokens=128)
        resp = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        responses.append(resp)

    return responses


# ============================================================
# 方法4: ReVoC — 可恢复缓存（关闭自适应，强制 full recovery）
# ============================================================
def run_revoc_multiturn(adapter, image, questions, device, config):
    engine = RevoCEngine(adapter, config, device)
    engine.start_session(image)
    responses = []
    for q in questions:
        stats = engine.chat(q)
        responses.append(stats.response)
    return responses


def main():
    parser = argparse.ArgumentParser(description='Adversarial Recovery Benchmark v2')
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--image-dir', type=str, default='data/textvqa/train_images')
    parser.add_argument('--num-images', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    from llava.conversation import conv_templates  # pre-import

    print(f"Loading model: {args.model_path}")
    tokenizer, model, image_processor, _ = load_model(args.model_path, device)

    adapter = LLaVAAdapter()
    adapter.model = model
    adapter.tokenizer = tokenizer
    adapter.image_processor = image_processor
    adapter.device = device

    # ReVoC config: DISABLE adaptive recovery, force full cluster unpack
    revoc_config = RevoCConfig(
        retriever_type="cosine",
        max_new_tokens=128,
        adaptive_recovery=False,  # force full recovery every round
    )

    # Collect images
    image_files = []
    if os.path.isdir(args.image_dir):
        for f in os.listdir(args.image_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_files.append(os.path.join(args.image_dir, f))

    if not image_files:
        print("No local images found, using demo image")
        image_files = ['https://llava-vl.github.io/static/images/view.jpg']

    random.shuffle(image_files)
    image_files = image_files[:args.num_images]

    methods = ['fastv_indep', 'fastv_cached', 'revoc']
    print(f"Testing {len(image_files)} images, 4 methods × 5 rounds")

    # ============================================================
    # Main loop
    # ============================================================
    per_round_sims = {m: {i: [] for i in range(5)} for m in methods}
    all_results = []

    for img_idx, img_path in enumerate(tqdm(image_files, desc="Adversarial eval")):
        try:
            image = load_image(img_path)
        except Exception:
            continue

        template = ADVERSARIAL_TEMPLATES[img_idx % len(ADVERSARIAL_TEMPLATES)]

        try:
            # Vanilla (reference)
            vanilla_resps = run_vanilla_multiturn(
                model, tokenizer, image_processor, image, template, device)

            # FastV-indep (re-encodes every round)
            fastv_i_resps = run_fastv_indep(
                model, tokenizer, image_processor, image, template, device)

            # FastV-cached (reuses Round 1 pruned tokens)
            fastv_c_resps = run_fastv_cached(
                model, tokenizer, image_processor, image, template, device)

            # ReVoC (recoverable cache)
            revoc_resps = run_revoc_multiturn(
                adapter, image, template, device, revoc_config)

            # Compute similarities
            img_result = {'image': os.path.basename(img_path), 'rounds': []}
            for r in range(5):
                sims = {}
                for method, resps in [('fastv_indep', fastv_i_resps),
                                       ('fastv_cached', fastv_c_resps),
                                       ('revoc', revoc_resps)]:
                    sim = compute_response_similarity(
                        model, tokenizer, vanilla_resps[r], resps[r], device)
                    per_round_sims[method][r].append(sim)
                    sims[method] = sim

                img_result['rounds'].append({
                    'round': r + 1,
                    'question': template[r],
                    'vanilla': vanilla_resps[r][:200],
                    'fastv_indep': fastv_i_resps[r][:200],
                    'fastv_cached': fastv_c_resps[r][:200],
                    'revoc': revoc_resps[r][:200],
                    **{f'{m}_sim': sims[m] for m in methods},
                })
            all_results.append(img_result)

        except Exception as e:
            print(f"  Skip: {e}")
            continue

    # ============================================================
    # 汇总报告
    # ============================================================
    n = len(all_results)
    print(f"\n{'='*85}")
    print(f"  Adversarial Recovery Benchmark v2 — {n} images")
    print(f"{'='*85}")

    round_labels = ['Initial description', 'Region A', 'Region B',
                    'RECOVERY (back to A)', 'Global summary']

    print(f"  {'Round':<8}{'Type':<25}{'FastV-ind':<12}{'FastV-cch':<12}{'ReVoC':<12}{'Winner':<12}")
    print(f"  {'-'*79}")

    for r in range(5):
        avgs = {}
        for m in methods:
            s = per_round_sims[m][r]
            avgs[m] = sum(s) / len(s) if s else 0

        # Winner between cached methods (fair comparison: both encode once)
        cache_winner = 'ReVoC' if avgs['revoc'] > avgs['fastv_cached'] else 'FastV-cch'
        marker = "  ←KEY" if r == 3 else ""

        print(f"  {r+1:<8}{round_labels[r]:<25}"
              f"{avgs['fastv_indep']:<12.4f}"
              f"{avgs['fastv_cached']:<12.4f}"
              f"{avgs['revoc']:<12.4f}"
              f"{cache_winner:<12}{marker}")

    print(f"{'='*85}")

    # Overall & Round 4
    for m in methods:
        all_s = [s for r in per_round_sims[m].values() for s in r]
        avg = sum(all_s) / len(all_s) if all_s else 0
        r4_s = per_round_sims[m][3]
        r4_avg = sum(r4_s) / len(r4_s) if r4_s else 0
        print(f"  {m:<16} overall={avg:.4f}  round4={r4_avg:.4f}")

    # The key comparison
    fc_r4 = per_round_sims['fastv_cached'][3]
    rv_r4 = per_round_sims['revoc'][3]
    fc_avg = sum(fc_r4) / len(fc_r4) if fc_r4 else 0
    rv_avg = sum(rv_r4) / len(rv_r4) if rv_r4 else 0
    print(f"\n  ★ Round 4 (RECOVERY) — fair comparison (both encode once):")
    print(f"    FastV-cached: {fc_avg:.4f}")
    print(f"    ReVoC:        {rv_avg:.4f}")
    print(f"    Gap:          {rv_avg - fc_avg:+.4f} "
          f"{'(ReVoC wins!)' if rv_avg > fc_avg else '(FastV-cached wins)'}")

    # Token count comparison
    print(f"\n  Token usage per 5-round conversation:")
    print(f"    Vanilla:       576×5 = 2880  (5 ViT encodes)")
    print(f"    FastV-indep:   144×5 = 720   (5 ViT encodes)")
    print(f"    FastV-cached:  144×5 = 720   (1 ViT encode, stale tokens)")
    print(f"    ReVoC:         576+140×4≈1136 (1 ViT encode, fresh recovered tokens)")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        'num_images': n,
        'per_round_avg': {},
        'round4_recovery': {},
    }
    for m in methods:
        output['per_round_avg'][m] = {
            str(r): sum(s)/len(s) if s else 0
            for r, s in per_round_sims[m].items()
        }
        r4 = per_round_sims[m][3]
        output['round4_recovery'][m] = sum(r4)/len(r4) if r4 else 0
    output['details'] = all_results

    out_path = os.path.join(args.output_dir, 'adversarial_recovery_v2.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
