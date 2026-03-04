"""
FastV 核心推理模块 (two-step approach)

方法:
  Step 1: 手动构建 inputs_embeds (展开 image tokens)
  Step 2: 完整 forward + hook 捕获第 K 层注意力
  Step 3: 根据注意力分数选择重要 image tokens
  Step 4: 用 pruned inputs_embeds 跑 generate()
"""

import torch
from PIL import Image
import requests
from io import BytesIO


def load_image(image_source: str) -> Image.Image:
    if image_source.startswith(('http://', 'https://')):
        response = requests.get(image_source)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_source)
    return image.convert('RGB')


def load_model(model_path, device='cuda'):
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


def prepare_input(tokenizer, image_processor, image, prompt, device='cuda'):
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


def get_multimodal_embeds(model, input_ids, image_tensor):
    """
    手动构建 inputs_embeds: 把 IMAGE_TOKEN_INDEX 替换成 image features
    返回 (full_embeds, image_start, num_image_tokens)
    """
    from llava.constants import IMAGE_TOKEN_INDEX

    image_features = model.encode_images(image_tensor)
    embed_tokens = model.get_model().embed_tokens

    # IMAGE_TOKEN_INDEX = -200, out of range for embedding table
    # Replace with 0 before embedding, then use prefix/suffix only
    safe_ids = input_ids.clone()
    image_mask = safe_ids[0] == IMAGE_TOKEN_INDEX
    safe_ids[0][image_mask] = 0
    input_embeds = embed_tokens(safe_ids)

    image_pos = image_mask.nonzero(as_tuple=True)[0]

    if len(image_pos) == 0:
        return input_embeds, -1, 0

    image_start = image_pos[0].item()
    num_image_tokens = image_features.shape[1]  # 576

    prefix_embeds = input_embeds[:, :image_start, :]
    suffix_embeds = input_embeds[:, image_start + 1:, :]  # +1 skip placeholder

    full_embeds = torch.cat([
        prefix_embeds,
        image_features,
        suffix_embeds,
    ], dim=1)

    return full_embeds, image_start, num_image_tokens


def run_vanilla(model, tokenizer, input_ids, image_tensor, device='cuda', max_new_tokens=256):
    """标准推理 (无剪枝)"""
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, images=image_tensor,
            do_sample=False, max_new_tokens=max_new_tokens,
        )

    new_tokens = output_ids[:, input_ids.shape[1]:]
    response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return response


def run_fastv(model, tokenizer, input_ids, image_tensor, device='cuda',
              fastv_k=2, fastv_r=0.75, max_new_tokens=256, verbose=False):
    """
    FastV 推理 (two-step approach)

    Step 1: 构建完整 inputs_embeds
    Step 2: forward + hook 捕获第 K 层注意力
    Step 3: 选择重要 image tokens
    Step 4: generate() with pruned inputs_embeds
    """
    with torch.no_grad():
        full_embeds, image_start, num_image_tokens = get_multimodal_embeds(
            model, input_ids, image_tensor
        )

    if image_start < 0:
        if verbose:
            print("  No image tokens found, falling back to vanilla")
        return run_vanilla(model, tokenizer, input_ids, image_tensor, device, max_new_tokens)

    seq_len = full_embeds.shape[1]
    if verbose:
        print(f"  原始序列: {seq_len} tokens, image: [{image_start}, {image_start + num_image_tokens})")

    # Hook to capture layer K attention
    attn_captured = {}
    llm = model.get_model()  # LlamaModel

    def capture_attn_hook(module, args, output):
        if len(output) > 1 and output[1] is not None:
            attn_captured['weights'] = output[1].detach()

    target_layer = llm.layers[fastv_k - 1]
    hook = target_layer.register_forward_hook(capture_attn_hook)

    with torch.no_grad():
        _ = model(
            inputs_embeds=full_embeds,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    hook.remove()

    if 'weights' not in attn_captured:
        if verbose:
            print("  警告: 未捕获到注意力，回退到 vanilla")
        return run_vanilla(model, tokenizer, input_ids, image_tensor, device, max_new_tokens)

    # Compute importance
    attn_weights = attn_captured['weights']  # (batch, heads, seq, seq)
    img_attn = attn_weights[:, :, -1, image_start:image_start + num_image_tokens]
    importance = img_attn.mean(dim=1)  # (batch, num_image_tokens)

    num_keep = int(num_image_tokens * (1 - fastv_r))
    _, top_indices = importance.topk(num_keep, dim=-1)
    top_indices = top_indices.sort(dim=-1).values  # keep original order

    if verbose:
        print(f"  FastV: 保留 {num_keep}/{num_image_tokens} image tokens, "
              f"剪掉 {num_image_tokens - num_keep} ({fastv_r*100:.0f}%)")

    # Build pruned inputs_embeds
    prefix_embeds = full_embeds[:, :image_start, :]
    selected_img_embeds = full_embeds[:, image_start + top_indices[0], :]
    suffix_start = image_start + num_image_tokens
    suffix_embeds = full_embeds[:, suffix_start:, :]

    pruned_embeds = torch.cat([prefix_embeds, selected_img_embeds, suffix_embeds], dim=1)
    pruned_len = pruned_embeds.shape[1]

    if verbose:
        print(f"  Pruned 序列: {pruned_len} tokens (原 {seq_len})")

    # Generate with pruned embeds
    attention_mask = torch.ones((1, pruned_len), dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=pruned_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response
