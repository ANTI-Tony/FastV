"""
FastV 推理 Demo

通过 monkey-patch LlamaModel.forward 实现原论文的 inplace token dropping:
在第 K 层后，根据注意力分数丢弃 R% 的 image tokens。

用法:
    bash run.sh python demo_fastv.py --model-path models/llava-v1.5-7b
    bash run.sh python demo_fastv.py --fastv-k 2 --fastv-r 0.75
    bash run.sh python demo_fastv.py --fastv-k 2 --fastv-r 0.5
"""

import argparse
import time
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


# ============================================================
#  FastV: Monkey-patch LlamaModel.forward
#  在第 K 层之后丢弃低注意力的 image tokens (原论文方式)
# ============================================================

def patch_model_for_fastv(model, fastv_k=2, fastv_r=0.75, image_token_start=35, image_token_length=576):
    """
    Monkey-patch 模型实现 FastV inplace token dropping

    原理: 修改 LlamaModel.forward()，在第 K 层输出后:
    1. 从该层的注意力权重中计算 image token 重要性
    2. 保留 top-(1-R) 的 image tokens
    3. 从 hidden_states 中物理删除不重要的 tokens
    4. 后续层处理更少的 tokens → 真正的加速
    """
    import transformers.models.llama.modeling_llama as llama_module

    # 保存原始 forward
    LlamaModel = llama_module.LlamaModel
    original_forward = LlamaModel.forward

    # FastV 状态
    fastv_state = {
        'enabled': True,
        'pruned_this_pass': False,
        'num_kept': 0,
        'num_pruned': 0,
    }

    def fastv_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        修改后的 LlamaModel.forward()

        和原版唯一的区别: 在第 K 层之后 drop image tokens
        """
        from transformers.modeling_outputs import BaseModelOutputWithPast
        from transformers.models.llama.modeling_llama import (
            _prepare_4d_causal_attention_mask,
        )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取 inputs_embeds
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length, _ = inputs_embeds.shape
        seq_length_with_past = seq_length

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length_with_past,
                dtype=torch.long, device=device
            ).unsqueeze(0)

        # 4D causal mask
        attention_mask_4d = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # 是否应该在这次 pass 中执行 FastV 剪枝
        # 只在 prefill (seq_length > 1) 且有足够 image tokens 时剪枝
        should_prune = (
            fastv_state['enabled']
            and seq_length > 1  # prefill, 不是 decode
            and past_key_values is None  # 首次 forward, 不是续生成
            and seq_length > image_token_start + image_token_length
        )

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # 在第 K 层, 强制 output_attentions 以获取注意力权重
            layer_output_attentions = output_attentions
            if should_prune and idx == fastv_k - 1:
                layer_output_attentions = True

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=layer_output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if layer_output_attentions else 1],)

            if layer_output_attentions:
                all_self_attns += (layer_outputs[1],)

            # ========== FastV: 在第 K 层之后执行 token dropping ==========
            if should_prune and idx == fastv_k - 1:
                attn_weights = layer_outputs[1]  # (batch, heads, seq, seq)

                # 计算 image tokens 的重要性
                # 取最后一个 token 对 image tokens 的注意力，对 heads 取平均
                img_attn = attn_weights[:, :, -1, image_token_start:image_token_start + image_token_length]
                importance = img_attn.mean(dim=1)  # (batch, image_token_length)

                # 选择要保留的 top-k image tokens
                num_keep = int(image_token_length * (1 - fastv_r))
                _, top_indices = importance.topk(num_keep, dim=-1)
                top_indices = top_indices.sort(dim=-1).values  # 保持原始顺序

                # 构建完整的保留索引
                device = hidden_states.device
                prefix_idx = torch.arange(image_token_start, device=device)
                selected_img_idx = top_indices[0] + image_token_start
                suffix_start = image_token_start + image_token_length
                suffix_idx = torch.arange(suffix_start, seq_length, device=device)
                keep_indices = torch.cat([prefix_idx, selected_img_idx, suffix_idx])

                # 物理删除 hidden_states 中不重要的 tokens
                hidden_states = hidden_states[:, keep_indices, :]

                # 更新 position_ids (保持原始位置，这对 RoPE 很重要)
                position_ids = keep_indices.unsqueeze(0)

                # 更新 attention mask (重建 4D mask)
                new_seq_length = keep_indices.shape[0]
                new_attention_mask = torch.ones(
                    (batch_size, new_seq_length), dtype=torch.long, device=device
                )
                attention_mask_4d = _prepare_4d_causal_attention_mask(
                    new_attention_mask, (batch_size, new_seq_length), hidden_states, 0
                )

                # 更新 KV cache (已缓存的层也需要剪枝)
                pruned_cache = ()
                for layer_cache in next_decoder_cache:
                    k, v = layer_cache[0], layer_cache[1]
                    pruned_cache += ((k[:, :, keep_indices, :], v[:, :, keep_indices, :]),)
                next_decoder_cache = pruned_cache

                seq_length = new_seq_length
                should_prune = False  # 只剪一次

                fastv_state['pruned_this_pass'] = True
                fastv_state['num_kept'] = num_keep
                fastv_state['num_pruned'] = image_token_length - num_keep

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # 应用 monkey-patch
    LlamaModel.forward = fastv_forward

    return fastv_state


def unpatch_model(model):
    """恢复原始 forward"""
    import transformers.models.llama.modeling_llama as llama_module
    # 重新导入会恢复原始方法
    import importlib
    importlib.reload(llama_module)


def run_generate(model, tokenizer, input_ids, image_tensor, device, max_new_tokens=256):
    """用 model.generate() 生成"""
    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    torch.cuda.synchronize()
    elapsed = time.time() - start

    new_tokens = output_ids[:, input_ids.shape[1]:]
    response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return response, elapsed


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
    print("  Vanilla 推理 (无 FastV)")
    print("=" * 60)
    vanilla_resp, vanilla_time = run_generate(model, tokenizer, input_ids, image_tensor, device)
    print(f"  耗时: {vanilla_time:.3f}s")
    print(f"  输出: {vanilla_resp[:500]}")

    # ========== FastV ==========
    print("\n" + "=" * 60)
    print(f"  FastV 推理 (K={args.fastv_k}, R={args.fastv_r})")
    print("=" * 60)

    # 检测 image token 起始位置
    img_placeholder_pos = (input_ids[0] == -200).nonzero(as_tuple=True)[0]
    image_start = img_placeholder_pos[0].item() if len(img_placeholder_pos) > 0 else 35

    # Patch 模型
    fastv_state = patch_model_for_fastv(
        model,
        fastv_k=args.fastv_k,
        fastv_r=args.fastv_r,
        image_token_start=image_start,
        image_token_length=576,
    )

    fastv_resp, fastv_time = run_generate(model, tokenizer, input_ids, image_tensor, device)

    if fastv_state['pruned_this_pass']:
        print(f"  FastV: 保留 {fastv_state['num_kept']}/576, "
              f"剪掉 {fastv_state['num_pruned']} ({args.fastv_r*100:.0f}%)")
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

    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  峰值显存:      {mem_gb:.1f} GB")

    print("\nDemo 完成!")


if __name__ == '__main__':
    main()
