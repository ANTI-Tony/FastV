"""
FastV 核心实现：在 LLM forward pass 中对 visual tokens 进行动态剪枝

原理：
1. 前 K 层正常处理所有 tokens
2. 在第 K 层，根据注意力分数对 image tokens 排序
3. 丢弃注意力最低的 R% image tokens
4. 后续层只处理保留的 tokens
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .fastv_config import FastVConfig


def compute_image_token_importance(
    attention_weights: torch.Tensor,
    image_start: int,
    image_length: int,
) -> torch.Tensor:
    """
    计算 image tokens 的重要性分数

    Args:
        attention_weights: shape (batch, num_heads, seq_len, seq_len)
        image_start: image tokens 的起始索引
        image_length: image tokens 的数量

    Returns:
        importance: shape (batch, image_length) - 每个 image token 的重要性分数
    """
    # 取最后一个 token (当前生成位置) 对所有 token 的注意力
    # shape: (batch, num_heads, seq_len)
    last_token_attn = attention_weights[:, :, -1, :]

    # 提取对 image tokens 的注意力, shape: (batch, num_heads, image_length)
    image_attn = last_token_attn[:, :, image_start:image_start + image_length]

    # 对所有 attention heads 取平均, shape: (batch, image_length)
    importance = image_attn.mean(dim=1)

    return importance


def select_important_tokens(
    importance: torch.Tensor,
    num_keep: int,
) -> torch.Tensor:
    """
    根据重要性选择要保留的 token 索引

    Args:
        importance: shape (batch, image_length)
        num_keep: 要保留的 token 数量

    Returns:
        keep_indices: shape (batch, num_keep) - 保留的 token 在 image 范围内的索引
    """
    _, indices = torch.topk(importance, k=num_keep, dim=-1, sorted=True)
    # 排序使得保留的 tokens 按原始顺序排列
    keep_indices, _ = torch.sort(indices, dim=-1)
    return keep_indices


def build_pruned_indices(
    seq_len: int,
    image_start: int,
    image_length: int,
    keep_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    构建剪枝后的完整序列索引

    保留: [system tokens] + [selected image tokens] + [text tokens]
    """
    batch_size = keep_indices.shape[0]

    # system/prefix tokens: [0, image_start)
    prefix_indices = torch.arange(image_start, device=device).unsqueeze(0).expand(batch_size, -1)

    # 被选中的 image tokens (加上偏移)
    selected_image_indices = keep_indices + image_start

    # text/suffix tokens: [image_start + image_length, seq_len)
    suffix_start = image_start + image_length
    suffix_indices = torch.arange(suffix_start, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # 拼接
    all_indices = torch.cat([prefix_indices, selected_image_indices, suffix_indices], dim=1)
    return all_indices


class FastVWrapper(nn.Module):
    """
    FastV 包装器，包装 LLaVA 模型实现 token 剪枝

    使用方式:
        model = LlavaForConditionalGeneration.from_pretrained(...)
        fastv_model = FastVWrapper(model, FastVConfig(fastv_k=2, fastv_r=0.75))
        output = fastv_model.generate(...)
    """

    def __init__(self, model, config: FastVConfig):
        super().__init__()
        self.model = model
        self.config = config
        self._attention_cache = {}
        self._pruned = False
        self._hooks = []
        self._setup_hooks()

    def _setup_hooks(self):
        """注册 attention hook 来捕获第 K 层的注意力权重"""
        self._remove_hooks()

        # 获取 LLM 的 transformer layers
        llm = self._get_llm()
        target_layer = llm.layers[self.config.fastv_k - 1]

        def attention_hook(module, args, output):
            # output 通常是 (hidden_states, attention_weights, past_key_value)
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self._attention_cache['layer_k_attn'] = output[1].detach()

        hook = target_layer.self_attn.register_forward_hook(attention_hook)
        self._hooks.append(hook)

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _get_llm(self):
        """获取底层 LLM 模型"""
        if hasattr(self.model, 'language_model'):
            # HuggingFace LLaVA
            return self.model.language_model.model
        elif hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'layers'):
                return self.model.model
            elif hasattr(self.model.model, 'model'):
                return self.model.model.model
        raise ValueError("无法找到 LLM layers，请检查模型结构")

    def _detect_image_token_range(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        """自动检测 image token 在序列中的位置"""
        if self.config.image_token_start_index is not None:
            return self.config.image_token_start_index, self.config.image_token_length

        # LLaVA 使用特殊 token (通常是 -200 或 32000) 标记图像位置
        # 在 prepare_inputs 后，image tokens 会替换这些特殊 token
        # 默认假设 image tokens 在 system prompt 之后
        # 对于 LLaVA-1.5: 通常 image_start ≈ 35, image_length = 576
        return 35, self.config.image_token_length

    def forward(self, **kwargs):
        """Forward pass with FastV token pruning"""
        return self.model(**kwargs)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[List] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        带 FastV 剪枝的生成

        核心流程:
        1. Prefill 阶段: 正常跑前 K 层，捕获注意力
        2. 根据注意力分数选择重要 tokens
        3. 剪枝不重要的 image tokens
        4. 用剪枝后的 KV cache 继续生成
        """
        if not self.config.use_fastv:
            return self.model.generate(
                input_ids=input_ids,
                images=images,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                **kwargs,
            )

        # 确保输出注意力权重（第 K 层的 hook 需要）
        kwargs['output_attentions'] = True

        # Step 1: 完整 prefill (所有层)，同时捕获第 K 层注意力
        outputs = self.model(
            input_ids=input_ids,
            images=images,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

        # Step 2: 从第 K 层的注意力中计算 image token 重要性
        if 'layer_k_attn' not in self._attention_cache:
            print("警告: 未捕获到注意力权重，回退到完整生成")
            return self.model.generate(
                input_ids=input_ids,
                images=images,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                **kwargs,
            )

        attn_weights = self._attention_cache['layer_k_attn']
        image_start, image_length = self._detect_image_token_range(input_ids)
        seq_len = outputs.logits.shape[1]

        importance = compute_image_token_importance(attn_weights, image_start, image_length)
        num_keep = self.config.num_tokens_to_keep
        keep_indices = select_important_tokens(importance, num_keep)

        # Step 3: 构建剪枝后的索引
        pruned_indices = build_pruned_indices(
            seq_len, image_start, image_length, keep_indices, input_ids.device
        )

        # Step 4: 剪枝 KV cache
        past_key_values = outputs.past_key_values
        pruned_past = []
        for layer_kv in past_key_values:
            k, v = layer_kv[0], layer_kv[1]
            # k, v shape: (batch, num_heads, seq_len, head_dim)
            pruned_k = torch.index_select(k, 2, pruned_indices[0])
            pruned_v = torch.index_select(v, 2, pruned_indices[0])
            pruned_past.append((pruned_k, pruned_v))
        pruned_past = tuple(pruned_past)

        # Step 5: 用剪枝后的 KV cache 继续自回归生成
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated_ids = [next_token]
        new_attention_mask = torch.ones(
            (1, pruned_indices.shape[1] + 1), dtype=torch.long, device=input_ids.device
        )

        max_new_tokens = kwargs.get('max_new_tokens', 256)
        eos_token_id = kwargs.get('eos_token_id',
                                  getattr(self.model.config, 'eos_token_id', 2))

        for _ in range(max_new_tokens - 1):
            out = self.model.language_model(
                input_ids=next_token,
                attention_mask=new_attention_mask,
                past_key_values=pruned_past,
                use_cache=True,
                return_dict=True,
            )
            pruned_past = out.past_key_values
            next_token_logits = out.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if next_token.item() == eos_token_id:
                break

            generated_ids.append(next_token)
            new_attention_mask = torch.cat([
                new_attention_mask,
                torch.ones((1, 1), dtype=torch.long, device=input_ids.device)
            ], dim=1)

        generated_ids = torch.cat(generated_ids, dim=1)

        # 清理缓存
        self._attention_cache.clear()

        return generated_ids


def apply_fastv_to_model(model, config: FastVConfig):
    """
    便捷函数：给模型加上 FastV

    用法:
        model = LlavaForConditionalGeneration.from_pretrained(...)
        model = apply_fastv_to_model(model, FastVConfig(fastv_k=2, fastv_r=0.75))
    """
    return FastVWrapper(model, config)


def fastv_forward_hook(config: FastVConfig):
    """
    返回一个可以直接注册到 LlamaModel 的 forward hook

    这是一种更轻量的集成方式，直接修改 hidden states 而不是包装整个模型。
    适合集成到已有的训练/评测 pipeline 中。
    """
    pruned = {'done': False, 'indices': None}

    def hook(module, args, output):
        """
        在第 K 层后执行，直接修改 hidden states

        注意: 此 hook 需要注册到第 K 层的 decoder layer 上
        """
        if pruned['done']:
            return output

        hidden_states = output[0]
        attn_weights = output[1] if len(output) > 1 else None

        if attn_weights is None:
            return output

        batch_size, seq_len, hidden_dim = hidden_states.shape
        image_start = config.image_token_start_index or 35
        image_length = config.image_token_length

        if seq_len <= image_start + image_length:
            return output

        # 计算重要性
        importance = compute_image_token_importance(attn_weights, image_start, image_length)
        num_keep = config.num_tokens_to_keep
        keep_indices = select_important_tokens(importance, num_keep)

        # 构建剪枝索引
        indices = build_pruned_indices(seq_len, image_start, image_length, keep_indices, hidden_states.device)
        pruned['indices'] = indices
        pruned['done'] = True

        # 剪枝 hidden states
        new_hidden = torch.index_select(hidden_states, 1, indices[0])

        return (new_hidden,) + output[1:]

    return hook
