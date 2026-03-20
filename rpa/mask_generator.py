"""
RPA Mask Generator — progressive visual token masking during generation.

Instead of evicting KV cache entries (which breaks model stability),
we progressively MASK visual tokens in the attention_mask. This:
  1. Keeps KV cache structure intact → model is stable → EOS works
  2. Effectively reduces visual attention as generation progresses
  3. Forces the model to rely on its own reasoning chain, not visual tokens
  4. May IMPROVE reasoning quality by reducing visual distraction

Two possible paper angles:
  A. "Reducing visual distraction improves VLM reasoning" (quality)
  B. "Progressive visual masking saves compute" (efficiency, with sparse attn)
"""

import torch
from typing import Optional
from dataclasses import dataclass

from .config import RPAConfig


@dataclass
class MaskGenerationResult:
    response: str
    total_generated: int
    initial_visual_tokens: int
    final_effective_tokens: int  # visual tokens still unmasked at end
    mask_steps: int              # number of masking operations
    mask_curve: list             # [(step, n_effective), ...]
    elapsed: float = 0.0


def rpa_masked_generate(
    model,
    tokenizer,
    inputs_embeds: torch.Tensor,
    visual_start: int,
    visual_end: int,
    config: RPAConfig,
    attention_mask: Optional[torch.Tensor] = None,
) -> MaskGenerationResult:
    """
    Generate with progressive visual token masking.

    Every check_interval tokens, we:
      1. Look at attention from recent generated tokens to visual tokens
      2. Mask the most-absorbed visual tokens (set attention_mask to 0)
      3. Continue generating with reduced visual attention

    The KV cache is NEVER modified — only the attention mask changes.
    This makes the method completely stable.
    """
    import time

    device = inputs_embeds.device
    t0 = time.time()

    n_visual = visual_end - visual_start
    n_protected = min(4, n_visual)  # attention sink tokens

    # Track which visual tokens are still "active" (not masked)
    visual_active = torch.ones(n_visual, dtype=torch.bool, device=device)
    n_active = n_visual
    mask_curve = [(0, n_visual)]
    mask_steps = 0

    # ---- Prefill ----
    seq_len = inputs_embeds.shape[1]
    if attention_mask is None:
        attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

    # Set up attention hook for absorption measurement
    attn_captured = {}
    llm = model.get_model()
    target_layer = llm.layers[config.ranking_layer - 1]

    def capture_hook(module, args, output):
        if len(output) > 1 and output[1] is not None:
            attn_captured['weights'] = output[1].detach()

    # Prefill (no hook needed — just build KV cache)
    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]

    # Cumulative absorption tracker
    cumulative_absorption = torch.zeros(n_visual, device=device)

    # ---- Autoregressive generation ----
    generated_ids = []
    eos_token_id = tokenizer.eos_token_id
    n_generated = 0

    for step in range(config.max_new_tokens):
        # Sample next token
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token.item())
        n_generated += 1

        if next_token.item() == eos_token_id:
            break

        # Check if we should mask more visual tokens
        should_mask = (
            n_generated >= config.warmup_tokens
            and n_generated % config.check_interval == 0
            and n_active > config.min_visual_tokens
        )

        if should_mask:
            # Forward WITH attention hook to measure absorption
            hook = target_layer.register_forward_hook(capture_hook)

            next_embeds = model.get_model().embed_tokens(next_token)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)],
                dim=1,
            )

            with torch.no_grad():
                step_out = model(
                    inputs_embeds=next_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_attentions=True,
                    use_cache=True,
                    return_dict=True,
                )

            past_key_values = step_out.past_key_values
            next_token_logits = step_out.logits[:, -1, :]
            hook.remove()

            # Measure absorption: attention from this token to visual tokens
            if 'weights' in attn_captured:
                cur_len = attention_mask.shape[1]
                attn_w = attn_captured['weights']
                # (batch, heads, 1, cur_len) → attention to visual positions
                vis_attn = attn_w[:, :, 0, visual_start:visual_end]  # (B, H, N_vis)
                step_absorption = vis_attn.mean(dim=(0, 1))  # (N_vis,)
                cumulative_absorption += step_absorption

            # Determine how many to keep
            target_n = max(int(n_active * config.compress_ratio), config.min_visual_tokens)
            n_to_mask = n_active - target_n

            if n_to_mask > 0:
                # Get absorption scores for currently active tokens
                active_indices = visual_active.nonzero(as_tuple=True)[0]
                active_scores = cumulative_absorption[active_indices]

                # Protect first few tokens (attention sinks)
                # by setting their scores to -inf (never masked)
                for i, idx in enumerate(active_indices):
                    if idx < n_protected:
                        active_scores[i] = -1e9

                # Mask tokens with HIGHEST absorption (info already in text)
                _, mask_order = active_scores.topk(n_to_mask, largest=True)
                tokens_to_mask = active_indices[mask_order]

                # Update attention mask: set masked visual positions to 0
                for idx in tokens_to_mask:
                    abs_pos = visual_start + idx.item()
                    attention_mask[0, abs_pos] = 0
                    visual_active[idx] = False

                n_active = visual_active.sum().item()
                mask_steps += 1
                mask_curve.append((n_generated, n_active))

        else:
            # Normal forward (no masking)
            next_embeds = model.get_model().embed_tokens(next_token)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)],
                dim=1,
            )

            with torch.no_grad():
                step_out = model(
                    inputs_embeds=next_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

            past_key_values = step_out.past_key_values
            next_token_logits = step_out.logits[:, -1, :]

    # Decode
    output_ids = torch.tensor([generated_ids], device=device)
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return MaskGenerationResult(
        response=response,
        total_generated=n_generated,
        initial_visual_tokens=n_visual,
        final_effective_tokens=n_active,
        mask_steps=mask_steps,
        mask_curve=mask_curve,
        elapsed=time.time() - t0,
    )
