"""
RPA Generator — modified autoregressive generation with periodic
visual token compression.

Key design decisions:
  1. Compression happens BETWEEN generation steps (not during)
  2. After compression, a recalibration forward pass produces new logits
     consistent with the compressed KV cache
  3. First few visual tokens (attention sinks) are never evicted
  4. Warmup period ensures model establishes generation pattern first
"""

import torch
from typing import Optional, Tuple
from dataclasses import dataclass, field

from .config import RPAConfig
from .absorption import AbsorptionTracker
from .kv_compressor import compress_kv_cache
from .scheduler import CompressionScheduler


@dataclass
class RPAGenerationResult:
    response: str
    total_generated: int
    initial_visual_tokens: int
    final_visual_tokens: int
    compressions: int
    abstraction_curve: list  # [(step, n_visual), ...]
    elapsed: float = 0.0


def rpa_generate(
    model,
    tokenizer,
    inputs_embeds: torch.Tensor,
    visual_start: int,
    visual_end: int,
    config: RPAConfig,
    attention_mask: Optional[torch.Tensor] = None,
) -> RPAGenerationResult:
    """
    Generate with progressive visual abstraction.

    The generation loop periodically:
      1. Measures absorption (cross-attention from text to visual tokens)
      2. Compresses visual KV cache (evict most-absorbed tokens)
      3. Re-runs a forward pass to get logits consistent with compressed cache
    """
    import time

    device = inputs_embeds.device
    t0 = time.time()

    n_visual_initial = visual_end - visual_start
    current_visual_start = visual_start
    current_visual_end = visual_end

    # Number of visual tokens to always protect (attention sinks)
    n_protected = min(4, n_visual_initial)

    tracker = AbsorptionTracker(n_visual_initial, device)
    scheduler = CompressionScheduler(config, n_visual_initial)

    # ---- Step 1: Prefill ----
    seq_len = inputs_embeds.shape[1]
    if attention_mask is None:
        attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]

    # ---- Step 2: Autoregressive generation ----
    generated_ids = []
    eos_token_id = tokenizer.eos_token_id
    n_generated = 0
    pending_compression = False  # flag: compress before next forward

    # Attention hook setup (only used when measuring absorption)
    attn_captured = {}
    llm = model.get_model()
    target_layer = llm.layers[config.ranking_layer - 1]

    def capture_hook(module, args, output):
        if len(output) > 1 and output[1] is not None:
            attn_captured['weights'] = output[1].detach()

    for step in range(config.max_new_tokens):
        # ---- Sample next token ----
        if config.do_sample and config.temperature > 0:
            probs = torch.softmax(next_token_logits / config.temperature, dim=-1)
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        else:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

        generated_ids.append(next_token.item())
        n_generated += 1

        if next_token.item() == eos_token_id:
            break

        # ---- Check if compression should happen BEFORE next forward ----
        should_compress = scheduler.step(n_generated)

        if should_compress:
            # First, do a forward WITH attention hook to measure absorption
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
            hook.remove()

            # Update absorption scores
            if 'weights' in attn_captured:
                cur_len = attention_mask.shape[1]
                tracker.update(
                    attn_captured['weights'],
                    current_visual_start, current_visual_end,
                    cur_len - 1, cur_len,
                )

            # ---- Compress KV cache ----
            target_n = scheduler.get_target_n()
            n_current = current_visual_end - current_visual_start

            if target_n < n_current:
                scores = tracker.get_scores(config.use_cumulative)

                # Protect attention sink tokens: set their absorption to -inf
                # so they're never evicted (kept as lowest absorption = least absorbed)
                if n_protected > 0 and scores.shape[0] > n_protected:
                    scores[:n_protected] = -1e9

                past_key_values, current_visual_start, current_visual_end, keep_mask = \
                    compress_kv_cache(
                        past_key_values,
                        scores,
                        current_visual_start,
                        current_visual_end,
                        target_n,
                        method=config.method,
                    )

                tracker.compress_indices(keep_mask)
                scheduler.after_compress(target_n)

                # Rebuild attention mask to match compressed cache
                new_kv_len = past_key_values[0][0].shape[2]
                attention_mask = torch.ones((1, new_kv_len), dtype=torch.long, device=device)

            # ---- Recalibration: get fresh logits from compressed cache ----
            # Generate a "dummy" forward using the last token to recalibrate
            # the model's internal state with the compressed cache
            # We use the token we JUST generated as a recalibration input
            recalib_embeds = model.get_model().embed_tokens(next_token)
            attention_mask_recalib = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)],
                dim=1,
            )

            with torch.no_grad():
                recalib_out = model(
                    inputs_embeds=recalib_embeds,
                    attention_mask=attention_mask_recalib,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

            past_key_values = recalib_out.past_key_values
            next_token_logits = recalib_out.logits[:, -1, :]
            attention_mask = attention_mask_recalib

        else:
            # ---- Normal forward (no compression) ----
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

    elapsed = time.time() - t0
    summary = scheduler.get_summary()

    return RPAGenerationResult(
        response=response,
        total_generated=n_generated,
        initial_visual_tokens=n_visual_initial,
        final_visual_tokens=summary['final'],
        compressions=summary['compressions'],
        abstraction_curve=summary['history'],
        elapsed=elapsed,
    )
