"""
RPA Generator — modified autoregressive generation with periodic
visual token compression.

Replaces the standard model.generate() with a manual token-by-token
loop that periodically:
  1. Captures attention from recent tokens to visual tokens
  2. Measures absorption (how much visual info is in the reasoning chain)
  3. Compresses visual KV cache entries (evict/merge)

This enables longer reasoning chains within the same compute budget.
"""

import torch
from typing import Optional, Tuple, List
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

    Instead of model.generate(), we do manual token-by-token generation
    with periodic KV cache compression.

    Args:
        model: LLaVA model (or any causal LM)
        tokenizer: tokenizer
        inputs_embeds: (1, seq_len, D) full embeddings with visual tokens
        visual_start: start index of visual tokens
        visual_end: end index of visual tokens
        config: RPAConfig
        attention_mask: optional (1, seq_len)

    Returns:
        RPAGenerationResult with response and abstraction stats
    """
    import time
    from transformers import LlamaForCausalLM

    device = inputs_embeds.device
    t0 = time.time()

    n_visual_initial = visual_end - visual_start
    current_visual_start = visual_start
    current_visual_end = visual_end

    # Initialize tracker and scheduler
    tracker = AbsorptionTracker(n_visual_initial, device)
    scheduler = CompressionScheduler(config, n_visual_initial)

    # ---- Step 1: Prefill ----
    # Run full forward pass with inputs_embeds to build initial KV cache
    seq_len = inputs_embeds.shape[1]
    if attention_mask is None:
        attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

    # Set up attention capture on ranking layer
    attn_captured = {}
    llm = model.get_model()
    target_layer = llm.layers[config.ranking_layer - 1]

    def capture_hook(module, args, output):
        if len(output) > 1 and output[1] is not None:
            attn_captured['weights'] = output[1].detach()

    hook = target_layer.register_forward_hook(capture_hook)

    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )

    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]

    # Initial absorption from prefill
    if 'weights' in attn_captured:
        tracker.update(
            attn_captured['weights'],
            current_visual_start, current_visual_end,
            seq_len - 1, seq_len,  # last prefill token
        )

    hook.remove()

    # ---- Step 2: Autoregressive generation with periodic compression ----
    generated_ids = []
    eos_token_id = tokenizer.eos_token_id
    n_generated = 0
    last_compress_step = 0

    for step in range(config.max_new_tokens):
        # Sample next token (greedy)
        if config.do_sample and config.temperature > 0:
            probs = torch.softmax(next_token_logits / config.temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

        generated_ids.append(next_token.item())
        n_generated += 1

        # Check EOS
        if next_token.item() == eos_token_id:
            break

        # ---- Periodic compression check ----
        if scheduler.step(n_generated):
            # We need attention for this step — do a forward with hook
            hook = target_layer.register_forward_hook(capture_hook)

            next_embeds = model.get_model().embed_tokens(next_token)
            new_mask = torch.ones((1, 1), dtype=torch.long, device=device)
            combined_mask = torch.cat([attention_mask, new_mask], dim=1)

            with torch.no_grad():
                step_outputs = model(
                    inputs_embeds=next_embeds,
                    attention_mask=combined_mask,
                    past_key_values=past_key_values,
                    output_attentions=True,
                    use_cache=True,
                    return_dict=True,
                )

            past_key_values = step_outputs.past_key_values
            next_token_logits = step_outputs.logits[:, -1, :]
            attention_mask = combined_mask

            hook.remove()

            # Update absorption with attention from recent tokens
            if 'weights' in attn_captured:
                current_seq_len = attention_mask.shape[1]
                tracker.update(
                    attn_captured['weights'],
                    current_visual_start, current_visual_end,
                    current_seq_len - 1, current_seq_len,
                )

            # Compress visual KV cache
            target_n = scheduler.get_target_n()
            n_current = current_visual_end - current_visual_start

            if target_n < n_current:
                scores = tracker.get_scores(config.use_cumulative)

                past_key_values, current_visual_start, current_visual_end, keep_mask = \
                    compress_kv_cache(
                        past_key_values,
                        scores,
                        current_visual_start,
                        current_visual_end,
                        target_n,
                        method=config.method,
                    )

                # Update tracker
                tracker.compress_indices(keep_mask)
                scheduler.after_compress(target_n)

                # Update attention mask (remove compressed positions)
                n_removed = n_current - target_n
                # Rebuild attention mask to match new KV cache length
                new_kv_len = past_key_values[0][0].shape[2]
                attention_mask = torch.ones((1, new_kv_len), dtype=torch.long, device=device)

            continue  # Already did forward pass with hook

        # ---- Normal forward (no compression needed) ----
        next_embeds = model.get_model().embed_tokens(next_token)
        new_mask = torch.ones((1, 1), dtype=torch.long, device=device)
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)

        with torch.no_grad():
            step_outputs = model(
                inputs_embeds=next_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        past_key_values = step_outputs.past_key_values
        next_token_logits = step_outputs.logits[:, -1, :]

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
