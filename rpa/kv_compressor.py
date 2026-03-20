"""
KV Cache Compressor — directly manipulate past_key_values during generation.

Three compression strategies:
  1. Evict: remove KV entries for most-absorbed visual tokens
  2. Merge: average adjacent similar visual token KV entries
  3. Hybrid: merge first, then evict remaining

All operations modify past_key_values in-place to reduce the
visual token portion of the KV cache.
"""

import torch
from typing import Tuple, List, Optional

from .config import RPAConfig


def compress_kv_cache(
    past_key_values: Tuple,
    absorption_scores: torch.Tensor,
    visual_start: int,
    visual_end: int,
    target_n: int,
    method: str = "evict",
) -> Tuple[Tuple, int, int, torch.Tensor]:
    """
    Compress visual token KV entries in the cache.

    Args:
        past_key_values: tuple of (key, value) per layer
            each shape: (batch, heads, seq_len, head_dim)
        absorption_scores: (n_visual,) higher = more absorbed = compress first
        visual_start: start index of visual tokens in sequence
        visual_end: end index of visual tokens
        target_n: target number of visual tokens after compression
        method: "evict" | "merge" | "hybrid"

    Returns:
        new_past_key_values: compressed cache
        new_visual_start: unchanged (same position)
        new_visual_end: visual_start + target_n
        keep_mask: (n_visual,) bool mask of kept tokens
    """
    n_visual = visual_end - visual_start

    if target_n >= n_visual:
        keep_mask = torch.ones(n_visual, dtype=torch.bool,
                               device=absorption_scores.device)
        return past_key_values, visual_start, visual_end, keep_mask

    if method == "evict":
        return _evict(past_key_values, absorption_scores,
                      visual_start, visual_end, target_n)
    elif method == "merge":
        return _merge(past_key_values, absorption_scores,
                      visual_start, visual_end, target_n)
    elif method == "hybrid":
        # First merge to reduce by half the gap, then evict the rest
        mid_target = (n_visual + target_n) // 2
        pv, vs, ve, mask1 = _merge(past_key_values, absorption_scores,
                                    visual_start, visual_end, mid_target)
        remaining_scores = absorption_scores[mask1]
        pv, vs, ve, mask2 = _evict(pv, remaining_scores, vs, ve, target_n)
        # Combine masks
        full_mask = mask1.clone()
        kept_indices = mask1.nonzero(as_tuple=True)[0]
        for i, idx in enumerate(kept_indices):
            if i < mask2.shape[0]:
                full_mask[idx] = mask2[i]
        return pv, vs, ve, full_mask
    else:
        raise ValueError(f"Unknown method: {method}")


def _evict(
    past_key_values: Tuple,
    absorption_scores: torch.Tensor,
    visual_start: int,
    visual_end: int,
    target_n: int,
) -> Tuple[Tuple, int, int, torch.Tensor]:
    """
    Evict the most-absorbed visual tokens (they're already captured in text).
    Keep the LEAST absorbed ones (still needed for future reasoning).
    """
    n_visual = visual_end - visual_start

    # Keep tokens with LOWEST absorption (most info still in visual only)
    _, keep_indices = absorption_scores.topk(target_n, largest=False)
    keep_indices = keep_indices.sort().values

    # Build full sequence keep mask
    keep_mask_visual = torch.zeros(n_visual, dtype=torch.bool,
                                   device=absorption_scores.device)
    keep_mask_visual[keep_indices] = True

    # Apply to each layer's KV cache
    new_past = []
    for key, value in past_key_values:
        seq_len = key.shape[2]

        # Build full sequence indices to keep
        prefix_indices = torch.arange(visual_start, device=key.device)
        visual_keep = visual_start + keep_indices.to(key.device)
        suffix_indices = torch.arange(visual_end, seq_len, device=key.device)
        all_keep = torch.cat([prefix_indices, visual_keep, suffix_indices])

        new_key = key[:, :, all_keep, :]
        new_value = value[:, :, all_keep, :]
        new_past.append((new_key, new_value))

    new_visual_end = visual_start + target_n
    return tuple(new_past), visual_start, new_visual_end, keep_mask_visual


def _merge(
    past_key_values: Tuple,
    absorption_scores: torch.Tensor,
    visual_start: int,
    visual_end: int,
    target_n: int,
) -> Tuple[Tuple, int, int, torch.Tensor]:
    """
    Merge pairs of most-absorbed adjacent visual tokens.
    Merged KV entries are averaged.
    """
    n_visual = visual_end - visual_start
    n_to_merge = n_visual - target_n

    if n_to_merge <= 0:
        keep_mask = torch.ones(n_visual, dtype=torch.bool,
                               device=absorption_scores.device)
        return past_key_values, visual_start, visual_end, keep_mask

    # Find pairs to merge: choose positions with highest sum of adjacent absorption
    pair_scores = absorption_scores[:-1] + absorption_scores[1:]
    _, merge_positions = pair_scores.topk(min(n_to_merge, pair_scores.shape[0]))
    merge_positions = merge_positions.sort().values

    # Track which indices survive (not the second element of merged pairs)
    merged_away = set()
    merge_pairs = []
    for pos in merge_positions.tolist():
        if pos not in merged_away and (pos + 1) not in merged_away:
            merge_pairs.append((pos, pos + 1))
            merged_away.add(pos + 1)  # second element gets merged into first
        if len(merge_pairs) >= n_to_merge:
            break

    # Build new KV cache with merged entries
    keep_mask = torch.ones(n_visual, dtype=torch.bool,
                           device=absorption_scores.device)
    for _, second in merge_pairs:
        keep_mask[second] = False

    new_past = []
    for key, value in past_key_values:
        seq_len = key.shape[2]

        # Average merged pairs in the visual region
        new_key = key.clone()
        new_value = value.clone()
        for first, second in merge_pairs:
            abs_first = visual_start + first
            abs_second = visual_start + second
            new_key[:, :, abs_first, :] = (key[:, :, abs_first, :] + key[:, :, abs_second, :]) / 2
            new_value[:, :, abs_first, :] = (value[:, :, abs_first, :] + value[:, :, abs_second, :]) / 2

        # Remove merged-away indices
        remove_set = {visual_start + s for _, s in merge_pairs}
        keep_seq = [i for i in range(seq_len) if i not in remove_set]
        keep_seq_t = torch.tensor(keep_seq, device=key.device)

        new_past.append((new_key[:, :, keep_seq_t, :],
                         new_value[:, :, keep_seq_t, :]))

    new_visual_end = visual_start + n_visual - len(merge_pairs)
    return tuple(new_past), visual_start, new_visual_end, keep_mask
