"""
Absorption Detection — measure how much visual information has been
transferred into the reasoning chain.

After generating K reasoning tokens, we examine the cross-attention
pattern from those K tokens to all remaining visual tokens. High
cumulative attention means the visual token's information has been
"absorbed" into the text representation → safe to compress.
"""

import torch
from typing import Optional


class AbsorptionTracker:
    """
    Tracks cumulative absorption of visual tokens across reasoning steps.

    absorption_score[i] = how much token i has been attended to so far
    Higher score → more info absorbed → safer to compress
    """

    def __init__(self, n_visual: int, device: str = "cuda"):
        self.n_visual = n_visual
        self.cumulative = torch.zeros(n_visual, device=device)
        self.step_count = 0

    def update(
        self,
        attn_weights: torch.Tensor,
        visual_start: int,
        visual_end: int,
        generated_start: int,
        generated_end: int,
    ) -> torch.Tensor:
        """
        Update absorption scores with attention from recently generated tokens.

        Args:
            attn_weights: (batch, heads, seq_len, seq_len) from one layer
            visual_start: start index of visual tokens in sequence
            visual_end: end index of visual tokens
            generated_start: start of recently generated tokens
            generated_end: end of recently generated tokens

        Returns:
            current_absorption: (n_remaining_visual,) absorption this step
        """
        n_visual_current = visual_end - visual_start

        # Attention from recent generated tokens to visual tokens
        # Shape: (batch, heads, K, n_visual)
        recent_to_visual = attn_weights[:, :, generated_start:generated_end,
                                        visual_start:visual_end]

        # Average across batch, heads, and generated positions
        # → per-visual-token absorption score
        step_absorption = recent_to_visual.mean(dim=(0, 1, 2))  # (n_visual,)

        # Update cumulative (only for current visual tokens)
        # After compression, n_visual shrinks, so we track by index
        if n_visual_current == self.cumulative.shape[0]:
            self.cumulative += step_absorption
        else:
            # Visual tokens were compressed since last update
            # Reset cumulative to match new size
            self.cumulative = step_absorption.clone()

        self.step_count += 1
        return step_absorption

    def get_scores(self, use_cumulative: bool = True) -> torch.Tensor:
        """Get absorption scores for compression decisions."""
        return self.cumulative if use_cumulative else self.cumulative / max(self.step_count, 1)

    def compress_indices(self, keep_mask: torch.Tensor):
        """Update tracker after visual tokens are compressed."""
        self.cumulative = self.cumulative[keep_mask]
        self.n_visual = self.cumulative.shape[0]

    def reset(self, n_visual: int):
        self.n_visual = n_visual
        self.cumulative = torch.zeros(n_visual, device=self.cumulative.device)
        self.step_count = 0
