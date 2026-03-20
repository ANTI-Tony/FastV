"""
Attention entropy computation and dialogue-adaptive importance tracking.

Key insight: attention entropy naturally separates tokens into:
  - High entropy → universally attended → good for global summary
  - Low entropy + high attention → specifically important → region clusters
  - Low entropy + low attention → rarely important → detail

EMA importance: tracks which clusters are relevant across dialogue turns,
enabling the retriever to leverage conversational context.
"""

import torch
import torch.nn.functional as F
from typing import Optional

from .config import RevoCConfig


def compute_attention_entropy(
    attn_weights: torch.Tensor,
    image_start: int,
    num_image_tokens: int,
) -> torch.Tensor:
    """
    Compute per-token attention entropy across all heads.

    High entropy → token is broadly attended to by diverse query types.
    Low entropy → token is specifically attended to by certain queries.

    Args:
        attn_weights: (batch, heads, seq_len, seq_len) from layer K
        image_start: start position of image tokens in sequence
        num_image_tokens: number of image tokens (576)

    Returns:
        entropy: (num_image_tokens,) entropy per image token
    """
    # Extract attention FROM all positions TO image tokens
    # Shape: (batch, heads, seq_len, num_image_tokens)
    img_attn = attn_weights[:, :, :, image_start:image_start + num_image_tokens]

    # Normalize to get distribution over source positions for each image token
    # For each image token, how is attention distributed across heads and positions?
    # We compute entropy of the attention TO each image token across source positions
    # Shape: (batch, heads, num_image_tokens) — mean over source positions
    attn_to_img = attn_weights[:, :, -1, image_start:image_start + num_image_tokens]
    # (batch, heads, num_img_tokens)

    # Clamp for numerical stability
    attn_to_img = attn_to_img.clamp(min=1e-8)

    # Entropy per head: H = -sum(p * log(p)) over the heads dimension
    # Normalize across image tokens to get a distribution
    p = attn_to_img / attn_to_img.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    entropy_per_head = -(p * p.log()).sum(dim=-1)  # (batch, heads)

    # Per-token entropy: for each image token, compute entropy of its attention
    # distribution across heads
    p_heads = attn_to_img / attn_to_img.sum(dim=1, keepdim=True).clamp(min=1e-8)
    # (batch, heads, num_img_tokens)
    token_entropy = -(p_heads * p_heads.log()).sum(dim=1)  # (batch, num_img_tokens)

    return token_entropy.squeeze(0)  # (num_image_tokens,)


def compute_importance_scores(
    attn_weights: torch.Tensor,
    image_start: int,
    num_image_tokens: int,
) -> torch.Tensor:
    """
    Compute attention-based importance (same as FastV for compatibility).

    Args:
        attn_weights: (batch, heads, seq_len, seq_len)
        image_start: start position of image tokens
        num_image_tokens: count of image tokens

    Returns:
        importance: (num_image_tokens,) importance scores
    """
    img_attn = attn_weights[:, :, -1, image_start:image_start + num_image_tokens]
    importance = img_attn.mean(dim=1).squeeze(0)  # (num_image_tokens,)
    return importance


class EMAImportanceTracker:
    """
    Tracks cluster importance across dialogue turns via exponential moving average.

    After each turn, generation attention reveals which clusters were actually
    useful. This information is accumulated across turns to bias future retrieval
    toward historically important regions.
    """

    def __init__(self, n_clusters: int, decay: float = 0.7, device: str = "cuda"):
        self.n_clusters = n_clusters
        self.decay = decay
        self.importance = torch.zeros(n_clusters, device=device)
        self.turn_count = 0

    def update(self, cluster_scores: torch.Tensor):
        """
        Update importance with scores from the current turn.

        Args:
            cluster_scores: (n_clusters,) relevance scores from current turn
        """
        cluster_scores = cluster_scores.to(self.importance.device)
        if self.turn_count == 0:
            self.importance = cluster_scores
        else:
            self.importance = (
                self.decay * self.importance +
                (1 - self.decay) * cluster_scores
            )
        self.turn_count += 1

    def get_bias(self) -> torch.Tensor:
        """Get normalized importance bias for retrieval blending."""
        if self.turn_count == 0:
            return torch.zeros(self.n_clusters, device=self.importance.device)
        return F.softmax(self.importance, dim=0)

    def reset(self):
        self.importance.zero_()
        self.turn_count = 0
