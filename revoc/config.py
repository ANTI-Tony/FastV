"""
ReVoC (Recoverable Visual Compression) configuration.

Paradigm shift: Compress, Don't Prune.
  - Existing methods (FastV, SparseVLM) irreversibly prune visual tokens.
  - ReVoC compresses tokens into cluster centers + residuals, enabling
    on-demand recovery of any token at any conversation turn.

Architecture:
  1. Global Summary (S_global): entropy-weighted spatial pooling
  2. Region Clusters (S_region): k-means clustering of visual tokens
  3. Residual Store (Δ): per-token residuals for exact recovery
  4. Cross-Attention Retriever: learned query→cluster selection
  5. Dialogue-Adaptive Importance: EMA-updated cluster importance
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RevoCConfig:
    # ---- compression ----
    n_global: int = 32               # global summary tokens
    n_clusters: int = 64             # number of region clusters
    image_token_length: int = 576    # total visual tokens (LLaVA-1.5)
    hidden_dim: int = 4096           # model hidden dimension

    # ---- spatial grid for global summary ----
    image_grid_h: int = 24
    image_grid_w: int = 24

    # ---- entropy-based partitioning ----
    ranking_layer: int = 2           # layer for attention/entropy computation
    entropy_global_percentile: float = 90.0   # top X% entropy → global pool

    # ---- retrieval ----
    n_retrieve_clusters: int = 12    # clusters to unpack per query
    retriever_type: str = "cross_attention"  # "cross_attention" | "cosine"

    # ---- cross-attention retriever ----
    retriever_heads: int = 8
    retriever_dropout: float = 0.0

    # ---- dialogue-adaptive importance (EMA) ----
    ema_decay: float = 0.7           # importance decay factor across turns
    history_weight: float = 0.3      # blend: α*query_sim + (1-α)*history

    # ---- adaptive recovery ----
    adaptive_recovery: bool = True   # enable probe-then-recover
    confidence_threshold: float = 1.5  # output entropy threshold for recovery trigger
    # Below threshold → confident → use centers only (~96 tokens)
    # Above threshold → uncertain → recover full clusters (~140 tokens)

    # ---- residual store ----
    quantize_residuals: bool = False  # FP16 residuals by default
    residual_device: str = "same"    # "same" | "cpu" (offload to save GPU)

    # ---- generation ----
    max_new_tokens: int = 256

    # ---- distillation training ----
    distill_lr: float = 1e-4
    distill_epochs: int = 3
    distill_batch_size: int = 4

    @property
    def tokens_per_retrieval(self) -> int:
        """Approximate tokens used per round 2+ (full recovery mode)."""
        avg_cluster_size = self.image_token_length // self.n_clusters  # ~9
        return self.n_global + self.n_retrieve_clusters * avg_cluster_size

    @property
    def tokens_centers_only(self) -> int:
        """Tokens used in center-only probe mode (no recovery)."""
        return self.n_global + self.n_clusters

    def validate(self):
        assert self.n_clusters >= self.n_retrieve_clusters, (
            f"n_retrieve_clusters ({self.n_retrieve_clusters}) > "
            f"n_clusters ({self.n_clusters})"
        )
        assert self.ranking_layer >= 1
        assert 0 < self.ema_decay < 1
        assert 0 <= self.history_weight <= 1
        assert self.retriever_type in ("cross_attention", "cosine")
