"""
MTVC (Multi-Granularity Visual Token Cache) configuration.

Three-level cache hierarchy:
  L1 (Global)  — 32 tokens  — spatial average pooling of 24x24 grid
  L2 (Region)  — 128 tokens — top-128 by attention importance
  L3 (Detail)  — 448 tokens — remaining tokens

Query-guided retrieval selects ~150 tokens per round:
  L1 all (32) + TopK(L2, 64) + TopK(L3, 54) = 150
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MTVCConfig:
    # ---- cache sizes ----
    l1_size: int = 32
    l2_size: int = 128
    l3_size: int = 448       # 576 - 32(virtual) - 128 = 448 actual remaining

    # ---- spatial pooling grid for L1 ----
    grid_h: int = 4          # 24 / 6 = 4 super-pixels vertically
    grid_w: int = 8          # 24 / 3 = 8 super-pixels horizontally
    image_grid_h: int = 24   # LLaVA-1.5 image feature grid height
    image_grid_w: int = 24   # LLaVA-1.5 image feature grid width

    # ---- retrieval defaults ----
    default_k2: int = 64     # top-K from L2 per query
    default_k3: int = 54     # top-K from L3 per query

    # ---- attention scoring ----
    ranking_layer: int = 2   # which layer's attention to use (1-indexed)
    image_token_length: int = 576  # total image tokens in LLaVA-1.5

    # ---- generation ----
    max_new_tokens: int = 256

    @property
    def total_retrieved(self) -> int:
        """Tokens retrieved per round 2+."""
        return self.l1_size + self.default_k2 + self.default_k3

    def validate(self):
        assert self.l2_size + self.l3_size == self.image_token_length, (
            f"L2 ({self.l2_size}) + L3 ({self.l3_size}) must equal "
            f"image_token_length ({self.image_token_length})"
        )
        assert self.grid_h * self.grid_w == self.l1_size, (
            f"grid_h*grid_w ({self.grid_h}*{self.grid_w}) must equal l1_size ({self.l1_size})"
        )
        assert self.default_k2 <= self.l2_size
        assert self.default_k3 <= self.l3_size
        assert self.ranking_layer >= 1
