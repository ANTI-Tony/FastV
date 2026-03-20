"""
VisualTokenCache — three-level visual token cache.

L1 (Global):  Spatial average pooling — reshape (576, D) to (24, 24, D),
              pool into (grid_h, grid_w, D) super-pixels, flatten to (32, D).
L2 (Region):  Top-128 image tokens ranked by attention importance.
L3 (Detail):  Remaining 448 image tokens.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional

from .config import MTVCConfig


@dataclass
class VisualTokenCache:
    """Stores the three cache levels for a single image."""
    l1: Optional[torch.Tensor] = None   # (l1_size, D)
    l2: Optional[torch.Tensor] = None   # (l2_size, D)
    l3: Optional[torch.Tensor] = None   # (l3_size, D)
    l2_indices: Optional[torch.Tensor] = None  # original positions in 576
    l3_indices: Optional[torch.Tensor] = None  # original positions in 576
    dim: int = 0

    @property
    def is_built(self) -> bool:
        return self.l1 is not None

    @classmethod
    def build(
        cls,
        image_features: torch.Tensor,
        importance_scores: torch.Tensor,
        config: MTVCConfig,
    ) -> "VisualTokenCache":
        """
        Build the 3-level cache from image features and importance scores.

        Args:
            image_features: (N, D) where N = image_token_length (576)
            importance_scores: (N,) attention-based importance per token
            config: MTVCConfig
        Returns:
            VisualTokenCache with L1, L2, L3 populated
        """
        config.validate()
        N, D = image_features.shape
        assert N == config.image_token_length, f"Expected {config.image_token_length} tokens, got {N}"

        # ---- L1: spatial average pooling ----
        l1 = _build_l1(image_features, config)

        # ---- L2 / L3: attention-ranked partition ----
        l2, l3, l2_idx, l3_idx = _build_l2_l3(image_features, importance_scores, config)

        return cls(l1=l1, l2=l2, l3=l3, l2_indices=l2_idx, l3_indices=l3_idx, dim=D)


def _build_l1(image_features: torch.Tensor, config: MTVCConfig) -> torch.Tensor:
    """
    Spatial average pooling: (576, D) -> (24, 24, D) -> (4, 8, D) -> (32, D)

    The 24x24 grid is divided into super-pixels of size (6, 3):
      - 24 / 4 = 6 rows per super-pixel
      - 24 / 8 = 3 cols per super-pixel
    """
    N, D = image_features.shape
    gh, gw = config.image_grid_h, config.image_grid_w  # 24, 24

    # reshape to spatial grid
    grid = image_features.view(gh, gw, D)  # (24, 24, D)

    # compute super-pixel sizes
    sp_h = gh // config.grid_h  # 24 // 4 = 6
    sp_w = gw // config.grid_w  # 24 // 8 = 3

    # reshape into super-pixel blocks and average
    # (24,24,D) -> (4, 6, 8, 3, D)
    grid = grid.view(config.grid_h, sp_h, config.grid_w, sp_w, D)
    # average over the spatial dims within each super-pixel
    l1 = grid.mean(dim=(1, 3))  # (4, 8, D)
    l1 = l1.reshape(config.l1_size, D)  # (32, D)
    return l1


def _build_l2_l3(
    image_features: torch.Tensor,
    importance_scores: torch.Tensor,
    config: MTVCConfig,
):
    """
    Partition tokens into L2 (top-128) and L3 (remaining 448) by importance.
    """
    N = image_features.shape[0]

    # sort by importance descending
    sorted_indices = torch.argsort(importance_scores, descending=True)

    l2_idx = sorted_indices[: config.l2_size]              # top-128
    l3_idx = sorted_indices[config.l2_size :]              # remaining 448

    # sort each group by original position to preserve spatial locality
    l2_idx, _ = torch.sort(l2_idx)
    l3_idx, _ = torch.sort(l3_idx)

    l2 = image_features[l2_idx]  # (128, D)
    l3 = image_features[l3_idx]  # (448, D)

    return l2, l3, l2_idx, l3_idx
