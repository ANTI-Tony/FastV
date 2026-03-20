"""
Visual Token Compressor — entropy-guided clustering with residual storage.

Compression pipeline:
  1. Compute attention entropy per image token
  2. High-entropy tokens → Global Summary (entropy-weighted spatial pooling)
  3. Remaining tokens → K-means clustering → Region Clusters
  4. Per-token residuals stored for on-demand exact recovery

This replaces the fixed L1/L2/L3 heuristic of MTVC with a principled,
entropy-driven approach.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from .config import RevoCConfig
from .importance import compute_attention_entropy, compute_importance_scores
from .residual_store import ResidualStore


@dataclass
class CompressedCache:
    """Output of the compression pipeline — the recoverable cache."""
    global_tokens: torch.Tensor          # (n_global, D) — scene-level summary
    cluster_centers: torch.Tensor        # (K, D) — region cluster centers
    residual_store: ResidualStore         # per-token residuals for recovery
    entropy: torch.Tensor                # (N,) per-token entropy
    importance: torch.Tensor             # (N,) per-token importance
    global_indices: torch.Tensor         # which tokens contributed to global
    cluster_assignments: torch.Tensor    # (N,) cluster id per token
    image_features: torch.Tensor         # (N, D) original features (for theory)

    @property
    def is_built(self) -> bool:
        return self.global_tokens is not None

    @property
    def n_clusters(self) -> int:
        return self.cluster_centers.shape[0]


def build_compressed_cache(
    image_features: torch.Tensor,
    attn_weights: torch.Tensor,
    image_start: int,
    num_image_tokens: int,
    config: RevoCConfig,
) -> CompressedCache:
    """
    Full compression pipeline.

    Args:
        image_features: (N, D) visual tokens (N=576, D=4096)
        attn_weights: (batch, heads, seq, seq) from ranking layer
        image_start: position of first image token in sequence
        num_image_tokens: count of image tokens
        config: RevoCConfig

    Returns:
        CompressedCache with global summary, clusters, and residuals
    """
    N, D = image_features.shape
    device = image_features.device

    # 1) Compute entropy and importance
    entropy = compute_attention_entropy(attn_weights, image_start, num_image_tokens)
    importance = compute_importance_scores(attn_weights, image_start, num_image_tokens)

    # 2) Entropy-based global token selection
    global_tokens, global_indices = _build_global_summary(
        image_features, entropy, importance, config
    )

    # 3) Cluster remaining tokens
    remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
    remaining_mask[global_indices] = False
    remaining_indices = remaining_mask.nonzero(as_tuple=True)[0]
    remaining_features = image_features[remaining_indices]

    cluster_centers, assignments_local = _kmeans_cluster(
        remaining_features, config.n_clusters
    )

    # Map local assignments back to global token indices
    full_assignments = torch.full((N,), -1, dtype=torch.long, device=device)
    full_assignments[remaining_indices] = assignments_local

    # 4) Build residual store
    residual_store = ResidualStore.build(
        remaining_features, cluster_centers, assignments_local,
        target_device=config.residual_device,
    )

    return CompressedCache(
        global_tokens=global_tokens,
        cluster_centers=cluster_centers,
        residual_store=residual_store,
        entropy=entropy,
        importance=importance,
        global_indices=global_indices,
        cluster_assignments=full_assignments,
        image_features=image_features,
    )


def _build_global_summary(
    image_features: torch.Tensor,
    entropy: torch.Tensor,
    importance: torch.Tensor,
    config: RevoCConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build global summary via entropy-weighted spatial pooling.

    Tokens with high attention entropy are "universally important" —
    they are broadly attended to regardless of query type.
    We select top-entropy tokens and pool them into n_global summaries.

    Args:
        image_features: (N, D)
        entropy: (N,) per-token entropy
        importance: (N,) per-token importance

    Returns:
        global_tokens: (n_global, D)
        global_indices: (M,) indices of tokens that contributed
    """
    N, D = image_features.shape
    n_global = config.n_global
    gh, gw = config.image_grid_h, config.image_grid_w

    # Select high-entropy tokens
    threshold = torch.quantile(entropy, config.entropy_global_percentile / 100.0)
    high_entropy_mask = entropy >= threshold

    # Ensure we have enough tokens (at least n_global)
    if high_entropy_mask.sum() < n_global:
        _, topk_idx = entropy.topk(n_global)
        high_entropy_mask = torch.zeros(N, dtype=torch.bool, device=entropy.device)
        high_entropy_mask[topk_idx] = True

    global_indices = high_entropy_mask.nonzero(as_tuple=True)[0]

    # Entropy-weighted spatial pooling into n_global super-pixels
    # Reshape to spatial grid
    grid_features = image_features.view(gh, gw, D)
    grid_weights = (entropy * importance).view(gh, gw)  # combined weight

    # Pool into super-pixel grid
    sp_h = gh * gw // n_global  # tokens per super-pixel
    # Reshape into n_global groups and weighted-average
    flat_features = image_features  # (N, D)
    flat_weights = (entropy * importance)  # (N,)
    flat_weights = F.softmax(flat_weights, dim=0)

    # Divide into n_global groups of contiguous tokens (spatial locality)
    group_size = N // n_global
    global_tokens = []
    for i in range(n_global):
        start = i * group_size
        end = start + group_size if i < n_global - 1 else N
        group_feats = flat_features[start:end]  # (group_size, D)
        group_w = flat_weights[start:end]        # (group_size,)
        group_w = group_w / group_w.sum().clamp(min=1e-8)
        pooled = (group_feats * group_w.unsqueeze(-1)).sum(dim=0)  # (D,)
        global_tokens.append(pooled)

    global_tokens = torch.stack(global_tokens)  # (n_global, D)
    return global_tokens, global_indices


def _kmeans_cluster(
    features: torch.Tensor,
    n_clusters: int,
    max_iter: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple k-means clustering in embedding space.

    Args:
        features: (M, D) tokens to cluster
        n_clusters: K
        max_iter: max iterations

    Returns:
        centers: (K, D) cluster centers
        assignments: (M,) cluster index per token
    """
    M, D = features.shape
    device = features.device

    if M <= n_clusters:
        # Fewer tokens than clusters — each token is its own cluster
        centers = features.clone()
        # Pad with zeros if needed
        if M < n_clusters:
            padding = torch.zeros(n_clusters - M, D, device=device, dtype=features.dtype)
            centers = torch.cat([centers, padding])
        assignments = torch.arange(M, device=device)
        return centers, assignments

    # Initialize with k-means++ style: first center random, rest by distance
    indices = [torch.randint(0, M, (1,)).item()]
    for _ in range(n_clusters - 1):
        dists = torch.cdist(features, features[indices])  # (M, len(indices))
        min_dists = dists.min(dim=1).values  # (M,)
        # Sample proportional to distance squared
        probs = min_dists ** 2
        probs = probs / probs.sum()
        next_idx = torch.multinomial(probs, 1).item()
        indices.append(next_idx)

    centers = features[indices].clone()  # (K, D)

    # Iterate
    for _ in range(max_iter):
        # Assign
        dists = torch.cdist(features, centers)  # (M, K)
        assignments = dists.argmin(dim=1)  # (M,)

        # Update centers
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(n_clusters, device=device)
        new_centers.scatter_add_(0, assignments.unsqueeze(1).expand(-1, D), features)
        counts.scatter_add_(0, assignments, torch.ones(M, device=device))
        mask = counts > 0
        new_centers[mask] = new_centers[mask] / counts[mask].unsqueeze(1)
        # Keep old center for empty clusters
        new_centers[~mask] = centers[~mask]

        if torch.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    # Final assignment
    dists = torch.cdist(features, centers)
    assignments = dists.argmin(dim=1)

    return centers, assignments
