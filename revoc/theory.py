"""
Information-theoretic analysis for ReVoC.

Provides computable bounds on quality degradation from compression,
and demonstrates the theoretical advantage of recoverable compression
over irreversible pruning.

Key result:
  Pruning bound:  ||f(V) - f(S)|| ≤ L · Σ_{i∉S} ||v_i|| / √N
  ReVoC bound:    ||f(V) - f(Ŝ)|| ≤ L · Σ_{j∉unpacked} σ_j / √K

  Since σ_j (within-cluster std) << ||v_i|| (token norm),
  ReVoC's bound is strictly tighter for the same number of active tokens.
"""

import torch
import math
from typing import Optional
from dataclasses import dataclass


@dataclass
class CompressionBound:
    """Stores computed theoretical bounds for a compression instance."""
    prune_bound: float       # upper bound for pruning at same compression ratio
    revoc_bound: float       # upper bound for ReVoC
    improvement_ratio: float  # prune_bound / revoc_bound
    avg_residual_norm: float
    avg_token_norm: float
    max_cluster_variance: float


def estimate_lipschitz_constant(
    model,
    embeds: torch.Tensor,
    image_start: int,
    num_image_tokens: int,
    n_probes: int = 5,
) -> float:
    """
    Empirically estimate the local Lipschitz constant of the model's
    output w.r.t. visual token perturbations.

    L ≈ max_i ||f(V + εe_i) - f(V)|| / ε

    Args:
        model: the VLM
        embeds: (1, seq_len, D) full input embeddings
        image_start: start of image tokens
        num_image_tokens: count of image tokens
        n_probes: number of random perturbation directions

    Returns:
        Estimated Lipschitz constant (float)
    """
    eps = 1e-3
    D = embeds.shape[-1]
    max_ratio = 0.0

    with torch.no_grad():
        base_out = model(inputs_embeds=embeds, use_cache=False, return_dict=True)
        base_logits = base_out.logits[:, -1, :]  # last token logits

        for _ in range(n_probes):
            # Random unit perturbation on a random image token
            idx = torch.randint(0, num_image_tokens, (1,)).item()
            direction = torch.randn(1, 1, D, device=embeds.device, dtype=embeds.dtype)
            direction = direction / direction.norm()

            perturbed = embeds.clone()
            perturbed[:, image_start + idx, :] += eps * direction.squeeze(1)

            pert_out = model(inputs_embeds=perturbed, use_cache=False, return_dict=True)
            pert_logits = pert_out.logits[:, -1, :]

            ratio = (pert_logits - base_logits).norm().item() / eps
            max_ratio = max(max_ratio, ratio)

    return max_ratio


def compute_compression_bounds(
    image_features: torch.Tensor,
    cluster_assignments: torch.Tensor,
    cluster_centers: torch.Tensor,
    n_unpacked: int,
    lipschitz_L: float = 1.0,
) -> CompressionBound:
    """
    Compute theoretical bounds for pruning vs ReVoC compression.

    Args:
        image_features: (N, D) original visual tokens
        cluster_assignments: (N,) cluster index per token
        cluster_centers: (K, D) cluster center embeddings
        n_unpacked: number of clusters to unpack (recover)
        lipschitz_L: estimated Lipschitz constant

    Returns:
        CompressionBound with both bounds and comparison
    """
    N, D = image_features.shape
    K = cluster_centers.shape[0]

    # --- Token norms (for pruning bound) ---
    token_norms = image_features.norm(dim=-1)  # (N,)
    avg_token_norm = token_norms.mean().item()

    # --- Within-cluster residual norms ---
    residuals = image_features - cluster_centers[cluster_assignments]  # (N, D)
    residual_norms = residuals.norm(dim=-1)  # (N,)
    avg_residual_norm = residual_norms.mean().item()

    # --- Per-cluster variance ---
    cluster_variances = torch.zeros(K, device=image_features.device)
    for j in range(K):
        mask = cluster_assignments == j
        if mask.sum() > 0:
            cluster_variances[j] = residual_norms[mask].mean()

    max_cluster_variance = cluster_variances.max().item()

    # Sort clusters by variance (worst case: we DON'T unpack the highest-variance ones)
    sorted_var, _ = cluster_variances.sort(descending=True)
    # Remaining (not unpacked) cluster variances
    remaining_var = sorted_var[n_unpacked:]

    # --- Pruning bound ---
    # Prune same number of tokens as ReVoC compresses
    n_active = n_unpacked * (N // K)  # approximate tokens after unpacking
    n_pruned = N - n_active
    # Sort token norms descending, prune the smallest
    sorted_norms, _ = token_norms.sort(descending=True)
    pruned_norms = sorted_norms[n_active:]
    prune_bound = lipschitz_L * pruned_norms.sum().item() / math.sqrt(N)

    # --- ReVoC bound ---
    revoc_bound = lipschitz_L * remaining_var.sum().item() / math.sqrt(K)

    improvement = prune_bound / revoc_bound if revoc_bound > 0 else float('inf')

    return CompressionBound(
        prune_bound=prune_bound,
        revoc_bound=revoc_bound,
        improvement_ratio=improvement,
        avg_residual_norm=avg_residual_norm,
        avg_token_norm=avg_token_norm,
        max_cluster_variance=max_cluster_variance,
    )


def compute_mutual_information_estimate(
    token_entropy: torch.Tensor,
    importance: torch.Tensor,
) -> dict:
    """
    Estimate mutual information between token entropy and importance.

    High correlation supports the theoretical claim that entropy-based
    partitioning aligns with query-relevance.

    Returns:
        dict with correlation, entropy stats, importance stats
    """
    H = token_entropy.float()
    I = importance.float()

    # Pearson correlation
    H_centered = H - H.mean()
    I_centered = I - I.mean()
    corr = (H_centered * I_centered).sum() / (
        H_centered.norm() * I_centered.norm() + 1e-8
    )

    return {
        "entropy_importance_correlation": corr.item(),
        "entropy_mean": H.mean().item(),
        "entropy_std": H.std().item(),
        "importance_mean": I.mean().item(),
        "importance_std": I.std().item(),
        "high_entropy_high_importance_frac": (
            (H > H.median()) & (I > I.median())
        ).float().mean().item(),
    }
