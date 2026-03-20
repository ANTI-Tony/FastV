"""
Residual Store — the key to recoverable compression.

Unlike pruning (irreversible), ReVoC stores residuals:
  Δ_i = token_i - cluster_center_{c(i)}

Recovery: token_i = cluster_center_{c(i)} + Δ_i  (exact)

Residuals can optionally be:
  - Kept in FP16 on GPU (default, fastest recovery)
  - Offloaded to CPU (saves GPU memory)
  - Quantized to INT8 (saves memory, near-lossless)
"""

import torch
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ResidualStore:
    """Stores per-token residuals and cluster membership for recovery."""

    residuals: Optional[torch.Tensor] = None       # (N, D) — Δ_i per token
    assignments: Optional[torch.Tensor] = None     # (N,) — cluster index per token
    cluster_members: dict = field(default_factory=dict)  # cluster_id → list of token indices
    N: int = 0
    D: int = 0
    device: str = "cuda"

    @property
    def is_built(self) -> bool:
        return self.residuals is not None

    @classmethod
    def build(
        cls,
        image_features: torch.Tensor,
        cluster_centers: torch.Tensor,
        assignments: torch.Tensor,
        target_device: str = "same",
    ) -> "ResidualStore":
        """
        Compute and store residuals.

        Args:
            image_features: (N, D) original visual tokens
            cluster_centers: (K, D) cluster centers
            assignments: (N,) cluster index per token
            target_device: "same" (keep on GPU), "cpu" (offload)

        Returns:
            ResidualStore with residuals and membership info
        """
        N, D = image_features.shape

        # Compute residuals: Δ_i = v_i - c_{a(i)}
        residuals = image_features - cluster_centers[assignments]

        # Build cluster membership index
        cluster_members = {}
        for i in range(N):
            c = assignments[i].item()
            if c not in cluster_members:
                cluster_members[c] = []
            cluster_members[c].append(i)

        # Optionally offload to CPU
        device = image_features.device
        if target_device == "cpu":
            residuals = residuals.cpu()
            device_str = "cpu"
        else:
            device_str = str(device)

        return cls(
            residuals=residuals,
            assignments=assignments,
            cluster_members=cluster_members,
            N=N,
            D=D,
            device=device_str,
        )

    def recover_cluster(
        self,
        cluster_id: int,
        cluster_center: torch.Tensor,
        target_device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Recover all original tokens belonging to a cluster.

        token_i = cluster_center + Δ_i  (exact recovery)

        Args:
            cluster_id: which cluster to unpack
            cluster_center: (D,) the cluster center embedding
            target_device: device for output tensors

        Returns:
            (M, D) recovered tokens where M = cluster size
        """
        assert self.is_built, "ResidualStore not built"

        member_indices = self.cluster_members.get(cluster_id, [])
        if not member_indices:
            return cluster_center.unsqueeze(0)

        idx = torch.tensor(member_indices, dtype=torch.long)
        deltas = self.residuals[idx]  # (M, D)

        # Move to target device if needed
        if target_device:
            deltas = deltas.to(target_device)
            cluster_center = cluster_center.to(target_device)

        # Exact recovery
        recovered = cluster_center.unsqueeze(0) + deltas  # (M, D)
        return recovered

    def recover_tokens_by_indices(
        self,
        token_indices: torch.Tensor,
        cluster_centers: torch.Tensor,
        target_device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Recover specific tokens by their original indices.

        Args:
            token_indices: (M,) indices of tokens to recover
            cluster_centers: (K, D) all cluster centers
            target_device: device for output

        Returns:
            (M, D) recovered tokens
        """
        assignments = self.assignments[token_indices]
        centers = cluster_centers[assignments]  # (M, D)
        deltas = self.residuals[token_indices]   # (M, D)

        if target_device:
            centers = centers.to(target_device)
            deltas = deltas.to(target_device)

        return centers + deltas

    def get_cluster_residual_norms(self, cluster_centers: torch.Tensor) -> torch.Tensor:
        """
        Get average residual norm per cluster (proxy for information loss
        when using only the cluster center).

        Returns: (K,) average residual norm per cluster
        """
        K = cluster_centers.shape[0]
        norms = torch.zeros(K, device=cluster_centers.device)
        for c_id, members in self.cluster_members.items():
            idx = torch.tensor(members, dtype=torch.long)
            cluster_residuals = self.residuals[idx].to(cluster_centers.device)
            norms[c_id] = cluster_residuals.norm(dim=-1).mean()
        return norms

    @property
    def memory_bytes(self) -> int:
        """Estimate memory usage of stored residuals."""
        if self.residuals is None:
            return 0
        elem_size = self.residuals.element_size()
        return self.residuals.numel() * elem_size
