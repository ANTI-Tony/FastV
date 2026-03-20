"""
Retriever — selects which clusters to unpack given a query.

Two modes:
  1. CrossAttentionRetriever (learned, ~4M params):
     - Single cross-attention layer: Q=query, K=V=cluster_centers
     - Trained via distillation to match full-token model output
     - Blends with dialogue history via EMA importance

  2. CosineRetriever (training-free fallback):
     - Cosine similarity between query embedding and cluster centers
     - Blended with EMA importance from previous turns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import RevoCConfig
from .compressor import CompressedCache
from .importance import EMAImportanceTracker


class CrossAttentionRetriever(nn.Module):
    """
    Learned retriever using a single multi-head cross-attention layer.

    Q = query text embeddings
    K = V = cluster center embeddings

    Output attention weights → cluster selection scores.
    Only ~4M trainable params (for D=4096, H=8).
    """

    def __init__(self, config: RevoCConfig):
        super().__init__()
        D = config.hidden_dim
        H = config.retriever_heads

        self.n_retrieve = config.n_retrieve_clusters
        self.history_weight = config.history_weight

        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.out_proj = nn.Linear(D, D, bias=False)

        self.num_heads = H
        self.head_dim = D // H
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(config.retriever_dropout)

    def forward(
        self,
        query_embeds: torch.Tensor,
        cluster_centers: torch.Tensor,
        history_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select clusters to unpack.

        Args:
            query_embeds: (1, q_len, D) text query embeddings
            cluster_centers: (1, K, D) cluster center embeddings
            history_bias: (K,) EMA importance from previous turns

        Returns:
            selected_indices: (n_retrieve,) indices of clusters to unpack
            scores: (K,) per-cluster selection scores
        """
        B, q_len, D = query_embeds.shape
        K = cluster_centers.shape[1]

        # Project
        Q = self.q_proj(query_embeds)   # (B, q_len, D)
        Kp = self.k_proj(cluster_centers)  # (B, K, D)

        # Reshape for multi-head attention
        Q = Q.view(B, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        Kp = Kp.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        # Q: (B, H, q_len, d), Kp: (B, H, K, d)

        # Attention scores
        attn = torch.matmul(Q, Kp.transpose(-2, -1)) * self.scale  # (B, H, q_len, K)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Pool across query positions and heads → per-cluster score
        scores = attn.mean(dim=2).mean(dim=1).squeeze(0)  # (K,)

        # Blend with dialogue history
        if history_bias is not None:
            alpha = self.history_weight
            scores = (1 - alpha) * scores + alpha * history_bias

        # Select top clusters
        _, selected = scores.topk(min(self.n_retrieve, K))
        selected = selected.sort().values  # preserve spatial order

        return selected, scores


class CosineRetriever:
    """
    Training-free fallback: cosine similarity retrieval.
    Uses model's embed_tokens for query encoding.
    """

    def __init__(self, model, tokenizer, config: RevoCConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.n_retrieve = config.n_retrieve_clusters
        self.history_weight = config.history_weight
        self._embed_fn = model.get_model().embed_tokens

    @torch.no_grad()
    def encode_query(self, text: str, device: str = "cuda") -> torch.Tensor:
        """Encode query text: embed_tokens + mean pool → (1, D)."""
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens.input_ids.to(device)
        embeds = self._embed_fn(input_ids)  # (1, seq_len, D)
        return embeds.mean(dim=1)  # (1, D)

    @torch.no_grad()
    def select_clusters(
        self,
        query_vec: torch.Tensor,
        cluster_centers: torch.Tensor,
        history_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select clusters by cosine similarity + history blend.

        Args:
            query_vec: (1, D) query embedding
            cluster_centers: (K, D) cluster centers
            history_bias: (K,) EMA importance

        Returns:
            selected_indices: (n_retrieve,) cluster indices
            scores: (K,) per-cluster scores
        """
        # Cosine similarity
        scores = F.cosine_similarity(
            cluster_centers, query_vec.expand_as(cluster_centers), dim=-1
        )  # (K,)

        # Blend with history
        if history_bias is not None:
            alpha = self.history_weight
            scores = (1 - alpha) * scores + alpha * history_bias

        _, selected = scores.topk(min(self.n_retrieve, cluster_centers.shape[0]))
        selected = selected.sort().values

        return selected, scores


class UnifiedRetriever:
    """
    Wraps both retriever types with a common interface.
    Handles query encoding, cluster selection, and token recovery.
    """

    def __init__(self, model, tokenizer, config: RevoCConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.ema_tracker = EMAImportanceTracker(config.n_clusters, config.ema_decay, device)

        if config.retriever_type == "cross_attention":
            self.learned_retriever = CrossAttentionRetriever(config).to(device)
            self.cosine_retriever = None
        else:
            self.learned_retriever = None
            self.cosine_retriever = CosineRetriever(model, tokenizer, config)

        self._embed_fn = model.get_model().embed_tokens
        self.tokenizer = tokenizer

    @torch.no_grad()
    def retrieve(
        self,
        cache: CompressedCache,
        query_text: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full retrieval pipeline: encode query → select clusters → recover tokens.

        Returns:
            retrieved_tokens: (M, D) — global tokens + recovered cluster tokens
            cluster_scores: (K,) — for EMA update
        """
        history_bias = self.ema_tracker.get_bias()

        if self.learned_retriever is not None:
            # Encode query
            tokens = self.tokenizer(query_text, return_tensors="pt", add_special_tokens=False)
            input_ids = tokens.input_ids.to(self.device)
            query_embeds = self._embed_fn(input_ids).unsqueeze(0) if input_ids.dim() == 1 else self._embed_fn(input_ids)
            # query_embeds: (1, q_len, D)

            selected, scores = self.learned_retriever(
                query_embeds,
                cache.cluster_centers.unsqueeze(0),
                history_bias,
            )
        else:
            query_vec = self.cosine_retriever.encode_query(query_text, self.device)
            selected, scores = self.cosine_retriever.select_clusters(
                query_vec, cache.cluster_centers, history_bias,
            )

        # Recover tokens from selected clusters
        recovered_parts = []
        target_dev = str(cache.cluster_centers.device)
        for c_id in selected.tolist():
            recovered = cache.residual_store.recover_cluster(
                c_id, cache.cluster_centers[c_id], target_device=target_dev,
            )
            recovered_parts.append(recovered)

        if recovered_parts:
            recovered_tokens = torch.cat(recovered_parts, dim=0)
        else:
            recovered_tokens = cache.cluster_centers[:1]  # fallback

        # Compose: global tokens + recovered cluster tokens
        retrieved = torch.cat([cache.global_tokens, recovered_tokens], dim=0)

        # Update EMA
        self.ema_tracker.update(scores)

        return retrieved, scores

    def load_retriever_weights(self, path: str):
        """Load trained cross-attention retriever weights."""
        if self.learned_retriever is not None:
            state = torch.load(path, map_location=self.device)
            self.learned_retriever.load_state_dict(state)

    def save_retriever_weights(self, path: str):
        if self.learned_retriever is not None:
            torch.save(self.learned_retriever.state_dict(), path)
