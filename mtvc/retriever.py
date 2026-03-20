"""
QueryGuidedRetriever — encode query text and retrieve relevant tokens from cache.

Retrieval formula per round:
  R(q) = L1(all 32) + TopK(L2, k2) + TopK(L3, k3)

Query encoding uses the model's own embed_tokens (no extra parameters).
"""

import torch
import torch.nn.functional as F

from .config import MTVCConfig
from .cache import VisualTokenCache


class QueryGuidedRetriever:
    def __init__(self, model, tokenizer, config: MTVCConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._embed_fn = model.get_model().embed_tokens

    @torch.no_grad()
    def encode_query(self, text: str, device: str = "cuda") -> torch.Tensor:
        """
        Encode query text into a single vector via embed_tokens + mean pooling.

        Returns: (1, D) tensor
        """
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens.input_ids.to(device)           # (1, seq_len)
        embeds = self._embed_fn(input_ids)                 # (1, seq_len, D)
        query_vec = embeds.mean(dim=1)                     # (1, D)
        return query_vec

    @torch.no_grad()
    def retrieve(
        self,
        cache: VisualTokenCache,
        query_vec: torch.Tensor,
        k2: int | None = None,
        k3: int | None = None,
    ) -> torch.Tensor:
        """
        Retrieve relevant visual tokens from the 3-level cache.

        Args:
            cache: populated VisualTokenCache
            query_vec: (1, D) query embedding
            k2: number of tokens from L2 (default: config.default_k2)
            k3: number of tokens from L3 (default: config.default_k3)

        Returns:
            (total_retrieved, D) tensor of selected visual tokens
        """
        assert cache.is_built, "Cache not built yet"
        k2 = k2 if k2 is not None else self.config.default_k2
        k3 = k3 if k3 is not None else self.config.default_k3

        # L1: always include all global tokens
        l1_tokens = cache.l1  # (32, D)

        # L2: cosine similarity TopK
        l2_tokens = self._topk_by_similarity(cache.l2, query_vec, k2)  # (k2, D)

        # L3: cosine similarity TopK
        l3_tokens = self._topk_by_similarity(cache.l3, query_vec, k3)  # (k3, D)

        # concatenate: L1 + L2_selected + L3_selected
        retrieved = torch.cat([l1_tokens, l2_tokens, l3_tokens], dim=0)
        return retrieved

    @staticmethod
    def _topk_by_similarity(
        tokens: torch.Tensor,
        query_vec: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Select top-k tokens by cosine similarity to query.

        Args:
            tokens: (M, D)
            query_vec: (1, D)
            k: number of tokens to select
        Returns:
            (k, D) selected tokens in original order
        """
        if k >= tokens.shape[0]:
            return tokens

        # cosine similarity: (M,)
        sim = F.cosine_similarity(tokens, query_vec.expand_as(tokens), dim=-1)
        # top-k indices
        _, topk_idx = torch.topk(sim, k)
        # sort to preserve positional order
        topk_idx, _ = torch.sort(topk_idx)
        return tokens[topk_idx]
