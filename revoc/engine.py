"""
Multi-Turn Dialogue Engine with Recoverable Visual Compression.

Round 1: Full 576 tokens → attention capture → build compressed cache → generate
Round N: Query-guided cluster selection → recover tokens → generate

Key differences from MTVC:
  - Tokens are compressed (clustered + residuals), not pruned
  - Any token can be exactly recovered at any turn
  - Cluster selection adapts across turns via EMA importance
  - Supports both learned (cross-attention) and training-free (cosine) retrieval
"""

import time
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from .config import RevoCConfig
from .compressor import CompressedCache, build_compressed_cache
from .retriever import UnifiedRetriever
from .model_adapter import VLMAdapter, LLaVAAdapter


@dataclass
class RoundStats:
    round_num: int
    image_tokens_used: int
    total_seq_len: int
    elapsed: float
    response: str
    clusters_unpacked: int = 0
    tokens_recovered: int = 0
    recovery_triggered: bool = False  # True if adaptive recovery was triggered
    probe_entropy: float = 0.0       # output entropy from probe pass


@dataclass
class Session:
    """State for one multi-turn conversation."""
    image: object = None
    image_tensor: Optional[torch.Tensor] = None
    cache: Optional[CompressedCache] = None
    history: list = field(default_factory=list)    # [(query, response), ...]
    round_stats: list = field(default_factory=list)


class RevoCEngine:
    """
    Main engine for multi-turn dialogue with recoverable visual compression.
    """

    def __init__(
        self,
        adapter: VLMAdapter,
        config: RevoCConfig = None,
        device: str = "cuda",
    ):
        self.adapter = adapter
        self.config = config or RevoCConfig()
        self.config.validate()
        self.device = device
        self.retriever = UnifiedRetriever(
            adapter.model, adapter.tokenizer, self.config, device
        )
        self.session: Optional[Session] = None

    def start_session(self, image) -> None:
        """Initialize a new conversation session with an image."""
        from llava.mm_utils import process_images
        self.session = Session(image=image)
        self.session.image_tensor = process_images(
            [image], self.adapter.image_processor, None
        ).to(self.device, dtype=torch.float16)
        self.retriever.ema_tracker.reset()

    def chat(self, query: str, verbose: bool = False) -> RoundStats:
        """Unified entry — dispatches to round 1 or round N."""
        assert self.session is not None, "Call start_session() first"

        round_num = len(self.session.history) + 1
        t0 = time.time()

        if round_num == 1:
            response, img_tokens, seq_len, n_clusters, n_recovered = \
                self._round1(query, verbose)
        else:
            response, img_tokens, seq_len, n_clusters, n_recovered = \
                self._round_n(query, verbose)

        elapsed = time.time() - t0
        self.session.history.append((query, response))

        stats = RoundStats(
            round_num=round_num,
            image_tokens_used=img_tokens,
            total_seq_len=seq_len,
            elapsed=elapsed,
            response=response,
            clusters_unpacked=n_clusters,
            tokens_recovered=n_recovered,
            recovery_triggered=n_clusters > 0 and round_num > 1,
            probe_entropy=0.0,
        )
        self.session.round_stats.append(stats)
        return stats

    # ------------------------------------------------------------------
    # Round 1: full processing + cache construction
    # ------------------------------------------------------------------
    def _round1(self, query: str, verbose: bool) -> Tuple:
        """
        Round 1: encode image, capture attention, build compressed cache, generate.
        Returns: (response, img_tokens, seq_len, clusters_unpacked, tokens_recovered)
        """
        # 1) Prepare input
        input_ids, image_tensor = self.adapter.prepare_input(
            self.session.image, query
        )

        # 2) Build full embeddings
        with torch.no_grad():
            full_embeds, image_start, num_img = self.adapter.build_multimodal_embeds(
                input_ids, self.session.image_tensor,
            )

        if verbose:
            print(f"  [Round 1] full_embeds: {full_embeds.shape}, "
                  f"image @ [{image_start}, {image_start + num_img})")

        # 3) Forward with attention capture
        _, attn_weights = self.adapter.forward_with_attention(
            full_embeds, self.config.ranking_layer - 1
        )

        if attn_weights is None:
            raise RuntimeError("Failed to capture attention weights")

        # 4) Extract image features and build compressed cache
        image_features = full_embeds[0, image_start:image_start + num_img, :]

        self.session.cache = build_compressed_cache(
            image_features, attn_weights, image_start, num_img, self.config
        )

        if verbose:
            cache = self.session.cache
            print(f"  [Round 1] Cache built: "
                  f"global={cache.global_tokens.shape}, "
                  f"clusters={cache.cluster_centers.shape}, "
                  f"residuals={cache.residual_store.memory_bytes / 1024:.0f} KB")

        # 5) Generate with full embeddings (no compression for round 1)
        seq_len = full_embeds.shape[1]
        attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.device)
        response = self.adapter.generate(full_embeds, attn_mask, self.config.max_new_tokens)

        return response, num_img, seq_len, 0, 0

    # ------------------------------------------------------------------
    # Round N: adaptive probe → decide → generate
    # ------------------------------------------------------------------
    def _round_n(self, query: str, verbose: bool) -> Tuple:
        """
        Round 2+ with adaptive recovery:
          Phase 1 (Probe): Use global + cluster CENTERS only (~96 tokens).
                           Forward pass → check output entropy.
          Phase 2 (Decide): If confident → generate directly (83% savings).
                            If uncertain → recover full cluster tokens → regenerate.

        This makes ReVoC's token budget adaptive — averaging fewer tokens than
        fixed-budget methods like FastV, while retaining exact recovery as safety net.

        Returns: (response, img_tokens, seq_len, clusters_unpacked, tokens_recovered)
        """
        from llava.constants import IMAGE_TOKEN_INDEX
        import torch.nn.functional as F

        cache = self.session.cache
        round_label = len(self.session.history) + 1

        # 1) Build multi-turn conversation input_ids (shared across probe/full)
        input_ids = self.adapter.build_multiturn_input(
            self.session.image, self.session.history, query
        )
        embed_tokens = self.adapter.get_embed_tokens()
        safe_ids = input_ids.clone()
        image_mask = safe_ids[0] == IMAGE_TOKEN_INDEX
        safe_ids[0][image_mask] = 0
        input_embeds = embed_tokens(safe_ids)

        image_pos = image_mask.nonzero(as_tuple=True)[0]
        if len(image_pos) == 0:
            # No image placeholder — generate with text only
            full_embeds = input_embeds
            seq_len = full_embeds.shape[1]
            attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.device)
            response = self.adapter.generate(full_embeds, attn_mask, self.config.max_new_tokens)
            return response, 0, seq_len, 0, 0

        img_start = image_pos[0].item()
        prefix = input_embeds[:, :img_start, :]
        suffix = input_embeds[:, img_start + 1:, :]

        # ============================================================
        # Phase 1: PROBE with centers only (32 global + 64 centers = 96 tokens)
        # ============================================================
        probe_tokens = torch.cat([cache.global_tokens, cache.cluster_centers], dim=0)
        probe_embeds = torch.cat([prefix, probe_tokens.unsqueeze(0), suffix], dim=1)
        probe_len = probe_embeds.shape[1]

        recovery_triggered = False
        probe_entropy = 0.0

        if self.config.adaptive_recovery:
            # Lightweight forward to check confidence
            with torch.no_grad():
                probe_out = self.adapter.model(
                    inputs_embeds=probe_embeds,
                    use_cache=False,
                    return_dict=True,
                )
                logits = probe_out.logits[:, -1, :]  # (1, vocab_size)
                probs = F.softmax(logits, dim=-1)
                probe_entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).item()

            recovery_triggered = probe_entropy > self.config.confidence_threshold

            if verbose:
                status = "RECOVER" if recovery_triggered else "CONFIDENT"
                print(f"  [Round {round_label}] Probe: entropy={probe_entropy:.3f}, "
                      f"threshold={self.config.confidence_threshold}, → {status}")
        else:
            # Adaptive disabled → always recover
            recovery_triggered = True

        # ============================================================
        # Phase 2: DECIDE — generate from probe or recover then generate
        # ============================================================
        if not recovery_triggered:
            # High confidence → generate directly from centers (83% savings!)
            num_tokens = probe_tokens.shape[0]
            attn_mask = torch.ones((1, probe_len), dtype=torch.long, device=self.device)
            response = self.adapter.generate(probe_embeds, attn_mask, self.config.max_new_tokens)

            # Still update EMA with probe-based scores (cosine sim of query to centers)
            query_vec = self.retriever.cosine_retriever.encode_query(query, self.device) \
                if self.retriever.cosine_retriever else None
            if query_vec is not None:
                scores = F.cosine_similarity(
                    cache.cluster_centers, query_vec.expand_as(cache.cluster_centers), dim=-1
                )
                self.retriever.ema_tracker.update(scores)

            if verbose:
                print(f"  [Round {round_label}] Centers-only: {num_tokens} tokens "
                      f"(savings: {(1 - num_tokens / self.config.image_token_length) * 100:.0f}%)")

            return response, num_tokens, probe_len, 0, 0
        else:
            # Low confidence → recover full cluster tokens for relevant clusters
            retrieved_tokens, cluster_scores = self.retriever.retrieve(cache, query)
            num_retrieved = retrieved_tokens.shape[0]
            n_clusters = self.config.n_retrieve_clusters

            full_embeds = torch.cat([prefix, retrieved_tokens.unsqueeze(0), suffix], dim=1)
            seq_len = full_embeds.shape[1]
            attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.device)
            response = self.adapter.generate(full_embeds, attn_mask, self.config.max_new_tokens)

            if verbose:
                print(f"  [Round {round_label}] Recovery: {num_retrieved} tokens "
                      f"({self.config.n_global} global + "
                      f"{num_retrieved - self.config.n_global} from {n_clusters} clusters)")

            return response, num_retrieved, seq_len, n_clusters, num_retrieved - self.config.n_global

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def get_summary(self) -> dict:
        """Session statistics."""
        if not self.session or not self.session.round_stats:
            return {}

        stats = self.session.round_stats
        total_img = sum(s.image_tokens_used for s in stats)
        vanilla_eq = self.config.image_token_length * len(stats)

        return {
            "num_rounds": len(stats),
            "total_image_tokens": total_img,
            "vanilla_equivalent": vanilla_eq,
            "savings_pct": (1 - total_img / vanilla_eq) * 100 if vanilla_eq > 0 else 0,
            "total_time": sum(s.elapsed for s in stats),
            "adaptive_recovery": self.config.adaptive_recovery,
            "rounds_with_recovery": sum(1 for s in stats if s.recovery_triggered),
            "rounds_centers_only": sum(1 for s in stats if not s.recovery_triggered and s.round_num > 1),
            "compression_info": {
                "n_global": self.config.n_global,
                "n_clusters": self.config.n_clusters,
                "n_retrieve_clusters": self.config.n_retrieve_clusters,
                "retriever_type": self.config.retriever_type,
            },
            "per_round": [
                {
                    "round": s.round_num,
                    "image_tokens": s.image_tokens_used,
                    "clusters_unpacked": s.clusters_unpacked,
                    "tokens_recovered": s.tokens_recovered,
                    "seq_len": s.total_seq_len,
                    "time": f"{s.elapsed:.3f}s",
                }
                for s in stats
            ],
        }
