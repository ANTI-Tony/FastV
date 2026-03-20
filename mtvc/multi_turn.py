"""
MultiTurnEngine — multi-turn dialogue with MTVC.

Round 1: full 576 image tokens + attention hook → build cache → generate
Round N: query-guided retrieval (~150 tokens) → build input → generate

Conversation history is accumulated across rounds. Each round builds the
full conversation context (with all prior Q&A pairs) to maintain coherence.
"""

import time
import torch
from dataclasses import dataclass, field
from typing import Optional

from .config import MTVCConfig
from .cache import VisualTokenCache
from .retriever import QueryGuidedRetriever


@dataclass
class RoundStats:
    round_num: int
    image_tokens_used: int
    total_seq_len: int
    elapsed: float
    response: str


@dataclass
class Session:
    """Holds state for one multi-turn conversation about an image."""
    image: object = None                     # PIL Image
    image_tensor: Optional[torch.Tensor] = None
    cache: Optional[VisualTokenCache] = None
    history: list = field(default_factory=list)  # [(query, response), ...]
    round_stats: list = field(default_factory=list)
    # saved from round 1
    image_features: Optional[torch.Tensor] = None


class MultiTurnEngine:
    def __init__(self, model, tokenizer, image_processor, config: MTVCConfig = None,
                 device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.config = config or MTVCConfig()
        self.device = device
        self.retriever = QueryGuidedRetriever(model, tokenizer, self.config)
        self.session: Optional[Session] = None

    def start_session(self, image) -> None:
        """Initialize a new conversation session with an image."""
        from llava.mm_utils import process_images

        self.session = Session(image=image)
        self.session.image_tensor = process_images(
            [image], self.image_processor, None
        ).to(self.device, dtype=torch.float16)

    def chat(self, query: str, verbose: bool = False) -> RoundStats:
        """Unified entry point — dispatches to round 1 or round N."""
        assert self.session is not None, "Call start_session() first"

        round_num = len(self.session.history) + 1
        t0 = time.time()

        if round_num == 1:
            response, img_tokens, seq_len = self._round1(query, verbose)
        else:
            response, img_tokens, seq_len = self._round_n(query, verbose)

        elapsed = time.time() - t0
        self.session.history.append((query, response))

        stats = RoundStats(
            round_num=round_num,
            image_tokens_used=img_tokens,
            total_seq_len=seq_len,
            elapsed=elapsed,
            response=response,
        )
        self.session.round_stats.append(stats)
        return stats

    # ------------------------------------------------------------------
    # Round 1: full processing + cache construction
    # ------------------------------------------------------------------
    def _round1(self, query: str, verbose: bool) -> tuple:
        """
        Round 1: encode image fully, capture attention, build cache, generate.
        Returns: (response, image_tokens_used, seq_len)
        """
        from fastv.core import prepare_input, get_multimodal_embeds
        from transformers import LlamaForCausalLM

        # 1) Prepare input with LLaVA conversation template
        input_ids, _ = prepare_input(
            self.tokenizer, self.image_processor,
            self.session.image, query, self.device,
        )

        # 2) Build full embeddings with all 576 image tokens
        with torch.no_grad():
            full_embeds, image_start, num_img = get_multimodal_embeds(
                self.model, input_ids, self.session.image_tensor,
            )

        if verbose:
            print(f"  [Round 1] full_embeds: {full_embeds.shape}, "
                  f"image @ [{image_start}, {image_start + num_img})")

        # 3) Forward pass with attention hook on ranking layer
        attn_captured = {}
        llm = self.model.get_model()
        target_layer = llm.layers[self.config.ranking_layer - 1]

        def capture_attn(module, args, output):
            if len(output) > 1 and output[1] is not None:
                attn_captured['weights'] = output[1].detach()

        hook = target_layer.register_forward_hook(capture_attn)
        with torch.no_grad():
            _ = self.model(
                inputs_embeds=full_embeds,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        hook.remove()

        # 4) Compute importance scores
        attn_w = attn_captured['weights']  # (B, heads, seq, seq)
        img_attn = attn_w[:, :, -1, image_start:image_start + num_img]
        importance = img_attn.mean(dim=1).squeeze(0)  # (576,)

        # 5) Extract raw image features and build cache
        image_features = full_embeds[0, image_start:image_start + num_img, :]  # (576, D)
        self.session.image_features = image_features
        self.session.cache = VisualTokenCache.build(image_features, importance, self.config)

        if verbose:
            print(f"  [Round 1] Cache built: L1={self.session.cache.l1.shape}, "
                  f"L2={self.session.cache.l2.shape}, L3={self.session.cache.l3.shape}")

        # 6) Generate using full embeddings (no pruning for round 1)
        seq_len = full_embeds.shape[1]
        attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.device)

        with torch.no_grad():
            output_ids = LlamaForCausalLM.generate(
                self.model,
                inputs_embeds=full_embeds,
                attention_mask=attn_mask,
                do_sample=False,
                max_new_tokens=self.config.max_new_tokens,
            )

        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return response, num_img, seq_len

    # ------------------------------------------------------------------
    # Round N: query-guided retrieval + generation
    # ------------------------------------------------------------------
    def _round_n(self, query: str, verbose: bool) -> tuple:
        """
        Round 2+: retrieve relevant tokens from cache, build context, generate.
        Returns: (response, image_tokens_used, seq_len)
        """
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from transformers import LlamaForCausalLM

        # 1) Build multi-turn conversation prompt
        conv = conv_templates["v1"].copy()

        # First round includes image token
        first_query = self.session.history[0][0]
        if DEFAULT_IMAGE_TOKEN not in first_query:
            first_query = DEFAULT_IMAGE_TOKEN + "\n" + first_query
        conv.append_message(conv.roles[0], first_query)
        conv.append_message(conv.roles[1], self.session.history[0][1])

        # Subsequent rounds
        for q, a in self.session.history[1:]:
            conv.append_message(conv.roles[0], q)
            conv.append_message(conv.roles[1], a)

        # Current query
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        # 2) Tokenize (with IMAGE_TOKEN_INDEX placeholder)
        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self.device)

        # 3) Query-guided retrieval
        query_vec = self.retriever.encode_query(query, self.device)
        retrieved_tokens = self.retriever.retrieve(self.session.cache, query_vec)
        num_retrieved = retrieved_tokens.shape[0]

        if verbose:
            print(f"  [Round {len(self.session.history) + 1}] "
                  f"Retrieved {num_retrieved} tokens from cache")

        # 4) Build embeddings: replace IMAGE_TOKEN_INDEX with retrieved tokens
        embed_tokens = self.model.get_model().embed_tokens

        safe_ids = input_ids.clone()
        image_mask = safe_ids[0] == IMAGE_TOKEN_INDEX
        safe_ids[0][image_mask] = 0
        input_embeds = embed_tokens(safe_ids)

        image_pos = image_mask.nonzero(as_tuple=True)[0]
        if len(image_pos) == 0:
            # no image placeholder — shouldn't happen, but handle gracefully
            full_embeds = input_embeds
        else:
            image_start = image_pos[0].item()
            prefix_embeds = input_embeds[:, :image_start, :]
            suffix_embeds = input_embeds[:, image_start + 1:, :]  # skip placeholder

            retrieved_embeds = retrieved_tokens.unsqueeze(0)  # (1, ~150, D)
            full_embeds = torch.cat([prefix_embeds, retrieved_embeds, suffix_embeds], dim=1)

        # 5) Generate
        seq_len = full_embeds.shape[1]
        attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.device)

        with torch.no_grad():
            output_ids = LlamaForCausalLM.generate(
                self.model,
                inputs_embeds=full_embeds,
                attention_mask=attn_mask,
                do_sample=False,
                max_new_tokens=self.config.max_new_tokens,
            )

        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return response, num_retrieved, seq_len

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def get_summary(self) -> dict:
        """Return summary statistics for the session."""
        if not self.session or not self.session.round_stats:
            return {}

        stats = self.session.round_stats
        total_img_tokens = sum(s.image_tokens_used for s in stats)
        vanilla_equivalent = self.config.image_token_length * len(stats)

        return {
            "num_rounds": len(stats),
            "total_image_tokens": total_img_tokens,
            "vanilla_equivalent": vanilla_equivalent,
            "savings_pct": (1 - total_img_tokens / vanilla_equivalent) * 100,
            "total_time": sum(s.elapsed for s in stats),
            "per_round": [
                {
                    "round": s.round_num,
                    "image_tokens": s.image_tokens_used,
                    "seq_len": s.total_seq_len,
                    "time": f"{s.elapsed:.3f}s",
                }
                for s in stats
            ],
        }
