"""
RPA (Reasoning-aware Progressive Abstraction) configuration.

Core idea: During chain-of-thought reasoning, visual tokens become
progressively redundant as their information is absorbed into the
reasoning chain. RPA periodically compresses visual KV cache entries
based on measured absorption, enabling longer reasoning within the
same compute budget.
"""

from dataclasses import dataclass


@dataclass
class RPAConfig:
    # ---- compression schedule ----
    check_interval: int = 32        # check absorption every K generated tokens
    compress_ratio: float = 0.75    # keep this fraction of visual tokens each step
    min_visual_tokens: int = 64     # never compress below this
    warmup_tokens: int = 64         # don't compress during first N generated tokens

    # ---- absorption detection ----
    ranking_layer: int = 2          # which layer's attention to monitor (1-indexed)
    absorption_threshold: float = 0.0  # 0 = use top-k, >0 = absolute threshold
    use_cumulative: bool = True     # accumulate absorption across steps

    # ---- compression method ----
    method: str = "evict"           # "evict" | "merge" | "hybrid"
    # evict: remove lowest-absorption visual KV entries
    # merge: merge similar adjacent visual tokens
    # hybrid: merge first, then evict

    # ---- model ----
    image_token_length: int = 576   # LLaVA-1.5 default

    # ---- generation ----
    max_new_tokens: int = 512       # longer for reasoning tasks
    do_sample: bool = False
    temperature: float = 0.0

    def validate(self):
        assert self.check_interval >= 8, "check_interval too small"
        assert 0 < self.compress_ratio < 1
        assert self.min_visual_tokens >= 16
        assert self.method in ("evict", "merge", "hybrid")
        assert self.ranking_layer >= 1
