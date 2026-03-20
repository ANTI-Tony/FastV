"""
Compression Schedulers — control when and how aggressively to compress.
"""

from .config import RPAConfig


class CompressionScheduler:
    """Base scheduler: compress every K tokens, keep ratio fraction."""

    def __init__(self, config: RPAConfig, initial_n: int):
        self.config = config
        self.initial_n = initial_n
        self.current_n = initial_n
        self.total_generated = 0
        self.compression_count = 0
        self.history = [(0, initial_n)]  # (step, n_visual) pairs

    def step(self, n_generated: int) -> bool:
        """Called after each token. Returns True if should compress now."""
        self.total_generated = n_generated
        if n_generated < self.config.warmup_tokens:
            return False
        if self.current_n <= self.config.min_visual_tokens:
            return False
        if n_generated % self.config.check_interval == 0 and n_generated > 0:
            return True
        return False

    def get_target_n(self) -> int:
        """How many visual tokens to keep after this compression."""
        target = int(self.current_n * self.config.compress_ratio)
        target = max(target, self.config.min_visual_tokens)
        return target

    def after_compress(self, new_n: int):
        """Update state after compression."""
        self.current_n = new_n
        self.compression_count += 1
        self.history.append((self.total_generated, new_n))

    def get_history(self):
        """Return the abstraction curve: [(step, n_visual), ...]"""
        return self.history

    def get_summary(self):
        return {
            'initial': self.initial_n,
            'final': self.current_n,
            'compressions': self.compression_count,
            'total_generated': self.total_generated,
            'reduction_pct': (1 - self.current_n / self.initial_n) * 100,
            'history': self.history,
        }
