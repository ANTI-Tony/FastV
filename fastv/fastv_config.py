"""FastV 配置类"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FastVConfig:
    """
    FastV 核心配置

    Args:
        use_fastv: 是否启用 FastV
        fastv_k: 在第 K 层进行 token 剪枝 (论文推荐 K=2)
        fastv_r: 剪掉的 image token 比例 (0.5 = 剪掉50%, 0.75 = 剪掉75%)
        image_token_start_index: image token 在序列中的起始位置
        image_token_length: image token 的数量 (LLaVA-1.5 默认 576)
    """
    use_fastv: bool = True
    fastv_k: int = 2
    fastv_r: float = 0.75
    image_token_start_index: Optional[int] = None  # 自动检测
    image_token_length: int = 576

    def __post_init__(self):
        assert 0 < self.fastv_r < 1, f"fastv_r 必须在 (0, 1) 之间, 当前值: {self.fastv_r}"
        assert self.fastv_k >= 1, f"fastv_k 必须 >= 1, 当前值: {self.fastv_k}"

    @property
    def num_tokens_to_keep(self) -> int:
        """保留的 image token 数量"""
        return int(self.image_token_length * (1 - self.fastv_r))

    @property
    def num_tokens_to_drop(self) -> int:
        """丢弃的 image token 数量"""
        return self.image_token_length - self.num_tokens_to_keep
