"""
注意力可视化工具

用于可视化 image tokens 在各层的注意力分布，
验证 FastV 的核心观察：第 2 层之后大量 image tokens 注意力趋近于零
"""

import torch
import numpy as np
import os
from typing import Optional


def visualize_attention(
    attention_weights_per_layer: list,
    image_start: int,
    image_length: int,
    save_path: str = "results/attention_heatmap.png",
    title: str = "Image Token Attention Across Layers",
):
    """
    可视化各层对 image tokens 的注意力分布

    Args:
        attention_weights_per_layer: list of tensors, 每层的注意力权重
            shape: (batch, num_heads, seq_len, seq_len)
        image_start: image tokens 起始位置
        image_length: image tokens 数量
        save_path: 保存路径
        title: 图标题
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("需要安装 matplotlib 和 seaborn: pip install matplotlib seaborn")
        return

    num_layers = len(attention_weights_per_layer)

    # 计算每层中，最后一个 token 对 image tokens 的平均注意力
    layer_avg_attn = []
    for layer_attn in attention_weights_per_layer:
        # 取最后一个 token 对 image tokens 的注意力
        img_attn = layer_attn[0, :, -1, image_start:image_start + image_length]
        # 对 heads 取平均
        avg_attn = img_attn.mean(dim=0).cpu().numpy()
        layer_avg_attn.append(avg_attn)

    attn_matrix = np.stack(layer_avg_attn)  # (num_layers, image_length)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. 热力图
    sns.heatmap(
        attn_matrix,
        ax=axes[0],
        cmap='viridis',
        xticklabels=False,
        yticklabels=range(num_layers),
    )
    axes[0].set_title('Attention Heatmap (Layer x Image Token)')
    axes[0].set_xlabel('Image Token Index')
    axes[0].set_ylabel('Layer')

    # 2. 每层平均注意力
    avg_per_layer = attn_matrix.mean(axis=1)
    axes[1].bar(range(num_layers), avg_per_layer, color='steelblue')
    axes[1].set_title('Average Image Token Attention per Layer')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Average Attention')
    axes[1].axvline(x=1.5, color='red', linestyle='--', label='FastV K=2')
    axes[1].legend()

    # 3. "低注意力" token 比例
    threshold = 0.001
    low_attn_ratio = (attn_matrix < threshold).sum(axis=1) / image_length
    axes[2].plot(range(num_layers), low_attn_ratio, 'o-', color='coral')
    axes[2].set_title(f'Ratio of "Dead" Tokens (attn < {threshold})')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Ratio')
    axes[2].set_ylim(0, 1)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"注意力可视化已保存到: {save_path}")


def print_attention_stats(
    attention_weights_per_layer: list,
    image_start: int,
    image_length: int,
):
    """打印各层 image token 注意力统计信息"""
    print(f"\n{'='*60}")
    print(f"  Image Token Attention Statistics")
    print(f"  Image range: [{image_start}, {image_start + image_length})")
    print(f"{'='*60}")
    print(f"{'Layer':>6} | {'Mean Attn':>10} | {'Max Attn':>10} | {'Dead%':>8} | {'Top10% Attn':>12}")
    print(f"{'-'*60}")

    for i, layer_attn in enumerate(attention_weights_per_layer):
        img_attn = layer_attn[0, :, -1, image_start:image_start + image_length]
        avg_attn = img_attn.mean(dim=0)

        mean_val = avg_attn.mean().item()
        max_val = avg_attn.max().item()
        dead_ratio = (avg_attn < 0.001).float().mean().item() * 100
        top10_val = avg_attn.topk(max(1, image_length // 10)).values.mean().item()

        print(f"{i:>6} | {mean_val:>10.6f} | {max_val:>10.6f} | {dead_ratio:>7.1f}% | {top10_val:>12.6f}")

    print(f"{'='*60}\n")
