"""
MTVC utilities — spatial indexing, visualization, and diagnostics.
"""

import torch


def token_index_to_spatial(index: int, grid_w: int = 24) -> tuple:
    """Convert flat token index to (row, col) in the image grid."""
    return index // grid_w, index % grid_w


def spatial_to_token_index(row: int, col: int, grid_w: int = 24) -> int:
    """Convert (row, col) to flat token index."""
    return row * grid_w + col


def visualize_cache_coverage(cache, config, save_path: str = "results/cache_coverage.png"):
    """
    Visualize which spatial positions are in L2 vs L3.
    Produces a 24x24 heatmap: L2 tokens in red, L3 in blue.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib required for visualization")
        return

    grid = np.zeros((config.image_grid_h, config.image_grid_w))

    for idx in cache.l2_indices.cpu().numpy():
        r, c = token_index_to_spatial(idx, config.image_grid_w)
        grid[r, c] = 2  # L2

    for idx in cache.l3_indices.cpu().numpy():
        r, c = token_index_to_spatial(idx, config.image_grid_w)
        grid[r, c] = 1  # L3

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cmap = plt.cm.colors.ListedColormap(['white', '#4A90D9', '#D94A4A'])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2)
    ax.set_title(f"Cache Coverage: L2 (red, {config.l2_size}) / L3 (blue, {config.l3_size})")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Cache coverage saved to {save_path}")


def print_session_summary(summary: dict):
    """Pretty-print session summary from MultiTurnEngine.get_summary()."""
    print("\n" + "=" * 60)
    print("  MTVC Session Summary")
    print("=" * 60)
    print(f"  Rounds:             {summary['num_rounds']}")
    print(f"  Total image tokens: {summary['total_image_tokens']}")
    print(f"  Vanilla equivalent: {summary['vanilla_equivalent']}")
    print(f"  Token savings:      {summary['savings_pct']:.1f}%")
    print(f"  Total time:         {summary['total_time']:.3f}s")
    print()
    print(f"  {'Round':<8}{'Img Tokens':<14}{'Seq Len':<12}{'Time':<10}")
    print(f"  {'-'*42}")
    for r in summary['per_round']:
        print(f"  {r['round']:<8}{r['image_tokens']:<14}{r['seq_len']:<12}{r['time']:<10}")
    print("=" * 60)


def compare_methods_table(vanilla_tokens: int, fastv_tokens: int, mtvc_tokens: int,
                          num_rounds: int):
    """Print comparison table of token usage across methods."""
    print(f"\n  {'Method':<12}{'Per Round':<14}{'Total Img Tokens':<20}{'Savings':<10}")
    print(f"  {'-'*54}")
    print(f"  {'Vanilla':<12}{vanilla_tokens // num_rounds:<14}{vanilla_tokens:<20}{'—':<10}")
    fastv_pct = (1 - fastv_tokens / vanilla_tokens) * 100
    print(f"  {'FastV':<12}{fastv_tokens // num_rounds:<14}{fastv_tokens:<20}"
          f"{fastv_pct:.0f}%")
    mtvc_pct = (1 - mtvc_tokens / vanilla_tokens) * 100
    print(f"  {'MTVC':<12}{'576+150x' + str(num_rounds - 1):<14}{mtvc_tokens:<20}"
          f"{mtvc_pct:.0f}%")
