"""
ReVoC utilities — visualization, diagnostics, summary printing.
"""

import torch
import os


def print_session_summary(summary: dict):
    """Pretty-print session summary from RevoCEngine.get_summary()."""
    print(f"\n{'=' * 70}")
    print(f"  ReVoC Session Summary")
    print(f"{'=' * 70}")
    print(f"  Rounds:             {summary['num_rounds']}")
    print(f"  Total image tokens: {summary['total_image_tokens']}")
    print(f"  Vanilla equivalent: {summary['vanilla_equivalent']}")
    print(f"  Token savings:      {summary['savings_pct']:.1f}%")
    print(f"  Total time:         {summary['total_time']:.3f}s")

    ci = summary['compression_info']
    print(f"  Retriever:          {ci['retriever_type']}")
    print(f"  Clusters:           {ci['n_clusters']} total, "
          f"{ci['n_retrieve_clusters']} unpacked/round")
    if summary.get('adaptive_recovery'):
        print(f"  Adaptive recovery:  ON")
        print(f"  Rounds confident:   {summary.get('rounds_centers_only', 0)} "
              f"(centers only, ~{ci['n_clusters'] + 32} tokens)")
        print(f"  Rounds recovered:   {summary.get('rounds_with_recovery', 0)} "
              f"(full cluster unpack)")
    print()
    print(f"  {'Round':<7}{'Img Tok':<10}{'Clusters':<10}{'Recovered':<12}"
          f"{'Seq Len':<10}{'Time':<8}")
    print(f"  {'-' * 55}")
    for r in summary['per_round']:
        print(f"  {r['round']:<7}{r['image_tokens']:<10}{r['clusters_unpacked']:<10}"
              f"{r['tokens_recovered']:<12}{r['seq_len']:<10}{r['time']:<8}")
    print(f"{'=' * 70}")


def compare_methods_table(results: dict, num_rounds: int):
    """
    Print comparison table across methods.

    results: dict of method_name → {'tokens': int, 'time': float}
    """
    print(f"\n  {'Method':<16}{'Total Tokens':<16}{'Savings':<12}{'Time (s)':<12}")
    print(f"  {'-' * 54}")
    vanilla_tok = results.get('Vanilla', {}).get('tokens', 1)
    for name, data in results.items():
        savings = (1 - data['tokens'] / vanilla_tok) * 100 if vanilla_tok > 0 else 0
        print(f"  {name:<16}{data['tokens']:<16}{savings:<12.1f}%{data['time']:<12.3f}")


def visualize_cluster_selection(
    cache,
    selected_clusters: list,
    config,
    query: str = "",
    save_path: str = "results/cluster_selection.png",
):
    """
    Visualize which spatial regions are selected for a given query.
    Shows a 24x24 heatmap with selected clusters highlighted.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib required for visualization")
        return

    gh, gw = config.image_grid_h, config.image_grid_w
    grid = np.zeros((gh, gw))

    # Map cluster assignments back to spatial positions
    assignments = cache.cluster_assignments
    for pos in range(config.image_token_length):
        c_id = assignments[pos].item()
        if c_id < 0:
            # Global token
            r, c = pos // gw, pos % gw
            grid[r, c] = 3  # global = highest
        elif c_id in selected_clusters:
            r, c = pos // gw, pos % gw
            grid[r, c] = 2  # selected cluster
        else:
            r, c = pos // gw, pos % gw
            grid[r, c] = 1  # compressed (not recovered)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cmap = plt.cm.colors.ListedColormap(['white', '#CCCCCC', '#4A90D9', '#D94A4A'])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=3)
    ax.set_title(f"ReVoC Selection: {len(selected_clusters)} clusters unpacked"
                 + (f"\nQuery: {query[:60]}" if query else ""))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D94A4A', label='Global (always used)'),
        Patch(facecolor='#4A90D9', label='Recovered (query-relevant)'),
        Patch(facecolor='#CCCCCC', label='Compressed (cluster center only)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {save_path}")


def visualize_entropy_distribution(
    entropy: torch.Tensor,
    importance: torch.Tensor,
    save_path: str = "results/entropy_dist.png",
):
    """
    Plot entropy vs importance scatter, showing the natural separation
    that motivates entropy-based partitioning.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib required")
        return

    H = entropy.cpu().numpy()
    I = importance.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Scatter: entropy vs importance
    axes[0].scatter(H, I, alpha=0.3, s=8)
    axes[0].set_xlabel("Attention Entropy")
    axes[0].set_ylabel("Attention Importance")
    axes[0].set_title("Entropy vs Importance")

    # Entropy histogram
    axes[1].hist(H, bins=50, alpha=0.7, color='#4A90D9')
    threshold = np.percentile(H, 90)
    axes[1].axvline(threshold, color='red', linestyle='--', label=f'90th pctl = {threshold:.3f}')
    axes[1].set_xlabel("Entropy")
    axes[1].set_title("Entropy Distribution")
    axes[1].legend()

    # Spatial heatmap of entropy
    grid_size = int(np.sqrt(len(H)))
    if grid_size * grid_size == len(H):
        grid = H.reshape(grid_size, grid_size)
    else:
        grid = H[:576].reshape(24, 24) if len(H) >= 576 else H.reshape(-1, 1)
    im = axes[2].imshow(grid, cmap='hot')
    axes[2].set_title("Spatial Entropy Map")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Entropy visualization saved to {save_path}")


def print_compression_bounds(bounds):
    """Print theoretical compression bounds."""
    print(f"\n  Theoretical Compression Bounds:")
    print(f"  {'─' * 45}")
    print(f"  Pruning bound:       {bounds.prune_bound:.4f}")
    print(f"  ReVoC bound:         {bounds.revoc_bound:.4f}")
    print(f"  Improvement ratio:   {bounds.improvement_ratio:.2f}x")
    print(f"  Avg residual norm:   {bounds.avg_residual_norm:.4f}")
    print(f"  Avg token norm:      {bounds.avg_token_norm:.4f}")
    print(f"  Max cluster var:     {bounds.max_cluster_variance:.4f}")
