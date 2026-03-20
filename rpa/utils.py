"""
RPA utilities — visualization, summary printing.
"""

import os


def print_generation_result(result, label: str = ""):
    """Pretty-print RPA generation result."""
    print(f"\n  {'─' * 55}")
    if label:
        print(f"  {label}")
    print(f"  Generated:    {result.total_generated} tokens")
    print(f"  Visual:       {result.initial_visual_tokens} → {result.final_visual_tokens} "
          f"({(1 - result.final_visual_tokens / result.initial_visual_tokens) * 100:.0f}% compressed)")
    print(f"  Compressions: {result.compressions}")
    print(f"  Time:         {result.elapsed:.2f}s")
    print(f"  Response:     {result.response[:300]}")
    print(f"  {'─' * 55}")


def print_abstraction_curve(result):
    """Print the abstraction curve as ASCII."""
    curve = result.abstraction_curve
    if not curve:
        return

    max_n = curve[0][1]
    width = 40

    print(f"\n  Abstraction Curve:")
    print(f"  {'Step':<8}{'Tokens':<10}{'Bar'}")
    for step, n in curve:
        bar_len = int(n / max_n * width)
        bar = '█' * bar_len
        print(f"  {step:<8}{n:<10}{bar}")


def plot_abstraction_curve(results: list, labels: list = None,
                           save_path: str = "results/abstraction_curve.png"):
    """Plot abstraction curves for multiple experiments."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, result in enumerate(results):
        curve = result.abstraction_curve
        steps = [s for s, _ in curve]
        tokens = [n for _, n in curve]
        label = labels[i] if labels else f"Run {i}"
        ax.plot(steps, tokens, 'o-', label=label, linewidth=2)

    ax.set_xlabel("Generated Tokens (Reasoning Steps)")
    ax.set_ylabel("Visual Tokens Remaining")
    ax.set_title("Progressive Visual Abstraction Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add reference lines
    if results:
        initial = results[0].initial_visual_tokens
        ax.axhline(y=initial, color='gray', linestyle='--', alpha=0.5, label='Vanilla (no compression)')
        ax.axhline(y=144, color='red', linestyle='--', alpha=0.5, label='FastV R=75%')

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Curve saved to {save_path}")


def compare_methods_table(vanilla_result, fastv_result, rpa_result):
    """Print comparison table."""
    print(f"\n  {'Method':<16}{'Tokens(start→end)':<22}{'Generated':<12}{'Time(s)':<10}")
    print(f"  {'─' * 58}")

    if vanilla_result:
        print(f"  {'Vanilla':<16}{'576→576':<22}"
              f"{vanilla_result.total_generated:<12}{vanilla_result.elapsed:<10.2f}")

    if fastv_result:
        print(f"  {'FastV R=75%':<16}{'144→144':<22}"
              f"{fastv_result.total_generated:<12}{fastv_result.elapsed:<10.2f}")

    if rpa_result:
        tok_str = f"{rpa_result.initial_visual_tokens}→{rpa_result.final_visual_tokens}"
        print(f"  {'RPA':<16}{tok_str:<22}"
              f"{rpa_result.total_generated:<12}{rpa_result.elapsed:<10.2f}")
