#!/usr/bin/env python3

import sys
from pathlib import Path

import jax.numpy as jnp
import immrax as irx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from go2_separating_input import (
    create_scenarios,
    SeparatingInputOptimizer,
    optimize_multistart,
)

def main():
    scenarios = create_scenarios()
    x0_ivl = irx.Interval(
        lower=jnp.array([-0.05, -0.05, -0.02]),
        upper=jnp.array([ 0.05,  0.05,  0.02]),
    )
    dt = 0.5
    num_steps = 10

    opt = SeparatingInputOptimizer(scenarios, x0_ivl, dt, num_steps)
    print("Computing optimal input (this may take a moment)...")
    u_opt, loss_opt, stats_opt = optimize_multistart(opt, num_restarts=200)
    print(f"Optimal loss = {loss_opt:.6f}, u_opt = {u_opt}")

    candidates = [
        ("Optimal u*",          u_opt),
        ("Forward only",        jnp.array([0.5, 0.0, 0.0])),
        ("Forward + turn",      jnp.array([0.5, 0.0, 0.3])),
        ("Circle (no lateral)", jnp.array([0.4, 0.0, 0.5])),
        ("Max vy + turn",       jnp.array([0.0, 0.6, 0.6])),
    ]

    results = []
    stats_list = []
    print(f"{'Input':<28} {'vx':>6} {'vy':>6} {'omega':>6}  {'Loss (m^2)':>12}")
    print("-" * 65)
    for name, u in candidates:
        loss = float(opt.loss_fn(u))
        stats = opt.evaluate(u)
        results.append((name, u, loss))
        stats_list.append(stats)
        print(f"{name:<28} {float(u[0]):>6.3f} {float(u[1]):>6.3f} {float(u[2]):>6.3f}  {loss:>12.6f}")

    names  = [r[0] for r in results]
    losses = [r[2] for r in results]
    bar_c = ['#2ecc71'] + ['#95a5a6'] * (len(results) - 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, losses, color=bar_c, alpha=0.85,
                  edgecolor='black', linewidth=1.5)
    bars[0].set_linewidth(3)
    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{val:.5f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Total overlap (m^2)', fontsize=12, fontweight='bold')
    ax.set_title('Separation Loss for Different Control Inputs\n'
                 '(green = optimal; lower is better)',
                 fontsize=12, fontweight='bold')
    ax.set_xticklabels(names, rotation=12, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(HERE / 'go2_baseline_comparison.pdf', dpi=150, bbox_inches='tight')
    print('Saved go2_baseline_comparison.pdf')

    colors      = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    edge_colors = ['#2980b9', '#c0392b', '#27ae60', '#8e44ad']

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (name, u, loss), stats in zip(axes, results, stats_list):
        ax.add_patch(Rectangle(
            (float(x0_ivl.lower[0]), float(x0_ivl.lower[1])),
            float(x0_ivl.upper[0] - x0_ivl.lower[0]),
            float(x0_ivl.upper[1] - x0_ivl.lower[1]),
            linewidth=1, edgecolor='black', facecolor='gold', alpha=0.3,
            linestyle='--', label='Initial set',
        ))
        for i, (s, iv) in enumerate(zip(scenarios, stats['position_intervals'])):
            w = float(iv.upper[0] - iv.lower[0])
            h = float(iv.upper[1] - iv.lower[1])
            ax.add_patch(Rectangle(
                (float(iv.lower[0]), float(iv.lower[1])), w, h,
                linewidth=2, edgecolor=edge_colors[i], facecolor=colors[i],
                alpha=0.4, label=s.name,
            ))
        ax.set_title(f"{name}\nu={tuple(round(float(x), 3) for x in u)}\nLoss={loss:.4f}")
        ax.set_xlabel('px (m)')
        ax.set_ylabel('py (m)')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(HERE / 'go2_baseline_intervals.pdf', dpi=150, bbox_inches='tight')
    print('Saved go2_baseline_intervals.pdf')

if __name__ == '__main__':
    main()
