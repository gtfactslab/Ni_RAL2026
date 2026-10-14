import sys
from pathlib import Path

import jax.numpy as jnp
import immrax as irx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from go2_separating_input import (
    create_scenarios,
    _propagate_history,
    position_interval,
    optimize_multistep,
)

def collect_position_history(x0_ivl, u_seq, scenario, dt, steps_per_segment):
    hist = _propagate_history(
        x0_ivl, u_seq, scenario.emb_system, scenario.p_interval, dt, steps_per_segment
    )
    ivls = []
    for k in range(hist.lower.shape[0]):
        x_iv = irx.Interval(lower=hist.lower[k], upper=hist.upper[k])
        ivls.append(position_interval(x_iv))
    return ivls

def draw_pairwise_overlaps(ax, ivs, t):
    n = len(ivs)
    for i in range(n):
        for j in range(i + 1, n):
            iv1, iv2 = ivs[i], ivs[j]
            xl = float(max(iv1.lower[0], iv2.lower[0]))
            xh = float(min(iv1.upper[0], iv2.upper[0]))
            yl = float(max(iv1.lower[1], iv2.lower[1]))
            yh = float(min(iv1.upper[1], iv2.upper[1]))
            if xh >= xl and yh >= yl:
                ax.plot([xl, xh, xh, xl, xl], [t] * 5, [yl, yl, yh, yh, yl],
                        color='black', linewidth=1.5, zorder=5)

def main():
    scenarios = create_scenarios()
    x0_ivl = irx.Interval(
        lower=jnp.array([-0.05, -0.05, -0.02]),
        upper=jnp.array([ 0.05,  0.05,  0.02]),
    )
    dt = 0.3
    steps_per_segment = 1
    num_segments = 12

    print("Optimising multistep separating input...")
    u_seq_opt, loss_opt, _ = optimize_multistep(
        scenarios, x0_ivl, dt=dt, steps_per_segment=steps_per_segment,
        num_segments=num_segments, learning_rate=0.01, num_iters=150,
        num_restarts=50, seed=42,
    )
    print(f"  loss = {loss_opt:.6f}, sequence shape = {u_seq_opt.shape}")

    histories = [
        collect_position_history(x0_ivl, u_seq_opt, s, dt, steps_per_segment)
        for s in scenarios
    ]
    step_indices = np.arange(num_segments)
    times = [k * dt for k in step_indices]

    cmap = plt.colormaps['tab10'].resampled(len(scenarios))

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    all_px, all_py = [], []
    for step_idx, t in zip(step_indices, times):
        ivs = [hist[step_idx] for hist in histories]
        for k_sc, iv in enumerate(ivs):
            color = cmap(k_sc)
            xl, xh = float(iv.lower[0]), float(iv.upper[0])
            yl, yh = float(iv.lower[1]), float(iv.upper[1])
            all_px += [xl, xh]; all_py += [yl, yh]
            verts = [[(xl, t, yl), (xh, t, yl), (xh, t, yh), (xl, t, yh)]]
            poly = Poly3DCollection(verts, alpha=0.3)
            poly.set_facecolor(color); poly.set_edgecolor(color)
            ax.add_collection3d(poly)
        draw_pairwise_overlaps(ax, ivs, t)

    pad = 0.05
    ax.set_xlim(min(all_px) - pad, max(all_px) + pad)
    ax.set_ylim(times[0], times[-1])
    ax.set_zlim(min(all_py) - pad, max(all_py) + pad)
    for k_sc, s in enumerate(scenarios):
        ax.plot([], [], [], color=cmap(k_sc), linewidth=3, label=s.name)
    ax.set_xlabel('$p_x$ (m)')
    ax.set_ylabel('Time (s)')
    ax.set_zlabel('$p_y$ (m)')
    ax.set_title('Go2: position-reachable-set history (unrefined multistep)')
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=15, azim=20, roll=0)
    plt.tight_layout()
    out = HERE / 'go2_position_intervals_3d.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
