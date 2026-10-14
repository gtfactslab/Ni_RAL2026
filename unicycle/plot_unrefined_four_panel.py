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

from car_separating_input import (
    create_scenarios,
    observed_output,
    _propagate_history,
    MultistepSequenceOptimizer,
    optimize_multistep_gpu,
)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': False,
})

scenarios = create_scenarios(actuator_alpha_lo=0.0, actuator_alpha_hi=0.5)
x0_ivl = irx.icentpert(jnp.array([0.1, 0.1, 0.0]), jnp.array([0.1, 0.1, 0.1]))
dt = 0.1
steps_per_segment = 1
num_segments = 50

print("Optimising non-refined separating input...")
opt = MultistepSequenceOptimizer(
    scenarios=scenarios,
    x0_ivl=x0_ivl,
    dt=dt,
    steps_per_segment=steps_per_segment,
    num_segments=num_segments,
)
u_seq_opt, loss_opt, _, _ = optimize_multistep_gpu(
    opt,
    num_restarts=50,
    learning_rate=0.05,
    num_iters=200,
    seed=42,
)
print(f"  loss = {loss_opt:.6f}, sequence shape = {u_seq_opt.shape}")

def collect_output_history_multistep(x0, u_seq, scenario, dt, steps_per_segment):
    hist = _propagate_history(x0, u_seq, scenario.emb_system, scenario.p_interval,
                              dt, steps_per_segment)
    ivls = []
    for k in range(hist.lower.shape[0]):
        x_iv = irx.Interval(lower=hist.lower[k], upper=hist.upper[k])
        ivls.append(observed_output(x_iv, scenario))
    return ivls

output_histories = [
    collect_output_history_multistep(x0_ivl, u_seq_opt, s, dt, steps_per_segment)
    for s in scenarios
]
for k, h in enumerate(output_histories):
    print(f"history for scenario {k} has {len(h)} steps")
step_indices = np.arange(0, num_segments, 1)
times = [k * dt for k in step_indices]

cmap = plt.colormaps['tab10'].resampled(len(scenarios))

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
                        color='black', linewidth=2.0, zorder=5)

pairs = [(0, 1), (0, 2), (1, 2)]
pair_titles = ['Nominal+Actuator', 'Nominal+Sensor', 'Actuator+Sensor']

def draw_pair_plot(ax, i, j, pair_title):
    all_px, all_py = [], []
    for step_idx, t in zip(step_indices, times):
        iv_i = output_histories[i][step_idx]
        iv_j = output_histories[j][step_idx]
        for iv, color in ((iv_i, cmap(i)), (iv_j, cmap(j))):
            xl, xh = float(iv.lower[0]), float(iv.upper[0])
            yl, yh = float(iv.lower[1]), float(iv.upper[1])
            all_px += [xl, xh]; all_py += [yl, yh]
            verts = [[(xl, t, yl), (xh, t, yl), (xh, t, yh), (xl, t, yh)]]
            poly = Poly3DCollection(verts, alpha=0.35)
            poly.set_facecolor(color); poly.set_edgecolor(color)
            ax.add_collection3d(poly)
        xl = float(max(iv_i.lower[0], iv_j.lower[0]))
        xh = float(min(iv_i.upper[0], iv_j.upper[0]))
        yl = float(max(iv_i.lower[1], iv_j.lower[1]))
        yh = float(min(iv_i.upper[1], iv_j.upper[1]))
        if xh >= xl and yh >= yl:
            ax.plot([xl, xh, xh, xl, xl], [t] * 5, [yl, yl, yh, yh, yl],
                    color='black', linewidth=2.0, zorder=5)
    pad = 0.05
    ax.set_xlim(min(all_px) - pad, max(all_px) + pad)
    ax.set_ylim(times[0], times[-1])
    ax.set_zlim(min(all_py) - pad, max(all_py) + pad)
    ax.plot([], [], [], color=cmap(i), linewidth=3, label=scenarios[i].name)
    ax.plot([], [], [], color=cmap(j), linewidth=3, label=scenarios[j].name)
    ax.set_xlabel('$y_1 = p_x$ (m)')
    ax.set_ylabel('Time (s)')
    ax.set_zlabel('$y_2 = p_y$ (m)')
    ax.set_title(pair_title)
    ax.view_init(elev=15, azim=20, roll=0)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
all_px, all_py = [], []
for step_idx, t in zip(step_indices, times):
    ivs = [hist[step_idx] for hist in output_histories]
    for k_sc, iv in enumerate(ivs):
        color = cmap(k_sc)
        xl, xh = float(iv.lower[0]), float(iv.upper[0])
        yl, yh = float(iv.lower[1]), float(iv.upper[1])
        all_px += [xl, xh]; all_py += [yl, yh]
        verts = [[(xl, t, yl), (xh, t, yl), (xh, t, yh), (xl, t, yh)]]
        poly = Poly3DCollection(verts, alpha=0.35)
        poly.set_facecolor(color); poly.set_edgecolor(color)
        ax.add_collection3d(poly)
    draw_pairwise_overlaps(ax, ivs, t)
pad = 0.05
ax.set_xlim(min(all_px) - pad, max(all_px) + pad)
ax.set_ylim(times[0], times[-1])
ax.set_zlim(min(all_py) - pad, max(all_py) + pad)
for k_sc, s in enumerate(scenarios):
    ax.plot([], [], [], color=cmap(k_sc), linewidth=3, label=s.name)
ax.set_xlabel('$y_1 = p_x$ (m)')
ax.set_ylabel('Time (s)')
ax.set_zlabel('$y_2 = p_y$ (m)')
ax.set_title('All three scenarios - unrefined')
ax.view_init(elev=15, azim=20, roll=0)
plt.tight_layout()
plt.savefig(HERE / 'unrefined_all.pdf', bbox_inches='tight')
plt.close(fig)

for idx, (i, j) in enumerate(pairs, start=1):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    draw_pair_plot(ax, i, j, f'Unrefined pair: {pair_titles[idx - 1]}')
    plt.tight_layout()
    fname = HERE / f'unrefined_pair_{pair_titles[idx - 1].replace("+", "_")}.pdf'
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)

fig = plt.figure(figsize=(20, 5))
axes = [fig.add_subplot(1, 4, k + 1, projection='3d') for k in range(4)]
all_px, all_py = [], []
for step_idx, t in zip(step_indices, times):
    ivs = [hist[step_idx] for hist in output_histories]
    for k_sc, iv in enumerate(ivs):
        color = cmap(k_sc)
        xl, xh = float(iv.lower[0]), float(iv.upper[0])
        yl, yh = float(iv.lower[1]), float(iv.upper[1])
        all_px += [xl, xh]; all_py += [yl, yh]
        verts = [[(xl, t, yl), (xh, t, yl), (xh, t, yh), (xl, t, yh)]]
        poly = Poly3DCollection(verts, alpha=0.35)
        poly.set_facecolor(color); poly.set_edgecolor(color)
        axes[0].add_collection3d(poly)
    draw_pairwise_overlaps(axes[0], ivs, t)
pad = 0.05
axes[0].set_xlim(min(all_px) - pad, max(all_px) + pad)
axes[0].set_ylim(times[0], times[-1])
axes[0].set_zlim(min(all_py) - pad, max(all_py) + pad)
scenario_handles = []
for k_sc, s in enumerate(scenarios):
    h, = axes[0].plot([], [], [], color=cmap(k_sc), linewidth=3, label=s.name)
    scenario_handles.append(h)
axes[0].set_xlabel('$y_1 = p_x$ (m)')
axes[0].set_ylabel('Time (s)')
axes[0].set_zlabel('$y_2 = p_y$ (m)')
axes[0].set_title('All three scenarios - unrefined')
axes[0].view_init(elev=15, azim=20, roll=0)

for idx, (i, j) in enumerate(pairs, start=1):
    draw_pair_plot(axes[idx], i, j, pair_titles[idx - 1])

int_h = plt.Line2D([0], [0], color='black', linewidth=2, label='Intersection')
fig.legend(handles=scenario_handles + [int_h], loc='upper center', ncol=4)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(HERE / f'unrefined_four_panel_dt{dt}.pdf', bbox_inches='tight')
plt.close(fig)

print("Plots written to disk.")
