import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jax
import jax.numpy as jnp
import immrax as irx

from quadrotor_separating_input import (
    create_scenarios, _propagate_history, _signed_separation_margin, QuadrotorSystem,
)
from solve_and_plot import (
    _draw_box3d, SCENARIO_COLORS, RESULTS_DIR, ALPHA_TRUE_FAULT,
)

NPZ_PATH = RESULTS_DIR / "quadrotor_separating_input.npz"
STATE_NAMES = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'vx', 'vy', 'vz', 'p', 'q', 'r']

def euler_point_rollout(x0, u_seq, alpha, dt):
    sys_ = QuadrotorSystem()

    def f(x, u):
        return sys_.f(jnp.zeros(()), x, u, alpha)

    xs = [x0]
    x = x0
    for k in range(u_seq.shape[0]):
        x = x + dt * f(x, u_seq[k])
        xs.append(x)
    return jnp.stack(xs)

def main():
    d = np.load(NPZ_PATH)
    u_seq = jnp.array(d["u_seq"])
    dt = float(d["dt"])
    x0_width = float(d["x0_width"])
    alpha_lo, alpha_hi = float(d["alpha_lo"]), float(d["alpha_hi"])
    disjoint_segments = [int(k) for k in d["disjoint_segments"]]
    if not disjoint_segments:
        raise RuntimeError(
            f"{NPZ_PATH} has no genuinely-disjoint segments (disjoint_segments "
            "is empty) -- nothing to plot. Re-run solve_and_plot.py first.")

    scenarios = create_scenarios(alpha_lo=alpha_lo, alpha_hi=alpha_hi)
    x0_ivl = irx.icentpert(jnp.zeros(12), jnp.full(12, x0_width))

    _emb_sys = scenarios[0].emb_system
    _p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )
    _hist_batch = jax.vmap(lambda p: _propagate_history(x0_ivl, u_seq, _emb_sys, p, dt, 1))(_p_batch)
    hist_by_name = {s.name: irx.Interval(lower=_hist_batch.lower[i], upper=_hist_batch.upper[i])
                    for i, s in enumerate(scenarios)}

    best_k, best_margin = None, -float("inf")
    per_pair_dim_at_best = {}
    for k in disjoint_segments:
        worst_margin_k = float("inf")
        pair_dims_k = {}
        for i in range(len(scenarios)):
            for j in range(i + 1, len(scenarios)):
                ivl_i = irx.Interval(lower=hist_by_name[scenarios[i].name].lower[k],
                                     upper=hist_by_name[scenarios[i].name].upper[k])
                ivl_j = irx.Interval(lower=hist_by_name[scenarios[j].name].lower[k],
                                     upper=hist_by_name[scenarios[j].name].upper[k])
                margins = np.array(_signed_separation_margin(ivl_i, ivl_j))
                m = float(np.max(margins))
                pair_dims_k[(scenarios[i].name, scenarios[j].name)] = STATE_NAMES[int(np.argmax(margins))]
                worst_margin_k = min(worst_margin_k, m)
        if worst_margin_k > best_margin:
            best_margin = worst_margin_k
            best_k = k
            per_pair_dim_at_best = pair_dims_k
    print(f"Best disjoint segment: {best_k} (t={best_k*dt:.2f}s), "
         f"worst-pairwise margin={best_margin:+.5f}")
    dims_used = sorted(set(per_pair_dim_at_best.values()))
    print(f"Dimensions actually carrying a pairwise separation: {dims_used}")
    for pair, dim in per_pair_dim_at_best.items():
        print(f"  {pair[0]:16s} vs {pair[1]:16s}  -> {dim}")

    panel_dims = {"vx", "vy", "vz", "p", "q", "r"}
    uncovered = set(dims_used) - panel_dims
    if uncovered:
        raise RuntimeError(
            f"Separating dimension(s) {uncovered} at segment {best_k} are not covered by "
            f"the velocity/body-rate panels ({panel_dims}) -- update this script's panel "
            "choice before trusting the figure.")

    box_lo_vel, box_hi_vel = {}, {}
    box_lo_rate, box_hi_rate = {}, {}
    point_traj_vel, point_traj_rate = {}, {}
    for s in scenarios:
        hist = hist_by_name[s.name]
        box_lo_vel[s.name] = np.array(hist.lower[best_k])[6:9]
        box_hi_vel[s.name] = np.array(hist.upper[best_k])[6:9]
        box_lo_rate[s.name] = np.array(hist.lower[best_k])[9:12]
        box_hi_rate[s.name] = np.array(hist.upper[best_k])[9:12]
        alpha_point = jnp.ones(4) if s.name == "Nominal" else (
            jnp.ones(4).at[int(s.name[-1]) - 1].set(ALPHA_TRUE_FAULT))
        traj = euler_point_rollout(jnp.zeros(12), u_seq[:best_k + 1], alpha_point, dt)
        point_traj_vel[s.name] = np.array(traj)[:, 6:9]
        point_traj_rate[s.name] = np.array(traj)[:, 9:12]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": 11,
    })

    fig = plt.figure(figsize=(12.5, 6.0))
    ax_vel = fig.add_subplot(121, projection="3d")
    ax_rate = fig.add_subplot(122, projection="3d")

    order_vel = sorted(scenarios, key=lambda s: -float(np.sum(box_hi_vel[s.name] - box_lo_vel[s.name])))
    order_rate = sorted(scenarios, key=lambda s: -float(np.sum(box_hi_rate[s.name] - box_lo_rate[s.name])))

    for s in order_vel:
        color = SCENARIO_COLORS[s.name]
        vel = point_traj_vel[s.name]
        ax_vel.plot(vel[:, 0], vel[:, 1], vel[:, 2], color=color, linewidth=1.4,
                   alpha=0.6, zorder=4)
        ax_vel.scatter(*vel[-1], color=color, s=14, zorder=6)
        _draw_box3d(ax_vel, box_lo_vel[s.name], box_hi_vel[s.name], color, alpha=0.9, linewidth=1.8)

    for s in order_rate:
        color = SCENARIO_COLORS[s.name]
        rate = point_traj_rate[s.name]
        ax_rate.plot(rate[:, 0], rate[:, 1], rate[:, 2], color=color, linewidth=1.4,
                    alpha=0.6, zorder=4)
        ax_rate.scatter(*rate[-1], color=color, s=14, zorder=6)
        _draw_box3d(ax_rate, box_lo_rate[s.name], box_hi_rate[s.name], color, alpha=0.9, linewidth=1.8)

    legend_handles = [Line2D([0], [0], color=SCENARIO_COLORS[s.name], lw=2, label=s.name)
                      for s in scenarios]
    fig.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.90, 0.85),
              frameon=False, fontsize=8)

    ax_vel.set_xlabel("$v_x$ (m/s)", labelpad=8)
    ax_vel.set_ylabel("$v_y$ (m/s)", labelpad=8)
    ax_vel.set_zlabel("$v_z$ (m/s)", labelpad=10)

    ax_rate.set_xlabel("$p$ (rad/s)", labelpad=8)
    ax_rate.set_ylabel("$q$ (rad/s)", labelpad=8)
    ax_rate.set_zlabel("$r$ (rad/s)", labelpad=10)

    fig.subplots_adjust(left=0.02, right=0.86, bottom=0.05, top=0.98, wspace=0.25)

    out_pdf = RESULTS_DIR / "quadrotor_separation_moment.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote {out_pdf}")

if __name__ == "__main__":
    main()
