import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import jax
import jax.numpy as jnp
import immrax as irx

from quadrotor_separating_input import (
    create_scenarios, optimize_multistep_gpu_rejit, time_jit,
    QuadrotorSystem, _overlap_volume, _signed_separation_margin,
)

RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DT = 0.1
NUM_SEGMENTS = 5
MARGIN_TAU = 0.01
NUM_RESTARTS = 40
GD_LEARNING_RATE = 5e-6
GD_NUM_ITERS = 50
X0_WIDTH = 0.01
ALPHA_LO, ALPHA_HI = 0.5, 0.9
ALPHA_TRUE_FAULT = 0.7

SCENARIO_COLORS = {
    "Nominal": "tab:blue", "ActuatorFault_1": "tab:orange",
    "ActuatorFault_2": "tab:green", "ActuatorFault_3": "tab:red",
    "ActuatorFault_4": "tab:purple",
}

_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
         (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]

def _draw_box3d(ax, lo3, hi3, color, alpha=0.55, linewidth=1.1):
    corners = [(x, y, z) for x in (lo3[0], hi3[0])
              for y in (lo3[1], hi3[1]) for z in (lo3[2], hi3[2])]
    segments = [(corners[i], corners[j]) for i, j in _EDGES]
    ax.add_collection3d(Line3DCollection(segments, colors=color, linewidths=linewidth,
                                         alpha=alpha))

def rk4_rollout(x0, u_seq, alpha, dt, sub_steps=10):
    sys_ = QuadrotorSystem()
    h = dt / sub_steps

    def f(x, u):
        return sys_.f(jnp.zeros(()), x, u, alpha)

    def rk4_step(x, u):
        k1 = f(x, u)
        k2 = f(x + 0.5 * h * k1, u)
        k3 = f(x + 0.5 * h * k2, u)
        k4 = f(x + h * k3, u)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    xs = [x0]
    x = x0
    for k in range(u_seq.shape[0]):
        u = u_seq[k]
        for _ in range(sub_steps):
            x = rk4_step(x, u)
            xs.append(x)
    return jnp.stack(xs)

def main():
    print(f"Solving multistep-unrefined separating input: {NUM_SEGMENTS} segments x "
         f"dt={DT}s = {NUM_SEGMENTS*DT:.2f}s horizon, x0_width={X0_WIDTH}, "
         f"alpha in [{ALPHA_LO},{ALPHA_HI}], margin_tau={MARGIN_TAU}")
    scenarios = create_scenarios(alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI)
    x0_ivl = irx.icentpert(jnp.zeros(12), jnp.full(12, X0_WIDTH))

    def full_multistart(seed):
        return optimize_multistep_gpu_rejit(
            x0_ivl, scenarios, dt=DT, steps_per_segment=1, num_segments=NUM_SEGMENTS,
            num_restarts=NUM_RESTARTS, learning_rate=GD_LEARNING_RATE, num_iters=GD_NUM_ITERS,
            seed=seed, margin_tau=MARGIN_TAU, normalize_grad=True)

    jitted_fn, compile_t, run_t, mem = time_jit(full_multistart, 0)
    print(f"compile {compile_t*1e3:8.2f} ms   run {run_t*1e3:7.3f} ms")

    u_seq, loss, u_all, losses_all = jitted_fn(0)
    n_zero = int(jnp.sum(losses_all == 0.0))
    print(f"Best margin-loss: {float(loss):.6e}  ({n_zero}/{NUM_RESTARTS} restarts hit "
         f"loss=0.0, a certified-disjoint-at-some-segment solution -- see "
         f"_soft_separation_loss's 'Bias correction' docstring for why loss=0.0 is safe here)")

    NUM_STEPS = int(u_seq.shape[0])
    for k in range(NUM_STEPS):
        print(f"  u_seq[{k:2d}] = [U1={float(u_seq[k,0]):.5f} N, U2={float(u_seq[k,1]):.6e}, "
             f"U3={float(u_seq[k,2]):.6e}, U4={float(u_seq[k,3]):.6e}]")

    from quadrotor_separating_input import _propagate_history
    _emb_sys = scenarios[0].emb_system
    _p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )
    _hist_batch = jax.vmap(lambda p: _propagate_history(x0_ivl, u_seq, _emb_sys, p, DT, 1))(_p_batch)
    hist_by_name = {s.name: irx.Interval(lower=_hist_batch.lower[i], upper=_hist_batch.upper[i])
                    for i, s in enumerate(scenarios)}
    print("Per-segment pairwise separation (raw, independent per-scenario boxes):")
    disjoint_segments = []
    for k in range(NUM_STEPS):
        all_disjoint_k = True
        worst_margin_k = float("inf")
        for i in range(len(scenarios)):
            for j in range(i + 1, len(scenarios)):
                ivl_i = irx.Interval(lower=hist_by_name[scenarios[i].name].lower[k],
                                     upper=hist_by_name[scenarios[i].name].upper[k])
                ivl_j = irx.Interval(lower=hist_by_name[scenarios[j].name].lower[k],
                                     upper=hist_by_name[scenarios[j].name].upper[k])
                m = float(jnp.max(_signed_separation_margin(ivl_i, ivl_j)))
                all_disjoint_k &= (m >= 0.0)
                worst_margin_k = min(worst_margin_k, m)
        if all_disjoint_k:
            disjoint_segments.append(k)
        print(f"  segment {k:2d}: all pairs disjoint={all_disjoint_k!s:5s}  "
             f"worst_pair_margin={worst_margin_k:+.5f}")
    all_separated = len(disjoint_segments) > 0
    print(f"Segments with ALL pairs GENUINELY disjoint: {disjoint_segments}")
    print(f"At least one genuinely-disjoint segment found: {all_separated}")

    box_lo, box_hi = {}, {}
    box_lo_att, box_hi_att = {}, {}
    point_traj, point_traj_att = {}, {}
    for s in scenarios:
        hist = hist_by_name[s.name]
        box_lo[s.name] = np.array(hist.lower)[:, 0:3]
        box_hi[s.name] = np.array(hist.upper)[:, 0:3]
        box_lo_att[s.name] = np.array(hist.lower)[:, 3:6]
        box_hi_att[s.name] = np.array(hist.upper)[:, 3:6]
        alpha_point = jnp.ones(4) if s.name == "Nominal" else (
            jnp.ones(4).at[int(s.name[-1]) - 1].set(ALPHA_TRUE_FAULT))
        traj = rk4_rollout(jnp.zeros(12), u_seq, alpha_point, DT)
        point_traj[s.name] = np.array(traj)[:, 0:3]
        point_traj_att[s.name] = np.array(traj)[:, 3:6]

    np.savez(RESULTS_DIR / "quadrotor_separating_input.npz",
            u_seq=np.array(u_seq), loss=float(loss), dt=DT, num_steps=NUM_STEPS,
            x0_width=X0_WIDTH, alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI,
            scenario_names=np.array([s.name for s in scenarios]),
            all_separated=all_separated,
            disjoint_segments=np.array(disjoint_segments, dtype=np.int32))
    print(f"Saved {RESULTS_DIR / 'quadrotor_separating_input.npz'}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": 11,
    })
    plot_segments = min(NUM_STEPS, max(disjoint_segments, default=0) + 3)
    plot_pts = plot_segments * 10 + 1

    fig = plt.figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    for s in scenarios:
        color = SCENARIO_COLORS[s.name]
        pos = point_traj[s.name][:plot_pts]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, linewidth=2.0,
               label=s.name, zorder=5)
        ax.scatter(pos[::10, 0], pos[::10, 1], pos[::10, 2], color=color, s=10, zorder=6)
        for k in range(plot_segments):
            _draw_box3d(ax, box_lo[s.name][k], box_hi[s.name][k], color)
    ax.set_xlabel("$x$ (m)", labelpad=8)
    ax.set_ylabel("$y$ (m)", labelpad=8)
    ax.set_zlabel("$z$ (m)", labelpad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.0, right=0.88, bottom=0.02, top=1.0)
    out_pdf = RESULTS_DIR / "quadrotor_position_3d.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote {out_pdf}")

    fig2 = plt.figure(figsize=(7.0, 6.0))
    ax2 = fig2.add_subplot(111, projection="3d")
    for s in scenarios:
        color = SCENARIO_COLORS[s.name]
        att = point_traj_att[s.name][:plot_pts]
        ax2.plot(att[:, 0], att[:, 1], att[:, 2], color=color, linewidth=2.0,
                 label=s.name, zorder=5)
        ax2.scatter(att[::10, 0], att[::10, 1], att[::10, 2], color=color, s=10, zorder=6)
        for k in range(plot_segments):
            _draw_box3d(ax2, box_lo_att[s.name][k], box_hi_att[s.name][k], color)
    ax2.set_xlabel(r"$\phi$ (rad)", labelpad=8)
    ax2.set_ylabel(r"$\theta$ (rad)", labelpad=8)
    ax2.set_zlabel(r"$\psi$ (rad)", labelpad=12)
    ax2.legend(loc="upper left", frameon=False, fontsize=8)
    fig2.subplots_adjust(left=0.0, right=0.88, bottom=0.02, top=1.0)
    out_pdf2 = RESULTS_DIR / "quadrotor_attitude_3d.pdf"
    fig2.savefig(out_pdf2)
    plt.close(fig2)
    print(f"Wrote {out_pdf2}")

if __name__ == "__main__":
    main()
