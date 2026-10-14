import argparse
import shutil
import sys
from pathlib import Path

import jax.numpy as jnp
import immrax as irx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from car_separating_input import (
    CarNomActSystem,
    create_scenarios,
    euler_step,
    observed_output,
    optimize_refined_gpu,
    collect_refinement_history,
)

DEFAULT_AZIM    = 45.0
DEFAULT_ELEV    = 20.0
LIMIT_PAD       = 0.05
CTRL_DT         = 1.0
CTRL_NUM_STEPS  = 5
DEFAULT_DT_TRAJ = 0.1
DEFAULT_SIM_DT  = 0.033
PAIR_I, PAIR_J  = 0, 1

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-panel 3-D refinement animation (matplotlib)."
    )
    p.add_argument(
        "--fault-mode", choices=["nominal", "actuator", "sensor"],
        default="actuator",
        help="Fault mode the trajectory is simulated under (default: actuator).",
    )
    p.add_argument(
        "--x0", nargs=3, type=float, metavar=("PX", "PY", "PHI"),
        default=[0.1, 0.1, 0.0],
        help="Initial state [px py phi] (default: 0.1 0.1 0.0).",
    )
    p.add_argument(
        "--ctrl-dt", type=float, default=CTRL_DT,
        help=f"Controller Euler step size in seconds (default: {CTRL_DT}).",
    )
    p.add_argument(
        "--ctrl-num-steps", type=int, default=CTRL_NUM_STEPS,
        help=f"Number of controller steps (default: {CTRL_NUM_STEPS}).",
    )
    p.add_argument(
        "--dt-traj", type=float, default=DEFAULT_DT_TRAJ,
        help=(
            f"Trajectory Euler sub-step in seconds (default: {DEFAULT_DT_TRAJ}). "
            "Smaller values give a smoother trajectory curve. "
            "Must be <= ctrl_dt."
        ),
    )
    p.add_argument(
        "--sim-dt", type=float, default=DEFAULT_SIM_DT,
        help=(
            f"Animation frame interval in seconds (default: {DEFAULT_SIM_DT} ~30 fps). "
            "Controls playback speed only."
        ),
    )
    p.add_argument(
        "--azim", type=float, default=DEFAULT_AZIM,
        help=f"Fixed camera azimuth in degrees (default: {DEFAULT_AZIM}).",
    )
    p.add_argument(
        "--elev", type=float, default=DEFAULT_ELEV,
        help=f"Fixed camera elevation in degrees (default: {DEFAULT_ELEV}).",
    )
    p.add_argument("--dpi", type=int, default=150, help="Output DPI (default: 150).")
    p.add_argument("--output", type=str, default=None, help="Output .mp4 path.")
    return p.parse_args()

def simulate_trajectory_fine(x0, u_seq, alpha, ctrl_dt, dt_traj):
    sys_obj = CarNomActSystem()
    p = jnp.array([alpha])
    x = x0
    traj  = [np.array(x)]
    times = [0.0]
    t     = 0.0
    steps_per_ctrl = max(1, round(ctrl_dt / dt_traj))
    actual_dt = ctrl_dt / steps_per_ctrl
    for k in range(u_seq.shape[0]):
        u = u_seq[k]
        for _ in range(steps_per_ctrl):
            dx = sys_obj.f(jnp.zeros(()), x, u, p)
            x  = x + dx * actual_dt
            t += actual_dt
            traj.append(np.array(x))
            times.append(t)
    return np.stack(traj), np.array(times)

def _intersect_obs(a, b):
    lo = jnp.maximum(a.lower, b.lower)
    hi = jnp.minimum(a.upper, b.upper)
    return irx.Interval(lower=lo, upper=hi) if bool(jnp.all(hi >= lo)) else None

def collect_unrefined_history(x0_ivl, u_seq, scenarios, dt, num_steps):
    n  = len(scenarios)
    xs = [euler_step(s.emb_system, x0_ivl, u_seq[0], s.p_interval, dt)
          for s in scenarios]
    steps = [{"t": dt, "obs": [observed_output(xs[k], scenarios[k]) for k in range(n)]}]
    for step_k in range(num_steps):
        xs = [euler_step(s.emb_system, xs[k], u_seq[step_k + 1], s.p_interval, dt)
              for k, s in enumerate(scenarios)]
        steps.append({"t": (step_k + 2) * dt,
                      "obs": [observed_output(xs[k], scenarios[k]) for k in range(n)]})
    return steps

def compute_validity(ref_steps, ref_pairs, traj_obs_px, traj_obs_py, ctrl_dt, dt_traj):
    n_pairs = len(ref_pairs)
    n_steps = len(ref_steps)
    valid   = np.ones((n_pairs, n_steps), dtype=bool)
    steps_per_ctrl = max(1, round(ctrl_dt / dt_traj))
    for pi in range(n_pairs):
        still_valid = True
        for k, step in enumerate(ref_steps):
            if not still_valid:
                valid[pi, k] = False
                continue
            obs_i, obs_j = step["pair_obs"][pi][0], step["pair_obs"][pi][1]
            traj_idx = min((k + 1) * steps_per_ctrl, len(traj_obs_px) - 1)
            yx = traj_obs_px[traj_idx]
            yy = traj_obs_py[traj_idx]
            inside_i = (float(obs_i.lower[0]) <= yx <= float(obs_i.upper[0]) and
                        float(obs_i.lower[1]) <= yy <= float(obs_i.upper[1]))
            inside_j = (float(obs_j.lower[0]) <= yx <= float(obs_j.upper[0]) and
                        float(obs_j.lower[1]) <= yy <= float(obs_j.upper[1]))
            if not (inside_i and inside_j):
                still_valid = False
                valid[pi, k] = False
    return valid

def _invalidation_step(valid, pi):
    for k in range(valid.shape[1]):
        if not valid[pi, k]:
            return k
    return None

def compute_static_limits(unref_steps, ref_steps, ref_pairs, pair_idx,
                          traj_obs_px, traj_obs_py):
    all_x, all_y = list(traj_obs_px), list(traj_obs_py)
    for step in unref_steps:
        for sc_idx in range(len(step["obs"])):
            obs = step["obs"][sc_idx]
            all_x += [float(obs.lower[0]), float(obs.upper[0])]
            all_y += [float(obs.lower[1]), float(obs.upper[1])]
    for step in ref_steps:
        obs_i, obs_j = step["pair_obs"][pair_idx][0], step["pair_obs"][pair_idx][1]
        for obs in (obs_i, obs_j):
            all_x += [float(obs.lower[0]), float(obs.upper[0])]
            all_y += [float(obs.lower[1]), float(obs.upper[1])]
    xl, xh = min(all_x), max(all_x)
    yl, yh = min(all_y), max(all_y)
    if xh - xl < 1e-4: xh += 0.05; xl -= 0.05
    if yh - yl < 1e-4: yh += 0.05; yl -= 0.05
    pad_x = LIMIT_PAD * (xh - xl)
    pad_y = LIMIT_PAD * (yh - yl)
    return xl - pad_x, xh + pad_x, yl - pad_y, yh + pad_y

def draw_unrefined_panel(ax, ctrl_k, traj_end_idx,
                         unref_steps, scenarios, cmap,
                         traj_obs_px, traj_obs_py, traj_times,
                         traj_color,
                         px_lo, px_hi, py_lo, py_hi,
                         total_time, azim, elev):
    ax.cla()
    for s_idx in range(min(ctrl_k, len(unref_steps))):
        t_s = unref_steps[s_idx]["t"]
        for sc_idx in range(len(scenarios)):
            obs   = unref_steps[s_idx]["obs"][sc_idx]
            color = cmap(sc_idx)
            xl = float(obs.lower[0]); xh = float(obs.upper[0])
            yl = float(obs.lower[1]); yh = float(obs.upper[1])
            verts = [[(xl, t_s, yl), (xh, t_s, yl), (xh, t_s, yh), (xl, t_s, yh)]]
            poly  = Poly3DCollection(verts, alpha=0.35)
            poly.set_facecolor(color)
            poly.set_edgecolor((*matplotlib.colors.to_rgb(color), 0.35))
            ax.add_collection3d(poly)
        obs_list = [unref_steps[s_idx]["obs"][k] for k in range(len(scenarios))]
        for (p, q) in [(0, 1), (0, 2), (1, 2)]:
            iv1, iv2 = obs_list[p], obs_list[q]
            xl = float(max(iv1.lower[0], iv2.lower[0]))
            xh = float(min(iv1.upper[0], iv2.upper[0]))
            yl = float(max(iv1.lower[1], iv2.lower[1]))
            yh = float(min(iv1.upper[1], iv2.upper[1]))
            if xh >= xl and yh >= yl:
                ax.plot([xl, xh, xh, xl, xl], [t_s] * 5, [yl, yl, yh, yh, yl],
                        color="black", linewidth=1.5, zorder=5)
    end = traj_end_idx + 1
    if end > 1:
        ax.plot(traj_obs_px[:end], traj_times[:end], traj_obs_py[:end],
                color=traj_color, linewidth=2.5, linestyle="-",
                marker="o", markersize=3, zorder=10)
    ax.scatter([traj_obs_px[traj_end_idx]], [traj_times[traj_end_idx]],
               [traj_obs_py[traj_end_idx]],
               color="red", s=80, zorder=15, depthshade=False)
    ax.set_xlim(px_lo, px_hi)
    ax.set_ylim(0.0, total_time)
    ax.set_zlim(py_lo, py_hi)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_title("Unrefined", fontsize=24, pad=8)

def draw_refined_panel(ax, ctrl_k, traj_end_idx,
                        ref_steps, pair_idx, valid, scenarios, cmap,
                        traj_obs_px, traj_obs_py, traj_times,
                        traj_color,
                        px_lo, px_hi, py_lo, py_hi,
                        total_time, azim, elev):
    ax.cla()
    inv_step = _invalidation_step(valid, pair_idx)
    for s_idx in range(min(ctrl_k, len(ref_steps))):
        t_s = ref_steps[s_idx]["t"]
        box_alpha = 0.15 if (inv_step is not None and s_idx >= inv_step) else 0.35
        obs_i, obs_j, inter = ref_steps[s_idx]["pair_obs"][pair_idx]
        for obs, color in [(obs_i, cmap(PAIR_I)), (obs_j, cmap(PAIR_J))]:
            xl = float(obs.lower[0]); xh = float(obs.upper[0])
            yl = float(obs.lower[1]); yh = float(obs.upper[1])
            verts = [[(xl, t_s, yl), (xh, t_s, yl), (xh, t_s, yh), (xl, t_s, yh)]]
            poly  = Poly3DCollection(verts, alpha=box_alpha)
            poly.set_facecolor(color)
            poly.set_edgecolor((*matplotlib.colors.to_rgb(color), box_alpha))
            ax.add_collection3d(poly)
        if inter is not None:
            xl = float(inter.lower[0]); xh = float(inter.upper[0])
            yl = float(inter.lower[1]); yh = float(inter.upper[1])
            ax.plot([xl, xh, xh, xl, xl], [t_s] * 5, [yl, yl, yh, yh, yl],
                    color="black", linewidth=1.5, alpha=box_alpha / 0.35, zorder=5)
    end = traj_end_idx + 1
    if end > 1:
        ax.plot(traj_obs_px[:end], traj_times[:end], traj_obs_py[:end],
                color=traj_color, linewidth=2.5, linestyle="-",
                marker="o", markersize=3, zorder=10)
    ax.scatter([traj_obs_px[traj_end_idx]], [traj_times[traj_end_idx]],
               [traj_obs_py[traj_end_idx]],
               color="red", s=80, zorder=15, depthshade=False)
    ax.set_xlim(px_lo, px_hi)
    ax.set_ylim(0.0, total_time)
    ax.set_zlim(py_lo, py_hi)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    t_now = traj_times[traj_end_idx]
    ax.set_title(f"Refined  |  t={t_now:.2f}s", fontsize=24, pad=8)

def compute_model_invalidation(unref_steps, scenarios, traj_obs_px, traj_obs_py,
                               ctrl_dt, dt_traj):
    n_sc = len(scenarios)
    steps_per_ctrl = max(1, round(ctrl_dt / dt_traj))
    result = {}
    for sc in range(n_sc):
        inv_k = None
        still_valid = True
        for k, step in enumerate(unref_steps):
            if not still_valid:
                break
            obs = step["obs"][sc]
            traj_idx = min((k + 1) * steps_per_ctrl, len(traj_obs_px) - 1)
            yx = float(traj_obs_px[traj_idx])
            yy = float(traj_obs_py[traj_idx])
            inside = (float(obs.lower[0]) <= yx <= float(obs.upper[0]) and
                      float(obs.lower[1]) <= yy <= float(obs.upper[1]))
            if not inside:
                inv_k = k
                still_valid = False
        result[sc] = inv_k
    return result

def draw_status_panel(ax_status, ctrl_k, scenarios, cmap, model_elim, ctrl_dt):
    ax_status.cla()
    ax_status.set_xlim(0, 1)
    ax_status.set_ylim(-0.5, len(scenarios) - 0.5)
    ax_status.axis("off")

    ax_status.set_title("Candidate Models", fontsize=30, fontweight="bold", pad=10)
    ax_status.axhline(len(scenarios) - 0.5, color="lightgrey", linewidth=1.0)

    for sc_idx, scenario in enumerate(scenarios):
        row_y = len(scenarios) - 1 - sc_idx
        elim_step = model_elim[sc_idx]
        eliminated = (elim_step is not None) and (ctrl_k > elim_step)

        if eliminated:
            inv_t = int(round(elim_step * ctrl_dt))
            label = f"✓  {scenario.name} \n INVALIDATED \n@ t = {inv_t} s"
            text_color = "#999999"
            fontweight = "normal"
        else:
            label = f"✓  {scenario.name}"
            text_color = matplotlib.colors.to_hex(cmap(sc_idx))
            fontweight = "bold"

        ax_status.text(
            0.08, row_y, label,
            ha="left", va="center",
            fontsize=24, color=text_color,
            fontweight=fontweight,
            fontfamily="DejaVu Serif",
            transform=ax_status.transData,
        )
        if eliminated:
            ax_status.axhline(
                row_y, xmin=0.04, xmax=0.96,
                color="#999999", linewidth=1.8, zorder=5,
            )
        ax_status.axhline(row_y - 0.5, color="#eeeeee", linewidth=0.8)

def main():
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found. Install it and re-run.", file=sys.stderr)
        sys.exit(1)

    fault_mode     = args.fault_mode
    x0_state       = jnp.array(args.x0)
    ctrl_dt        = args.ctrl_dt
    ctrl_num_steps = args.ctrl_num_steps
    dt_traj        = min(args.dt_traj, ctrl_dt)
    sim_dt         = args.sim_dt
    azim           = args.azim
    elev           = args.elev
    fps            = max(1, round(1.0 / sim_dt))
    total_time     = ctrl_dt * ctrl_num_steps

    print(f"Controller : ctrl_dt={ctrl_dt}s x {ctrl_num_steps} steps -> {total_time:.2f}s")
    print(f"Trajectory : dt_traj={dt_traj}s (sub-step for smooth curve)")
    print(f"Animation  : sim_dt={sim_dt:.4f}s -> {fps} fps")
    print(f"Camera     : azim={azim} elev={elev} (fixed)")

    scenarios = create_scenarios(actuator_alpha_lo=0.0, actuator_alpha_hi=0.5)
    x0_ivl    = irx.icentpert(jnp.array([0.1, 0.1, 0.0]), jnp.array([0.1, 0.1, 0.1]))

    _mode_map = {"nominal": (0, 1.0), "actuator": (1, 0.25), "sensor": (2, 1.0)}
    scenario_idx, alpha = _mode_map[fault_mode]
    traj_scenario = scenarios[scenario_idx]
    print(f"Fault mode : {traj_scenario.name}  (alpha={alpha})")

    print("Optimising separating input ...")
    u_opt, loss_opt, _, _ = optimize_refined_gpu(
        x0_ivl=x0_ivl, scenarios=scenarios, dt=ctrl_dt,
        num_restarts=500, learning_rate=1, num_iters=50, seed=42,
        num_steps=ctrl_num_steps,
    )
    print(f"  loss={loss_opt:.6f}  u_opt shape={u_opt.shape}")

    print("Simulating fine trajectory ...")
    traj_fine, traj_times_fine = simulate_trajectory_fine(
        x0_state, u_opt, alpha, ctrl_dt, dt_traj)
    obs_off     = np.array(traj_scenario.obs_offset)
    obs_scale   = float(np.array(traj_scenario.obs_scale)[0])
    traj_obs_px = traj_fine[:, 0] * obs_scale + obs_off[0]
    traj_obs_py = traj_fine[:, 1] * obs_scale + obs_off[1]
    print(f"  fine traj shape={traj_fine.shape}  ({len(traj_times_fine)} points)")

    print("Collecting interval histories ...")
    unref_steps = collect_unrefined_history(
        x0_ivl, u_opt, scenarios, ctrl_dt, ctrl_num_steps)
    ref_steps, ref_pairs = collect_refinement_history(
        x0_ivl, u_opt, scenarios, ctrl_dt, ctrl_num_steps)
    pair_idx = ref_pairs.index((PAIR_I, PAIR_J))
    print(f"  {len(ref_steps)} ctrl steps  |  pair_idx={pair_idx} "
          f"({scenarios[PAIR_I].name} vs {scenarios[PAIR_J].name})")

    print("Computing validity ...")
    valid = compute_validity(ref_steps, ref_pairs, traj_obs_px, traj_obs_py,
                             ctrl_dt, dt_traj)
    inv = _invalidation_step(valid, pair_idx)
    inv_label = f"t={inv*ctrl_dt}s" if inv is not None else "never"
    print(f"  Pair invalidated at: {inv_label}")

    print("Computing static axis limits ...")
    px_lo, px_hi, py_lo, py_hi = compute_static_limits(
        unref_steps, ref_steps, ref_pairs, pair_idx, traj_obs_px, traj_obs_py)
    print(f"  px=[{px_lo:.3f}, {px_hi:.3f}]  py=[{py_lo:.3f}, {py_hi:.3f}]")

    num_anim_frames = max(1, round(total_time / sim_dt)) + 1
    anim_times      = np.linspace(0.0, total_time, num_anim_frames)
    traj_idx_for_frame = np.round(anim_times / dt_traj).astype(int).clip(0, len(traj_times_fine) - 1)
    ctrl_step_for_frame = np.floor(anim_times / ctrl_dt).astype(int).clip(0, ctrl_num_steps)
    print(f"Animation  : {num_anim_frames} frames at {fps} fps")

    plt.rcParams.update({"font.family": "DejaVu Serif", "text.usetex": False})
    cmap       = plt.colormaps["tab10"].resampled(len(scenarios))
    traj_color = cmap(scenario_idx)

    model_elim = compute_model_invalidation(
        unref_steps, scenarios, traj_obs_px, traj_obs_py, ctrl_dt, dt_traj)
    for sc_idx, scenario in enumerate(scenarios):
        es = model_elim[sc_idx]
        label = f"t={int(round(es * ctrl_dt))}s (step {es})" if es is not None else "never"
        print(f"  Model '{scenario.name}' invalidated at: {label}")

    fig = plt.figure(figsize=(22, 7))
    gs  = fig.add_gridspec(
        1, 3,
        width_ratios=[5, 5, 2],
        left=0.03, right=0.97,
        top=0.88, bottom=0.05,
        wspace=0.08,
    )
    ax_left   = fig.add_subplot(gs[0], projection="3d")
    ax_right  = fig.add_subplot(gs[1], projection="3d")
    ax_status = fig.add_subplot(gs[2])

    legend_handles = [
        plt.Line2D([0], [0], color=cmap(0), linewidth=3, label=scenarios[0].name),
        plt.Line2D([0], [0], color=cmap(1), linewidth=3, label=scenarios[1].name),
        plt.Line2D([0], [0], color=cmap(2), linewidth=3, label=scenarios[2].name),
        plt.Line2D([0], [0], color="black", linewidth=1.5, label="Intersection"),
        plt.Line2D([0], [0], color=traj_color, linewidth=2.5,
                   marker="o", markersize=4,
                   label=f"Trajectory ({traj_scenario.name})"),
        plt.Line2D([0], [0], color="red", marker="o", markersize=6,
                   linestyle="None", label="Current"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=1,
               fontsize=16, bbox_to_anchor=(0.43, 1.0))

    def update(frame_idx):
        ctrl_k   = int(ctrl_step_for_frame[frame_idx])
        traj_idx = int(traj_idx_for_frame[frame_idx])
        draw_unrefined_panel(
            ax_left, ctrl_k, traj_idx,
            unref_steps, scenarios, cmap,
            traj_obs_px, traj_obs_py, traj_times_fine,
            traj_color,
            px_lo, px_hi, py_lo, py_hi,
            total_time, azim, elev,
        )
        draw_refined_panel(
            ax_right, ctrl_k, traj_idx,
            ref_steps, pair_idx, valid, scenarios, cmap,
            traj_obs_px, traj_obs_py, traj_times_fine,
            traj_color,
            px_lo, px_hi, py_lo, py_hi,
            total_time, azim, elev,
        )
        draw_status_panel(ax_status, ctrl_k, scenarios, cmap, model_elim, ctrl_dt)
        return []

    frame_interval_ms = round(sim_dt * 1000)
    print(f"Rendering {num_anim_frames} frames (interval={frame_interval_ms}ms, {fps}fps) ...")
    ani = FuncAnimation(fig, update, frames=num_anim_frames,
                        interval=frame_interval_ms, blit=False)

    out_path = args.output or str(
        HERE / f"refinement_anim_twopanel_{fault_mode}_ctrl{ctrl_dt}x{ctrl_num_steps}.mp4"
    )
    ani.save(out_path, writer="ffmpeg", fps=fps, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    main()
