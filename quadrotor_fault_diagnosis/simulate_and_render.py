import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import jax
import jax.numpy as jnp
import immrax as irx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
import matplotlib.animation as animation

from quadrotor_separating_input import (
    QuadrotorSystem,
    create_scenarios,
    MultistepSequenceOptimizer,
    optimize_multistep_gpu,
    _HOVER_THRUST,
)

DT_DESIGN = 0.01
STEPS_PER_SEGMENT = 50
NUM_SEGMENTS = 2
X0_WIDTH = 0.01
ALPHA_LO, ALPHA_HI = 0.5, 0.9
FAULT_ALPHA = 0.5 * (ALPHA_LO + ALPHA_HI)

SIM_DT = 0.002
ARM_LENGTH = 0.15
SLOWDOWN = 4.0
FPS = 25

OUT_MP4 = _HERE / "quadrotor_fault_diagnosis_flight.mp4"

_COLORS = {
    "Nominal": "#2a78d6",
    "ActuatorFault_1": "#eb6834",
    "ActuatorFault_2": "#3fa34d",
    "ActuatorFault_3": "#c0392b",
    "ActuatorFault_4": "#8e44ad",
}

def design_controller():
    scenarios = create_scenarios(alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI)
    x0_ivl = irx.icentpert(jnp.zeros(12), jnp.full(12, X0_WIDTH))
    ms_opt = MultistepSequenceOptimizer(
        scenarios, x0_ivl, dt=DT_DESIGN,
        steps_per_segment=STEPS_PER_SEGMENT, num_segments=NUM_SEGMENTS,
    )
    u_seq_opt, loss_opt, _, _ = optimize_multistep_gpu(
        ms_opt, num_restarts=10, learning_rate=0.02, num_iters=20, seed=42,
    )
    print(f"Controller re-derived: loss={float(loss_opt):.3e}")
    print("u_seq:")
    for k in range(NUM_SEGMENTS):
        print(f"  segment {k}: U1={float(u_seq_opt[k,0]):.4f} N, "
              f"U2={float(u_seq_opt[k,1]):.4f}, U3={float(u_seq_opt[k,2]):.4f}, "
              f"U4={float(u_seq_opt[k,3]):.4f} N*m")
    return np.array(u_seq_opt), np.zeros(12)

def rk4_step(f, t, x, u, p, dt):
    k1 = f(t, x, u, p)
    k2 = f(t + dt / 2, x + dt / 2 * k1, u, p)
    k3 = f(t + dt / 2, x + dt / 2 * k2, u, p)
    k4 = f(t + dt, x + dt * k3, u, p)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def simulate(sys_, x0, u_seq, segment_duration, alpha, sim_dt):
    p = jnp.array(alpha)
    steps_per_segment = int(round(segment_duration / sim_dt))
    x = jnp.array(x0)
    t = 0.0
    xs, ts = [np.array(x)], [t]
    for k in range(u_seq.shape[0]):
        u_k = jnp.array(u_seq[k])
        for _ in range(steps_per_segment):
            x = rk4_step(sys_.f, jnp.array(t), x, u_k, p, sim_dt)
            t += sim_dt
            xs.append(np.array(x))
            ts.append(t)
    return np.array(ts), np.array(xs)

def rotation_matrix(phi, theta, psi):
    sphi, cphi = np.sin(phi), np.cos(phi)
    stheta, ctheta = np.sin(theta), np.cos(theta)
    spsi, cpsi = np.sin(psi), np.cos(psi)
    return np.array([
        [ctheta * cpsi, sphi * stheta * cpsi - cphi * spsi, cphi * stheta * cpsi + sphi * spsi],
        [ctheta * spsi, sphi * stheta * spsi + cphi * cpsi, cphi * stheta * spsi - sphi * cpsi],
        [-stheta, sphi * ctheta, cphi * ctheta],
    ])

def arm_tip_offsets(arm_length):
    angles = np.deg2rad([45, 135, 225, 315])
    return arm_length * np.stack([np.cos(angles), np.sin(angles), np.zeros(4)], axis=1)

def main():
    print("Devices:", jax.devices())
    sys_ = QuadrotorSystem()

    u_seq, x0 = design_controller()
    segment_duration = STEPS_PER_SEGMENT * DT_DESIGN

    scenario_alphas = {
        "Nominal": [1.0, 1.0, 1.0, 1.0],
        "ActuatorFault_1": [FAULT_ALPHA, 1.0, 1.0, 1.0],
        "ActuatorFault_2": [1.0, FAULT_ALPHA, 1.0, 1.0],
        "ActuatorFault_3": [1.0, 1.0, FAULT_ALPHA, 1.0],
        "ActuatorFault_4": [1.0, 1.0, 1.0, FAULT_ALPHA],
    }

    print(f"\nSimulating {len(scenario_alphas)} trajectories, "
          f"dt={SIM_DT}s, horizon={NUM_SEGMENTS * segment_duration}s ...")
    trajectories = {}
    for name, alpha in scenario_alphas.items():
        ts, xs = simulate(sys_, x0, u_seq, segment_duration, alpha, SIM_DT)
        trajectories[name] = (ts, xs)
        z_final = xs[-1, 2]
        print(f"  {name:16s}: final position (x,y,z)=({xs[-1,0]:+.3f}, {xs[-1,1]:+.3f}, {xs[-1,2]:+.3f}) m")

    render_every = max(1, int(round(1.0 / (SIM_DT * SLOWDOWN * FPS))))
    num_raw_frames = trajectories["Nominal"][1].shape[0]
    frame_indices = list(range(0, num_raw_frames, render_every))
    print(f"\nRendering {len(frame_indices)} frames at {FPS} fps "
          f"({SLOWDOWN}x slowdown vs. real time) ...")

    all_pos = np.concatenate([xs[:, :3] for _, xs in trajectories.values()], axis=0)
    pad = 0.15
    xlim = (all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad)
    ylim = (all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad)
    zlim = (all_pos[:, 2].min() - pad, all_pos[:, 2].max() + pad)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    arm_offsets = arm_tip_offsets(ARM_LENGTH)
    frame_lines, trail_lines, trail_data = {}, {}, {}
    for name in trajectories:
        color = _COLORS[name]
        (fl1,) = ax.plot([], [], [], '-', color=color, linewidth=2.5)
        (fl2,) = ax.plot([], [], [], '-', color=color, linewidth=2.5)
        (tl,) = ax.plot([], [], [], '--', color=color, linewidth=1.0, alpha=0.6, label=name)
        frame_lines[name] = (fl1, fl2)
        trail_lines[name] = tl
        trail_data[name] = ([], [], [])

    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m) [up]")
        ax.set_title("Quadrotor fault diagnosis: same control, 5 fault hypotheses")
        ax.legend(loc='upper right', fontsize=8)
        return []

    def update(frame_idx):
        artists = []
        for name, (ts, xs) in trajectories.items():
            k = min(frame_idx, xs.shape[0] - 1)
            pos = xs[k, :3]
            phi, theta, psi = xs[k, 3], xs[k, 4], xs[k, 5]
            R = rotation_matrix(phi, theta, psi)
            tips = pos + arm_offsets @ R.T

            fl1, fl2 = frame_lines[name]
            fl1.set_data([tips[0, 0], tips[2, 0]], [tips[0, 1], tips[2, 1]])
            fl1.set_3d_properties([tips[0, 2], tips[2, 2]])
            fl2.set_data([tips[1, 0], tips[3, 0]], [tips[1, 1], tips[3, 1]])
            fl2.set_3d_properties([tips[1, 2], tips[3, 2]])

            tx, ty, tz = trail_data[name]
            tx.append(pos[0]); ty.append(pos[1]); tz.append(pos[2])
            trail_lines[name].set_data(tx, ty)
            trail_lines[name].set_3d_properties(tz)

            artists.extend([fl1, fl2, trail_lines[name]])

        time_text.set_text(f"t = {trajectories['Nominal'][0][min(frame_idx, num_raw_frames-1)]:.2f} s")
        artists.append(time_text)
        return artists

    ani = animation.FuncAnimation(
        fig, update, frames=frame_indices, init_func=init,
        blit=False, interval=1000.0 / FPS,
    )

    writer = animation.FFMpegWriter(fps=FPS, bitrate=1800)
    ani.save(str(OUT_MP4), writer=writer)
    plt.close(fig)
    print(f"\nWrote video: {OUT_MP4}")

if __name__ == "__main__":
    main()
