import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXAMPLES_DIR = _HERE.parent
for _p in (_HERE, _EXAMPLES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import jax
import jax.numpy as jnp
import immrax as irx

from quadrotor_separating_input import create_scenarios, _propagate_history
from adaptive_spoofing.crazyflie_12d import CrazyflieSystem

RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DT = 0.1
X0_WIDTH = 0.01
ALPHA_LO, ALPHA_HI = 0.5, 0.9
FINE_SUBSTEPS = 20
GT_SUBSTEPS = 20

N_INBOUND = 2000
N_ADV = 500
SEED = 0
_ADV_X0_MARGIN_LO, _ADV_X0_MARGIN_HI = 1.0, 1.5
_ADV_ALPHA_MARGIN = 0.1

MODEL_NAMES = ["Nominal", "ActuatorFault_1", "ActuatorFault_2",
              "ActuatorFault_3", "ActuatorFault_4"]

_gt_sys = CrazyflieSystem()

def _gt_f(x, u, alpha):
    return _gt_sys.f(jnp.zeros(()), x, u, alpha)

def rk4_rollout_batch(x0_batch: jnp.ndarray, u_seq: jnp.ndarray,
                      alpha_batch: jnp.ndarray, dt: float, sub_steps: int) -> jnp.ndarray:
    h = dt / sub_steps

    def rk4_step(x, u, alpha):
        k1 = _gt_f(x, u, alpha)
        k2 = _gt_f(x + 0.5 * h * k1, u, alpha)
        k3 = _gt_f(x + 0.5 * h * k2, u, alpha)
        k4 = _gt_f(x + h * k3, u, alpha)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rollout_one(x0, alpha):
        def seg_scan(x, u_k):
            def body(x, _):
                return rk4_step(x, u_k, alpha), None
            x_end, _ = jax.lax.scan(body, x, xs=None, length=sub_steps)
            return x_end, x_end
        _, hist = jax.lax.scan(seg_scan, x0, u_seq)
        return hist

    return jax.vmap(rollout_one)(x0_batch, alpha_batch)

_rk4_rollout_batch_jit = jax.jit(rk4_rollout_batch, static_argnums=(4,))

def predicted_histories(scenarios, x0_ivl, u_seq, steps_per_segment):
    per_step_dt = DT / steps_per_segment
    emb_sys = scenarios[0].emb_system
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )

    def prop_one(p_ivl_single):
        return _propagate_history(x0_ivl, u_seq, emb_sys, p_ivl_single,
                                  per_step_dt, steps_per_segment)

    hist_batch = jax.vmap(prop_one)(p_batch)
    return {s.name: (np.array(hist_batch.lower[i]), np.array(hist_batch.upper[i]))
            for i, s in enumerate(scenarios)}

def sample_x0(rng, batch, adversarial):
    x0 = np.zeros((batch, 12))
    if not adversarial:
        x0[:] = rng.uniform(-X0_WIDTH, X0_WIDTH, size=(batch, 12))
        return x0
    for b in range(batch):
        dim = rng.integers(0, 12)
        sign = rng.choice([-1.0, 1.0])
        extra = rng.uniform(_ADV_X0_MARGIN_LO, _ADV_X0_MARGIN_HI) * X0_WIDTH
        row = rng.uniform(-X0_WIDTH, X0_WIDTH, size=12)
        row[dim] = sign * (X0_WIDTH + extra)
        x0[b] = row
    return x0

def sample_alpha(rng, true_mode, batch, adversarial):
    alpha = np.ones((batch, 4))
    if true_mode == "Nominal":
        return alpha
    ch = int(true_mode[-1]) - 1
    if not adversarial:
        alpha[:, ch] = rng.uniform(ALPHA_LO, ALPHA_HI, size=batch)
        return alpha
    lo_lo, lo_hi = max(ALPHA_LO - _ADV_ALPHA_MARGIN, 0.0), ALPHA_LO
    hi_lo, hi_hi = ALPHA_HI, ALPHA_HI + _ADV_ALPHA_MARGIN
    for b in range(batch):
        if lo_hi > lo_lo and (rng.random() < 0.5 or hi_hi <= hi_lo):
            alpha[b, ch] = rng.uniform(lo_lo, lo_hi)
        else:
            alpha[b, ch] = rng.uniform(hi_lo, hi_hi)
    return alpha

def _classify(alive, true_idx, outcomes, suffix=""):
    num_alive = alive.sum(axis=0)
    true_alive = alive[true_idx]
    for i in range(alive.shape[1]):
        if true_alive[i] and num_alive[i] == 1:
            outcomes["correct" + suffix] += 1
        elif true_alive[i]:
            outcomes["inconclusive" + suffix] += 1
        else:
            outcomes["missed" + suffix] += 1
            if num_alive[i] == 1:
                outcomes["wrong" + suffix] += 1

def run_cell(rng, u_seq, predicted_fine, predicted_coarse, true_mode, n_trials, adversarial,
            num_steps):
    outcomes = {k + suffix: 0 for k in ("correct", "inconclusive", "missed", "wrong")
               for suffix in ("", "_coarse")}
    true_idx = MODEL_NAMES.index(true_mode)

    x0_true = sample_x0(rng, n_trials, adversarial)
    alpha_true = sample_alpha(rng, true_mode, n_trials, adversarial)

    state_hist = np.array(_rk4_rollout_batch_jit(
        jnp.asarray(x0_true, dtype=jnp.float32), jnp.asarray(u_seq, dtype=jnp.float32),
        jnp.asarray(alpha_true, dtype=jnp.float32), DT, GT_SUBSTEPS))

    for predicted, prefix in ((predicted_fine, ""), (predicted_coarse, "_coarse")):
        alive = np.ones((len(MODEL_NAMES), n_trials), dtype=bool)
        for k in range(num_steps):
            x_k = state_hist[:, k, :]
            for m, name in enumerate(MODEL_NAMES):
                lo, hi = predicted[name]
                contained = np.all((x_k >= lo[k]) & (x_k <= hi[k]), axis=-1)
                alive[m] &= contained
        _classify(alive, true_idx, outcomes, prefix)
    return outcomes

def main(out_path: Path):
    d = np.load(RESULTS_DIR / "quadrotor_separating_input.npz", allow_pickle=True)
    u_seq = jnp.array(d["u_seq"])
    num_steps = int(u_seq.shape[0])
    disjoint_segments = list(d["disjoint_segments"])
    assert list(d["scenario_names"]) == MODEL_NAMES, \
        f"scenario order mismatch: {list(d['scenario_names'])} vs {MODEL_NAMES}"

    max_cutoff = max(disjoint_segments, default=num_steps - 1) + 1
    print(f"Loaded u_seq: {num_steps} segments x dt={DT}s = {num_steps*DT:.2f}s horizon "
         f"(disjoint_segments={disjoint_segments}); sweeping diagnosis cutoff 1..{max_cutoff}")

    scenarios = create_scenarios(alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI)
    x0_ivl = irx.icentpert(jnp.zeros(12), jnp.full(12, X0_WIDTH))

    print("Computing predicted state-interval histories (coarse + fine)...")
    predicted_coarse = predicted_histories(scenarios, x0_ivl, u_seq, 1)
    predicted_fine = predicted_histories(scenarios, x0_ivl, u_seq, FINE_SUBSTEPS)

    rng = np.random.default_rng(SEED)
    rows = []
    t_start = time.perf_counter()
    for cutoff in range(1, max_cutoff + 1):
        print(f"=== diagnosis cutoff = {cutoff} (through segment {cutoff - 1}) ===")
        for true_mode in MODEL_NAMES:
            for adversarial, n_trials in ((False, N_INBOUND), (True, N_ADV)):
                out = run_cell(rng, u_seq, predicted_fine, predicted_coarse, true_mode,
                               n_trials, adversarial, cutoff)
                rows.append(dict(cutoff=cutoff, true_mode=true_mode, adversarial=adversarial,
                                 n_trials=n_trials, **out))
                tag = "adversarial" if adversarial else "in-bound"
                print(f"  {true_mode:16s} [{tag:11s}] n={n_trials}: {out}")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nTotal wall time: {total_elapsed:.1f}s over "
         f"{sum(r['n_trials'] for r in rows)} trials")

    np.savez(out_path, rows=np.array(rows, dtype=object),
            dt=DT, num_steps=num_steps, max_cutoff=max_cutoff,
            disjoint_segments=np.array(disjoint_segments, dtype=np.int32), x0_width=X0_WIDTH,
            alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI,
            fine_substeps=FINE_SUBSTEPS, gt_substeps=GT_SUBSTEPS,
            n_inbound=N_INBOUND, n_adv=N_ADV, seed=SEED,
            total_elapsed_s=total_elapsed)
    print(f"Wrote {out_path}")
    return rows

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR / "quadrotor_diagnosis_qps.npz"
    main(out)
