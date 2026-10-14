import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import jax
import jax.numpy as jnp
import immrax as irx
import numpy as np

from quadrotor_separating_input import (
    Scenario,
    _HOVER_THRUST,
    _U_HI,
    _U_LO,
    _memory_snapshot,
    _overlap_volume,
    _propagate_history,
    _run_unrolled_or_loop_nocheckpoint,
    _soft_separation_loss,
    create_scenarios,
    get_system_and_embedding,
    propagate_with_refinement,
    separation_loss_multistep,
)

NUM_STATES = 12
NUM_INPUTS = 4

OUTPUT_IDX = (3, 4, 5, 9, 10, 11)
NUM_OUTPUTS = len(OUTPUT_IDX)
NUM_GAINS = NUM_INPUTS * NUM_OUTPUTS
THETA_LEN = NUM_GAINS + NUM_INPUTS

_U_HALFRANGE = (_U_HI - _U_LO) / 2

_K_REL = 10.0
_K_MAX_VEC = _K_REL * _U_HALFRANGE

_THETA_LO = jnp.concatenate([
    jnp.repeat(-_K_MAX_VEC, NUM_OUTPUTS), _U_LO])
_THETA_HI = jnp.concatenate([
    jnp.repeat(_K_MAX_VEC, NUM_OUTPUTS), _U_HI])
_STEP_SCALE = (_THETA_HI - _THETA_LO) / 2

def _project_theta(theta: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(theta, _THETA_LO, _THETA_HI)

def theta_to_K_r(theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return theta[:NUM_GAINS].reshape(NUM_INPUTS, NUM_OUTPUTS), theta[NUM_GAINS:]

def K_r_to_theta(K: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([K.flatten(), r])

def _gain_mask_matrix(structure: str) -> jnp.ndarray:
    if structure == 'dense':
        return jnp.ones((NUM_INPUTS, NUM_OUTPUTS))
    if structure == 'diagonal':
        return jnp.eye(NUM_INPUTS, NUM_OUTPUTS)
    if structure == 'rate':
        m = np.zeros((NUM_INPUTS, NUM_OUTPUTS))
        for u_i, y_i in ((1, 3), (2, 4), (3, 5)):
            m[u_i, y_i] = 1.0
        return jnp.asarray(m)
    raise ValueError(f"unknown gain structure {structure!r}")

GAIN_STRUCTURES = ('dense', 'diagonal', 'rate')

def _select_outputs(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([x[i:i + 1] for i in OUTPUT_IDX])

def closed_loop_input(theta_step: jnp.ndarray, y: jnp.ndarray,
                      mask: jnp.ndarray) -> jnp.ndarray:
    K, r = theta_to_K_r(theta_step)
    return jnp.clip((K * mask) @ y + r, _U_LO, _U_HI)

def expand_theta(theta: jnp.ndarray, num_steps: int) -> jnp.ndarray:
    if theta.ndim == 1:
        return jnp.broadcast_to(theta, (num_steps, THETA_LEN))
    return theta

def make_cl_embedding(structure: str = 'dense', **phys):
    ol_sys, _ = get_system_and_embedding(**phys)
    mask = _gain_mask_matrix(structure)

    class QuadrotorOutputFeedbackCLSystem(irx.System):

        def __init__(self):
            self.evolution = 'continuous'
            self.xlen = NUM_STATES

        def f(self, t, x, u, p):
            return ol_sys.f(t, x, closed_loop_input(u, _select_outputs(x), mask), p)

    return irx.natemb(QuadrotorOutputFeedbackCLSystem())

def create_cl_scenarios(structure: str = 'dense', alpha_lo: float = 0.5,
                        alpha_hi: float = 0.9, **phys) -> List[Scenario]:
    emb = make_cl_embedding(structure, **phys)
    return [dataclasses.replace(s, emb_system=emb)
            for s in create_scenarios(alpha_lo=alpha_lo, alpha_hi=alpha_hi, **phys)]

def refined_separation_loss(theta_seq: jnp.ndarray, x0_ivl: irx.Interval,
                            cl_scenarios: List[Scenario], dt: float,
                            num_steps: int, pair_cost_fn=_soft_separation_loss):
    return propagate_with_refinement(x0_ivl, theta_seq, cl_scenarios, dt,
                                     num_steps, pair_cost_fn)

_MARGIN_TAU = 0.01

def unrefined_separation_loss(theta_seq: jnp.ndarray, x0_ivl: irx.Interval,
                              cl_scenarios: List[Scenario], dt: float,
                              num_steps: int, margin_tau=_MARGIN_TAU):
    return separation_loss_multistep(theta_seq, x0_ivl, cl_scenarios, dt, 1,
                                     margin_tau)

def theta0_from_open_loop(u_seq: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate(
        [jnp.zeros((u_seq.shape[0], NUM_GAINS)), u_seq], axis=-1)

def _init_theta(key, num_restarts: int, num_steps: int, shared_theta: bool,
                init_std_gain: float, theta0_mean: Optional[jnp.ndarray]):
    shape = (num_restarts,) if shared_theta else (num_restarts, num_steps)
    k_gain, k_ff = jax.random.split(key)
    gains = (jax.random.normal(k_gain, (*shape, NUM_GAINS)) * init_std_gain
             * jnp.repeat(_K_MAX_VEC, NUM_OUTPUTS))
    r_hover = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])
    ff = r_hover + jax.random.normal(k_ff, (*shape, NUM_INPUTS)) * 0.1 * _U_HALFRANGE
    theta0 = jnp.concatenate([gains, ff], axis=-1)
    if theta0_mean is not None:
        theta0 = theta0.at[0].set(jnp.zeros_like(theta0[0])) + theta0_mean[None, ...]
    return _project_theta(theta0)

def optimize_output_feedback_gpu(x0_ivl: irx.Interval, cl_scenarios: List[Scenario],
                                 dt: float, num_steps: int,
                                 num_restarts: int = 32, learning_rate: float = 0.05,
                                 num_iters: int = 60, seed: int = 42,
                                 init_std_gain: float = 0.3,
                                 shared_theta: bool = True,
                                 theta0_mean: Optional[jnp.ndarray] = None,
                                 restart_chunk: Optional[int] = None,
                                 optimizer: str = 'adam',
                                 gain_mask=1.0,
                                 pair_cost_fn=_soft_separation_loss,
                                 refined: bool = True):
    theta0 = _init_theta(jax.random.PRNGKey(seed), num_restarts, num_steps,
                         shared_theta, init_std_gain, theta0_mean)

    def _mask(theta):
        return theta.at[..., :NUM_GAINS].multiply(gain_mask)

    def loss_fn(theta):
        theta_seq = expand_theta(_mask(theta), num_steps)
        if refined:
            return refined_separation_loss(theta_seq, x0_ivl, cl_scenarios, dt,
                                           num_steps, pair_cost_fn)
        return unrefined_separation_loss(theta_seq, x0_ivl, cl_scenarios, dt,
                                         num_steps)

    value_and_grad = jax.value_and_grad(loss_fn)
    if restart_chunk is None or restart_chunk >= num_restarts:
        batched = jax.vmap(value_and_grad)
    elif num_restarts % restart_chunk:
        raise ValueError(
            f"restart_chunk ({restart_chunk}) must divide num_restarts "
            f"({num_restarts}) exactly -- a remainder makes lax.map compile a "
            "second vmapped body for the leftovers.")
    else:
        batched = lambda tb: jax.lax.map(value_and_grad, tb, batch_size=restart_chunk)

    if optimizer == 'gd':
        def body(carry, _i):
            theta, m, v, t = carry
            _, g = batched(theta)
            return (_project_theta(theta - learning_rate * g), m, v, t)
    elif optimizer == 'adam':
        b1, b2, eps = 0.9, 0.999, 1e-8

        def body(carry, _i):
            theta, m, v, t = carry
            _, g = batched(theta)
            t = t + 1
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            step = (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
            return (_project_theta(theta - learning_rate * _STEP_SCALE * step),
                    m, v, t)
    else:
        raise ValueError(f"optimizer must be 'adam' or 'gd', got {optimizer!r}")

    zeros = jnp.zeros_like(theta0)
    theta_final, _, _, _ = _run_unrolled_or_loop_nocheckpoint(
        body, (theta0, zeros, zeros, jnp.zeros((), jnp.int32)), num_iters)
    losses, _ = batched(theta_final)
    theta_final = _mask(theta_final)

    losses_valid = jnp.where(jnp.isnan(losses), jnp.inf, losses)
    best_idx = jnp.argmin(losses_valid)
    return theta_final[best_idx], losses[best_idx], theta_final, losses

def raw_state_histories(x0_ivl: irx.Interval, theta_seq: jnp.ndarray,
                        cl_scenarios: List[Scenario], dt: float) -> irx.Interval:
    emb_sys = cl_scenarios[0].emb_system
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in cl_scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in cl_scenarios]),
    )
    return jax.vmap(
        lambda p: _propagate_history(x0_ivl, theta_seq, emb_sys, p, dt, 1)
    )(p_batch)

def raw_pair_separated(hist: irx.Interval) -> np.ndarray:
    lo, hi = np.asarray(hist.lower), np.asarray(hist.upper)
    ii, jj = np.triu_indices(lo.shape[0], k=1)
    return np.any((lo[ii] > hi[jj]) | (lo[jj] > hi[ii]), axis=-1)

def raw_disjoint_segments(hist: irx.Interval) -> List[int]:
    sep = raw_pair_separated(hist)
    return [k for k in range(sep.shape[1]) if bool(np.all(sep[:, k]))]

def raw_disjoint_pair_counts(hist: irx.Interval) -> np.ndarray:
    return raw_pair_separated(hist).sum(axis=0)

def state_box_growth(x0_ivl: irx.Interval, hist: irx.Interval) -> float:
    w0 = float(jnp.max(x0_ivl.upper - x0_ivl.lower))
    w = float(np.max(np.asarray(hist.upper) - np.asarray(hist.lower)))
    return w / w0 if w0 > 0 else float('inf')

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Solve a quadrotor output-feedback separating controller "
                    "(intersection refinement).")
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--steps', type=int, default=5, help="horizon in steps")
    p.add_argument('--restarts', type=int, default=32)
    p.add_argument('--iters', type=int, default=60)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--x0-halfwidth', type=float, default=0.01)
    p.add_argument('--init-std-gain', type=float, default=0.3)
    p.add_argument('--gain-structure', choices=GAIN_STRUCTURES, default='dense')
    p.add_argument('--pair-cost', choices=['soft', 'overlap-volume'], default='soft',
                   help="'soft' is the margin proxy; 'overlap-volume' is the "
                        "12-dimension width product kept for compatibility")
    p.add_argument('--optimizer', choices=['adam', 'gd'], default='adam')
    p.add_argument('--unrefined', action='store_true',
                   help="skip intersection refinement (propagate every "
                        "scenario independently). Correct choice at --steps 1, "
                        "where refinement is vacuous: it first acts BETWEEN "
                        "steps 1 and 2, so over one Euler step the refined and "
                        "unrefined losses coincide by construction")
    p.add_argument('--restart-chunk', type=int, default=None)
    p.add_argument('--shared-theta', action=argparse.BooleanOptionalAction,
                   default=True,
                   help="single-stage: ONE time-invariant (K, r) for the whole "
                        "horizon (default)")
    p.add_argument('--baseline', action='store_true',
                   help="also solve the gain_mask=0 open-loop arm for a "
                        "like-for-like comparison (shares one compile)")
    p.add_argument('--out', type=str, default='results/quadrotor_output_feedback.npz')
    return p.parse_args(argv)

def main(argv=None):
    args = _parse_args(argv)
    print(f"Devices: {jax.devices()}")

    pair_cost_fn = _soft_separation_loss if args.pair_cost == 'soft' else _overlap_volume
    cl_scenarios = create_cl_scenarios(structure=args.gain_structure)
    n = len(cl_scenarios)
    n_pairs = n * (n - 1) // 2
    n_gains = int(_gain_mask_matrix(args.gain_structure).sum())
    print(f"{n} closed-loop scenarios / {n_pairs} pairs; "
          f"gain structure '{args.gain_structure}' ({n_gains} free gains), "
          f"outputs {OUTPUT_IDX}")

    x0_ivl = irx.icentpert(jnp.zeros(NUM_STATES),
                           jnp.ones(NUM_STATES) * args.x0_halfwidth)
    print(f"Horizon: {args.steps} x {args.dt}s = {args.steps*args.dt:.3f}s   "
          f"restarts={args.restarts}  iters={args.iters}  lr={args.lr}  "
          f"pair-cost={args.pair_cost}  "
          f"loss={'unrefined' if args.unrefined else 'refined'}")

    def run(seed, gain_mask):
        return optimize_output_feedback_gpu(
            x0_ivl=x0_ivl, cl_scenarios=cl_scenarios, dt=args.dt,
            num_steps=args.steps, num_restarts=args.restarts,
            learning_rate=args.lr, num_iters=args.iters, seed=seed,
            init_std_gain=args.init_std_gain, shared_theta=args.shared_theta,
            restart_chunk=args.restart_chunk, optimizer=args.optimizer,
            gain_mask=gain_mask, pair_cost_fn=pair_cost_fn,
            refined=not args.unrefined)

    jitted = jax.jit(run)
    print("\nCompiling (trig-heavy embedding: tan(theta), 1/cos(theta)) ...")
    t0 = time.perf_counter()
    out = jax.block_until_ready(jitted(args.seed, jnp.float32(1.0)))
    compile_t = time.perf_counter() - t0
    print(f"Compile time: {compile_t:.1f}s")

    t0 = time.perf_counter()
    out = jax.block_until_ready(jitted(args.seed, jnp.float32(1.0)))
    run_t = time.perf_counter() - t0
    print(f"Post-compile run time: {run_t*1e3:.2f}ms")
    print(f"Memory: {_memory_snapshot()}")

    arms = [('feedback', out)]
    if args.baseline:
        arms.append(('open-loop', jax.block_until_ready(
            jitted(args.seed, jnp.float32(0.0)))))

    rows = []
    for arm, (best_theta, best_loss, _, losses) in arms:
        theta_seq = expand_theta(best_theta, args.steps)
        K, r = theta_to_K_r(theta_seq[0])
        hist = raw_state_histories(x0_ivl, theta_seq, cl_scenarios, args.dt)
        counts = raw_disjoint_pair_counts(hist)
        full = raw_disjoint_segments(hist)
        growth = state_box_growth(x0_ivl, hist)
        n_nan = int(jnp.sum(jnp.isnan(losses)))
        print(f"\n{arm}:  loss={float(best_loss):.6e}  ({n_nan}/{args.restarts} NaN)")
        print(f"  max pairs separated = {int(counts.max())}/{n_pairs}   "
              f"per step {list(map(int, counts))}")
        print(f"  fully-separated steps = {full}   state-box growth = {growth:.1f}x")
        print(f"  |K|max per channel = "
              f"{[float(v) for v in jnp.max(jnp.abs(K), axis=1)]}")
        print(f"  r = {[float(v) for v in r]}  (hover thrust {_HOVER_THRUST:.4f})")
        rows.append(dict(arm=arm, loss=float(best_loss), growth=growth,
                         max_sep=int(counts.max()), n_full=len(full),
                         first_full=(full[0] if full else -1),
                         theta=np.asarray(best_theta)))

    if len(rows) == 2:
        fb, ol = rows[0], rows[1]
        excess = fb['growth'] / ol['growth'] if ol['growth'] > 0 else float('inf')
        print(f"\nFeedback vs open-loop: box growth {fb['growth']:.1f}x vs "
              f"{ol['growth']:.1f}x ({excess:.2f}x excess), "
              f"first fully-separated step {fb['first_full']} vs {ol['first_full']}")
        if excess > 2.0:
            print("  WARNING: feedback more than doubles the baseline's box "
                  "growth -- its interval conservatism is dominating; try "
                  "--gain-structure rate/diagonal or a smaller _K_REL.")
    elif rows[0]['growth'] > 10.0:
        print(f"\nNote: box growth {rows[0]['growth']:.1f}x. Re-run with "
              "--baseline to see whether feedback caused it -- this system's "
              "open-loop arm grows comparably on its own.")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _HERE / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             **{k: np.array([r[k] for r in rows])
                for k in ('arm', 'loss', 'growth', 'max_sep', 'n_full',
                          'first_full')},
             theta=np.stack([r['theta'] for r in rows]),
             dt=args.dt, num_steps=args.steps,
             gain_structure=args.gain_structure, output_idx=np.array(OUTPUT_IDX),
             refined=not args.unrefined,
             shared_theta=args.shared_theta, k_rel=_K_REL,
             x0_halfwidth=args.x0_halfwidth,
             compile_t=compile_t, run_t=run_t)
    print(f"\nSaved {out_path}")

if __name__ == "__main__":
    main()
