import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import jax
import jax.numpy as jnp
import immrax as irx

from quadrotor_separating_input import (
    create_scenarios,
    SeparatingInputOptimizer,
    optimize_parallel_gpu,
    MultistepSequenceOptimizer,
    optimize_multistep_gpu,
    separation_loss_multistep,
    refined_overlap_loss,
    optimize_refined_gpu,
    time_jit,
    _HOVER_THRUST,
)

DT = 0.01
X0_WIDTH = 0.01

def _print_scenario_report(title, scenarios, x_ivls, u_desc):
    n = len(scenarios)
    print(f"\n{title}")
    print(f"  separating input: {u_desc}")
    print("  pairwise state-interval overlaps (0 = fully separated):")
    for i in range(n):
        for j in range(i + 1, n):
            from quadrotor_separating_input import _overlap_volume
            ov = float(_overlap_volume(x_ivls[i], x_ivls[j]))
            print(f"    {scenarios[i].name:16s} vs {scenarios[j].name:16s}: {ov:.6e}")
    print("  reachable-set volumes:")
    for s, iv in zip(scenarios, x_ivls):
        vol = float(jnp.prod(iv.upper - iv.lower))
        print(f"    {s.name:16s}: {vol:.6e}")

def run_single_step():
    print("=" * 78)
    print("LAYER 1: single-step separating input  (num_steps=2)")
    print("=" * 78)
    scenarios = create_scenarios()
    x0 = irx.icentpert(jnp.zeros(12), jnp.full(12, X0_WIDTH))
    opt = SeparatingInputOptimizer(scenarios, x0, dt=DT, num_steps=2)

    def full_multistart(seed):
        return optimize_parallel_gpu(opt, num_restarts=20, learning_rate=0.02, num_iters=30, seed=seed)

    jitted_fn, compile_t, run_t, mem = time_jit(full_multistart, 42)
    print(f"compile {compile_t*1e3:8.2f} ms   run {run_t*1e3:7.3f} ms   mem {mem}")

    u_opt, loss_opt, _, _ = jitted_fn(42)
    stats = opt.evaluate(u_opt)
    u_str = f"[U1={float(u_opt[0]):.4f} N, U2={float(u_opt[1]):.4f}, U3={float(u_opt[2]):.4f}, U4={float(u_opt[3]):.4f} N*m]"
    _print_scenario_report(f"Best loss: {float(loss_opt):.6e}", scenarios, stats['state_intervals'], u_str)

def run_multistep_unrefined():
    print("\n" + "=" * 78)
    print("LAYER 2: multistep UNREFINED separating input  (steps_per_segment=1, num_segments=5)")
    print("=" * 78)
    scenarios = create_scenarios()
    x0 = irx.icentpert(jnp.zeros(12), jnp.full(12, X0_WIDTH))
    ms_opt = MultistepSequenceOptimizer(scenarios, x0, dt=DT, steps_per_segment=1, num_segments=5)

    def full_multistart(seed):
        return optimize_multistep_gpu(ms_opt, num_restarts=10, learning_rate=0.02, num_iters=20, seed=seed)

    jitted_fn, compile_t, run_t, mem = time_jit(full_multistart, 42)
    print(f"compile {compile_t*1e3:8.2f} ms   run {run_t*1e3:7.3f} ms   mem {mem}")

    u_seq_opt, loss_opt, _, _ = jitted_fn(42)
    from quadrotor_separating_input import _propagate_history
    _emb_sys = scenarios[0].emb_system
    _p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )
    _hist_batch = jax.vmap(lambda p: _propagate_history(x0, u_seq_opt, _emb_sys, p, DT, 1))(_p_batch)
    x_ivls = [irx.Interval(lower=_hist_batch.lower[i, -1], upper=_hist_batch.upper[i, -1])
              for i in range(len(scenarios))]
    u_str = f"u_seq[0] = [U1={float(u_seq_opt[0,0]):.4f}, U2={float(u_seq_opt[0,1]):.4f}, U3={float(u_seq_opt[0,2]):.4f}, U4={float(u_seq_opt[0,3]):.4f}]"
    _print_scenario_report(f"Best loss: {float(loss_opt):.6e}", scenarios, x_ivls, u_str)
    return u_seq_opt, scenarios, x0

def run_refined(u_seq_unrefined, scenarios, x0):
    print("\n" + "=" * 78)
    print("LAYER 3: multistep REFINED separating input  (num_steps=2)")
    print("=" * 78)

    def full_multistart(seed):
        return optimize_refined_gpu(x0, scenarios, dt=DT, num_steps=2, num_restarts=10,
                                    learning_rate=0.02, num_iters=15, seed=seed)

    jitted_fn, compile_t, run_t, mem = time_jit(full_multistart, 42)
    print(f"compile {compile_t*1e3:8.2f} ms   run {run_t*1e3:7.3f} ms   mem {mem}")

    u_seq_opt, loss_opt, _, _ = jitted_fn(42)
    print(f"\nBest refined loss: {float(loss_opt):.6e}")
    print(f"  separating input: u_seq[0] = [U1={float(u_seq_opt[0,0]):.4f}, U2={float(u_seq_opt[0,1]):.4f}, "
          f"U3={float(u_seq_opt[0,2]):.4f}, U4={float(u_seq_opt[0,3]):.4f}]")

    u_seq_cmp = u_seq_unrefined[:2]
    unrefined_loss_here = float(separation_loss_multistep(u_seq_cmp, x0, scenarios, dt=DT, steps_per_segment=1))
    refined_loss_here = float(refined_overlap_loss(u_seq_cmp, x0, scenarios, dt=DT, num_steps=2))
    print(f"\nSame 2-step control sequence (unrefined-optimal u_seq, truncated):")
    print(f"  unrefined loss: {unrefined_loss_here:.6e}")
    print(f"  refined loss:   {refined_loss_here:.6e}  "
          f"({'<=' if refined_loss_here <= unrefined_loss_here + 1e-6 else '>'} unrefined, as expected)")

if __name__ == "__main__":
    print("Devices:", jax.devices())
    print(f"Hover thrust (mg): {_HOVER_THRUST:.4f} N")

    run_single_step()
    u_seq_unrefined, scenarios, x0 = run_multistep_unrefined()
    run_refined(u_seq_unrefined, scenarios, x0)

    print("\nDone.")
