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
    MultistepSequenceOptimizer,
    optimize_multistep_gpu,
    _propagate_history,
    _overlap_volume,
    time_jit,
    _HOVER_THRUST,
)

DT = 0.2
STEPS_PER_SEGMENT = 1
NUM_SEGMENTS = 5
X0_WIDTH = 0.01

def main():
    print("Devices:", jax.devices())
    print(f"Hover thrust (mg): {_HOVER_THRUST:.4f} N")
    print("=" * 78)
    print(f"multistep UNREFINED separating input  "
          f"(steps_per_segment={STEPS_PER_SEGMENT}, num_segments={NUM_SEGMENTS}, "
          f"dt={DT}s -> {STEPS_PER_SEGMENT*DT}s/segment, {NUM_SEGMENTS*STEPS_PER_SEGMENT*DT}s total)")
    print("=" * 78)

    scenarios = create_scenarios()
    x0 = irx.icentpert(jnp.zeros(12), jnp.full(12, X0_WIDTH))
    ms_opt = MultistepSequenceOptimizer(
        scenarios, x0, dt=DT, steps_per_segment=STEPS_PER_SEGMENT, num_segments=NUM_SEGMENTS,
    )

    def full_multistart(seed):
        return optimize_multistep_gpu(ms_opt, num_restarts=10, learning_rate=0.02, num_iters=20, seed=seed)

    jitted_fn, compile_t, run_t, mem = time_jit(full_multistart, 42)
    print(f"compile {compile_t*1e3:8.2f} ms   run {run_t*1e3:7.3f} ms   mem {mem}")

    u_seq_opt, loss_opt, _, _ = jitted_fn(42)
    print(f"\nBest loss (min over all {NUM_SEGMENTS} measurement points): {float(loss_opt):.6e}")
    print("separating input per segment:")
    for k in range(NUM_SEGMENTS):
        t_lo, t_hi = k * STEPS_PER_SEGMENT * DT, (k + 1) * STEPS_PER_SEGMENT * DT
        print(f"  segment {k} (t=[{t_lo:.1f},{t_hi:.1f}]s): "
              f"U1={float(u_seq_opt[k,0]):.4f} N, U2={float(u_seq_opt[k,1]):.4f}, "
              f"U3={float(u_seq_opt[k,2]):.4f}, U4={float(u_seq_opt[k,3]):.4f} N*m")

    n = len(scenarios)
    _emb_sys = scenarios[0].emb_system
    _p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )
    _hist_batch = jax.vmap(
        lambda p: _propagate_history(x0, u_seq_opt, _emb_sys, p, DT, STEPS_PER_SEGMENT)
    )(_p_batch)
    x_hists = [irx.Interval(lower=_hist_batch.lower[i], upper=_hist_batch.upper[i])
              for i in range(n)]
    for k in range(NUM_SEGMENTS):
        t = (k + 1) * STEPS_PER_SEGMENT * DT
        print(f"\npairwise state-interval overlaps at t={t:.1f}s (measurement {k+1}, 0 = fully separated):")
        for i in range(n):
            for j in range(i + 1, n):
                xi = irx.Interval(lower=x_hists[i].lower[k], upper=x_hists[i].upper[k])
                xj = irx.Interval(lower=x_hists[j].lower[k], upper=x_hists[j].upper[k])
                ov = float(_overlap_volume(xi, xj))
                print(f"  {scenarios[i].name:16s} vs {scenarios[j].name:16s}: {ov:.6e}")
        print(f"reachable-set volumes at t={t:.1f}s:")
        for i, s in enumerate(scenarios):
            xi = irx.Interval(lower=x_hists[i].lower[k], upper=x_hists[i].upper[k])
            vol = float(jnp.prod(xi.upper - xi.lower))
            print(f"  {s.name:16s}: {vol:.6e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
