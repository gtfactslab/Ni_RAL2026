import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import jax
import jax.numpy as jnp
import immrax as irx

from go2_separating_input import (
    create_scenarios,
    MultistepSequenceOptimizer,
    optimize_multistep_gpu,
    propagate_scenario_multistep,
    position_interval,
    _overlap_volume,
    time_jit,
)

def main():
    print("Devices:", jax.devices())

    scenarios = create_scenarios()
    x0_ivl = irx.Interval(
        lower=jnp.array([-0.05, -0.05, -0.02]),
        upper=jnp.array([ 0.05,  0.05,  0.02]),
    )
    dt, steps_per_segment, num_segments = 0.25, 2, 10
    print(f"Horizon: {num_segments} segments x {steps_per_segment} steps x {dt} s "
          f"= {num_segments * steps_per_segment * dt} s total")

    ms_opt = MultistepSequenceOptimizer(scenarios, x0_ivl, dt, steps_per_segment, num_segments)

    def full_multistart(seed):
        return optimize_multistep_gpu(ms_opt, num_restarts=50, learning_rate=0.01, num_iters=150, seed=seed)

    jitted_fn, compile_t, run_t, mem = time_jit(full_multistart, 42)
    print(f"full multistart:   compile {compile_t * 1e3:8.2f} ms   run {run_t * 1e3:7.3f} ms")

    u_seq_opt, loss_opt, _, _ = jitted_fn(42)
    print(f"\nBest loss: {float(loss_opt):.6f}")
    print(f"Optimal control sequence (first 5 segments, [vx, vy, omega]): "
          f"{[[round(float(v), 3) for v in row] for row in u_seq_opt[:5]]}")

    pos_ivls = [
        position_interval(propagate_scenario_multistep(x0_ivl, u_seq_opt, s, dt, steps_per_segment))
        for s in scenarios
    ]
    n = len(scenarios)
    print("\nPairwise output-space overlaps (at final segment):")
    for i in range(n):
        for j in range(i + 1, n):
            ov = float(_overlap_volume(pos_ivls[i], pos_ivls[j]))
            print(f"  {scenarios[i].name} vs {scenarios[j].name}: {ov:.6f}")
    print("\nOutput-interval volumes (at final segment):")
    for s, iv in zip(scenarios, pos_ivls):
        print(f"  {s.name}: {float(jnp.prod(iv.upper - iv.lower)):.6f}")
    print(f"\nMemory snapshot: {mem}")

if __name__ == "__main__":
    main()
