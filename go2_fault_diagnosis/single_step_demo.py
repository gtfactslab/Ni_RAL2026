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
    SeparatingInputOptimizer,
    optimize_parallel_gpu,
    time_jit,
)

def main():
    print("Devices:", jax.devices())

    scenarios = create_scenarios()
    print(f"{len(scenarios)} fault scenarios:")
    for s in scenarios:
        print(f"  - {s.name}")

    x0_ivl = irx.Interval(
        lower=jnp.array([-0.05, -0.05, -0.02]),
        upper=jnp.array([ 0.05,  0.05,  0.02]),
    )
    dt, num_steps = 0.5, 10
    print(f"\nPropagation: {num_steps} x {dt} s = {num_steps * dt} s total")

    opt = SeparatingInputOptimizer(scenarios, x0_ivl, dt, num_steps)

    def full_multistart(seed):
        return optimize_parallel_gpu(opt, num_restarts=50, learning_rate=0.01, num_iters=150, seed=seed)

    jitted_fn, compile_t, run_t, mem = time_jit(full_multistart, 42)
    print(f"\nfull multistart:   compile {compile_t * 1e3:8.2f} ms   run {run_t * 1e3:7.3f} ms")

    u_opt, loss_opt, _, _ = jitted_fn(42)
    stats = opt.evaluate(u_opt)

    print(f"\nOptimal separating input: vx={float(u_opt[0]):+.4f}  "
          f"vy={float(u_opt[1]):+.4f}  omega={float(u_opt[2]):+.4f}")
    print(f"Best loss: {float(loss_opt):.6f}")
    print("\nPairwise output-space overlaps:")
    for k, v in stats['pairwise_overlaps'].items():
        print(f"  {k}: {v:.6f}")
    print("\nOutput-interval volumes:")
    for k, v in stats['volumes'].items():
        print(f"  {k}: {v:.6f}")
    print(f"\nMemory snapshot: {mem}")

if __name__ == "__main__":
    main()
