import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import jax
import jax.numpy as jnp
import immrax as irx

from admire_separating_input import (
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

    x0_nom = jnp.zeros(9).at[0].set(343.0 * 0.3)
    x0_ivl = irx.icentpert(x0_nom, jnp.ones(9) * 0.01)
    dt, num_steps = 0.1, 10
    print(f"\nPropagation: {num_steps} x {dt} s = {num_steps * dt} s total")
    print("Separation objective: angular rates (pb, qb, rb)")

    opt = SeparatingInputOptimizer(scenarios, x0_ivl, dt, num_steps)

    def full_multistart(seed):
        return optimize_parallel_gpu(opt, num_restarts=10, learning_rate=0.05, num_iters=50, seed=seed)

    print("\nCompiling (division-heavy dynamics -- may take a while, see module docstring) ...")
    jitted_fn, compile_t, run_t, mem = time_jit(full_multistart, 42)
    print(f"\nfull multistart:   compile {compile_t:8.2f} s   run {run_t * 1e3:7.3f} ms")

    u_opt, loss_opt, _, _ = jitted_fn(42)
    stats = opt.evaluate(u_opt)

    print(f"\nOptimal separating input (rad): {[round(float(v), 4) for v in u_opt]}")
    print(f"Best loss: {float(loss_opt):.6f}")
    print("\nPairwise output-space overlaps (nonzero only):")
    for k, v in stats['pairwise_overlaps'].items():
        if v > 0:
            print(f"  {k}: {v:.6f}")
    print(f"\nMemory snapshot: {mem}")

if __name__ == "__main__":
    main()
