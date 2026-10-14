import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import jax
import jax.numpy as jnp
import immrax as irx

from car_separating_input import (
    create_scenarios,
    optimize_multistep,
    optimize_refined_gpu,
    refined_overlap_loss,
    separation_loss_multistep,
    time_jit,
)

def main():
    print("Devices:", jax.devices())

    scenarios = create_scenarios()
    x0_ivl = irx.icentpert(jnp.array([0.1, 0.1, 0.0]), jnp.array([0.05, 0.05, 0.05]))
    dt = 0.15
    num_steps = 8

    print("\nOptimizing UNREFINED (uninformed) separating input sequence...")
    u_seq_unrefined, loss_unrefined, _ = optimize_multistep(
        scenarios, x0_ivl, dt=dt, steps_per_segment=1, num_segments=num_steps,
        learning_rate=0.05, num_iters=100, num_restarts=30, verbose=False,
    )
    print(f"  unrefined best loss: {loss_unrefined:.6f}")

    def full_refined_multistart(seed):
        return optimize_refined_gpu(
            x0_ivl, scenarios, dt=dt, num_steps=num_steps,
            num_restarts=30, learning_rate=0.05, num_iters=100, seed=seed,
        )

    jitted_refined_fn, compile_t, run_t, mem = time_jit(full_refined_multistart, 42)
    print(f"\nrefined multistart: compile {compile_t * 1e3:8.2f} ms   run {run_t * 1e3:7.3f} ms")

    u_seq_refined, loss_refined, _, _ = jitted_refined_fn(42)
    print(f"  refined best loss:   {float(loss_refined):.6f}")

    refined_loss_at_unrefined_u = float(
        refined_overlap_loss(u_seq_unrefined, x0_ivl, scenarios, dt=dt, num_steps=num_steps)
    )
    print(f"\nSame control sequence (unrefined-optimal u_seq):")
    print(f"  unrefined loss: {loss_unrefined:.6f}")
    print(f"  refined loss:   {refined_loss_at_unrefined_u:.6f}"
          f"  ({'<=' if refined_loss_at_unrefined_u <= loss_unrefined + 1e-5 else '>'} unrefined, as expected)")

    print(f"\nMemory snapshot: {mem}")

if __name__ == "__main__":
    main()
