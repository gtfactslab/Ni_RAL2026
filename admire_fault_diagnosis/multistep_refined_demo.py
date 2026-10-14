import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import jax
import jax.numpy as jnp
import immrax as irx

from admire_separating_input import create_scenarios, _propagate_history, optimize_refined_gpu

DT = 0.1
NUM_STEPS = 10
NUM_RESTARTS = 6
NUM_ITERS = 15              
LEARNING_RATE = 0.05
SEED = 42
X0_HALFWIDTH = 5e-3

def main():
    print(f"Devices: {jax.devices()}")
    scenarios = create_scenarios(fault_effectiveness=0.0)
    print(f"{len(scenarios)} fault scenarios (Nominal + 10 total-actuator-loss faults)")

    x0_nom = jnp.zeros(9).at[0].set(343.0 * 0.3)
    x0_ivl = irx.icentpert(x0_nom, jnp.ones(9) * X0_HALFWIDTH)
    print(f"Horizon: {NUM_STEPS} x {DT}s = {NUM_STEPS*DT:.2f}s   "
          f"restarts={NUM_RESTARTS}  iters={NUM_ITERS}  lr={LEARNING_RATE}")

    def run(seed):
        return optimize_refined_gpu(
            x0_ivl=x0_ivl, scenarios=scenarios, dt=DT, num_steps=NUM_STEPS,
            num_restarts=NUM_RESTARTS, learning_rate=LEARNING_RATE,
            num_iters=NUM_ITERS, seed=seed,
        )

    jitted = jax.jit(run)
    print("\nCompiling (see module docstring -- expect several minutes, "
          "likely more than the unrefined script) ...")
    t0 = time.perf_counter()
    out = jax.block_until_ready(jitted(SEED))
    compile_t = time.perf_counter() - t0
    print(f"Compile time: {compile_t:.1f}s")

    t0 = time.perf_counter()
    out = jax.block_until_ready(jitted(SEED))
    run_t = time.perf_counter() - t0
    print(f"Post-compile run time: {run_t*1e3:.2f}ms")

    best_u, best_loss, u_final, losses = out
    n_nan = int(jnp.sum(jnp.isnan(losses)))
    print(f"\nBest (refined) loss: {float(best_loss):.6f}  ({n_nan}/{NUM_RESTARTS} restarts NaN)")
    print(f"Loss distribution: min={float(jnp.nanmin(losses)):.6f}  "
          f"mean={float(jnp.nanmean(losses)):.6f}  max={float(jnp.nanmax(losses)):.6f}")

    hist_by_name = {
        s.name: _propagate_history(x0_ivl, best_u, s.emb_system, s.p_interval, DT, 1)
        for s in scenarios
    }
    n = len(scenarios)
    disjoint_segments = []
    for k in range(NUM_STEPS):
        all_disjoint_k = True
        for i in range(n):
            for j in range(i + 1, n):
                oi = irx.Interval(lower=hist_by_name[scenarios[i].name].lower[k, 3:6],
                                  upper=hist_by_name[scenarios[i].name].upper[k, 3:6])
                oj = irx.Interval(lower=hist_by_name[scenarios[j].name].lower[k, 3:6],
                                  upper=hist_by_name[scenarios[j].name].upper[k, 3:6])
                disjoint = bool(jnp.any((oi.lower > oj.upper) | (oj.lower > oi.upper)))
                all_disjoint_k &= disjoint
        if all_disjoint_k:
            disjoint_segments.append(k)
    print(f"\nSegments with ALL {n*(n-1)//2} pairs genuinely (RAW, independent-box) "
          f"disjoint: {disjoint_segments}")
    print(f"At least one genuinely-disjoint segment: {len(disjoint_segments) > 0}")

    np.savez(_HERE / "multistep_refined_result.npz",
             u_seq=np.array(best_u), loss=float(best_loss), dt=DT,
             num_steps=NUM_STEPS, compile_t=compile_t, run_t=run_t,
             disjoint_segments=np.array(disjoint_segments, dtype=np.int32))
    print(f"\nSaved {_HERE / 'multistep_refined_result.npz'}")

if __name__ == "__main__":
    main()
