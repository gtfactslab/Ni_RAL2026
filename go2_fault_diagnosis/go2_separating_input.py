import os
import sys
import time
import resource
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import jax
import jax.numpy as jnp
from jax import lax
import immrax as irx
import numpy as np

_U_LO = jnp.array([-.6, -.6, -.6])
_U_HI = jnp.array([ .6,  .6,  .6])

def _project_u(u: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(u, _U_LO, _U_HI)

def _overlap_volume(ivl1: irx.Interval, ivl2: irx.Interval) -> jnp.ndarray:
    lo = jnp.maximum(ivl1.lower, ivl2.lower)
    hi = jnp.minimum(ivl1.upper, ivl2.upper)

    def has_overlap(operands):
        lower, upper = operands
        return jnp.prod(upper - lower)

    def no_overlap(operands):
        del operands
        return jnp.array(0.0)

    has_no_overlap = jnp.any(hi < lo)
    return lax.cond(jnp.logical_not(has_no_overlap), has_overlap, no_overlap, (lo, hi))

def _scan_loop(step_fn, init, n: int):
    def body(carry, _):
        return step_fn(carry), None
    carry, _ = jax.lax.scan(body, init, xs=None, length=n)
    return carry

def _memory_snapshot() -> Dict[str, float]:
    try:
        stats = jax.devices()[0].memory_stats()
        if stats:
            return {k: float(v) for k, v in stats.items() if 'bytes' in k}
    except Exception:
        pass
    return {'ru_maxrss_kb': float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}

def time_jit(fn, *args, **kwargs):
    jitted_fn = jax.jit(fn)

    t0 = time.perf_counter()
    out = jax.block_until_ready(jitted_fn(*args, **kwargs))
    compile_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = jax.block_until_ready(jitted_fn(*args, **kwargs))
    run_time_s = time.perf_counter() - t0

    del out
    return jitted_fn, compile_time_s, run_time_s, _memory_snapshot()

class Go2NomActSystem(irx.System):

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 3

    def f(self, t, x, u, p):
        vx, vy, omega = u[0], u[1], u[2]
        alpha = p[0]
        beta = p[1]
        theta = x[2]
        return jnp.array([
            vx * jnp.cos(theta) - alpha * vy * jnp.sin(theta),
            vx * jnp.sin(theta) + alpha * vy * jnp.cos(theta),
            beta * omega,
        ])

class Go2SensorFaultSystem(irx.System):

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 3

    def f(self, t, x, u, p):
        vx       = u[0]
        vy       = u[1]
        vy_noise = p[0]
        alpha = p[1]
        beta = p[2]
        omega    = u[2]
        theta    = x[2]
        vy_corrupted = (1 - omega ** 2) * alpha * vy + omega ** 2 * vy_noise
        return jnp.array([
            vx * jnp.cos(theta) - vy_corrupted * jnp.sin(theta),
            vx * jnp.sin(theta) + vy_corrupted * jnp.cos(theta),
            beta * omega,
        ])

_NOM_ACT_SYS = Go2NomActSystem()
_NOM_ACT_EMB = irx.natemb(_NOM_ACT_SYS)

_SF_SYS = Go2SensorFaultSystem()
_SF_EMB = irx.natemb(_SF_SYS)

@dataclass
class Scenario:
    name: str
    emb_system: object
    p_interval: irx.Interval

def create_scenarios(
    actuator_alpha_lo: float = 0.60,
    actuator_alpha_hi: float = 0.80,
    actuator_beta_low: float = 0.60,
    actuator_beta_high: float = .80,
    sensor_noise_bound: float = 0.25,
) -> List[Scenario]:
    return [
        Scenario(
            name="Nominal",
            emb_system=_NOM_ACT_EMB,
            p_interval=irx.icentpert(jnp.array([1.0, 1.0]), jnp.zeros(2)),
        ),
        Scenario(
            name="Actuator Fault",
            emb_system=_NOM_ACT_EMB,
            p_interval=irx.Interval(
                lower=jnp.array([actuator_alpha_lo, actuator_beta_low]),
                upper=jnp.array([actuator_alpha_hi, actuator_beta_high]),
            ),
        ),
        Scenario(
            name="Sensor Fault",
            emb_system=_SF_EMB,
            p_interval=irx.Interval(
                lower=jnp.array([-sensor_noise_bound, 1.0, 1.0]),
                upper=jnp.array([ sensor_noise_bound, 1.0, 1.0]),
            ),
        ),
        Scenario(
            name="Simultaneous Fault",
            emb_system=_SF_EMB,
            p_interval=irx.Interval(
                lower=jnp.array([-sensor_noise_bound, actuator_alpha_lo / 2, actuator_beta_low / 2]),
                upper=jnp.array([ sensor_noise_bound, actuator_alpha_hi / 2, actuator_beta_high / 2]),
            ),
        ),
    ]

def euler_step(emb_sys, x_ivl: irx.Interval, u: jnp.ndarray,
               p_ivl: irx.Interval, dt: float) -> irx.Interval:
    _t    = jnp.zeros(())
    x_ut  = irx.i2ut(x_ivl)
    dx_ut = emb_sys.f(_t, x_ut, u, p_ivl)
    return irx.ut2i(dx_ut * dt + x_ut)

def propagate_scenario(x0_ivl: irx.Interval, u: jnp.ndarray,
                       scenario: Scenario,
                       dt: float, num_steps: int) -> irx.Interval:
    emb = scenario.emb_system
    p   = scenario.p_interval

    @jax.checkpoint
    def body(x_carry, _):
        return euler_step(emb, x_carry, u, p, dt), None

    x_final, _ = jax.lax.scan(body, x0_ivl, xs=None, length=num_steps)
    return x_final

def _propagate_history(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                       emb_sys, p_ivl: irx.Interval,
                       dt: float, steps_per_segment: int) -> irx.Interval:
    def segment(x_ivl, u_k):
        @jax.checkpoint
        def euler_body(x, _): return euler_step(emb_sys, x, u_k, p_ivl, dt), None
        x_end, _ = jax.lax.scan(euler_body, x_ivl, xs=None, length=steps_per_segment)
        return x_end, x_end

    _, x_hist = jax.lax.scan(jax.checkpoint(segment), x0_ivl, u_seq)
    return x_hist

def propagate_scenario_multistep(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                                 scenario: Scenario,
                                 dt: float, steps_per_segment: int) -> irx.Interval:
    x_hist = _propagate_history(
        x0_ivl, u_seq, scenario.emb_system, scenario.p_interval, dt, steps_per_segment
    )
    return irx.Interval(lower=x_hist.lower[-1], upper=x_hist.upper[-1])

def position_interval(x_ivl: irx.Interval) -> irx.Interval:
    return irx.Interval(lower=x_ivl.lower[:2], upper=x_ivl.upper[:2])

def separation_loss(u: jnp.ndarray,
                    x0_ivl: irx.Interval,
                    scenarios: List[Scenario],
                    dt: float,
                    num_steps: int) -> jnp.ndarray:
    pos_ivls = [
        position_interval(propagate_scenario(x0_ivl, u, s, dt, num_steps))
        for s in scenarios
    ]
    n = len(pos_ivls)
    total = jnp.array(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            total = total + _overlap_volume(pos_ivls[i], pos_ivls[j])
    return total

def separation_loss_multistep(u_seq: jnp.ndarray,
                              x0_ivl: irx.Interval,
                              scenarios: List[Scenario],
                              dt: float,
                              steps_per_segment: int) -> jnp.ndarray:
    num_segments = u_seq.shape[0]
    n = len(scenarios)

    emb_groups: Dict[int, tuple] = {}
    for idx, s in enumerate(scenarios):
        eid = id(s.emb_system)
        if eid not in emb_groups:
            emb_groups[eid] = (s.emb_system, [], [])
        emb_groups[eid][1].append(idx)
        emb_groups[eid][2].append(s.p_interval)

    x_hist_all: List[irx.Interval] = [None] * n
    for emb_sys, indices, p_ivls in emb_groups.values():
        p_batch = irx.Interval(
            lower=jnp.stack([p.lower for p in p_ivls]),
            upper=jnp.stack([p.upper for p in p_ivls]),
        )

        def prop_one(p_ivl_single):
            return _propagate_history(
                x0_ivl, u_seq, emb_sys, p_ivl_single, dt, steps_per_segment
            )

        x_hist_batch = jax.vmap(prop_one)(p_batch)

        for local_i, global_i in enumerate(indices):
            x_hist_all[global_i] = irx.Interval(
                lower=x_hist_batch.lower[local_i],
                upper=x_hist_batch.upper[local_i],
            )

    def overlap_at_k(k):
        pos_ivls_k = [
            irx.Interval(
                lower=x_hist_all[i].lower[k, :2],
                upper=x_hist_all[i].upper[k, :2],
            )
            for i in range(n)
        ]
        total = jnp.array(0.0)
        for i in range(n):
            for j in range(i + 1, n):
                total = total + _overlap_volume(pos_ivls_k[i], pos_ivls_k[j])
        return total

    segment_overlaps = jnp.stack([overlap_at_k(k) for k in range(num_segments)])
    return jnp.min(segment_overlaps)

class SeparatingInputOptimizer:

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval,
                 dt: float, num_steps: int):
        self.scenarios = scenarios
        self.x0_ivl    = x0_ivl
        self.dt        = dt
        self.num_steps = num_steps

        _loss = partial(separation_loss,
                        x0_ivl=x0_ivl, scenarios=scenarios,
                        dt=dt, num_steps=num_steps)
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))

    def optimize(self,
                 u_init: Optional[jnp.ndarray] = None,
                 learning_rate: float = 0.01,
                 num_iters: int = 150,
                 verbose: bool = False) -> Tuple[jnp.ndarray, float]:
        if u_init is None:
            u_init = jnp.array([0.5, 0.0, 0.3])
        u = u_init

        for i in range(num_iters):
            g = self.grad_fn(u)
            u = _project_u(u - learning_rate * g)

            if verbose and (i % 20 == 0 or i == num_iters - 1):
                loss  = float(self.loss_fn(u))
                gnorm = float(jnp.linalg.norm(g))
                print(f"  Iter {i:4d}  loss={loss:.6f}  "
                      f"u=[{float(u[0]):+.3f},{float(u[1]):+.3f},{float(u[2]):+.3f}]"
                      f"  |g|={gnorm:.4f}")

        return u, float(self.loss_fn(u))

    def evaluate(self, u: jnp.ndarray) -> Dict:
        pos_ivls = [
            position_interval(
                propagate_scenario(self.x0_ivl, u, s, self.dt, self.num_steps)
            )
            for s in self.scenarios
        ]
        n = len(self.scenarios)
        overlaps = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{self.scenarios[i].name} vs {self.scenarios[j].name}"
                overlaps[key] = float(_overlap_volume(pos_ivls[i], pos_ivls[j]))

        volumes = {
            s.name: float(jnp.prod(iv.upper - iv.lower))
            for s, iv in zip(self.scenarios, pos_ivls)
        }
        return {
            'position_intervals': pos_ivls,
            'pairwise_overlaps':  overlaps,
            'volumes':            volumes,
        }

class MultistepSequenceOptimizer:

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval,
                 dt: float, steps_per_segment: int, num_segments: int):
        self.scenarios = scenarios
        self.x0_ivl = x0_ivl
        self.dt = dt
        self.steps_per_segment = steps_per_segment
        self.num_segments = num_segments

        _loss = partial(
            separation_loss_multistep,
            x0_ivl=x0_ivl,
            scenarios=scenarios,
            dt=dt,
            steps_per_segment=steps_per_segment,
        )
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))

def optimize_multistart(opt: 'SeparatingInputOptimizer',
                        num_restarts: int = 3,
                        learning_rate: float = 0.01,
                        num_iters: int = 150,
                        verbose: bool = False,
                        ) -> Tuple[jnp.ndarray, float, Dict]:
    best_u, best_loss, best_stats = None, float('inf'), None
    key  = jax.random.PRNGKey(42)
    restart_times: List[float] = []

    for r in range(num_restarts):
        key, subkey = jax.random.split(key)
        u_init = (jax.random.normal(subkey, (3,)) * 0.3
                  + jnp.array([0.5, 0.0, 0.3]))

        if verbose:
            print(f"\n{'-'*60}\nRestart {r+1}/{num_restarts}\n{'-'*60}")

        _t0 = time.perf_counter()
        u_opt, loss = opt.optimize(u_init, learning_rate, num_iters, verbose)
        restart_times.append(time.perf_counter() - _t0)

        stats = opt.evaluate(u_opt)

        if loss < best_loss:
            best_loss, best_u, best_stats = loss, u_opt, stats
            if verbose:
                print(f"  -> New best: {loss:.6f}")
        if loss == 0:
            break

    best_stats['restart_times_s'] = restart_times
    return best_u, best_loss, best_stats

def optimize_parallel(
    opt: 'SeparatingInputOptimizer',
    num_restarts: int = 100,
    learning_rate: float = 0.01,
    num_iters: int = 150,
    max_workers: Optional[int] = None,
    verbose: bool = False,
    seed: int = 42,
) -> Tuple[jnp.ndarray, float, Dict]:
    if num_restarts <= 0:
        raise ValueError("num_restarts must be >= 1")

    subkeys = jax.random.split(jax.random.PRNGKey(seed), num_restarts)

    _u_warm = jnp.array([0.5, 0.0, 0.3])
    _ = opt.loss_fn(_u_warm)
    _ = opt.grad_fn(_u_warm)

    if max_workers is None:
        max_workers = min(num_restarts, max(1, (os.cpu_count() or 1)))
    else:
        max_workers = max(1, min(max_workers, num_restarts))

    restart_times: List[float] = [0.0] * num_restarts
    best_u, best_loss, best_stats = None, float("inf"), None

    def _run_one(restart_idx: int):
        key = subkeys[restart_idx]
        u_init = (jax.random.normal(key, (3,)) * 0.3
                  + jnp.array([0.5, 0.0, 0.3]))
        t0 = time.perf_counter()
        u_opt, loss = opt.optimize(u_init, learning_rate, num_iters, verbose=False)
        elapsed = time.perf_counter() - t0
        stats = opt.evaluate(u_opt)
        return restart_idx, u_opt, float(loss), stats, elapsed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one, r): r
            for r in range(num_restarts)
        }

        for fut in as_completed(futures):
            r, u_opt, loss, stats, elapsed = fut.result()
            restart_times[r] = elapsed

            if verbose:
                print(f"Restart {r+1}/{num_restarts}: loss={loss:.6f} time={elapsed:.3f}s")

            if loss < best_loss:
                best_u, best_loss, best_stats = u_opt, loss, stats
                if verbose:
                    print(f"  -> New best from restart {r+1}: {best_loss:.6f}")

    best_stats['restart_times_s'] = restart_times
    return best_u, best_loss, best_stats

def optimize_parallel_gpu(opt, num_restarts=100, learning_rate=0.01, num_iters=150, seed=42):
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(key, (num_restarts, 3)) * 0.3 + jnp.array([0.5, 0.0, 0.3])

    batched_loss = jax.vmap(opt.loss_fn)
    batched_grad = jax.vmap(opt.grad_fn)

    def body(u):
        g = batched_grad(u)
        return _project_u(u - learning_rate * g)

    u_final = _scan_loop(body, u0, num_iters)
    losses = batched_loss(u_final)

    best_idx = jnp.argmin(losses)
    best_u = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses

def optimize_parallel_gpu_rejit(
    x0_ivl: irx.Interval,
    scenarios: List[Scenario],
    dt: float,
    num_steps: int,
    num_restarts: int = 100,
    learning_rate: float = 0.01,
    num_iters: int = 150,
    seed: int = 42,
):
    return optimize_parallel_gpu(
        SeparatingInputOptimizer(
            scenarios=scenarios,
            x0_ivl=x0_ivl,
            dt=dt,
            num_steps=num_steps,
        ),
        num_restarts=num_restarts,
        learning_rate=learning_rate,
        num_iters=num_iters,
        seed=seed,
    )

def optimize_multistep_gpu(
    opt: 'MultistepSequenceOptimizer',
    num_restarts: int = 100,
    learning_rate: float = 0.01,
    num_iters: int = 150,
    seed: int = 42,
):
    key = jax.random.PRNGKey(seed)
    u0 = (
        jax.random.normal(key, (num_restarts, opt.num_segments, 3)) * 0.1
        + jnp.array([0.5, 0.0, 0.3])
    )

    batched_loss = jax.vmap(opt.loss_fn)
    batched_grad = jax.vmap(opt.grad_fn)

    def body(u_batch):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final = _scan_loop(body, u0, num_iters)
    losses = batched_loss(u_final)

    best_idx = jnp.argmin(losses)
    best_u = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses

def optimize_multistep_gpu_rejit(
    x0_ivl: irx.Interval,
    scenarios: List[Scenario],
    dt: float,
    steps_per_segment: int,
    num_segments: int,
    num_restarts: int = 100,
    learning_rate: float = 0.01,
    num_iters: int = 150,
    seed: int = 42,
):
    key = jax.random.PRNGKey(seed)
    u0 = (
        jax.random.normal(key, (num_restarts, num_segments, 3)) * 0.1
        + jnp.array([0.5, 0.0, 0.3])
    )

    def loss_fn_multistep(u):
        return separation_loss_multistep(
            u,
            x0_ivl=x0_ivl,
            scenarios=scenarios,
            dt=dt,
            steps_per_segment=steps_per_segment,
        )

    batched_loss = jax.vmap(loss_fn_multistep)
    batched_grad = jax.vmap(jax.grad(loss_fn_multistep))

    def body(u_batch):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final = _scan_loop(body, u0, num_iters)
    losses = batched_loss(u_final)

    best_idx = jnp.argmin(losses)
    best_u = u_final[best_idx]
    best_loss = losses[best_idx]
    return best_u, best_loss, u_final, losses

def optimize_multistep(scenarios: List[Scenario],
                       x0_ivl: irx.Interval,
                       dt: float,
                       steps_per_segment: int,
                       num_segments: int,
                       learning_rate: float = 0.01,
                       num_iters: int = 300,
                       num_restarts: int = 100,
                       verbose: bool = False,
                       seed: int = 42,
                       ) -> Tuple[jnp.ndarray, float, Dict]:
    if num_restarts <= 0:
        raise ValueError("num_restarts must be >= 1")

    ms_opt = MultistepSequenceOptimizer(
        scenarios=scenarios,
        x0_ivl=x0_ivl,
        dt=dt,
        steps_per_segment=steps_per_segment,
        num_segments=num_segments,
    )

    _t0 = time.perf_counter()
    u_seq, loss_opt_jax, _, final_losses = optimize_multistep_gpu(
        opt=ms_opt,
        num_restarts=num_restarts,
        learning_rate=learning_rate,
        num_iters=num_iters,
        seed=seed,
    )
    elapsed = time.perf_counter() - _t0

    best_idx = int(jnp.argmin(final_losses))
    loss_opt = float(loss_opt_jax)

    if verbose:
        mean_loss = float(jnp.mean(final_losses))
        print(
            f"Multistep GPU multistart complete: best_loss={loss_opt:.6f}  "
            f"mean_final_loss={mean_loss:.6f}  best_restart={best_idx+1}/{num_restarts}"
        )

    pos_ivls = [
        position_interval(
            propagate_scenario_multistep(x0_ivl, u_seq, s, dt, steps_per_segment)
        )
        for s in scenarios
    ]
    n = len(scenarios)
    overlaps = {}
    for i in range(n):
        for j in range(i + 1, n):
            key_ij = f"{scenarios[i].name} vs {scenarios[j].name}"
            overlaps[key_ij] = float(_overlap_volume(pos_ivls[i], pos_ivls[j]))
    volumes = {
        s.name: float(jnp.prod(iv.upper - iv.lower))
        for s, iv in zip(scenarios, pos_ivls)
    }
    stats = {
        'position_intervals':  pos_ivls,
        'pairwise_overlaps':   overlaps,
        'volumes':             volumes,
        'optimization_time_s': elapsed,
        'best_restart_idx':    best_idx,
        'all_restart_losses':  np.array(final_losses),
        'num_restarts':        num_restarts,
    }
    return u_seq, loss_opt, stats
