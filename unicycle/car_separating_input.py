import sys
import time
import resource
from pathlib import Path
from dataclasses import dataclass, field
from functools import partial
from typing import List, Tuple, Optional, Dict

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import jax
import jax.numpy as jnp
import immrax as irx
import numpy as np

_U_LO = jnp.array([-1.0, -1.0])
_U_HI = jnp.array([1.0, 1.0])

def _pair_indices(n: int):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return jnp.array([i for i, j in pairs]), jnp.array([j for i, j in pairs])

def _project_u(u: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(u, _U_LO, _U_HI)

def _overlap_volume(ivl1: irx.Interval, ivl2: irx.Interval) -> jnp.ndarray:
    widths = jnp.maximum(
        jnp.minimum(ivl1.upper, ivl2.upper) - jnp.maximum(ivl1.lower, ivl2.lower),
        0.0,
    )
    return jnp.prod(widths)

def _scan_loop(step_fn, init, n: int):
    def body(carry, _):
        return step_fn(carry), None
    carry, _ = jax.lax.scan(body, init, xs=None, length=n)
    return carry

def gd_early_stop(loss_fn, u0: jnp.ndarray, learning_rate: float,
                  max_iters: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    grad_fn = jax.grad(loss_fn)

    def cond_fn(carry):
        _, loss, i = carry
        return jnp.logical_and(loss > 0.0, i < max_iters)

    def body_fn(carry):
        u, _, i = carry
        u_next = _project_u(u - learning_rate * grad_fn(u))
        return u_next, loss_fn(u_next), i + 1

    init = (u0, loss_fn(u0), jnp.array(0, dtype=jnp.int32))
    u_final, loss_final, num_iters = jax.lax.while_loop(cond_fn, body_fn, init)
    return u_final, loss_final, num_iters

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

class CarNomActSystem(irx.System):

    def __init__(self):
        self.evolution = 'continuous'
        self.xlen = 3

    def f(self, t, x, u, p):
        v, omega = u[0], u[1]
        alpha = p[0]
        phi = x[2]
        return jnp.array([
            v * jnp.cos(phi),
            v * jnp.sin(phi),
            alpha * omega,
        ])

_NOM_ACT_SYS = CarNomActSystem()
_NOM_ACT_EMB = irx.natemb(_NOM_ACT_SYS)

def get_system_and_embedding() -> Tuple[CarNomActSystem, object]:
    return _NOM_ACT_SYS, _NOM_ACT_EMB

@dataclass
class Scenario:
    name: str
    emb_system: object
    p_interval: irx.Interval
    obs_offset: jnp.ndarray = field(default_factory=lambda: jnp.zeros(2))
    obs_scale: jnp.ndarray = field(default_factory=lambda: jnp.ones(1))

def create_scenarios(
    actuator_alpha_lo: float = 0.0,
    actuator_alpha_hi: float = 0.5,
    sensor_obs_offset: Tuple[float, float] = (0.2, 0.2),
    sensor_obs_scale: float = 0.95,
) -> List[Scenario]:
    _, emb = get_system_and_embedding()
    return [
        Scenario(
            name="Nominal",
            emb_system=emb,
            p_interval=irx.icentpert(jnp.array([1.0]), jnp.zeros(1)),
        ),
        Scenario(
            name="Actuator Fault",
            emb_system=emb,
            p_interval=irx.Interval(
                lower=jnp.array([actuator_alpha_lo]),
                upper=jnp.array([actuator_alpha_hi]),
            ),
        ),
        Scenario(
            name="Sensor Fault",
            emb_system=emb,
            p_interval=irx.icentpert(jnp.array([1.0]), jnp.zeros(1)),
            obs_offset=jnp.array(sensor_obs_offset),
            obs_scale=jnp.array([sensor_obs_scale]),
        ),
    ]

def observed_output(x_ivl: irx.Interval, scenario: Scenario) -> irx.Interval:
    pos_lower, pos_upper = x_ivl.lower[:2], x_ivl.upper[:2]
    scale = scenario.obs_scale[0]
    return irx.Interval(
        lower=scale * pos_lower + scenario.obs_offset,
        upper=scale * pos_upper + scenario.obs_offset,
    )

def _invert_observation(y_ivl: irx.Interval, scenario: Scenario,
                        phi_source: irx.Interval) -> irx.Interval:
    scale = scenario.obs_scale[0]
    pos_lower = (y_ivl.lower - scenario.obs_offset) / scale
    pos_upper = (y_ivl.upper - scenario.obs_offset) / scale
    return irx.Interval(
        lower=jnp.concatenate([pos_lower, phi_source.lower[2:3]]),
        upper=jnp.concatenate([pos_upper, phi_source.upper[2:3]]),
    )

def euler_step(emb_sys, x_ivl: irx.Interval, u: jnp.ndarray,
               p_ivl: irx.Interval, dt: float) -> irx.Interval:
    _t = jnp.zeros(())
    x_ut = irx.i2ut(x_ivl)
    dx_ut = emb_sys.f(_t, x_ut, u, p_ivl)
    return irx.ut2i(dx_ut * dt + x_ut)

def propagate_scenario(x0_ivl: irx.Interval, u: jnp.ndarray,
                       scenario: Scenario,
                       dt: float, num_steps: int) -> irx.Interval:
    def body(x_carry, _):
        return euler_step(scenario.emb_system, x_carry, u, scenario.p_interval, dt), None
    x_final, _ = jax.lax.scan(body, x0_ivl, xs=None, length=num_steps)
    return x_final

def separation_loss(u: jnp.ndarray,
                    x0_ivl: irx.Interval,
                    scenarios: List[Scenario],
                    dt: float,
                    num_steps: int) -> jnp.ndarray:
    y_ivls = [
        observed_output(propagate_scenario(x0_ivl, u, s, dt, num_steps), s)
        for s in scenarios
    ]
    n = len(y_ivls)
    lo_stack = jnp.stack([iv.lower for iv in y_ivls])
    hi_stack = jnp.stack([iv.upper for iv in y_ivls])
    pair_i, pair_j = _pair_indices(n)
    ivl_i = irx.Interval(lower=lo_stack[pair_i], upper=hi_stack[pair_i])
    ivl_j = irx.Interval(lower=lo_stack[pair_j], upper=hi_stack[pair_j])
    return jnp.sum(jax.vmap(_overlap_volume)(ivl_i, ivl_j))

class SeparatingInputOptimizer:

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval,
                 dt: float, num_steps: int):
        self.scenarios = scenarios
        self.x0_ivl = x0_ivl
        self.dt = dt
        self.num_steps = num_steps

        _loss = partial(separation_loss, x0_ivl=x0_ivl, scenarios=scenarios,
                        dt=dt, num_steps=num_steps)
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))

    def optimize(self, u_init: Optional[jnp.ndarray] = None,
                 learning_rate: float = 0.05, num_iters: int = 200,
                 verbose: bool = False) -> Tuple[jnp.ndarray, float]:
        if u_init is None:
            u_init = jnp.array([0.5, 0.3])
        u = u_init
        for i in range(num_iters):
            g = self.grad_fn(u)
            u = _project_u(u - learning_rate * g)
            if verbose and (i % 20 == 0 or i == num_iters - 1):
                loss = float(self.loss_fn(u))
                print(f"  Iter {i:4d}  loss={loss:.6f}  "
                      f"u=[{float(u[0]):+.3f},{float(u[1]):+.3f}]  "
                      f"|g|={float(jnp.linalg.norm(g)):.4f}")
        return u, float(self.loss_fn(u))

    def evaluate(self, u: jnp.ndarray) -> Dict:
        y_ivls = [
            observed_output(propagate_scenario(self.x0_ivl, u, s, self.dt, self.num_steps), s)
            for s in self.scenarios
        ]
        n = len(self.scenarios)
        overlaps = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{self.scenarios[i].name} vs {self.scenarios[j].name}"
                overlaps[key] = float(_overlap_volume(y_ivls[i], y_ivls[j]))
        volumes = {
            s.name: float(jnp.prod(iv.upper - iv.lower))
            for s, iv in zip(self.scenarios, y_ivls)
        }
        return {'output_intervals': y_ivls, 'pairwise_overlaps': overlaps, 'volumes': volumes}

def optimize_parallel_gpu(opt: 'SeparatingInputOptimizer', num_restarts: int = 100,
                          learning_rate: float = 0.05, num_iters: int = 200,
                          seed: int = 42):
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(key, (num_restarts, 2)) * 0.3 + jnp.array([0.5, 0.3])

    batched_loss = jax.vmap(opt.loss_fn)
    batched_grad = jax.vmap(opt.grad_fn)

    def body(u):
        g = batched_grad(u)
        return _project_u(u - learning_rate * g)

    u_final = _scan_loop(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.argmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses

def optimize_parallel_gpu_rejit(x0_ivl: irx.Interval, scenarios: List[Scenario],
                                dt: float, num_steps: int, num_restarts: int = 100,
                                learning_rate: float = 0.05, num_iters: int = 200,
                                seed: int = 42):
    return optimize_parallel_gpu(
        SeparatingInputOptimizer(scenarios=scenarios, x0_ivl=x0_ivl, dt=dt, num_steps=num_steps),
        num_restarts=num_restarts, learning_rate=learning_rate, num_iters=num_iters, seed=seed,
    )

def _propagate_history(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                       emb_sys, p_ivl: irx.Interval,
                       dt: float, steps_per_segment: int) -> irx.Interval:
    def segment(x_ivl, u_k):
        def euler_body(x, _): return euler_step(emb_sys, x, u_k, p_ivl, dt), None
        x_end, _ = jax.lax.scan(euler_body, x_ivl, xs=None, length=steps_per_segment)
        return x_end, x_end

    _, x_hist = jax.lax.scan(segment, x0_ivl, u_seq)
    return x_hist

def propagate_scenario_multistep(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                                 scenario: Scenario, dt: float,
                                 steps_per_segment: int) -> irx.Interval:
    x_hist = _propagate_history(
        x0_ivl, u_seq, scenario.emb_system, scenario.p_interval, dt, steps_per_segment
    )
    return irx.Interval(lower=x_hist.lower[-1], upper=x_hist.upper[-1])

def separation_loss_multistep(u_seq: jnp.ndarray, x0_ivl: irx.Interval,
                              scenarios: List[Scenario], dt: float,
                              steps_per_segment: int) -> jnp.ndarray:
    num_segments = u_seq.shape[0]
    n = len(scenarios)
    emb_sys = scenarios[0].emb_system

    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )

    def prop_one(p_ivl_single):
        return _propagate_history(x0_ivl, u_seq, emb_sys, p_ivl_single, dt, steps_per_segment)

    x_hist_batch = jax.vmap(prop_one)(p_batch)

    pair_i, pair_j = _pair_indices(n)

    def overlap_at_k(k):
        y_ivls_k = [
            observed_output(
                irx.Interval(lower=x_hist_batch.lower[i, k], upper=x_hist_batch.upper[i, k]),
                scenarios[i],
            )
            for i in range(n)
        ]
        lo_stack = jnp.stack([iv.lower for iv in y_ivls_k])
        hi_stack = jnp.stack([iv.upper for iv in y_ivls_k])
        ivl_i = irx.Interval(lower=lo_stack[pair_i], upper=hi_stack[pair_i])
        ivl_j = irx.Interval(lower=lo_stack[pair_j], upper=hi_stack[pair_j])
        return jnp.sum(jax.vmap(_overlap_volume)(ivl_i, ivl_j))

    segment_overlaps = jnp.stack([overlap_at_k(k) for k in range(num_segments)])
    return jnp.min(segment_overlaps)

class MultistepSequenceOptimizer:

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval,
                 dt: float, steps_per_segment: int, num_segments: int):
        self.scenarios = scenarios
        self.x0_ivl = x0_ivl
        self.dt = dt
        self.steps_per_segment = steps_per_segment
        self.num_segments = num_segments

        _loss = partial(separation_loss_multistep, x0_ivl=x0_ivl, scenarios=scenarios,
                        dt=dt, steps_per_segment=steps_per_segment)
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))

def optimize_multistep_gpu(opt: 'MultistepSequenceOptimizer', num_restarts: int = 100,
                           learning_rate: float = 0.05, num_iters: int = 200, seed: int = 42):
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(key, (num_restarts, opt.num_segments, 2)) * 0.3 + jnp.array([0.5, 0.3])

    batched_loss = jax.vmap(opt.loss_fn)
    batched_grad = jax.vmap(opt.grad_fn)

    def body(u_batch):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final = _scan_loop(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.argmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses

def optimize_multistep_gpu_rejit(x0_ivl: irx.Interval, scenarios: List[Scenario],
                                 dt: float, steps_per_segment: int, num_segments: int,
                                 num_restarts: int = 100, learning_rate: float = 0.05,
                                 num_iters: int = 200, seed: int = 42):
    return optimize_multistep_gpu(
        MultistepSequenceOptimizer(scenarios=scenarios, x0_ivl=x0_ivl, dt=dt,
                                   steps_per_segment=steps_per_segment, num_segments=num_segments),
        num_restarts=num_restarts, learning_rate=learning_rate, num_iters=num_iters, seed=seed,
    )

def optimize_multistep(scenarios: List[Scenario], x0_ivl: irx.Interval, dt: float,
                       steps_per_segment: int, num_segments: int,
                       learning_rate: float = 0.05, num_iters: int = 300,
                       num_restarts: int = 100, verbose: bool = False,
                       seed: int = 42) -> Tuple[jnp.ndarray, float, Dict]:
    if num_restarts <= 0:
        raise ValueError("num_restarts must be >= 1")

    ms_opt = MultistepSequenceOptimizer(
        scenarios=scenarios, x0_ivl=x0_ivl, dt=dt,
        steps_per_segment=steps_per_segment, num_segments=num_segments,
    )

    _t0 = time.perf_counter()
    u_seq, loss_opt_jax, _, final_losses = optimize_multistep_gpu(
        opt=ms_opt, num_restarts=num_restarts, learning_rate=learning_rate,
        num_iters=num_iters, seed=seed,
    )
    elapsed = time.perf_counter() - _t0

    best_idx = int(jnp.argmin(final_losses))
    loss_opt = float(loss_opt_jax)

    if verbose:
        mean_loss = float(jnp.mean(final_losses))
        print(f"Multistep GPU multistart complete: best_loss={loss_opt:.6f}  "
              f"mean_final_loss={mean_loss:.6f}  best_restart={best_idx+1}/{num_restarts}")

    y_ivls = [
        observed_output(propagate_scenario_multistep(x0_ivl, u_seq, s, dt, steps_per_segment), s)
        for s in scenarios
    ]
    n = len(scenarios)
    overlaps = {}
    for i in range(n):
        for j in range(i + 1, n):
            overlaps[f"{scenarios[i].name} vs {scenarios[j].name}"] = float(_overlap_volume(y_ivls[i], y_ivls[j]))
    volumes = {s.name: float(jnp.prod(iv.upper - iv.lower)) for s, iv in zip(scenarios, y_ivls)}
    stats = {
        'output_intervals': y_ivls,
        'pairwise_overlaps': overlaps,
        'volumes': volumes,
        'optimization_time_s': elapsed,
        'best_restart_idx': best_idx,
        'all_restart_losses': np.array(final_losses),
        'num_restarts': num_restarts,
    }
    return u_seq, loss_opt, stats

def _refine_and_step_pair(
    x_curr_i: irx.Interval, x_curr_j: irx.Interval,
    scenario_i: Scenario, scenario_j: Scenario,
    u_k: jnp.ndarray, dt: float,
):
    obs_i = observed_output(x_curr_i, scenario_i)
    obs_j = observed_output(x_curr_j, scenario_j)

    y_lo = jnp.maximum(obs_i.lower, obs_j.lower)
    y_hi = jnp.minimum(obs_i.upper, obs_j.upper)
    has_overlap = jnp.all(y_hi >= y_lo)
    y_int = irx.Interval(lower=y_lo, upper=y_hi)

    x_ref_i_overlap = _invert_observation(y_int, scenario_i, x_curr_i)
    x_ref_j_overlap = _invert_observation(y_int, scenario_j, x_curr_j)

    x_ref_i = irx.Interval(
        lower=jnp.where(has_overlap, x_ref_i_overlap.lower, x_curr_i.lower),
        upper=jnp.where(has_overlap, x_ref_i_overlap.upper, x_curr_i.upper),
    )
    x_ref_j = irx.Interval(
        lower=jnp.where(has_overlap, x_ref_j_overlap.lower, x_curr_j.lower),
        upper=jnp.where(has_overlap, x_ref_j_overlap.upper, x_curr_j.upper),
    )

    x_next_i = euler_step(scenario_i.emb_system, x_ref_i, u_k, scenario_i.p_interval, dt)
    x_next_j = euler_step(scenario_j.emb_system, x_ref_j, u_k, scenario_j.p_interval, dt)

    obs_next_i = observed_output(x_next_i, scenario_i)
    obs_next_j = observed_output(x_next_j, scenario_j)
    raw_cost = _overlap_volume(obs_next_i, obs_next_j)
    pair_cost = jnp.where(has_overlap, raw_cost, jnp.array(0.0))

    return x_next_i, x_next_j, obs_next_i, obs_next_j, pair_cost, has_overlap

def propagate_with_refinement(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                              scenarios: List[Scenario], dt: float,
                              num_steps: int = 2) -> jnp.ndarray:
    n = len(scenarios)
    pair_i, pair_j = _pair_indices(n)
    xlen = x0_ivl.lower.shape[0]

    def ivl_to_arr(ivl: irx.Interval) -> jnp.ndarray:
        return jnp.concatenate([ivl.lower, ivl.upper])

    def arr_to_ivl(arr: jnp.ndarray) -> irx.Interval:
        return irx.Interval(lower=arr[:xlen], upper=arr[xlen:])

    x1_ivls = [euler_step(s.emb_system, x0_ivl, u_seq[0], s.p_interval, dt) for s in scenarios]
    y1_ivls = [observed_output(x1, s) for x1, s in zip(x1_ivls, scenarios)]

    y1_lo = jnp.stack([iv.lower for iv in y1_ivls])
    y1_hi = jnp.stack([iv.upper for iv in y1_ivls])
    step1_cost = jnp.sum(jax.vmap(_overlap_volume)(
        irx.Interval(lower=y1_lo[pair_i], upper=y1_hi[pair_i]),
        irx.Interval(lower=y1_lo[pair_j], upper=y1_hi[pair_j]),
    ))

    x1_lo = jnp.stack([iv.lower for iv in x1_ivls])
    x1_hi = jnp.stack([iv.upper for iv in x1_ivls])
    pxi_arr = jnp.concatenate([x1_lo[pair_i], x1_hi[pair_i]], axis=-1)
    pxj_arr = jnp.concatenate([x1_lo[pair_j], x1_hi[pair_j]], axis=-1)

    emb_sys = scenarios[0].emb_system
    p_lo = jnp.stack([s.p_interval.lower for s in scenarios])
    p_hi = jnp.stack([s.p_interval.upper for s in scenarios])
    obs_offset = jnp.stack([s.obs_offset for s in scenarios])
    obs_scale = jnp.stack([s.obs_scale for s in scenarios])

    def _gather(arr, idx):
        return arr[idx]

    p_pi = irx.Interval(lower=p_lo[pair_i], upper=p_hi[pair_i])
    p_pj = irx.Interval(lower=p_lo[pair_j], upper=p_hi[pair_j])
    off_pi, off_pj = _gather(obs_offset, pair_i), _gather(obs_offset, pair_j)
    scale_pi, scale_pj = _gather(obs_scale, pair_i), _gather(obs_scale, pair_j)

    class _ScenLike:
        __slots__ = ("obs_offset", "obs_scale")

        def __init__(self, obs_offset, obs_scale):
            self.obs_offset = obs_offset
            self.obs_scale = obs_scale

    def step_one_pair(pxi_arr_1, pxj_arr_1, p_i1, off_i1, scale_i1,
                      p_j1, off_j1, scale_j1, u_k):
        x_curr_i = arr_to_ivl(pxi_arr_1)
        x_curr_j = arr_to_ivl(pxj_arr_1)
        scen_i = _ScenLike(off_i1, scale_i1)
        scen_j = _ScenLike(off_j1, scale_j1)

        obs_i = observed_output(x_curr_i, scen_i)
        obs_j = observed_output(x_curr_j, scen_j)
        y_lo = jnp.maximum(obs_i.lower, obs_j.lower)
        y_hi = jnp.minimum(obs_i.upper, obs_j.upper)
        has_overlap = jnp.all(y_hi >= y_lo)
        y_int = irx.Interval(lower=y_lo, upper=y_hi)

        x_ref_i_overlap = _invert_observation(y_int, scen_i, x_curr_i)
        x_ref_j_overlap = _invert_observation(y_int, scen_j, x_curr_j)

        x_ref_i = irx.Interval(
            lower=jnp.where(has_overlap, x_ref_i_overlap.lower, x_curr_i.lower),
            upper=jnp.where(has_overlap, x_ref_i_overlap.upper, x_curr_i.upper),
        )
        x_ref_j = irx.Interval(
            lower=jnp.where(has_overlap, x_ref_j_overlap.lower, x_curr_j.lower),
            upper=jnp.where(has_overlap, x_ref_j_overlap.upper, x_curr_j.upper),
        )

        x_next_i = euler_step(emb_sys, x_ref_i, u_k, p_i1, dt)
        x_next_j = euler_step(emb_sys, x_ref_j, u_k, p_j1, dt)

        raw_cost = _overlap_volume(
            observed_output(x_next_i, scen_i),
            observed_output(x_next_j, scen_j),
        )
        pair_cost = jnp.where(has_overlap, raw_cost, jnp.array(0.0))
        return ivl_to_arr(x_next_i), ivl_to_arr(x_next_j), pair_cost

    step_all_pairs = jax.vmap(step_one_pair, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None))

    def step_body(carry, u_k):
        pxi_arr, pxj_arr, min_cost = carry
        new_pxi, new_pxj, pair_costs = step_all_pairs(
            pxi_arr, pxj_arr, p_pi, off_pi, scale_pi, p_pj, off_pj, scale_pj, u_k
        )
        step_cost = jnp.sum(pair_costs)
        new_carry = (new_pxi, new_pxj, jnp.minimum(min_cost, step_cost))
        return new_carry, None

    init_carry = (pxi_arr, pxj_arr, step1_cost)
    (_, _, min_cost_final), _ = jax.lax.scan(step_body, init_carry, xs=u_seq[1:num_steps])
    return min_cost_final

def refined_overlap_loss(u_seq: jnp.ndarray, x0_ivl: irx.Interval,
                         scenarios: List[Scenario], dt: float,
                         num_steps: int = 2) -> jnp.ndarray:
    return propagate_with_refinement(x0_ivl, u_seq, scenarios, dt, num_steps)

def optimize_refined_gpu(x0_ivl: irx.Interval, scenarios: List[Scenario], dt: float,
                         num_steps: int = 2, num_restarts: int = 50,
                         learning_rate: float = 0.05, num_iters: int = 200,
                         seed: int = 42) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(key, (num_restarts, num_steps, 2)) * 0.3 + jnp.array([0.5, 0.3])

    def loss_fn_refined(u_seq):
        return refined_overlap_loss(u_seq, x0_ivl, scenarios, dt, num_steps)

    batched_loss = jax.vmap(loss_fn_refined)
    batched_grad = jax.vmap(jax.grad(loss_fn_refined))

    def body(u_batch):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final = _scan_loop(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.argmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses

def collect_refinement_history(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                               scenarios: List[Scenario], dt: float, num_steps: int):
    n = len(scenarios)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    x1 = [euler_step(s.emb_system, x0_ivl, u_seq[0], s.p_interval, dt) for s in scenarios]
    obs1 = [observed_output(xi, s) for xi, s in zip(x1, scenarios)]

    def intersect_or_none(a, b):
        lo = jnp.maximum(a.lower, b.lower)
        hi = jnp.minimum(a.upper, b.upper)
        return irx.Interval(lower=lo, upper=hi) if bool(jnp.all(hi >= lo)) else None

    steps = [{'t': dt, 'pair_obs': [
        (obs1[i], obs1[j], intersect_or_none(obs1[i], obs1[j])) for i, j in pairs
    ]}]
    pair_states = [(x1[i], x1[j]) for i, j in pairs]

    for k in range(num_steps - 1):
        new_states, new_pair_obs = [], []
        for (i, j), (xi, xj) in zip(pairs, pair_states):
            x_next_i, x_next_j, obs_next_i, obs_next_j, _, _ = _refine_and_step_pair(
                xi, xj, scenarios[i], scenarios[j], u_seq[k + 1], dt
            )
            new_pair_obs.append((obs_next_i, obs_next_j, intersect_or_none(obs_next_i, obs_next_j)))
            new_states.append((x_next_i, x_next_j))
        steps.append({'t': (k + 2) * dt, 'pair_obs': new_pair_obs})
        pair_states = new_states

    return steps, pairs
