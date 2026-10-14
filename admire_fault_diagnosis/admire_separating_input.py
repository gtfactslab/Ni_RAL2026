import sys
import time
import resource
from pathlib import Path
from dataclasses import dataclass, field
from functools import partial, lru_cache
from typing import List, Tuple, Optional, Dict

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
_ADMIRE_DIR = _EXAMPLES_DIR / "admire"
for _d in (_EXAMPLES_DIR, _ADMIRE_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import jax
from jax import lax
import jax.numpy as jnp
import immrax as irx
import numpy as np

from admire import AdmireNineDoFLinAct

_U_LO = jnp.ones(10) * -0.05
_U_HI = jnp.ones(10) * 0.05

def _project_u(u: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(u, _U_LO, _U_HI)

def overlap_size_log(interval1: irx.Interval, interval2: irx.Interval) -> jnp.ndarray:
    intersection_lower = jnp.maximum(interval1.lower, interval2.lower)
    intersection_upper = jnp.minimum(interval1.upper, interval2.upper)

    def has_overlap(operands):
        lower, upper = operands
        return jnp.sum(jnp.log(upper - lower + 1.0))

    def no_overlap(operands):
        _ = operands
        return jnp.array(0.0)

    has_no_overlap = jnp.any(intersection_upper < intersection_lower)

    return lax.cond(
        jnp.logical_not(has_no_overlap),
        has_overlap,
        no_overlap,
        (intersection_lower, intersection_upper),
    )

@lru_cache(maxsize=None)
def _pair_idx_arrays(n: int) -> Tuple[np.ndarray, np.ndarray]:
    ii, jj = np.triu_indices(n, k=1)
    return ii, jj

def _overlap_size_log_pair(lo1: jnp.ndarray, up1: jnp.ndarray,
                           lo2: jnp.ndarray, up2: jnp.ndarray) -> jnp.ndarray:
    width = jnp.minimum(up1, up2) - jnp.maximum(lo1, lo2)
    disjoint = jnp.any(width < 0.0)
    total = jnp.sum(jnp.log(jnp.maximum(width, 0.0) + 1.0))
    return jnp.where(disjoint, 0.0, total)

def _pairwise_overlap_sum(lo: jnp.ndarray, up: jnp.ndarray, n: int) -> jnp.ndarray:
    ii, jj = _pair_idx_arrays(n)
    cost = _overlap_size_log_pair
    for _ in range(lo.ndim - 2):
        cost = jax.vmap(cost)
    per_pair = jax.vmap(cost)(lo[ii], up[ii], lo[jj], up[jj])
    return jnp.sum(per_pair, axis=0)

def _scan_loop(step_fn, init, n: int):
    def body(carry, _):
        return step_fn(carry), None
    carry, _ = lax.scan(body, init, xs=None, length=n)
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

_ADMIRE_SYS = AdmireNineDoFLinAct()
_ADMIRE_EMB = irx.natemb(_ADMIRE_SYS)

def get_system_and_embedding() -> Tuple[AdmireNineDoFLinAct, object]:
    return _ADMIRE_SYS, _ADMIRE_EMB

@dataclass
class Scenario:
    name: str
    emb_system: object
    p_interval: irx.Interval

_SURFACE_NAMES = [
    "Right Canard", "Left Canard", "Right Outer Elev", "Right Inner Elev",
    "Left Inner Elev", "Left Outer Elev", "Rudder", "Flap", "Yaw TV", "Pitch TV",
]

def create_scenarios(fault_effectiveness: float = 0.0) -> List[Scenario]:
    p_nominal = jnp.ones(10)
    scenarios = [Scenario("Nominal", _ADMIRE_EMB, irx.icentpert(p_nominal, jnp.zeros(10)))]
    for i, name in enumerate(_SURFACE_NAMES):
        p_fault = p_nominal.at[i].set(fault_effectiveness)
        scenarios.append(Scenario(name, _ADMIRE_EMB, irx.icentpert(p_fault, jnp.zeros(10))))
    return scenarios

def output_interval(x_ivl: irx.Interval) -> irx.Interval:
    return irx.Interval(lower=x_ivl.lower[3:6], upper=x_ivl.upper[3:6])

def euler_step(emb_sys, x_ivl: irx.Interval, u: jnp.ndarray,
               p_ivl: irx.Interval, dt: float) -> irx.Interval:
    _t = jnp.zeros(())
    x_ut = irx.i2ut(x_ivl)
    dx_ut = emb_sys.f(_t, x_ut, u, p_ivl)
    return irx.ut2i(dx_ut * dt + x_ut)

def propagate_scenario(x0_ivl: irx.Interval, u: jnp.ndarray,
                       scenario: Scenario, dt: float, num_steps: int) -> irx.Interval:
    return _propagate_scenario_p(x0_ivl, u, scenario.emb_system,
                                 scenario.p_interval, dt, num_steps)

def _propagate_scenario_p(x0_ivl: irx.Interval, u: jnp.ndarray, emb_sys,
                          p_ivl: irx.Interval, dt: float, num_steps: int) -> irx.Interval:
    def body(x_carry, _):
        return euler_step(emb_sys, x_carry, u, p_ivl, dt), None
    x_final, _ = lax.scan(body, x0_ivl, xs=None, length=num_steps)
    return x_final

def separation_loss(u: jnp.ndarray, x0_ivl: irx.Interval, scenarios: List[Scenario],
                    dt: float, num_steps: int) -> jnp.ndarray:
    n = len(scenarios)
    emb_sys = scenarios[0].emb_system
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )
    x_batch = jax.vmap(
        lambda p: _propagate_scenario_p(x0_ivl, u, emb_sys, p, dt, num_steps))(p_batch)
    return _pairwise_overlap_sum(x_batch.lower[:, 3:6], x_batch.upper[:, 3:6], n)

class SeparatingInputOptimizer:

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval, dt: float, num_steps: int):
        self.scenarios, self.x0_ivl, self.dt, self.num_steps = scenarios, x0_ivl, dt, num_steps
        _loss = partial(separation_loss, x0_ivl=x0_ivl, scenarios=scenarios, dt=dt, num_steps=num_steps)
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))

    def evaluate(self, u: jnp.ndarray) -> Dict:
        out_ivls = [output_interval(propagate_scenario(self.x0_ivl, u, s, self.dt, self.num_steps))
                    for s in self.scenarios]
        n = len(self.scenarios)
        overlaps = {f"{self.scenarios[i].name} vs {self.scenarios[j].name}":
                    float(overlap_size_log(out_ivls[i], out_ivls[j]))
                    for i in range(n) for j in range(i + 1, n)}
        volumes = {s.name: float(jnp.prod(iv.upper - iv.lower)) for s, iv in zip(self.scenarios, out_ivls)}
        return {'output_intervals': out_ivls, 'pairwise_overlaps': overlaps, 'volumes': volumes}

def optimize_parallel_gpu(opt: SeparatingInputOptimizer, num_restarts: int = 50,
                          learning_rate: float = 0.05, num_iters: int = 200, seed: int = 42):
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.uniform(key, (num_restarts, 10), minval=-0.1, maxval=0.1)

    batched_value_and_grad = jax.vmap(jax.value_and_grad(opt.loss_fn))

    def body(u):
        _, g = batched_value_and_grad(u)
        return _project_u(u - learning_rate * g)

    u_final = _scan_loop(body, u0, num_iters)
    losses, _ = batched_value_and_grad(u_final)
    losses_valid = jnp.where(jnp.isnan(losses), jnp.inf, losses)
    best_idx = jnp.argmin(losses_valid)
    return u_final[best_idx], losses[best_idx], u_final, losses

def _propagate_history(x0_ivl: irx.Interval, u_seq: jnp.ndarray, emb_sys,
                       p_ivl: irx.Interval, dt: float, steps_per_segment: int) -> irx.Interval:
    def segment(x_ivl, u_k):
        def euler_body(x, _): return euler_step(emb_sys, x, u_k, p_ivl, dt), None
        x_end, _ = lax.scan(euler_body, x_ivl, xs=None, length=steps_per_segment)
        return x_end, x_end

    _, x_hist = lax.scan(segment, x0_ivl, u_seq)
    return x_hist

def propagate_scenario_multistep(x0_ivl: irx.Interval, u_seq: jnp.ndarray, scenario: Scenario,
                                 dt: float, steps_per_segment: int) -> irx.Interval:
    x_hist = _propagate_history(x0_ivl, u_seq, scenario.emb_system, scenario.p_interval, dt, steps_per_segment)
    return irx.Interval(lower=x_hist.lower[-1], upper=x_hist.upper[-1])

def separation_loss_multistep(u_seq: jnp.ndarray, x0_ivl: irx.Interval, scenarios: List[Scenario],
                              dt: float, steps_per_segment: int) -> jnp.ndarray:
    n = len(scenarios)
    emb_sys = scenarios[0].emb_system

    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )

    def prop_one(p_ivl_single):
        return _propagate_history(x0_ivl, u_seq, emb_sys, p_ivl_single, dt, steps_per_segment)

    x_hist_batch = jax.vmap(prop_one)(p_batch)

    segment_overlaps = _pairwise_overlap_sum(
        x_hist_batch.lower[:, :, 3:6], x_hist_batch.upper[:, :, 3:6], n)
    return jnp.min(segment_overlaps)

class MultistepSequenceOptimizer:

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval, dt: float,
                 steps_per_segment: int, num_segments: int):
        self.scenarios, self.x0_ivl, self.dt = scenarios, x0_ivl, dt
        self.steps_per_segment, self.num_segments = steps_per_segment, num_segments
        _loss = partial(separation_loss_multistep, x0_ivl=x0_ivl, scenarios=scenarios,
                        dt=dt, steps_per_segment=steps_per_segment)
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))

    def evaluate(self, u: jnp.ndarray) -> Dict:
        if u.ndim != 2 or u.shape != (self.num_segments, 10):
            raise ValueError(f"u must have shape ({self.num_segments}, 10), got {tuple(u.shape)}")
        out_ivls = [output_interval(propagate_scenario_multistep(self.x0_ivl, u, s, self.dt, self.steps_per_segment))
                    for s in self.scenarios]
        n = len(self.scenarios)
        overlaps = {f"{self.scenarios[i].name} vs {self.scenarios[j].name}":
                    float(overlap_size_log(out_ivls[i], out_ivls[j]))
                    for i in range(n) for j in range(i + 1, n)}
        volumes = {s.name: float(jnp.prod(iv.upper - iv.lower)) for s, iv in zip(self.scenarios, out_ivls)}
        return {'output_intervals': out_ivls, 'pairwise_overlaps': overlaps, 'volumes': volumes}

def optimize_multistep_gpu(x0_ivl: irx.Interval, scenarios: List[Scenario], dt: float,
                           steps_per_segment: int, num_segments: int,
                           learning_rate: float = 0.05, num_iters: int = 200,
                           num_restarts: int = 50, seed: int = 42, init_scale: float = 0.05,
                           u0_mean: Optional[jnp.ndarray] = None):
    key = jax.random.PRNGKey(seed)
    noise = jax.random.uniform(key, (num_restarts, num_segments, 10), minval=-init_scale, maxval=init_scale)
    u0 = noise if u0_mean is None else noise + u0_mean[None, :, :]

    def loss_fn(u):
        return separation_loss_multistep(u_seq=u, x0_ivl=x0_ivl, scenarios=scenarios,
                                         dt=dt, steps_per_segment=steps_per_segment)

    batched_value_and_grad = jax.vmap(jax.value_and_grad(loss_fn))

    def body(u_batch):
        _, g = batched_value_and_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final = _scan_loop(body, u0, num_iters)
    losses, _ = batched_value_and_grad(u_final)

    losses_valid = jnp.where(jnp.isnan(losses), jnp.inf, losses)
    best_idx = jnp.argmin(losses_valid)
    return u_final[best_idx], losses[best_idx], u_final, losses

def optimize_multistep(scenarios: List[Scenario], x0_ivl: irx.Interval, dt: float,
                       steps_per_segment: int, num_segments: int,
                       learning_rate: float = 0.05, num_iters: int = 200,
                       num_restarts: int = 50, verbose: bool = False,
                       seed: int = 42) -> Tuple[jnp.ndarray, float, Dict]:
    if num_restarts <= 0:
        raise ValueError("num_restarts must be >= 1")

    _t0 = time.perf_counter()
    u_seq, loss_opt_jax, _, final_losses = optimize_multistep_gpu(
        x0_ivl=x0_ivl, scenarios=scenarios, dt=dt, steps_per_segment=steps_per_segment,
        num_segments=num_segments, learning_rate=learning_rate, num_iters=num_iters,
        num_restarts=num_restarts, seed=seed,
    )
    elapsed = time.perf_counter() - _t0

    losses_valid = jnp.where(jnp.isnan(final_losses), jnp.inf, final_losses)
    best_idx = int(jnp.argmin(losses_valid))
    loss_opt = float(loss_opt_jax)

    if verbose:
        mean_loss = float(jnp.nanmean(final_losses))
        print(f"Multistep GPU multistart complete: best_loss={loss_opt:.6f}  "
              f"mean_final_loss={mean_loss:.6f}  best_restart={best_idx+1}/{num_restarts}")

    out_ivls = [output_interval(propagate_scenario_multistep(x0_ivl, u_seq, s, dt, steps_per_segment))
                for s in scenarios]
    n = len(scenarios)
    overlaps = {f"{scenarios[i].name} vs {scenarios[j].name}": float(overlap_size_log(out_ivls[i], out_ivls[j]))
                for i in range(n) for j in range(i + 1, n)}
    volumes = {s.name: float(jnp.prod(iv.upper - iv.lower)) for s, iv in zip(scenarios, out_ivls)}
    stats = {
        'output_intervals': out_ivls, 'pairwise_overlaps': overlaps, 'volumes': volumes,
        'optimization_time_s': elapsed, 'best_restart_idx': best_idx,
        'all_restart_losses': np.array(final_losses), 'num_restarts': num_restarts,
    }
    return u_seq, loss_opt, stats

def _step_one_pair(xi_ut: jnp.ndarray, xj_ut: jnp.ndarray, pi_ut: jnp.ndarray,
                   pj_ut: jnp.ndarray, u_step: jnp.ndarray, emb_sys, dt: float, _t):
    y_lo = jnp.maximum(xi_ut[3:6], xj_ut[3:6])
    y_hi = jnp.minimum(xi_ut[12:15], xj_ut[12:15])
    has_overlap = jnp.all(y_hi >= y_lo)

    y_lo_safe = jnp.where(has_overlap, y_lo, xi_ut[3:6])
    y_hi_safe = jnp.where(has_overlap, y_hi, xi_ut[12:15])

    xi_ref_overlap = jnp.concatenate([xi_ut[:9].at[3:6].set(y_lo_safe), xi_ut[9:].at[3:6].set(y_hi_safe)])
    xj_ref_overlap = jnp.concatenate([xj_ut[:9].at[3:6].set(y_lo_safe), xj_ut[9:].at[3:6].set(y_hi_safe)])

    xi_ref = jnp.where(has_overlap, xi_ref_overlap, xi_ut)
    xj_ref = jnp.where(has_overlap, xj_ref_overlap, xj_ut)

    xn_i_ut = emb_sys.f(_t, xi_ref, u_step, irx.ut2i(pi_ut)) * dt + xi_ref
    xn_j_ut = emb_sys.f(_t, xj_ref, u_step, irx.ut2i(pj_ut)) * dt + xj_ref

    ov_lo = jnp.maximum(xn_i_ut[3:6], xn_j_ut[3:6])
    ov_hi = jnp.minimum(xn_i_ut[12:15], xn_j_ut[12:15])
    raw = jnp.prod(jnp.maximum(ov_hi - ov_lo, 0.0))
    pair_cost = jnp.where(has_overlap, raw, 0.0)

    return xn_i_ut, xn_j_ut, pair_cost

def _pair_indices(n: int) -> Tuple[jnp.ndarray, jnp.ndarray, List[Tuple[int, int]]]:
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return jnp.array([i for i, j in pairs]), jnp.array([j for i, j in pairs]), pairs

def propagate_with_refinement(u_seq: jnp.ndarray, x0_ivl: irx.Interval, scenarios: List[Scenario],
                              dt: float, num_steps: int) -> jnp.ndarray:
    n = len(scenarios)
    emb_sys = scenarios[0].emb_system
    _t = jnp.zeros(())
    pair_i, pair_j, pairs = _pair_indices(n)

    p_stack = jnp.stack([irx.i2ut(s.p_interval) for s in scenarios])
    pi_stack, pj_stack = p_stack[pair_i], p_stack[pair_j]

    x0_ut = irx.i2ut(x0_ivl)

    def one_scenario_step(p_ut, u):
        dx_ut = emb_sys.f(_t, x0_ut, u, irx.ut2i(p_ut))
        return dx_ut * dt + x0_ut

    x1_stack = jax.vmap(one_scenario_step, in_axes=(0, None))(p_stack, u_seq[0])
    xi_stack, xj_stack = x1_stack[pair_i], x1_stack[pair_j]

    def pair_overlap_cost(xi_ut, xj_ut):
        ov_lo = jnp.maximum(xi_ut[3:6], xj_ut[3:6])
        ov_hi = jnp.minimum(xi_ut[12:15], xj_ut[12:15])
        return jnp.prod(jnp.maximum(ov_hi - ov_lo, 0.0))

    cost0 = jnp.sum(jax.vmap(pair_overlap_cost)(xi_stack, xj_stack))

    step_all_pairs = jax.vmap(
        lambda xi, xj, pi, pj, u: _step_one_pair(xi, xj, pi, pj, u, emb_sys, dt, _t),
        in_axes=(0, 0, 0, 0, None),
    )

    def scan_body(carry, u_step):
        xi_stack, xj_stack, min_cost = carry
        xn_i_stack, xn_j_stack, pair_costs = step_all_pairs(xi_stack, xj_stack, pi_stack, pj_stack, u_step)
        step_cost = jnp.sum(pair_costs)
        return (xn_i_stack, xn_j_stack, jnp.minimum(min_cost, step_cost)), None

    init_carry = (xi_stack, xj_stack, cost0)
    (_, _, scan_min_cost), _ = lax.scan(jax.checkpoint(scan_body), init_carry, u_seq[1:num_steps])
    return jnp.minimum(cost0, scan_min_cost)

def refined_overlap_loss(u_seq, x0_ivl, scenarios, dt, num_steps):
    return propagate_with_refinement(u_seq, x0_ivl, scenarios, dt, num_steps)

def optimize_refined_gpu(x0_ivl: irx.Interval, scenarios: List[Scenario], dt: float,
                         num_steps: int = 5, num_restarts: int = 50,
                         learning_rate: float = 0.05, num_iters: int = 200,
                         seed: int = 42, init_scale: float = 0.1):
    key = jax.random.PRNGKey(seed)
    u0 = jax.random.uniform(key, (num_restarts, num_steps, 10), minval=-init_scale, maxval=init_scale)

    def loss_fn(u_seq):
        return propagate_with_refinement(u_seq, x0_ivl, scenarios, dt, num_steps)

    batched_value_and_grad = jax.vmap(jax.value_and_grad(loss_fn))

    def body(u_batch):
        _, g = batched_value_and_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final = _scan_loop(body, u0, num_iters)
    losses, _ = batched_value_and_grad(u_final)

    losses_valid = jnp.where(jnp.isnan(losses), jnp.inf, losses)
    best_idx = jnp.argmin(losses_valid)
    return u_final[best_idx], losses[best_idx], u_final, losses

def collect_refinement_history(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                               scenarios: List[Scenario], dt: float, num_steps: int):
    n = len(scenarios)
    emb_sys = scenarios[0].emb_system
    _t = jnp.zeros(())
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def to_ivl(x_ut):
        return irx.Interval(lower=x_ut[3:6], upper=x_ut[12:15])

    def intersect_or_none(a, b):
        lo = jnp.maximum(a.lower, b.lower)
        hi = jnp.minimum(a.upper, b.upper)
        return irx.Interval(lower=lo, upper=hi) if bool(jnp.all(hi >= lo)) else None

    x0_ut = irx.i2ut(x0_ivl)
    p_uts = [irx.i2ut(s.p_interval) for s in scenarios]

    def one_step(x_ut, p_ut, u):
        return emb_sys.f(_t, x_ut, u, irx.ut2i(p_ut)) * dt + x_ut

    x1 = [one_step(x0_ut, p_uts[k], u_seq[0]) for k in range(n)]
    obs1 = [to_ivl(x) for x in x1]

    steps = [{'t': dt, 'pair_obs': [
        (obs1[i], obs1[j], intersect_or_none(obs1[i], obs1[j])) for i, j in pairs
    ]}]
    pair_states = [(x1[i], x1[j]) for i, j in pairs]

    for k in range(num_steps - 1):
        new_states, new_pair_obs = [], []
        for (i, j), (xi_ut, xj_ut) in zip(pairs, pair_states):
            xn_i, xn_j, _ = _step_one_pair(xi_ut, xj_ut, p_uts[i], p_uts[j], u_seq[k + 1], emb_sys, dt, _t)
            obs_i, obs_j = to_ivl(xn_i), to_ivl(xn_j)
            new_pair_obs.append((obs_i, obs_j, intersect_or_none(obs_i, obs_j)))
            new_states.append((xn_i, xn_j))
        steps.append({'t': (k + 2) * dt, 'pair_obs': new_pair_obs})
        pair_states = new_states

    return steps, pairs
