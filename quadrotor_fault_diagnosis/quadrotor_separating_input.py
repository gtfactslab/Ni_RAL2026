import sys
import time
import resource
from pathlib import Path
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple, Optional, Dict

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import jax
import jax.numpy as jnp
import immrax as irx
import numpy as np

_M = 0.03589
_G = 9.81
_IXX = 2.3951e-5
_IYY = 2.3951e-5
_IZZ = 3.2346e-5

_HOVER_THRUST = _M * _G

_U_LO = jnp.array([0.5 * _HOVER_THRUST, -9.8645e-5, -9.8645e-5, -7.3505e-5])
_U_HI = jnp.array([1.5 * _HOVER_THRUST, 9.8645e-5, 9.8645e-5, 7.3505e-5])

_U_NOISE_SCALE = jnp.array([0.1 * _HOVER_THRUST, 0.1 * 9.8645e-5, 0.1 * 9.8645e-5, 0.1 * 7.3505e-5])

_UNROLL_THRESHOLD = 64

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

def _signed_separation_margin(ivl1: irx.Interval, ivl2: irx.Interval) -> jnp.ndarray:
    return jnp.maximum(ivl2.lower - ivl1.upper, ivl1.lower - ivl2.upper)

def _soft_separation_loss(ivl1: irx.Interval, ivl2: irx.Interval, tau: float = 0.01) -> jnp.ndarray:
    margins = _signed_separation_margin(ivl1, ivl2)
    n_dims = margins.shape[-1]
    soft_best_margin = tau * (jax.scipy.special.logsumexp(margins / tau) - jnp.log(n_dims))
    return jnp.maximum(-soft_best_margin, 0.0)

def _run_unrolled_or_loop(step_fn, init, n: int, unroll_threshold: int = _UNROLL_THRESHOLD):
    if n <= unroll_threshold:
        def body(carry):
            for i in range(n):
                carry = step_fn(carry, i)
            return carry
        return jax.checkpoint(body)(init)
    else:
        @jax.checkpoint
        def scan_body(carry, i):
            return step_fn(carry, i), None
        carry, _ = jax.lax.scan(scan_body, init, xs=jnp.arange(n))
        return carry

def _run_unrolled_or_loop_nocheckpoint(step_fn, init, n: int):
    def body(carry, i):
        return step_fn(carry, i), None
    carry, _ = jax.lax.scan(body, init, xs=jnp.arange(n))
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

class QuadrotorSystem(irx.System):

    def __init__(self, m: float = _M, g: float = _G,
                 Ixx: float = _IXX, Iyy: float = _IYY, Izz: float = _IZZ):
        self.evolution = 'continuous'
        self.xlen = 12
        self.m = m
        self.g = g
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz

    def f(self, t, x, u, p):
        alpha = p
        U1, U2, U3, U4 = alpha * u

        phi, theta, psi = x[3], x[4], x[5]
        ub, vb, wb = x[6], x[7], x[8]
        pr, qr, rr = x[9], x[10], x[11]

        sphi, cphi = jnp.sin(phi), jnp.cos(phi)
        stheta, ctheta, ttheta = jnp.sin(theta), jnp.cos(theta), jnp.tan(theta)
        spsi, cpsi = jnp.sin(psi), jnp.cos(psi)

        xdot = ctheta * cpsi * ub + (sphi * stheta * cpsi - cphi * spsi) * vb \
            + (cphi * stheta * cpsi + sphi * spsi) * wb
        ydot = ctheta * spsi * ub + (sphi * stheta * spsi + cphi * cpsi) * vb \
            + (cphi * stheta * spsi - sphi * cpsi) * wb
        zdot = -stheta * ub + sphi * ctheta * vb + cphi * ctheta * wb

        phidot = pr + sphi * ttheta * qr + cphi * ttheta * rr
        thetadot = cphi * qr - sphi * rr
        psidot = (sphi / ctheta) * qr + (cphi / ctheta) * rr

        ubdot = self.g * stheta - qr * wb + rr * vb
        vbdot = -self.g * sphi * ctheta - rr * ub + pr * wb
        wbdot = U1 / self.m - self.g * cphi * ctheta - pr * vb + qr * ub

        prdot = ((self.Iyy - self.Izz) / self.Ixx) * qr * rr + U2 / self.Ixx
        qrdot = ((self.Izz - self.Ixx) / self.Iyy) * pr * rr + U3 / self.Iyy
        rrdot = ((self.Ixx - self.Iyy) / self.Izz) * pr * qr + U4 / self.Izz

        return jnp.array([
            xdot, ydot, zdot, phidot, thetadot, psidot,
            ubdot, vbdot, wbdot, prdot, qrdot, rrdot,
        ])

_EMB_CACHE: Dict[Tuple[float, float, float, float, float], Tuple[QuadrotorSystem, object]] = {}

def get_system_and_embedding(m: float = _M, g: float = _G, Ixx: float = _IXX,
                             Iyy: float = _IYY, Izz: float = _IZZ) -> Tuple[QuadrotorSystem, object]:
    key = (m, g, Ixx, Iyy, Izz)
    if key not in _EMB_CACHE:
        sys_ = QuadrotorSystem(m, g, Ixx, Iyy, Izz)
        emb = irx.natemb(sys_)
        _EMB_CACHE[key] = (sys_, emb)
    return _EMB_CACHE[key]

@dataclass
class Scenario:
    name: str
    emb_system: object
    p_interval: irx.Interval

def create_scenarios(
    m: float = _M, g: float = _G, Ixx: float = _IXX, Iyy: float = _IYY, Izz: float = _IZZ,
    alpha_lo: float = 0.5, alpha_hi: float = 0.9,
) -> List[Scenario]:
    _, emb = get_system_and_embedding(m, g, Ixx, Iyy, Izz)

    ones_4 = jnp.ones(4)
    channel_names = ["ActuatorFault_1", "ActuatorFault_2", "ActuatorFault_3", "ActuatorFault_4"]

    scenarios = [
        Scenario("Nominal", emb, irx.Interval(lower=ones_4, upper=ones_4)),
    ]
    for i, name in enumerate(channel_names):
        alpha_lo_vec = ones_4.at[i].set(alpha_lo)
        alpha_hi_vec = ones_4.at[i].set(alpha_hi)
        scenarios.append(Scenario(name, emb, irx.Interval(lower=alpha_lo_vec, upper=alpha_hi_vec)))

    return scenarios

def euler_step(emb_sys, x_ivl: irx.Interval, u: jnp.ndarray,
               p_ivl: irx.Interval, dt: float) -> irx.Interval:
    _t = jnp.zeros(())
    x_ut = irx.i2ut(x_ivl)
    dx_ut = emb_sys.f(_t, x_ut, u, p_ivl)
    return irx.ut2i(dx_ut * dt + x_ut)

def _propagate_by_params(x0_ivl: irx.Interval, u: jnp.ndarray, emb_sys,
                         p_ivl: irx.Interval, dt: float, num_steps: int) -> irx.Interval:
    def step(x_carry, _i):
        return euler_step(emb_sys, x_carry, u, p_ivl, dt)
    return _run_unrolled_or_loop(step, x0_ivl, num_steps)

def propagate_scenario(x0_ivl: irx.Interval, u: jnp.ndarray,
                       scenario: Scenario,
                       dt: float, num_steps: int) -> irx.Interval:
    return _propagate_by_params(x0_ivl, u, scenario.emb_system, scenario.p_interval, dt, num_steps)

def _propagate_all_scenarios(x0_ivl: irx.Interval, u: jnp.ndarray,
                             scenarios: List[Scenario], dt: float, num_steps: int) -> List[irx.Interval]:
    emb_sys = scenarios[0].emb_system
    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )
    x_final_batch = jax.vmap(lambda p: _propagate_by_params(x0_ivl, u, emb_sys, p, dt, num_steps))(p_batch)
    return [
        irx.Interval(lower=x_final_batch.lower[i], upper=x_final_batch.upper[i])
        for i in range(len(scenarios))
    ]

def separation_loss(u: jnp.ndarray,
                    x0_ivl: irx.Interval,
                    scenarios: List[Scenario],
                    dt: float,
                    num_steps: int) -> jnp.ndarray:
    x_ivls = _propagate_all_scenarios(x0_ivl, u, scenarios, dt, num_steps)
    n = len(x_ivls)
    lo_stack = jnp.stack([iv.lower for iv in x_ivls])
    hi_stack = jnp.stack([iv.upper for iv in x_ivls])
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
                 learning_rate: float = 0.01, num_iters: int = 150,
                 verbose: bool = False) -> Tuple[jnp.ndarray, float]:
        if u_init is None:
            u_init = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])
        u = u_init
        for i in range(num_iters):
            g = self.grad_fn(u)
            u = _project_u(u - learning_rate * g)
            if verbose and (i % 20 == 0 or i == num_iters - 1):
                loss = float(self.loss_fn(u))
                print(f"  Iter {i:4d}  loss={loss:.6f}  u={np.array(u).round(4)}  |g|={float(jnp.linalg.norm(g)):.4f}")
        return u, float(self.loss_fn(u))

    def evaluate(self, u: jnp.ndarray) -> Dict:
        x_ivls = _propagate_all_scenarios(self.x0_ivl, u, self.scenarios, self.dt, self.num_steps)
        n = len(self.scenarios)
        overlaps = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{self.scenarios[i].name} vs {self.scenarios[j].name}"
                overlaps[key] = float(_overlap_volume(x_ivls[i], x_ivls[j]))
        volumes = {
            s.name: float(jnp.prod(iv.upper - iv.lower))
            for s, iv in zip(self.scenarios, x_ivls)
        }
        return {'state_intervals': x_ivls, 'pairwise_overlaps': overlaps, 'volumes': volumes}

def optimize_parallel_gpu(opt: 'SeparatingInputOptimizer', num_restarts: int = 100,
                          learning_rate: float = 0.01, num_iters: int = 150,
                          seed: int = 42):
    key = jax.random.PRNGKey(seed)
    u_init = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])
    noise_scale = _U_NOISE_SCALE
    u0 = jax.random.normal(key, (num_restarts, 4)) * noise_scale + u_init

    batched_loss = jax.vmap(opt.loss_fn)
    batched_grad = jax.vmap(opt.grad_fn)

    def body(u, _i):
        g = batched_grad(u)
        return _project_u(u - learning_rate * g)

    u_final = _run_unrolled_or_loop_nocheckpoint(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.argmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses

def optimize_parallel_gpu_rejit(x0_ivl: irx.Interval, scenarios: List[Scenario],
                                dt: float, num_steps: int, num_restarts: int = 100,
                                learning_rate: float = 0.01, num_iters: int = 150,
                                seed: int = 42):
    return optimize_parallel_gpu(
        SeparatingInputOptimizer(scenarios=scenarios, x0_ivl=x0_ivl, dt=dt, num_steps=num_steps),
        num_restarts=num_restarts, learning_rate=learning_rate, num_iters=num_iters, seed=seed,
    )

def _propagate_history(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                       emb_sys, p_ivl: irx.Interval,
                       dt: float, steps_per_segment: int) -> irx.Interval:
    def segment(x_ivl, u_k):
        def step(x_carry, _):
            return euler_step(emb_sys, x_carry, u_k, p_ivl, dt), None
        x_end, _ = jax.lax.scan(step, x_ivl, xs=None, length=steps_per_segment)
        return x_end, x_end

    _, x_hist = jax.lax.scan(jax.checkpoint(segment), x0_ivl, u_seq)
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
                              steps_per_segment: int,
                              margin_tau: Optional[float] = None) -> jnp.ndarray:
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

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    i_idx = jnp.array([i for i, j in pairs])
    j_idx = jnp.array([j for i, j in pairs])

    def overlap_at_k(k):
        lower_i, upper_i = x_hist_batch.lower[i_idx, k], x_hist_batch.upper[i_idx, k]
        lower_j, upper_j = x_hist_batch.lower[j_idx, k], x_hist_batch.upper[j_idx, k]
        if margin_tau is None:
            widths = jnp.maximum(jnp.minimum(upper_i, upper_j) - jnp.maximum(lower_i, lower_j), 0.0)
            return jnp.sum(jnp.prod(widths, axis=-1))
        else:
            margins = jnp.maximum(lower_j - upper_i, lower_i - upper_j)
            n_dims = margins.shape[-1]
            soft_best_margin = margin_tau * (jax.scipy.special.logsumexp(
                margins / margin_tau, axis=-1) - jnp.log(n_dims))
            return jnp.sum(jnp.maximum(-soft_best_margin, 0.0))

    segment_overlaps = jax.vmap(overlap_at_k)(jnp.arange(num_segments))
    return jnp.min(segment_overlaps)

class MultistepSequenceOptimizer:

    def __init__(self, scenarios: List[Scenario], x0_ivl: irx.Interval,
                 dt: float, steps_per_segment: int, num_segments: int,
                 margin_tau: Optional[float] = None):
        self.scenarios = scenarios
        self.x0_ivl = x0_ivl
        self.dt = dt
        self.steps_per_segment = steps_per_segment
        self.num_segments = num_segments

        _loss = partial(separation_loss_multistep, x0_ivl=x0_ivl, scenarios=scenarios,
                        dt=dt, steps_per_segment=steps_per_segment, margin_tau=margin_tau)
        self.loss_fn = jax.jit(_loss)
        self.grad_fn = jax.jit(jax.grad(_loss))

def optimize_multistep_gpu(opt: 'MultistepSequenceOptimizer', num_restarts: int = 100,
                           learning_rate: float = 0.01, num_iters: int = 150, seed: int = 42,
                           normalize_grad: bool = False):
    key = jax.random.PRNGKey(seed)
    u_init = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])
    noise_scale = _U_NOISE_SCALE
    u0 = jax.random.normal(key, (num_restarts, opt.num_segments, 4)) * noise_scale + u_init

    batched_loss = jax.vmap(opt.loss_fn)
    batched_grad = jax.vmap(opt.grad_fn)

    def body(u_batch, _i):
        g = batched_grad(u_batch)
        if normalize_grad:
            flat = g.reshape(g.shape[0], -1)
            g = g / (jnp.linalg.norm(flat, axis=-1).reshape(-1, 1, 1) + 1e-12)
        return _project_u(u_batch - learning_rate * g)

    u_final = _run_unrolled_or_loop_nocheckpoint(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.argmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses

def optimize_multistep_gpu_rejit(x0_ivl: irx.Interval, scenarios: List[Scenario],
                                 dt: float, steps_per_segment: int, num_segments: int,
                                 num_restarts: int = 100, learning_rate: float = 0.01,
                                 num_iters: int = 150, seed: int = 42,
                                 margin_tau: Optional[float] = None,
                                 normalize_grad: bool = False):
    return optimize_multistep_gpu(
        MultistepSequenceOptimizer(scenarios=scenarios, x0_ivl=x0_ivl, dt=dt,
                                   steps_per_segment=steps_per_segment, num_segments=num_segments,
                                   margin_tau=margin_tau),
        num_restarts=num_restarts, learning_rate=learning_rate, num_iters=num_iters, seed=seed,
        normalize_grad=normalize_grad,
    )

def optimize_multistep(scenarios: List[Scenario], x0_ivl: irx.Interval, dt: float,
                       steps_per_segment: int, num_segments: int,
                       learning_rate: float = 0.01, num_iters: int = 300,
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

    x_ivls = [
        propagate_scenario_multistep(x0_ivl, u_seq, s, dt, steps_per_segment)
        for s in scenarios
    ]
    n = len(scenarios)
    overlaps = {}
    for i in range(n):
        for j in range(i + 1, n):
            overlaps[f"{scenarios[i].name} vs {scenarios[j].name}"] = float(_overlap_volume(x_ivls[i], x_ivls[j]))
    volumes = {s.name: float(jnp.prod(iv.upper - iv.lower)) for s, iv in zip(scenarios, x_ivls)}
    stats = {
        'state_intervals': x_ivls,
        'pairwise_overlaps': overlaps,
        'volumes': volumes,
        'optimization_time_s': elapsed,
        'best_restart_idx': best_idx,
        'all_restart_losses': np.array(final_losses),
        'num_restarts': num_restarts,
    }
    return u_seq, loss_opt, stats

def propagate_with_refinement(x0_ivl: irx.Interval, u_seq: jnp.ndarray,
                              scenarios: List[Scenario], dt: float,
                              num_steps: int = 2,
                              pair_cost_fn=_overlap_volume) -> jnp.ndarray:
    n = len(scenarios)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    N = x0_ivl.lower.shape[0]
    emb_sys = scenarios[0].emb_system

    def ivl_to_arr(ivl: irx.Interval) -> jnp.ndarray:
        return jnp.concatenate([ivl.lower, ivl.upper])

    def arr_to_ivl(arr: jnp.ndarray) -> irx.Interval:
        return irx.Interval(lower=arr[:N], upper=arr[N:])

    p_i_batch = irx.Interval(
        lower=jnp.stack([scenarios[i].p_interval.lower for i, j in pairs]),
        upper=jnp.stack([scenarios[i].p_interval.upper for i, j in pairs]),
    )
    p_j_batch = irx.Interval(
        lower=jnp.stack([scenarios[j].p_interval.lower for i, j in pairs]),
        upper=jnp.stack([scenarios[j].p_interval.upper for i, j in pairs]),
    )

    p_batch = irx.Interval(
        lower=jnp.stack([s.p_interval.lower for s in scenarios]),
        upper=jnp.stack([s.p_interval.upper for s in scenarios]),
    )
    x1_batch = jax.vmap(lambda p: euler_step(emb_sys, x0_ivl, u_seq[0], p, dt))(p_batch)
    x1_ivls = [irx.Interval(lower=x1_batch.lower[i], upper=x1_batch.upper[i]) for i in range(n)]

    pair_i, pair_j = _pair_indices(n)
    step1_cost = jnp.sum(jax.vmap(pair_cost_fn)(
        irx.Interval(lower=x1_batch.lower[pair_i], upper=x1_batch.upper[pair_i]),
        irx.Interval(lower=x1_batch.lower[pair_j], upper=x1_batch.upper[pair_j]),
    ))

    pxi_arr = jnp.stack([ivl_to_arr(x1_ivls[i]) for i, j in pairs])
    pxj_arr = jnp.stack([ivl_to_arr(x1_ivls[j]) for i, j in pairs])

    def refine_one_pair(xi_arr, xj_arr, p_i, p_j, u_k):
        x_curr_i = arr_to_ivl(xi_arr)
        x_curr_j = arr_to_ivl(xj_arr)

        x_lo = jnp.maximum(x_curr_i.lower, x_curr_j.lower)
        x_hi = jnp.minimum(x_curr_i.upper, x_curr_j.upper)
        has_overlap = jnp.all(x_hi >= x_lo)
        x_int = irx.Interval(lower=x_lo, upper=x_hi)

        x_ref_i = irx.Interval(
            lower=jnp.where(has_overlap, x_int.lower, x_curr_i.lower),
            upper=jnp.where(has_overlap, x_int.upper, x_curr_i.upper),
        )
        x_ref_j = irx.Interval(
            lower=jnp.where(has_overlap, x_int.lower, x_curr_j.lower),
            upper=jnp.where(has_overlap, x_int.upper, x_curr_j.upper),
        )

        x_next_i = euler_step(emb_sys, x_ref_i, u_k, p_i, dt)
        x_next_j = euler_step(emb_sys, x_ref_j, u_k, p_j, dt)

        raw_cost = pair_cost_fn(x_next_i, x_next_j)
        pair_cost = jnp.where(has_overlap, raw_cost, jnp.array(0.0))
        return ivl_to_arr(x_next_i), ivl_to_arr(x_next_j), pair_cost

    def step_body(carry, k):
        pxi_arr, pxj_arr, min_cost = carry
        u_k = u_seq[k + 1]
        new_pxi, new_pxj, pair_costs = jax.vmap(
            refine_one_pair, in_axes=(0, 0, 0, 0, None)
        )(pxi_arr, pxj_arr, p_i_batch, p_j_batch, u_k)
        step_cost = jnp.sum(pair_costs)
        return (new_pxi, new_pxj, jnp.minimum(min_cost, step_cost))

    init_carry = (pxi_arr, pxj_arr, step1_cost)
    _, _, min_cost_final = _run_unrolled_or_loop(step_body, init_carry, num_steps - 1)
    return min_cost_final

def refined_overlap_loss(u_seq: jnp.ndarray, x0_ivl: irx.Interval,
                         scenarios: List[Scenario], dt: float,
                         num_steps: int = 2,
                         pair_cost_fn=_overlap_volume) -> jnp.ndarray:
    return propagate_with_refinement(x0_ivl, u_seq, scenarios, dt, num_steps, pair_cost_fn)

def optimize_refined_gpu(x0_ivl: irx.Interval, scenarios: List[Scenario], dt: float,
                         num_steps: int = 2, num_restarts: int = 50,
                         learning_rate: float = 0.05, num_iters: int = 200,
                         seed: int = 42,
                         pair_cost_fn=_overlap_volume) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    key = jax.random.PRNGKey(seed)
    u_init = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])
    noise_scale = _U_NOISE_SCALE
    u0 = jax.random.normal(key, (num_restarts, num_steps, 4)) * noise_scale + u_init

    def loss_fn_refined(u_seq):
        return refined_overlap_loss(u_seq, x0_ivl, scenarios, dt, num_steps, pair_cost_fn)

    batched_loss = jax.vmap(loss_fn_refined)
    batched_grad = jax.vmap(jax.grad(loss_fn_refined))

    def body(u_batch, _i):
        g = batched_grad(u_batch)
        return _project_u(u_batch - learning_rate * g)

    u_final = _run_unrolled_or_loop_nocheckpoint(body, u0, num_iters)
    losses = batched_loss(u_final)
    best_idx = jnp.argmin(losses)
    return u_final[best_idx], losses[best_idx], u_final, losses
