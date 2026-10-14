import os; os.environ['JAX_PLATFORMS'] = 'cpu'
import sys
from pathlib import Path

import jax.numpy as jnp
import immrax as irx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from admire_separating_input import (
    create_scenarios,
    euler_step,
    output_interval,
)

CONTROLLER_FILE = HERE / 'admire_u_opt.npz'

_result_file = CONTROLLER_FILE
if not _result_file.exists():
    raise FileNotFoundError(
        f"{_result_file} not found — it ships next to this script; restore it "
        "from git rather than re-solving."
    )

_data = np.load(_result_file)
u_opt_np          = _data['u_opt']
dt                = float(_data['dt'])
num_steps         = int(_data['num_steps']) 
fault_effectiveness = float(_data['fault_effectiveness'])
num_scenarios     = int(_data['num_scenarios'])
x0_ivl            = irx.Interval(
    lower=jnp.array(_data['x0_ivl_lower']),
    upper=jnp.array(_data['x0_ivl_upper']),
)
u_seq = jnp.array(u_opt_np)

print(f"Loaded {_result_file.name}")
print(f"  u_seq shape : {u_seq.shape}")
print(f"  dt={dt}s  num_steps={num_steps}  horizon={num_steps*dt:.2f}s")

scenarios = create_scenarios(fault_effectiveness=fault_effectiveness)[:num_scenarios]
print(f"{len(scenarios)} scenarios: " + ", ".join(s.name for s in scenarios))

pairs = [(i, j) for i in range(0, len(scenarios)) for j in range(i)]
n_pairs = len(pairs)

def _obs(x_ivl):
    return output_interval(x_ivl)

def _has_intersection(a, b):
    lo = jnp.maximum(a.lower, b.lower)
    hi = jnp.minimum(a.upper, b.upper)
    return bool(jnp.all(hi >= lo))

print("Collecting unrefined output histories …")
output_histories = []
for k, s in enumerate(scenarios):
    print(f"  scenario {k+1}/{len(scenarios)}: {s.name}", flush=True)
    hist = [_obs(x0_ivl)]
    x = x0_ivl
    for step in range(num_steps):
        x = euler_step(s.emb_system, x, u_seq[step], s.p_interval, dt)
        hist.append(_obs(x))
    output_histories.append(hist)

t_unref = [k * dt for k in range(num_steps + 1)]
unref_valid = [True] * n_pairs
unref_counts = []
for t_idx in range(num_steps + 1):
    for k, (i, j) in enumerate(pairs):
        if unref_valid[k] and not _has_intersection(
            output_histories[i][t_idx], output_histories[j][t_idx]
        ):
            unref_valid[k] = False
    unref_counts.append(sum(unref_valid))

print("Collecting refined histories …")

x1 = [euler_step(s.emb_system, x0_ivl, u_seq[0], s.p_interval, dt) for s in scenarios]
pair_states = [(x1[i], x1[j]) for i, j in pairs]
obs1 = [(_obs(x1[i]), _obs(x1[j])) for i, j in pairs]

t_ref = [(k + 1) * dt for k in range(num_steps)]
ref_valid = [True] * n_pairs
for k, (oi, oj) in enumerate(obs1):
    if not _has_intersection(oi, oj):
        ref_valid[k] = False
ref_counts = [sum(ref_valid)]

for step in range(num_steps - 1):
    u_k = u_seq[step + 1]
    new_pair_states = []
    new_obs = []
    for k, ((i, j), (xi, xj)) in enumerate(zip(pairs, pair_states)):
        oi = _obs(xi)
        oj = _obs(xj)
        y_lo = jnp.maximum(oi.lower, oj.lower)
        y_hi = jnp.minimum(oi.upper, oj.upper)
        hov  = bool(jnp.all(y_hi >= y_lo))
        fb   = (xi.lower[3:6] + xi.upper[3:6]) / 2
        ys_lo = jnp.where(hov, y_lo, fb)
        ys_hi = jnp.where(hov, y_hi, fb)

        xi_ref = irx.Interval(lower=xi.lower.at[3:6].set(ys_lo),
                              upper=xi.upper.at[3:6].set(ys_hi))
        xj_ref = irx.Interval(lower=xj.lower.at[3:6].set(ys_lo),
                              upper=xj.upper.at[3:6].set(ys_hi))

        xn_i = euler_step(scenarios[i].emb_system, xi_ref, u_k, scenarios[i].p_interval, dt)
        xn_j = euler_step(scenarios[j].emb_system, xj_ref, u_k, scenarios[j].p_interval, dt)
        new_pair_states.append((xn_i, xn_j))
        new_obs.append((_obs(xn_i), _obs(xn_j)))

    pair_states = new_pair_states
    for k, (oi, oj) in enumerate(new_obs):
        if ref_valid[k] and not _has_intersection(oi, oj):
            ref_valid[k] = False
    ref_counts.append(sum(ref_valid))
t_ref.insert(0, 0)
ref_counts.insert(0, n_pairs)
print(f"  unrefined counts: {unref_counts}")
print(f"  refined counts  : {ref_counts}")

plt.rcParams.update({'font.family': 'serif', 'font.size': 11})

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(t_unref[::10], unref_counts[::10], color='steelblue', linewidth=2,
        marker='o', markersize=4, label='Baseline')
ax.plot(t_ref[::10],   ref_counts[::10],   color='darkorange', linewidth=2,
        marker='s', markersize=4, label='Intersection Refinement')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Number of unseparated model pairs')
ax.set_ylim(-0.5, n_pairs + 0.5)
ax.set_yticks(range(0, n_pairs + 1, 5))
ax.legend(frameon=True, fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

fname = HERE / 'admire_valid_pairs.pdf'
plt.savefig(fname, bbox_inches='tight')
plt.close(fig)
print(f"Saved {fname}")
