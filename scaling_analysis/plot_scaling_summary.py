import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

_CSV_PATH = _HERE / "scaling_summary.csv"

_COLOR_UNREFINED = "#2a78d6"
_COLOR_REFINED = "#eb6834"
_INK = "0.15"

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif'],
    'text.usetex': False,
})

_SWEEPS = {
    "horizon": {
        "out": _HERE / "scaling_summary_horizon.pdf",
        "xlabel": "Number of planning steps",
    },
    "scenario": {
        "out": _HERE / "scaling_summary_scenario.pdf",
        "xlabel": "Number of fault scenarios",
        "integer_xticks": True,
    },
    "dimensionality": {
        "out": _HERE / "scaling_summary_dimensionality.pdf",
        "xlabel": "Integrator-chain order $N$",
    },
}

def load_summary(csv_path: Path = _CSV_PATH):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {sweep: {"unrefined": [], "refined": []} for sweep in _SWEEPS}
    for r in rows:
        row = dict(r)
        row["x"] = int(float(row["x"]))
        for key in ("compile_time_ms", "run_time_ms", "vram_mb", "host_ram_mb"):
            row[key] = float(row[key])
        out[row["sweep"]][row["optimizer"]].append(row)
    for sweep in out:
        for optimizer in out[sweep]:
            out[sweep][optimizer].sort(key=lambda r: r["x"])
    return out

def make_plot(unrefined_rows, refined_rows, xlabel: str, out_path: Path,
              integer_xticks: bool = False):
    fig, ax_run = plt.subplots(figsize=(6.5, 4.4))
    ax_vram = ax_run.twinx()

    handles = []
    for rows, color, label in (
        (unrefined_rows, _COLOR_UNREFINED, "Unrefined"),
        (refined_rows, _COLOR_REFINED, "Refined"),
    ):
        if not rows:
            continue
        xs = [r["x"] for r in rows]
        l_run, = ax_run.plot(xs, [r["run_time_ms"] for r in rows],
                             color=color, linewidth=2, linestyle='-',
                             marker='o', markersize=5, label=f"{label} -- run time", zorder=3)
        l_vram, = ax_vram.plot(xs, [r["vram_mb"] for r in rows],
                               color=color, linewidth=2, linestyle='--',
                               marker='s', markersize=4.5, markerfacecolor='white',
                               markeredgewidth=1.0, label=f"{label} -- GPU VRAM", zorder=2)
        handles += [l_run, l_vram]

    if integer_xticks:
        ax_run.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax_run.set_xlabel(xlabel)
    ax_run.set_ylabel("Run time (ms)")
    ax_vram.set_ylabel("GPU VRAM (MB)")
    ax_run.grid(True, color="0.85", linewidth=0.8)
    ax_vram.grid(False)
    ax_run.spines["top"].set_visible(False)
    ax_vram.spines["top"].set_visible(False)
    ax_run.tick_params(axis="y", labelcolor=_INK)
    ax_vram.tick_params(axis="y", labelcolor=_INK)

    ax_run.legend(handles=handles, frameon=False, loc="upper left",
                  fontsize=9, labelcolor=_INK, handlelength=2.4)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path

if __name__ == "__main__":
    data = load_summary()
    for sweep, cfg in _SWEEPS.items():
        unrefined_rows = data[sweep]["unrefined"]
        refined_rows = data[sweep]["refined"]
        print(f"{sweep}: {len(unrefined_rows)} unrefined points, {len(refined_rows)} refined points")
        out = make_plot(unrefined_rows, refined_rows, cfg["xlabel"], cfg["out"],
                        integer_xticks=cfg.get("integer_xticks", False))
        print(f"  wrote {out}")
