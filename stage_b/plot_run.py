"""
Plots of a course run and of run comparisons.

Design rules:
- Small multiples, no dual axes: every metric in its own panel on a shared
  time axis.
- Categorical, fixed colours (same as the report figures); the baseline is
  grey dashed because it is a reference, not a method.
- Direct labels at the right end of each curve, so identity does not rely
  on colour alone.
- Events as dotted vertical lines with a label above the top panel.

Usage:
    python plot_run.py                      # from course_*.npz files
    python plot_run.py --files course_M0.npz course_M2.npz
    python plot_run.py --walk walk_log.npz  # a single run_walk.py run
"""

import argparse
from pathlib import Path
import numpy as np
import sys
import matplotlib
# Agg has no window. live_plot.py may already have opened an interactive
# backend; forcing Agg would kill the live window, so only force it if
# pyplot is not loaded yet.
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- categorical palette (validated) --------------------------------------
COLORS = {
    "none": "#8a8a85",     # grey -- reference, not a method
    "M0I":  "#e87ba4",
    "M0":   "#2a78d6",
    "M0F":  "#eda100",
    "M1":   "#eb6834",
    "M2":   "#1baf7a",
}
LABELS = {
    "none": "no adaptation",
    "M0I": "M0I  integral (6)",
    "M0": "M0  regressor (12)",
    "M0F": "M0F  phase (24)",
    "M1": "M1  RBF (390)",
    "M2": "M2  DNN (208)",
}
STYLE = {"none": "--", "M0I": "-", "M0": "-", "M0F": "-", "M1": "-", "M2": "-"}

INK = "#2b2b28"
MUTED = "#6e6e68"
GRID = "#e3e3df"


def _setup(ax, ylabel, last=False):
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    if not last:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("time (s)", fontsize=9, color=INK)


def _events(ax, course, show_text=False):
    """Event lines; the text sits above the axis, not over the data."""
    for t_e, kind, val, label in course:
        ax.axvline(t_e, color=MUTED, linestyle=":", linewidth=1.0,
                   zorder=1, alpha=0.8)
        if show_text:
            ax.annotate(label, xy=(t_e, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 6), textcoords="offset points",
                        va="bottom", ha="center", fontsize=7,
                        color=MUTED, zorder=5)


def _place_labels(ax, items, pad_frac=0.085):
    """
    Place the direct labels at the right edge without overlap: sort in axis
    fractions and enforce at least pad_frac between neighbours.

    items : [(y_last, text, colour), ...]
    """
    if not items:
        return
    y0, y1 = ax.get_ylim()
    span = max(y1 - y0, 1e-12)
    rows = sorted(((np.clip((y - y0) / span, 0.0, 1.0), txt, col)
                   for y, txt, col in items), key=lambda r: r[0])
    placed = []
    for frac, txt, col in rows:
        if placed and frac - placed[-1][0] < pad_frac:
            frac = placed[-1][0] + pad_frac
        placed.append((frac, txt, col))
    over = placed[-1][0] - 1.0
    if over > 0:
        placed = [(f - over, t, c) for f, t, c in placed]
    for frac, txt, col in placed:
        ax.annotate(txt, xy=(1.0, np.clip(frac, 0.0, 1.0)),
                    xycoords="axes fraction", xytext=(5, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=7.5, color=col, zorder=6,
                    annotation_clip=False)


def plot_course(results, course, T, out="path.png", terrain=False,
                obstacles=False):
    """
    results : {kind: (L, rows)}  -- from scenario.py
    course  : event list
    """
    panels = []
    if terrain:
        # on terrain the ground changes with position, read through x(t)
        panels.append(("x", "progress x (m)", 1.0))
        panels.append(("terrain_h", "estimated ground (m)", 1.0))
    panels += [
        ("pitch", "pitch (rad)", 1.0),
        ("roll", "roll (rad)", 1.0),
        ("z", "body z (m)", 1.0),
        ("pred_err", "prediction error", 1.0),
        ("theta_norm", r"$\|\hat{\theta}\|$", 1.0),
        ("pen", "foot penetration (mm)", 1000.0),
        ("tau_max", "max torque (Nm)", 1.0),
    ]
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(10.5, 1.55 * n), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")

    order = [k for k in ["none", "M0I", "M0", "M0F", "M1", "M2"] if k in results]
    for pi, (key, ylab, scale) in enumerate(panels):
        ax = axes[pi]
        ax.set_facecolor("#fcfcfb")
        _setup(ax, ylab, last=(pi == n - 1))
        labels_here = []
        for k in order:
            L = results[k][0]
            t = L["t"]
            if key == "theta_norm":
                if L["theta"].size == 0:
                    continue
                y = np.linalg.norm(L["theta"], axis=1)
            elif key not in L:
                continue
            else:
                y = L[key] * scale
            if key == "pred_err" and k == "none":
                continue          # the baseline has no predictor
            if key == "theta_norm" and k == "none":
                continue
            ax.plot(t, y, STYLE[k], color=COLORS[k], linewidth=1.6,
                    zorder=3, label=LABELS[k])
            if len(y):
                labels_here.append((float(y[-1]),
                                    "baseline" if k == "none" else k, COLORS[k]))
        # mark runs that fell
        for k in order:
            L = results[k][0]
            if L.get("fell") is not None:
                ax.axvline(L["fell"], color=COLORS[k], linewidth=1.0,
                           alpha=0.5, zorder=2)
        ax.set_xlim(0, T)
        ax.margins(y=0.12)
        _events(ax, course, show_text=(pi == 0))
        _place_labels(ax, labels_here)

    # reference lines
    names = [p[0] for p in panels]
    if "z" in names and not terrain:
        axes[names.index("z")].axhline(
            0.27, color=MUTED, linewidth=0.8, alpha=0.5, zorder=1)
    ti = names.index("tau_max")
    axes[ti].axhline(33.5, color="#e34948", linewidth=0.8, alpha=0.6, zorder=1)
    axes[ti].annotate("torque limit 33.5", xy=(0.3, 33.5), fontsize=7,
                      color="#e34948", va="bottom")
    if terrain:
        # zone boundaries in the x panel
        from course_def import TERRAIN_ZONES
        zones = list(TERRAIN_ZONES)
        zf = Path(__file__).resolve().parent / "obstacle_zone.json"
        if obstacles and zf.exists():
            import json
            z = json.loads(zf.read_text(encoding="utf-8"))
            if z["base"] == "flat":
                zones = [(0.0, z["x0"], "flat"),
                         (z["x0"], z["x1"], f"OBSTACLES ({z['n']})")]
            else:
                zones = [t for t in zones if t[0] < z["x0"]]
                zones.append((z["x0"], z["x1"], f"OBSTACLES ({z['n']})"))
        axz = axes[names.index("x")]
        for x0, x1, lab in zones:
            axz.axhline(x0, color=MUTED, linewidth=0.6, linestyle=":",
                        alpha=0.7, zorder=1)
            axz.annotate(lab, xy=(0.012, x0), xycoords=("axes fraction", "data"),
                         xytext=(0, 2), textcoords="offset points",
                         fontsize=6.5, color=MUTED, va="bottom", zorder=5)

    axes[0].legend(loc="upper left", fontsize=8, frameon=False,
                   ncol=len(order), bbox_to_anchor=(0, 1.62))
    axes[0].set_title(("TerrAdapt terrain course -- ramp, plateau and rough "
                       "section" if terrain else
                       "TerrAdapt course run -- ground and payload change "
                       "while walking"),
                      fontsize=11, color=INK, loc="left", pad=46)
    fig.tight_layout()
    fig.subplots_adjust(right=0.93)
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(Path(out).resolve())


def plot_single(npz, out=None):
    """Single-run plot for run_walk.py's walk_log.npz."""
    L = dict(np.load(npz, allow_pickle=True))
    t = L["t"]
    panels = [("z", "body z (m)", 1.0),
              ("pitch", "pitch (rad)", 1.0),
              ("roll", "roll (rad)", 1.0),
              ("vx", "vx (m/s)", 1.0),
              ("swing_err", "swing error (mm)", 1000.0),
              ("tau_max", "max torque (Nm)", 1.0)]
    panels = [p for p in panels if p[0] in L]
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(9.5, 1.5 * n), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    if n == 1:
        axes = [axes]
    for pi, (key, ylab, sc) in enumerate(panels):
        ax = axes[pi]
        ax.set_facecolor("#fcfcfb")
        _setup(ax, ylab, last=(pi == n - 1))
        ax.plot(t, L[key] * sc, "-", color=COLORS["M2"], linewidth=1.6,
                zorder=3)
    axes[0].set_title(f"{Path(npz).name} -- single run", fontsize=11,
                      color=INK, loc="left")
    fig.tight_layout()
    out = out or (Path(npz).stem + ".png")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(Path(out).resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=None,
                    help="course_*.npz files")
    ap.add_argument("--walk", default=None, help="walk_log.npz")
    ap.add_argument("--out", default="path.png")
    ap.add_argument("--T", type=float, default=20.0)
    args = ap.parse_args()

    if args.walk:
        print("written:", plot_single(args.walk))
        return

    files = args.files
    if files is None:
        files = sorted(Path(".").glob("course_*.npz"))
    if not files:
        print("no course_*.npz found -- run scenario.py first")
        return

    results = {}
    for f in files:
        f = Path(f)
        kind = f.stem.replace("course_", "")
        L = dict(np.load(f, allow_pickle=True))
        L["fell"] = None
        results[kind] = (L, None)
        print(f"read: {f.name}  ({kind})")

    from course_def import default_course
    print("written:", plot_course(results, default_course(), args.T,
                                  args.out))


if __name__ == "__main__":
    main()
