"""
Report figures. Reads ONLY recorded data:
  - sonuclar/*/izgara.csv  (3-seed sweeps)  -> heatmap, pendulum cost
  - report/data/*.npz      (seed-0 time series, illustration only;
                            written by report/figure_data.py)

The recorded sweep folders and their CSV column names are kept exactly as
they were written (the measurements are not modified); rd() maps the
original column names to English keys.

    python report/make_figures.py       # all figures whose data exist
Output: report/figures/*.pdf
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

KOK = Path(__file__).resolve().parent.parent
SON = KOK / "sonuclar"
DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Categorical, fixed order, validated (dataviz validate_palette.js, light):
# entity -> colour never changes across figures.
COL = {"none": "#8a8a85", "M0": "#2a78d6", "M1": "#eb6834",
       "M2": "#1baf7a", "M0F": "#eda100", "M0I": "#e87ba4"}
LAB = {"none": "baseline (no adaptation)", "M0I": "M0I (6)", "M0": "M0 (12)",
       "M0F": "M0F (24)", "M1": "M1 RBF (390)", "M2": "M2 Lb-DNN (208)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]  # 0..3 of 3 (sequential)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.2, "pdf.fonttype": 42,
})
W1 = 3.45   # IEEE single column, inches


# original column names of the recorded CSVs -> English keys
COLUMN_MAP = {"yontem": "method", "tohum": "seed", "bitti": "done",
              "filtre": "filter", "kol": "arm", "takildi": "stalled",
              "dustu": "fell", "x_son": "x_final", "ayakta": "standing",
              "sarkac_std_deg": "pendulum_std_deg"}
VALUE_MAP = {"SARKAC": "PENDULUM", "RIJIT": "RIGID"}


def rd(path):
    """Read a recorded CSV with English column names and arm labels."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{COLUMN_MAP.get(k, k): VALUE_MAP.get(v, v) for k, v in r.items()}
            for r in rows]


# ---------------------------------------------------------------------------
def fig_window():
    """Completion out of 3 seeds, method x gain, combined course."""
    rows = [r for r in rd(SON / "dayaniklilik_y+0.12+0.05_g100-300-1000_t0-1-2_filtre-kiyas/izgara.csv")
            if r["filter"] == "0"]
    for sub in ["filtresiz_M0F", "filtresiz_M0I"]:
        rows += rd(SON / f"dayaniklilik_y+0.12+0.05_g100-300-1000_t0-1-2_{sub}/izgara.csv")
    rows += rd(SON / "dayaniklilik_y+0.12+0.05_g2000-3000_t0-1-2_filtresiz_M0/izgara.csv")
    meth = ["M0I", "M0", "M0F", "M1", "M2"]
    gams = [100, 300, 1000, 2000, 3000]
    fig, ax = plt.subplots(figsize=(W1, 1.75))
    ax.grid(False)
    for i, m in enumerate(meth):
        for j, g in enumerate(gams):
            rr = [r for r in rows if r["method"] == m and float(r["gamma"]) == g]
            if not rr:
                ax.add_patch(Rectangle((j - .48, i - .46), .96, .92,
                             facecolor="white", edgecolor=GRID, hatch="////",
                             linewidth=0.5))
                ax.text(j, i, "not run", ha="center", va="center",
                        color=INK2, fontsize=6)
                continue
            n = sum(int(r["done"]) for r in rr)
            ax.add_patch(Rectangle((j - .48, i - .46), .96, .92,
                         facecolor=BLUE[n], edgecolor="white", linewidth=1))
            ax.text(j, i, f"{n}/{len(rr)}", ha="center", va="center",
                    color="white" if n >= 2 else INK, fontsize=7,
                    fontweight="bold" if n == 3 else "normal")
    ax.set_xlim(-.5, len(gams) - .5)
    ax.set_ylim(len(meth) - .5, -.5)
    ax.set_xticks(range(len(gams)), [f"{g}" for g in gams])
    ax.set_yticks(range(len(meth)), [LAB[m] for m in meth])
    ax.set_xlabel(r"adaptation gain $\gamma$ (matched across methods)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_window.pdf")
    plt.close(fig)
    print("fig_window.pdf")


# ---------------------------------------------------------------------------
def fig_pendulum_cost():
    """Pendulum cost = pendulum arm minus rigid arm, same cell, median of 9."""
    base = "sarkac_L0.04x_vx0.3_g100-300-1000_t0-1-2"
    rows = rd(SON / base / "izgara.csv")
    for s in ["_M0F", "_M0I"]:
        rows += rd(SON / (base + s) / "izgara.csv")
    d = {(r["arm"], r["method"], float(r["gamma"]), int(r["seed"])): r for r in rows}
    cells = [(g, s) for g in [100., 300., 1000.] for s in range(3)]
    meth = ["M0I", "M0", "M0F", "M1", "M2"]
    cost = {}
    for m in meth:
        cp = [100 * (float(d[("PENDULUM", m, g, s)]["pred_err"]) /
                     float(d[("RIGID", m, g, s)]["pred_err"]) - 1) for g, s in cells]
        cr = [100 * (float(d[("PENDULUM", m, g, s)]["roll_std"]) /
                     float(d[("RIGID", m, g, s)]["roll_std"]) - 1) for g, s in cells]
        cost[m] = (np.median(cp), np.median(cr))
    fig, axs = plt.subplots(1, 2, figsize=(W1, 1.6), sharey=True)
    for ax, k, ttl in [(axs[0], 0, "estimation error  $\\|\\tilde e\\|$"),
                       (axs[1], 1, "roll std")]:
        ax.set_axisbelow(True)
        y = np.arange(len(meth))
        v = [cost[m][k] for m in meth]
        ax.barh(y, v, height=0.62, color=[COL[m] for m in meth],
                edgecolor="white", linewidth=1)
        for yi, vi in zip(y, v):
            ax.text(vi + 0.6, yi, f"+{vi:.0f}%", va="center", color=INK,
                    fontsize=7)
        ax.set_title(ttl, color=INK)
        ax.set_xlim(0, 32)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=INK2, linewidth=0.6)
    axs[0].set_yticks(range(len(meth)), [LAB[m] for m in meth])
    axs[0].invert_yaxis()
    fig.supxlabel("increase caused by the swinging payload (median of 9 cells)",
                  fontsize=7, color=INK2)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_pendulum_cost.pdf")
    plt.close(fig)
    print("fig_pendulum_cost.pdf", {m: tuple(round(x, 1) for x in cost[m]) for m in meth})


# ---------------------------------------------------------------------------
def smooth(y, n):
    k = np.ones(n) / n
    return np.convolve(y, k, mode="same")


def label_end(ax, t, y, txt, col, dy=0.0):
    ax.text(t[-1] + 0.6, y[-1] + dy, txt, color=INK, fontsize=6.5, va="center")
    ax.plot([t[-1]], [y[-1]], "o", ms=3, color=col)


def fig_baseline():
    """Combined course, gamma=300, seed 0: progress and pitch."""
    meth = ["none", "M0", "M1", "M2"]
    D = {m: np.load(DATA / f"combined_{m}_g{0 if m == 'none' else 300}.npz") for m in meth}
    fig, axs = plt.subplots(2, 1, figsize=(W1, 3.0), sharex=True)
    ax = axs[0]
    for m in meth:
        t, x = D[m]["t"], D[m]["x"]
        ax.plot(t, x, color=COL[m], linestyle="--" if m == "none" else "-",
                label=LAB[m])
    ax.axhline(10.68, color=INK2, linewidth=0.6, linestyle=":")
    ax.text(0.5, 11.0, "finish line 10.68 m", fontsize=6.5, color=INK2)
    tn = D["none"]["t"]
    ax.annotate("baseline stuck\nat 9.01 m", xy=(tn[-1], D["none"]["x"][-1]),
                xytext=(tn[-1] - 3.5, 5.6), fontsize=6.5, color=INK,
                ha="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.axvline(6.0, color=INK2, linewidth=0.6, linestyle=":")
    ax.text(6.4, 0.2, "3 kg payload", fontsize=6.5, color=INK2)
    ax.set_ylabel("x (m)")
    ax.set_ylim(-0.3, 11.8)
    ax.legend(loc="upper left", frameon=False, ncol=1, bbox_to_anchor=(0.0, 0.94))
    ax = axs[1]
    n = int(round(0.4 / (D["M0"]["t"][1] - D["M0"]["t"][0])))   # one gait period
    for m in ["none", "M0"]:
        t, p = D[m]["t"], np.degrees(smooth(D[m]["pitch"], n))
        ax.plot(t, p, color=COL[m], linestyle="--" if m == "none" else "-")
    ax.axvline(6.0, color=INK2, linewidth=0.6, linestyle=":")
    ax.text(13, 9.0, "baseline", fontsize=6.5, color=INK)
    ax.text(13, 2.4, r"M0, $\gamma=300$", fontsize=6.5, color=INK)
    ax.set_ylabel("pitch (deg)")
    ax.set_xlabel("time (s)")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_baseline.pdf")
    plt.close(fig)
    print("fig_baseline.pdf")


def fig_window_xt():
    """Combined course, gamma=1000, seed 0: M0 finishes, M1/M2 stall."""
    meth = ["M0", "M1", "M2"]
    D = {m: np.load(DATA / f"combined_{m}_g1000.npz") for m in meth}
    fig, ax = plt.subplots(figsize=(W1, 1.7))
    dy = {"M0": 0.0, "M1": 0.75, "M2": -0.75}
    for m in meth:
        t, x = D[m]["t"], D[m]["x"]
        ax.plot(t, x, color=COL[m], label=LAB[m])
        st = ("finished" if D[m]["done"] > 0 else
              "fell" if D[m]["fell"] > 0 else "stalled")
        txt = f"{m}: {st}" if st == "finished" else f"{m}: {st} at {x[-1]:.2f} m"
        ax.text(t[-1] + 0.8, x[-1] + dy[m] - (0.5 if st == "finished" else 0), txt,
                fontsize=6.5, color=INK, va="center")
    ax.axhline(10.68, color=INK2, linewidth=0.6, linestyle=":")
    ax.set_xlim(0, 48)
    ax.set_ylim(-0.3, 11.5)
    ax.set_ylabel("x (m)")
    ax.set_xlabel(r"time (s), $\gamma = 1000$")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_window_xt.pdf")
    plt.close(fig)
    print("fig_window_xt.pdf")


def fig_pendulum_trace():
    """Pendulum angle (baseline) and body roll (pendulum arm), seed 0."""
    D = {m: np.load(DATA / f"pendulum_{m}_g{0 if m == 'none' else 300}_pendulum.npz")
         for m in ["none", "M0", "M0F", "M2"]}
    fig, axs = plt.subplots(2, 1, figsize=(W1, 2.9), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1.4]))
    t = D["none"]["t"]
    w = (t >= 14) & (t <= 16)
    axs[0].plot(t[w], np.degrees(D["none"]["pendulum"][w]), color=INK2)
    axs[0].set_ylabel("pendulum (deg)")
    axs[0].text(0.01, 1.04, "(a) pendulum angle, baseline: 2.5 Hz, locked to trot",
                transform=axs[0].transAxes, fontsize=7, color=INK)
    for m in ["M0", "M0F", "M2"]:
        tm = D[m]["t"]
        wm = (tm >= 14) & (tm <= 16)
        r = D[m]["roll"][wm]
        axs[1].plot(tm[wm], np.degrees(r - r.mean()), color=COL[m], label=LAB[m])
    axs[1].set_ylabel("roll (deg)")
    axs[1].text(0.01, 1.04, r"(b) body roll, mean removed, $\gamma=300$",
                transform=axs[1].transAxes, fontsize=7, color=INK)
    axs[1].set_ylim(-0.9, 1.15)
    axs[1].set_xlabel("time (s)")
    axs[1].legend(loc="upper center", frameon=False, ncol=3,
                  bbox_to_anchor=(0.5, 1.0), handlelength=1.5,
                  columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_pendulum_trace.pdf")
    plt.close(fig)
    print("fig_pendulum_trace.pdf")


def fig_course():
    """Schematic height profile of the combined course (bulgular.md, Sec. A)."""
    x = np.linspace(0, 11.2, 2000)
    h = np.zeros_like(x)
    up = (x >= 1.5) & (x < 3.2)
    h[up] = 0.20 * (1 - np.cos(np.pi * (x[up] - 1.5) / 1.7)) / 2
    h[(x >= 3.2) & (x < 5.7)] = 0.20
    bump = (x >= 4.0) & (x < 5.7)
    h[bump] += 0.035 * np.sin(2 * np.pi * (x[bump] - 4.0) / 0.42)
    dn = (x >= 5.7) & (x < 7.4)
    h[dn] = 0.20 * (1 + np.cos(np.pi * (x[dn] - 5.7) / 1.7)) / 2
    fig, ax = plt.subplots(figsize=(2 * W1, 1.35))
    ax.fill_between(x, -0.05, h, color="#e4e3df", linewidth=0)
    ax.plot(x, h, color=INK2, linewidth=1.0)
    rng = np.random.default_rng(0)
    ox = np.sort(rng.uniform(7.75, 9.85, 40))
    oh = rng.uniform(0.015, 0.06, 40)
    ax.bar(ox, oh, width=0.05, bottom=0, color="#86b6ef", edgecolor="white",
           linewidth=0.5)
    for xa, xb, lab in [(0, 1.5, "flat"), (1.5, 3.2, "ramp up 10°"),
                        (3.2, 4.0, "plateau"), (4.0, 5.7, "rough ±3.5 cm"),
                        (5.7, 7.4, "ramp down"), (7.7, 9.9, "84 fixed obstacles")]:
        ax.text((xa + xb) / 2, 0.30, lab, ha="center", fontsize=6.5, color=INK)
        ax.axvline(xa, color=GRID, linewidth=0.6)
    ax.axvline(10.68, color=INK2, linestyle=":", linewidth=0.8)
    ax.text(10.72, 0.12, "finish\n10.68 m", fontsize=6.5, color=INK2)
    ax.annotate("3 kg payload\nat t = 6 s (x ≈ 1.8 m)", xy=(1.8, 0.02),
                xytext=(0.2, 0.15), fontsize=6.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(-0.03, 0.36)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("height (m)")
    ax.grid(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_course.pdf")
    plt.close(fig)
    print("fig_course.pdf")


if __name__ == "__main__":
    fig_course()
    fig_window()
    fig_pendulum_cost()
    for f, need in [(fig_baseline, "combined_M2_g300.npz"),
                    (fig_window_xt, "combined_M2_g1000.npz"),
                    (fig_pendulum_trace, "pendulum_M2_g300_pendulum.npz")]:
        if (DATA / need).exists():
            f()
        else:
            print(f"(skipped {f.__name__}: {need} not yet generated)")
