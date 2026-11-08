"""
Live plot panel -- a window next to the simulation, for screen recording.

Design choices:
1. Fixed axis limits. Auto-scaling makes the axes jump and the video
   unreadable (curve growing vs. axis shrinking cannot be told apart).
2. Limited redraw rate (default 15 Hz). Drawing at every physics step
   slows the simulation ~10x; 15 Hz looks smooth.
3. Five panels only; the nine-panel measurement plot (plot_run.py) is too
   dense for a video.
4. Colours match the report figures (rapor -> report/make_figures.py).
5. Backend: plot_run.py pins matplotlib to Agg (no window). This module
   must be imported first, which scenario.py does.

Usage:  python scenario.py --obstacles --adaptive M2 --view --live
"""

import time
import numpy as np
import matplotlib

import matplotlib.pyplot as plt   # noqa: E402


def pick_backend():
    """
    Switch to a backend that can open a window. Returns success.

    matplotlib.use() only records the name; with tkinter missing it returns
    fine and the crash comes at the first plt.subplots(). switch_backend()
    actually loads the module, so it is used here.
    """
    if not matplotlib.get_backend().lower().startswith("agg"):
        return True                      # already interactive
    for bk in ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX"):
        try:
            plt.switch_backend(bk)
            return True
        except Exception:
            continue
    return False


COLORS = {"none": "#8a8a85", "M0I": "#e87ba4", "M0": "#2a78d6",
          "M0F": "#eda100", "M1": "#eb6834", "M2": "#1baf7a"}
LABELS = {"none": "no adaptation (baseline MPC)",
          "M0I": "M0I  integral action (6 par.)",
          "M0": "M0  classical regressor (12 par.)",
          "M0F": "M0F  phase-aware regressor (24 par.)",
          "M1": "M1  RBF network (390 par.)",
          "M2": "M2  Lb-DNN (208 par.)"}
INK, MUTED, GRID = "#2b2b28", "#6e6e68", "#e3e3df"
ACCENT = "#e34948"


class LivePanel:
    """Plot window updated while the simulation runs."""

    def __init__(self, kind, T, x_max=10.5, course=None, zones=None,
                 hz=15.0, tau_limit=33.5, z_nominal=0.2504):
        self.kind = kind
        self.col = COLORS.get(kind, "#2a78d6")
        self.hz = float(hz)
        self._last = 0.0
        self.ok = pick_backend()
        if not self.ok:
            print("  (live plot skipped: no matplotlib backend can open a "
                  "window -- install pyqt5 or use a Python with tkinter. "
                  "The run continues.)")
            return

        self.buf = {k: [] for k in
                    ("t", "x", "pitch", "roll", "z", "terrain_h",
                     "theta", "tau")}

        try:
            self._build(kind, T, x_max, course, zones, tau_limit)
        except Exception as exc:
            # backend loaded but no window (remote session, no display...)
            self.ok = False
            print(f"  (live plot could not open: {exc}. The run continues.)")

    def _build(self, kind, T, x_max, course, zones, tau_limit):
        plt.ion()
        self.fig, axes = plt.subplots(5, 1, figsize=(5.4, 8.6), sharex=True)
        self.fig.patch.set_facecolor("#fcfcfb")
        self.fig.canvas.manager.set_window_title(
            f"TerrAdapt  --  {LABELS.get(kind, kind)}")
        self.ax = axes

        specs = [
            ("progress x (m)", (0.0, x_max)),
            ("pitch / roll (rad)", (-0.35, 0.35)),
            ("body z and ground (m)", (-0.02, 0.52)),
            (r"$\|\hat{\theta}\|$  estimated disturbance", (0.0, 60.0)),
            ("max torque (Nm)", (0.0, 36.0)),
        ]
        for a, (lab, ylim) in zip(axes, specs):
            a.set_facecolor("#fcfcfb")
            a.set_ylabel(lab, fontsize=9, color=INK)
            a.set_xlim(0.0, T)
            a.set_ylim(*ylim)          # fixed (design choice 1)
            a.grid(True, color=GRID, linewidth=0.6)
            a.set_axisbelow(True)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                a.spines[sp].set_color(GRID)
            a.tick_params(colors=MUTED, labelsize=8, length=3)
        axes[-1].set_xlabel("time (s)", fontsize=9, color=INK)

        # zone lines in the x panel, so the robot's position can be read
        if zones:
            for x0, x1, lab in zones:
                axes[0].axhline(x0, color=MUTED, lw=0.6, ls=":", alpha=0.7)
                axes[0].annotate(lab, xy=(0.012, x0),
                                 xycoords=("axes fraction", "data"),
                                 xytext=(0, 2), textcoords="offset points",
                                 fontsize=6.5, color=MUTED, va="bottom")

        # events (payload, ...)
        for t_e, _k, _v, lab in (course or []):
            for a in axes:
                a.axvline(t_e, color=MUTED, ls=":", lw=1.0, alpha=0.8)
            axes[0].annotate(lab, xy=(t_e, 1.0),
                             xycoords=("data", "axes fraction"),
                             xytext=(2, 4), textcoords="offset points",
                             fontsize=7, color=MUTED, va="bottom")

        axes[4].axhline(tau_limit, color=ACCENT, lw=0.9, alpha=0.7)
        axes[4].annotate(f"torque limit {tau_limit:.1f}", xy=(0.02, tau_limit),
                         xycoords=("axes fraction", "data"),
                         xytext=(0, 2), textcoords="offset points",
                         fontsize=7, color=ACCENT, va="bottom")

        c = self.col
        (self.l_x,) = axes[0].plot([], [], "-", color=c, lw=2.0)
        (self.l_pitch,) = axes[1].plot([], [], "-", color=c, lw=1.8,
                                       label="pitch")
        (self.l_roll,) = axes[1].plot([], [], "-", color=ACCENT, lw=1.4,
                                      alpha=0.85, label="roll")
        axes[1].legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
        (self.l_z,) = axes[2].plot([], [], "-", color=c, lw=1.8,
                                   label="body z")
        (self.l_g,) = axes[2].plot([], [], "-", color=MUTED, lw=1.4,
                                   label="estimated ground")
        axes[2].legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
        (self.l_th,) = axes[3].plot([], [], "-", color=c, lw=1.8)
        (self.l_tau,) = axes[4].plot([], [], "-", color=c, lw=1.2)

        self.head = self.fig.text(
            0.012, 0.985, "", fontsize=10.5, color=INK, va="top",
            fontweight="bold")
        self.sub = self.fig.text(0.012, 0.962, "", fontsize=8.5,
                                 color=MUTED, va="top")
        self.fig.tight_layout(rect=(0, 0, 1, 0.945))
        self.fig.show()
        self.fig.canvas.flush_events()
        self._zones = zones or []

    # -----------------------------------------------------------------
    def _zone_of(self, x):
        for x0, x1, lab in self._zones:
            if x0 <= x < x1:
                return lab
        return "-"

    def update(self, t, x, pitch, roll, z, terrain_h, theta, tau,
               force=False):
        """Append data; redraw respecting the rate limit."""
        if not self.ok:
            return
        b = self.buf
        b["t"].append(t); b["x"].append(x)
        b["pitch"].append(pitch); b["roll"].append(roll)
        b["z"].append(z); b["terrain_h"].append(terrain_h)
        b["theta"].append(theta); b["tau"].append(tau)

        now = time.perf_counter()
        if not force and (now - self._last) < 1.0 / self.hz:
            return
        self._last = now

        tt = b["t"]
        self.l_x.set_data(tt, b["x"])
        self.l_pitch.set_data(tt, b["pitch"])
        self.l_roll.set_data(tt, b["roll"])
        self.l_z.set_data(tt, b["z"])
        self.l_g.set_data(tt, b["terrain_h"])
        self.l_th.set_data(tt, b["theta"])
        self.l_tau.set_data(tt, b["tau"])

        self.head.set_text(f"{LABELS.get(self.kind, self.kind)}")
        self.sub.set_text(
            f"t = {t:5.1f} s    x = {x:5.2f} m    zone: {self._zone_of(x)}"
            f"    torque {tau:4.1f} Nm")
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            self.ok = False        # window may have been closed

    def finish(self, result):
        if not self.ok:
            return
        self.sub.set_text(result)
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def close(self, wait=0.0):
        if not self.ok:
            return
        if wait > 0:
            time.sleep(wait)
        try:
            plt.close(self.fig)
        except Exception:
            pass
