"""
Fig. 1 -- system block diagram (a) and the five theta-hat maps (b).
Same style/palette as make_figures.py. No data is read; every label was
checked against the code (8 Nov):
  controller.py  compute(): terrain height = LPF mean of stance feet,
                 slope = LSQ plane through last 8 touchdowns (td_buf_len=8);
                 stance tau = -J^T R^T F, swing tau = J^T R^T f_PD + G + ff + fric
                 adaptive layer every control step (2 ms), MPC every 20 ms
  mpc.py         X = A_qp x0 + B_qp U + C_qp theta_hat (build_disturbance)
  adaptive.py    features(): 11-D [v, w, roll, pitch, z-h-z0, sin, cos];
                 ||theta_hat|| <= 80; u to the predictor = applied F

    python report/fig_architecture.py  ->  figures/fig_arch.pdf, fig_maps.pdf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from make_figures import OUT, INK, INK2, COL

RED = "#c0392b"
FILL = "#f4f3ef"
AFILL = "#fff7e6"


def box(ax, cx, cy, w, h, txt, fs=6.6, dashed=False, fc=FILL, ec=INK2, lw=0.8):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0,rounding_size=1.2",
                       fc=fc, ec=ec, lw=lw, ls="--" if dashed else "-")
    ax.add_patch(p)
    ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs, color=INK,
            linespacing=1.25)
    return dict(l=cx - w / 2, r=cx + w / 2, b=cy - h / 2, t=cy + h / 2, x=cx, y=cy)


def arrow(ax, pts, col=INK2, lw=0.8, txt=None, tpos=None, tfs=6.4, ha="center",
          va="bottom", tcol=None, z=2):
    # Shaft: the WHOLE polyline as a plain line (poppler dropped the shafts
    # of FancyArrowPatch arrows in the PDF); the patch only draws the head.
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=col, lw=lw, solid_capstyle="butt",
            solid_joinstyle="miter", zorder=z)
    (x1, y1), (x0, y0) = pts[-1], pts[-2]
    d = max(np.hypot(x1 - x0, y1 - y0), 1e-9)
    hl = min(1.5, d)
    tail = (x1 - (x1 - x0) / d * hl, y1 - (y1 - y0) / d * hl)
    ax.annotate("", xy=pts[-1], xytext=tail,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                mutation_scale=7 if lw < 1.5 else 10,
                                shrinkA=0, shrinkB=0), zorder=z)
    if txt:
        ax.text(*tpos, txt, fontsize=tfs, ha=ha, va=va, color=tcol or INK,
                zorder=5)


def fig_arch():
    fig, ax = plt.subplots(figsize=(7.0, 3.35))
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 67)
    ax.axis("off")

    # ---- top: planners --------------------------------------------------
    ter = box(ax, 22, 56, 35, 9,
              "Terrain estimator\nheight $\\hat h$: stance feet (low-pass)\n"
              "slope: LSQ plane, last 8 touchdowns")
    gait = box(ax, 64, 56, 34, 9,
               "Gait planner (trot, period 400 ms)\n"
               "phase $\\varphi$, contact schedule,\nfoothold targets")
    # ---- middle: main loop ------------------------------------------------
    ref = box(ax, 22, 38, 20, 9, "Reference\ngenerator")
    mpc = box(ax, 64, 38, 30, 13,
              "Convex SRB-MPC\n13 states, horizon 10$\\times$30 ms\n"
              "solved every 20 ms\nprediction model $\\ni C_{qp}\\hat{\\theta}$")
    low = box(ax, 99.5, 38, 27.5, 13,
              "Low level, every 2 ms\nstance: $-J^\\top R^\\top F$\n"
              "swing: $J^\\top R^\\top f_{PD}+G(q)$\n+ inertia + damping + friction",
              fs=6.2)
    rob = box(ax, 128, 38, 15, 9, "Robot\n(MuJoCo)")

    arrow(ax, [(1.5, 38), (ref["l"], 38)], txt="$v_{cmd}$", tpos=(4, 39.2),
          ha="left")
    arrow(ax, [(ref["r"], 38), (mpc["l"], 38)], txt="$x_d$",
          tpos=(40.5, 39.2))
    arrow(ax, [(mpc["r"], 38), (low["l"], 38)], txt="$F$", tpos=(83.0, 39.2))
    arrow(ax, [(low["r"], 38), (rob["l"], 38)], txt="$\\tau$",
          tpos=(116.3, 39.2))
    # foot positions -> terrain estimator (top bus)
    arrow(ax, [(rob["x"], rob["t"]), (rob["x"], 64.5), (ter["x"], 64.5),
               (ter["x"], ter["t"])], txt="foot positions",
          tpos=(96, 65.0))
    # terrain -> reference (z ref)
    arrow(ax, [(ter["x"], ter["b"]), (ref["x"], ref["t"])], txt="$\\hat h$",
          tpos=(23.5, 46.5), ha="left", va="center")
    # gait -> MPC, gait -> low level
    arrow(ax, [(gait["x"], gait["b"]), (gait["x"], mpc["t"])],
          txt="schedule", tpos=(65.5, 48.0), ha="left", va="center")
    arrow(ax, [(gait["r"], 56), (low["x"], 56), (low["x"], low["t"])],
          txt="$\\varphi$, footholds", tpos=(101, 53.5), ha="left",
          va="center")

    # ---- feedback bus ----------------------------------------------------
    YB = 25.5
    ax.plot([rob["x"], rob["x"], 9], [rob["b"], YB, YB], color=INK2, lw=0.8,
            zorder=2)
    ax.text(129.2, 28.6, "$x$ (13)", fontsize=6.4, ha="left", color=INK)
    arrow(ax, [(9, YB), (9, ref["y"] - 2), (ref["l"], ref["y"] - 2)])

    # ---- adaptive part -----------------------------------------------------
    feat = box(ax, 22, 9, 24, 11,
               "Feature vector $\\xi$ (11)\n$v,\\ \\omega$, roll, pitch,\n"
               "$z-\\hat h$, $\\sin\\varphi$, $\\cos\\varphi$")
    emag = box(ax, 39, 19.5, 13, 5, "$\\|e\\|=\\|x-x_d\\|$", fs=6.2)
    appr = box(ax, 64, 9, 30, 12,
               "Adaptive layer (swappable)\n"
               "M0I / M0 / M0F / M1 / M2\n"
               "$\\hat\\theta=f(\\xi,\\|e\\|;\\hat w)$, $\\dot{\\hat w}$ from $s$",
               dashed=True, fc=AFILL, ec="#b07800", lw=1.1)
    proj = box(ax, 64, 19.8, 13, 4.6, "$\\|\\hat\\theta\\|\\leq 80$", fs=6.4)
    pred = box(ax, 112, 9, 34, 12,
               "L1 state predictor (every 2 ms)\n"
               "$\\dot{\\hat x}=A\\hat x+Bu+E\\hat\\theta-\\Lambda(\\hat x-x)$\n"
               "$\\tilde e=\\hat x-x,\\quad s=-E^\\top P\\tilde e$")

    arrow(ax, [(feat["x"], YB), (feat["x"], feat["t"])], txt="$x$",
          tpos=(23.2, 20.0), ha="left", va="center")
    arrow(ax, [(1.5, 9), (feat["l"], 9)], txt="$\\hat h,\\ \\varphi$",
          tpos=(2.2, 10.2), ha="left")
    arrow(ax, [(feat["r"], 9), (appr["l"], 9)], txt="$\\xi$",
          tpos=(42, 10.2))
    arrow(ax, [(emag["x"], YB), (emag["x"], emag["t"])])
    ax.text(37.6, 23.6, "$x,\\ x_d$", fontsize=6.0, ha="right", va="center",
            color=INK)
    arrow(ax, [(emag["r"], emag["y"]), (53, emag["y"]), (53, appr["t"])])
    ax.text(46.0, 20.3, "M0, M0F", fontsize=5.6, color=INK2, ha="left")
    arrow(ax, [(pred["l"], 9), (appr["r"], 9)], txt="$s$",
          tpos=(87, 10.2))
    arrow(ax, [(appr["x"], appr["t"]), (proj["x"], proj["b"])])
    # x and applied F -> predictor
    arrow(ax, [(124, YB), (124, pred["t"])], txt="$x$", tpos=(125.2, 19.5),
          ha="left", va="center")
    arrow(ax, [(103, low["b"]), (103, pred["t"])])
    ax.text(104.2, 19.5, "$u=F$ (applied)", fontsize=6.0, ha="left",
            va="center", color=INK)
    # theta -> predictor
    arrow(ax, [(proj["r"], proj["y"]), (97, proj["y"]), (97, pred["t"])],
          txt="$\\hat\\theta$", tpos=(84, 20.6))
    # KEY ARROW: theta -> MPC prediction model (not an added control input)
    arrow(ax, [(proj["x"], proj["t"]), (proj["x"], mpc["b"])], col=RED,
          lw=2.0, z=4)
    ax.text(66.3, 28.6,
            "$\\hat\\theta$ enters the MPC prediction model:\n"
            "$X=A_{qp}x_0+B_{qp}U+C_{qp}\\hat\\theta$",
            fontsize=6.0, color=RED, ha="left", va="center", zorder=5)
    fig.tight_layout(pad=0.1)
    fig.savefig(OUT / "fig_arch.pdf")
    plt.close(fig)
    print("fig_arch.pdf")


def fig_maps():
    fig, ax = plt.subplots(figsize=(7.0, 1.25))
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 25)
    ax.axis("off")
    inp = box(ax, 10, 12.5, 18, 17,
              "shared inputs\n$s=-E^\\top P\\tilde e$\n$\\xi$ (11)\n$\\|e\\|$")
    rows = [("M0I", "6", "$\\hat\\theta=\\hat\\beta$", "no regressor"),
            ("M0", "12", "$\\hat\\theta=\\hat\\alpha\\|e\\|+\\hat\\beta$",
             "[1],\nEq. (27)/(30)"),
            ("M0F", "24", "$\\hat\\alpha\\|e\\|+\\hat\\beta$\n"
             "$+\\hat c\\sin\\varphi+\\hat d\\cos\\varphi$", "phase-aware,\nclassical"),
            ("M1", "390", "$\\hat\\theta=\\hat W^\\top\\phi(\\xi)$",
             "64 Gaussians\n+ bias, local"),
            ("M2", "208", "$V_k^\\top\\phi_k(\\cdots V_1^\\top\\phi_1(V_0^\\top\\xi))$",
             "Lb-DNN [2],\nglobal")]
    x0, w, gap = 22.5, 18.4, 1.3
    for i, (m, n, f, note) in enumerate(rows):
        cx = x0 + w / 2 + i * (w + gap)
        b = box(ax, cx, 12.5, w, 17, "", fc=AFILL, ec=COL[m], lw=1.1,
                dashed=True)
        ax.text(cx, 18.6, f"{m} ({n})", ha="center", fontsize=6.8,
                fontweight="bold", color=INK)
        ax.text(cx, 12.6, f, ha="center", va="center", fontsize=5.6,
                color=INK, linespacing=1.3)
        ax.text(cx, 6.3, note, ha="center", va="center", fontsize=5.5,
                color=INK2)
    xr = x0 + 5 * (w + gap) - gap
    out = box(ax, 131, 12.5, 16, 17,
              "shared output\n$\\|\\hat\\theta\\|\\leq80$\n$\\to$ MPC model\n"
              "$\\to$ predictor")
    arrow(ax, [(inp["r"], 12.5), (x0, 12.5)])
    arrow(ax, [(xr, 12.5), (out["l"], 12.5)])
    fig.tight_layout(pad=0.1)
    fig.savefig(OUT / "fig_maps.pdf")
    plt.close(fig)
    print("fig_maps.pdf")


if __name__ == "__main__":
    fig_arch()
    fig_maps()
