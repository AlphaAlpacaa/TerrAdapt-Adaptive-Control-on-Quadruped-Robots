"""
Swinging-payload experiment -- a disturbance M0 cannot represent.

The speed axis did not separate the methods. The remaining candidate is a
zero-mean disturbance locked to the gait PHASE. M0 = alpha*||e|| + beta:
beta solves the constant part, but the oscillating part has no phase in
M0's regressor (||e|| is unsigned). M1/M2 see roll, pitch, omega and
sin/cos(phase).

Calibration (baseline only, before looking at any method)
---------------------------------------------------------
flat, vx = 0.3, payload (0.12, 0.05, 0.06), 3 kg at t = 6 s:

    axis   L      pendulum std  peak   dominant f
    x      4 cm   10.8 deg      21.1   2.50 Hz   <- chosen
    x      8 cm    7.3          15.2   2.50
    x     15 cm    3.0           7.0   2.50
    y      4 cm    2.0           5.9   5.00     (mean -7 deg: static offset)

x / 4 cm resonates (g/(2 pi 2.5)^2 = 3.98 cm), locks to the trot and has
near-zero mean.

Design -- two matched arms
--------------------------
  PENDULUM : pivot (0.12, 0.05, 0.06), mass 4 cm below, x axis
  RIGID    : the same 3 kg welded at the pendulum's rest position
             -> same static moment; the difference is the oscillation only.
Every (method, gamma, seed) cell is run in both arms.

Decision rule (written before running, not changed afterwards)
--------------------------------------------------------------
Metrics for t >= 8 s: prediction error ||e~|| (pred) and roll std.
Pendulum cost = PENDULUM - RIGID (same cell).

  A method Mk (M1 or M2) beats M0 if:
    (a) in the pendulum arm pred(Mk) < pred(M0) in >= 8 of 9 matched cells;
    (b) the advantage is larger in the pendulum arm than in the rigid arm
        ([pred(M0)-pred(Mk)]_pendulum > [ ... ]_rigid, >= 8/9);
    (c) the median roll-std difference in the pendulum arm is <= +10%.
  (b) is essential: on flat ground M2 may already be the better estimator,
  and that advantage does not belong to the oscillation.

M0F rule (written before M0F was run)
-------------------------------------
M0F = M0 + [sin phi, cos phi] (24 parameters, classical).
  (1) does M0F pass (a)(b)(c) against M0?
  (2) does it match M2? In the pendulum arm, median over the 9 cells:
        roll_std(M0F)/roll_std(M2) <= 1.10  AND
        pendulum cost pred% (M0F) <= pendulum cost pred% (M2) + 5 points
  (1) and (2) -> phase knowledge suffices; no universal approximation needed.

Usage:
    python make_pendulum_model.py        # once
    python sweep_pendulum.py             # 3 methods x 3 gains x 3 seeds x 2 arms
    python sweep_pendulum.py --methods M0F
"""

import argparse
import csv
import io
import contextlib
import sys
from pathlib import Path
import numpy as np

from scenario import run_course, Tee, RESULTS_ROOT
from course_def import terrain_course

T0 = 8.0     # payload at 6 s; the 2 s transient is excluded


def run(kind, gamma, seed, ppos, T, vx, pendulum):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        L, _ = run_course(kind, gamma, vx, T, terrain_course(), seed, False,
                          payload_pos=ppos, terrain=False, obstacles=False,
                          use_filter=False, emag_source="tracking",
                          pendulum=pendulum)
    t = L["t"]
    ms = t >= T0
    fell = L["fell"] is not None
    stalled = L["stuck"] is not None
    ok = ms.sum() > 10
    a = L["pendulum"][ms] if ok else np.array([np.nan])
    return dict(
        standing=not fell and not stalled, fell=fell, stalled=stalled,
        x=float(L["x_final"]),
        roll=float(np.std(L["roll"][ms])) if ok else np.nan,
        pitch=float(np.mean(np.abs(L["pitch"][ms]))) if ok else np.nan,
        pred=float(np.mean(L["pred_err"][ms])) if ok else np.nan,
        pendulum_std=float(np.degrees(np.std(a))),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["M0", "M1", "M2"])
    ap.add_argument("--gammas", type=float, nargs="+",
                    default=[100.0, 300.0, 1000.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--L", type=float, default=0.04)
    ap.add_argument("--axis", default="x", choices=["x", "y"])
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--T", type=float, default=20.0)
    a = ap.parse_args()

    piv = np.array([0.12, 0.05, 0.06])
    rigid_pos = piv - np.array([0.0, 0.0, a.L])
    pend = {"L": a.L, "axis": a.axis}
    arms = [("PENDULUM", piv, pend), ("RIGID", rigid_pos, None)]

    kok = Path(__file__).resolve().parent
    label = (f"pendulum_L{a.L:g}{a.axis}_vx{a.vx:g}"
             f"_g{'-'.join(f'{g:g}' for g in a.gammas)}"
             f"_s{'-'.join(str(s) for s in a.seeds)}"
             + ("" if a.methods == ["M0", "M1", "M2"]
                else "_" + "-".join(a.methods)))
    odir = kok / RESULTS_ROOT / label
    if odir.exists() and any(odir.iterdir()):
        sys.exit(f"!! {odir} is not empty -- not overwriting.")
    odir.mkdir(parents=True, exist_ok=True)
    tee = Tee(odir / "summary.txt")
    sys.stdout = tee

    print("=" * 74)
    print("SWINGING PAYLOAD -- does an approximator win where M0 is blind?")
    print("=" * 74)
    print(f"pendulum : pivot {piv.tolist()}, L={a.L} m, axis {a.axis}, 3 kg @ 6 s")
    print(f"rigid    : 3 kg welded @ {rigid_pos.tolist()} (pendulum rest position)")
    print(f"ground   : FLAT, vx={a.vx}, T={a.T} s, metrics for t >= {T0} s")
    print(f"total    : {2*len(a.methods)*len(a.gammas)*len(a.seeds)} runs + 2 baselines")

    res = {}
    for arm, pos, p in arms:
        r = run("none", None, 0, pos, a.T, a.vx, p)
        res[(arm, "none", 0, 0)] = r
        print(f"\n[{arm}] baseline: {'STANDING' if r['standing'] else 'BROKE'} "
              f"x={r['x']:.2f}  roll std {r['roll']:.4f}  "
              f"pendulum std {r['pendulum_std']:.1f} deg")

    for kind in a.methods:
        print(f"\n--- {kind} " + "-" * 64)
        print("  g      seed  |  PENDULUM: pred  roll   pend_std |"
              "  RIGID: pred   roll   | status")
        for g in a.gammas:
            for sd in a.seeds:
                rr = {}
                for arm, pos, p in arms:
                    rr[arm] = run(kind, g, sd, pos, a.T, a.vx, p)
                    res[(arm, kind, g, sd)] = rr[arm]
                S, R = rr["PENDULUM"], rr["RIGID"]
                st = ("ok" if S["standing"] and R["standing"] else
                      f"P:{'F' if S['fell'] else ('S' if S['stalled'] else 'ok')} "
                      f"R:{'F' if R['fell'] else ('S' if R['stalled'] else 'ok')}")
                print(f"  {g:6.0f} {sd:4d}  |  {S['pred']:7.4f} {S['roll']:7.4f}"
                      f"  {S['pendulum_std']:6.1f}  |  {R['pred']:7.4f} "
                      f"{R['roll']:7.4f}  | {st}", flush=True)

    # --- decision --------------------------------------------------------
    print("\n" + "=" * 74)
    print("DECISION (rule written before the run -- see the docstring)")
    print("=" * 74)
    cells = [(g, s) for g in a.gammas for s in a.seeds]
    n = len(cells)
    for kind in [k for k in a.methods if k != "M0"]:
        if "M0" not in a.methods:
            break
        ca = cb = 0
        droll = []
        for g, s in cells:
            p0s, pks = res[("PENDULUM", "M0", g, s)]["pred"], res[("PENDULUM", kind, g, s)]["pred"]
            p0r, pkr = res[("RIGID", "M0", g, s)]["pred"], res[("RIGID", kind, g, s)]["pred"]
            if pks < p0s:
                ca += 1
            if (p0s - pks) > (p0r - pkr):
                cb += 1
            r0, rk = res[("PENDULUM", "M0", g, s)]["roll"], res[("PENDULUM", kind, g, s)]["roll"]
            droll.append((rk - r0) / r0)
        med = float(np.nanmedian(droll))
        A, B, C = ca >= n - 1, cb >= n - 1, med <= 0.10
        print(f"\n  {kind} vs M0:")
        print(f"    (a) pendulum pred({kind}) < pred(M0)           : {ca}/{n}  {'YES' if A else 'no'}")
        print(f"    (b) advantage larger with pendulum than rigid : {cb}/{n}  {'YES' if B else 'no'}")
        print(f"    (c) median roll-std difference {med*100:+.1f}% (<= +10%) : {'YES' if C else 'no'}")
        print(f"    -> {'beats M0 on this disturbance' if A and B and C else 'rule NOT met'}")

    print("\n  Method means (t >= 8 s, runs that stayed up):")
    print("  method  arm        standing  pred     roll std   pendulum std")
    for kind in a.methods:
        for arm, _, _ in arms:
            rs = [res[(arm, kind, g, s)] for g, s in cells]
            ok = [r for r in rs if r["standing"]]
            f = lambda k: np.nanmean([r[k] for r in ok]) if ok else np.nan
            print(f"  {kind:5s}  {arm:9s}  {len(ok):2d}/{n}   {f('pred'):7.4f}  "
                  f"{f('roll'):8.4f}   {f('pendulum_std'):6.1f}")

    with open(odir / "grid.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "method", "gamma", "seed", "standing", "fell",
                    "stalled", "x_final", "pred_err", "roll_std", "pitch_abs",
                    "pendulum_std_deg"])
        for (arm, kind, g, s), r in res.items():
            w.writerow([arm, kind, g, s, int(r["standing"]), int(r["fell"]),
                        int(r["stalled"]), f"{r['x']:.4f}", f"{r['pred']:.6f}",
                        f"{r['roll']:.6f}", f"{r['pitch']:.6f}",
                        f"{r['pendulum_std']:.3f}"])
    print(f"\nALL OUTPUTS: {odir}")
    sys.stdout = tee.std
    tee.close()


if __name__ == "__main__":
    main()
