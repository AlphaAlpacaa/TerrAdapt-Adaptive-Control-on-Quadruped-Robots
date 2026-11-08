"""
Three-seed repetition of the estimation-vs-control comparison
("better estimation is not better control").

Two claims had come from one seed only:
  (i)  "M1 is the best estimator on all four segments"
  (ii) the pitch ranking (M0 best, M2 worst)

Setup:
  terrain   : scene_terrain.xml, T = 35 s
  obstacles : scene_obstacles.xml on a FLAT base, FIXED obstacles,
              x = 1.5..3.9 m, T = 20 s (make_obstacle_model.py --base flat --fixed)
  gamma = 300, payload (0.12, 0.05, 0.06) 3 kg at 6 s, vx = 0.3
  metric: segment_stats (scenario.py), steady half: pitch, prediction error

Decision rule (written before running)
--------------------------------------
  (i) holds <=> in each segment the lowest steady prediction error among
      M0/M1/M2 is M1's, in >= 2 of 3 seeds; otherwise report in how many
      segments it holds.
  (ii) does the seed-mean ranking M0 <= M1 <= M2 hold?
  Regression: obstacles / seed 0 / M0 loaded segment must give 0.2701.

Usage:
    python sweep_1b.py --course terrain
    python make_obstacle_model.py --base flat --fixed
    python sweep_1b.py --course obstacles
    python make_obstacle_model.py --fixed          # rebuild the combined course
"""

import argparse
import contextlib
import csv
import io
import json
import sys
from pathlib import Path
import numpy as np

from scenario import run_course, segment_stats, Tee, RESULTS_ROOT
from course_def import terrain_course
from make_course_model import find_model_dir

METHODS = ["none", "M0", "M0F", "M1", "M2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", required=True, choices=["terrain", "obstacles"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gamma", type=float, default=300.0)
    ap.add_argument("--methods", nargs="+", default=METHODS,
                    help="default: none M0 M0F M1 M2; with a subset the "
                         "decision section is skipped")
    a = ap.parse_args()

    pos = np.array([0.12, 0.05, 0.06])
    course = terrain_course()
    if a.course == "terrain":
        T, kw = 35.0, dict(terrain=True, obstacles=False)
    else:
        z = json.loads((find_model_dir() / "obstacle_zone.json").read_text())
        if z["base"] != "flat" or z["loose"]:
            sys.exit(f"!! wrong obstacle scene: {z}\n"
                     "   first: python make_obstacle_model.py --base flat --fixed")
        T, kw = float(z["T"]), dict(terrain=False, obstacles=True)

    kok = Path(__file__).resolve().parent
    odir = kok / RESULTS_ROOT / (f"1b-repeat_{a.course}_g{a.gamma:g}_vx0.3"
                                 f"_s{'-'.join(map(str, a.seeds))}"
                                 + ("" if a.methods == METHODS
                                    else "_" + "-".join(a.methods)))
    if odir.exists() and any(odir.iterdir()):
        sys.exit(f"!! {odir} is not empty -- not overwriting.")
    odir.mkdir(parents=True, exist_ok=True)
    tee = Tee(odir / "summary.txt")
    sys.stdout = tee
    print(f"1b REPEAT -- course {a.course}, T={T:.0f} s, gamma={a.gamma:g}, "
          f"seeds {a.seeds}")

    rows = []
    for sd in a.seeds:
        print(f"\n--- seed {sd} " + "-" * 60)
        print(f"  {'method':6s} {'outcome':10s} {'x':>6s}  "
              f"{'pitch base':>11s} {'pitch load':>10s}  "
              f"{'pred. base':>11s} {'pred. load':>10s}")
        for k in a.methods:
            with contextlib.redirect_stdout(io.StringIO()):
                L, _ = run_course(k, None if k == "none" else a.gamma, 0.3, T,
                                  course, sd, False, payload_pos=pos, **kw)
            seg = segment_stats(L, course, T)
            st = ("FELL" if L["fell"] is not None else
                  "STALLED" if L["stuck"] is not None else
                  "COMPLETED" if L["done"] is not None else "TIME_UP")
            s0, s1 = seg[0], seg[1] if len(seg) > 1 else seg[0]
            r = dict(seed=sd, method=k, outcome=st, x=L["x_final"],
                     pitch0=s0["pitch"], pitch1=s1["pitch"],
                     pred0=s0["pred"], pred1=s1["pred"],
                     roll1=s1["roll_std"])
            rows.append(r)
            print(f"  {k:6s} {st:10s} {r['x']:6.2f}  {r['pitch0']:+11.4f} "
                  f"{r['pitch1']:+10.4f}  {r['pred0']:11.4f} {r['pred1']:10.4f}",
                  flush=True)

    # --- decision ---------------------------------------------------------
    print("\n" + "=" * 74)
    print("DECISION (rule written before the run)")
    print("=" * 74)
    g = lambda sd, k: next(r for r in rows if r["seed"] == sd and r["method"] == k)
    full = all(k in a.methods for k in ["M0", "M0F", "M1", "M2"])
    if not full:
        print("  (decision skipped: M0/M0F/M1/M2 not all in this run)")
    for seg, name in ([("pred0", "base"), ("pred1", "load")] if full else []):
        winner = [min(["M0", "M1", "M2"], key=lambda k: g(sd, k)[seg])
                  for sd in a.seeds]
        n1 = winner.count("M1")
        print(f"  {a.course}, {name:5s}: best estimator (M0/M1/M2) per seed "
              f"{winner} -> M1 {n1}/{len(a.seeds)} "
              f"{'HOLDS' if n1 >= 2 else 'DOES NOT HOLD'}")
        m0f = sum(g(sd, "M0F")[seg] < min(g(sd, k)[seg] for k in ["M0", "M1", "M2"])
                  for sd in a.seeds)
        print(f"         (M0F better than all: {m0f}/{len(a.seeds)})")
    print("\n  seed means (loaded segment, steady):")
    print("  method   done    |pitch|    pred.      roll std")
    for k in a.methods:
        rs = [g(sd, k) for sd in a.seeds]
        print(f"  {k:6s}   {sum(r['outcome'] == 'COMPLETED' for r in rs)}/{len(rs)}"
              f"    {np.mean([abs(r['pitch1']) for r in rs]):.4f}    "
              f"{np.mean([r['pred1'] for r in rs]):.4f}    "
              f"{np.mean([r['roll1'] for r in rs]):.4f}")

    with open(odir / "grid.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nALL OUTPUTS: {odir}")
    sys.stdout = tee.std
    tee.close()


if __name__ == "__main__":
    main()
