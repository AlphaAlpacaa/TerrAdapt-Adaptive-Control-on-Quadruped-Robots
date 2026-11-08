"""
Robustness sweep -- who completes the course reproducibly?

For every method, completion over a (gain x seed) grid. The metric is
binary -- in how many cells is the course completed? -- because the claim
being tested ("the result is not reproducible") is binary; a mean roll
std would hide it.

Reading:
  every method 9/9            -> no fragility
  one method fails at one gain -> a tuning issue, not the architecture
  one method fails at all gains -> the architecture

Usage:
    python sweep_robust.py --baseline              # 3 gains x 3 seeds + baseline
    python sweep_robust.py --gammas 300            # one gain, quick
    python sweep_robust.py --methods M0 M2
    python sweep_robust.py --filter-compare        # C(s) on vs off, every cell
    python sweep_robust.py --flat --vx 0.9 --T 20 --baseline
    python sweep_robust.py --terrain --vx 0.9 --T 20 --baseline

Without --flat / --terrain the scene is scene_obstacles.xml, i.e. whatever
make_obstacle_model.py last built (combined course: make_obstacle_model.py
--fixed).
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

from scenario import run_course, Tee, RESULTS_ROOT
from make_course_model import find_model_dir
from course_def import terrain_course


def single_run(kind, gamma, seed, pos, T, quiet=True, use_filter=True,
               emag="tracking", vx=0.3, flat=False, terrain_only=False):
    """
    One run; returns the outcome only.

    Success criterion depends on the course:
      - terrain / obstacle course -> crossed the finish line (L["done"])
        (terrain_only: scene_terrain.xml only, finish at x = 8.0 m)
      - flat ground               -> no finish line; neither falling nor
                                     stalling for T seconds
    """
    import io
    import contextlib
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
    with ctx:
        L, _ = run_course(kind, gamma, vx, T, terrain_course(), seed,
                          False, payload_pos=pos,
                          terrain=not flat, obstacles=not (flat or terrain_only),
                          use_filter=use_filter, emag_source=emag)
    t = L["t"]
    msk = t >= 7.0                      # 1 s after the payload
    fell = L["fell"] is not None
    stalled = L["stuck"] is not None
    done = (not fell and not stalled) if flat else (L["done"] is not None)
    return dict(
        done=done, fell=fell, stalled=stalled,
        x=float(L["x_final"]),
        pitch=float(np.mean(np.abs(L["pitch"][msk]))) if msk.sum() > 2 else np.nan,
        roll=float(np.std(L["roll"][msk])) if msk.sum() > 2 else np.nan,
        pred=float(np.mean(L["pred_err"][msk])) if msk.sum() > 2 else np.nan,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["M0", "M1", "M2"])
    ap.add_argument("--gammas", type=float, nargs="+",
                    default=[100.0, 300.0, 1000.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--payload-pos", dest="ppos", type=float, nargs=3,
                    default=[0.12, 0.05, 0.06])
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--baseline", action="store_true",
                    help="also run the baseline (gain independent, once)")
    ap.add_argument("--filter", action="store_true",
                    help="turn C(s) on (default off -- measured worse)")
    ap.add_argument("--filter-compare", dest="compare", action="store_true",
                    help="run EVERY cell with and without C(s)")
    ap.add_argument("--emag", default="tracking",
                    choices=["tracking", "prediction"])
    ap.add_argument("--vx", type=float, default=0.3,
                    help="commanded speed; 0.3 is the tuning point")
    ap.add_argument("--flat", action="store_true",
                    help="flat ground (no terrain, no obstacles); criterion "
                         "is staying up and walking for T seconds")
    ap.add_argument("--terrain", action="store_true",
                    help="terrain course only (scene_terrain.xml, no obstacles)")
    a = ap.parse_args()
    if a.flat and a.terrain:
        ap.error("--flat and --terrain cannot be combined")
    arms = [True, False] if a.compare else [a.filter]

    kok = Path(__file__).resolve().parent
    label = (f"robustness_y{a.ppos[0]:+.2f}{a.ppos[1]:+.2f}"
             f"_g{'-'.join(f'{g:g}' for g in a.gammas)}"
             f"_s{'-'.join(str(x) for x in a.seeds)}"
             + ("_filter-compare" if a.compare else
                ("_filtered" if a.filter else "_unfiltered"))
             + ("" if a.emag == "tracking" else "_emag-prediction")
             + (f"_vx{a.vx:g}" if a.vx != 0.3 else "")
             + ("_flat" if a.flat else "")
             + ("_terrain" if a.terrain else "")
             + ("" if a.methods == ["M0", "M1", "M2"]
                else "_" + "-".join(a.methods)))
    odir = kok / RESULTS_ROOT / label
    odir.mkdir(parents=True, exist_ok=True)
    tee = Tee(odir / "summary.txt")
    sys.stdout = tee
    print(f"OUTPUT FOLDER: {odir}")

    T = a.T
    if T is None:
        f = find_model_dir() / "obstacle_zone.json"
        T = float(json.loads(f.read_text(encoding="utf-8"))["T"]) if f.exists() else 60.0
    pos = np.array(a.ppos, float)

    print("=" * 74)
    print("ROBUSTNESS SWEEP -- who completes the course reproducibly?")
    print("=" * 74)
    print(f"payload    : {pos.tolist()} m  "
          f"(pitch {3*9.81*abs(pos[0]):.2f} N.m, roll {3*9.81*abs(pos[1]):.2f} N.m)")
    print(f"gains      : {a.gammas}")
    print(f"seeds      : {a.seeds}")
    print(f"duration   : {T:.0f} s")
    print(f"speed      : {a.vx:g} m/s" + ("  (tuning point)" if a.vx == 0.3 else
                                          "  <-- outside the tuning point"))
    print("ground     : " + ("FLAT (no terrain/obstacles)" if a.flat
                              else ("TERRAIN ONLY (scene_terrain.xml)"
                                    if a.terrain else
                                    "scene_obstacles.xml (combined course)")))
    if a.flat:
        print(f"criterion  : neither fall nor stall for {T:.0f} s")
    print("C(s) filter: "
          + ("ON and OFF (comparison)" if a.compare
             else ("ON (paper Eqs. 33/45)" if a.filter
                   else "OFF  (default)")))
    print(f"regressor  : ||e{'~' if a.emag == 'prediction' else ''}||  "
          f"({a.emag})")
    print(f"total      : "
          f"{len(arms)*len(a.methods)*len(a.gammas)*len(a.seeds)} runs"
          + (" + baseline" if a.baseline else ""))

    if a.baseline:
        r = single_run("none", None, 0, pos, T, vx=a.vx, flat=a.flat,
                       terrain_only=a.terrain)
        st = "COMPLETED" if r["done"] else ("STALLED" if r["stalled"] else "FELL")
        print(f"\nbaseline (no adaptation): {st}, {r['x']:.2f} m "
              "-- independent of gain and seed")

    res = {}
    for flt in arms:
        if a.compare:
            print("\n" + "#" * 74)
            print("# C(s) LOW-PASS FILTER: " + ("ON" if flt else "OFF"))
            print("#" * 74)
        for kind in a.methods:
            print(f"\n--- {kind} " + "-" * 60)
            print("  seed \\ gain    "
                  + "".join(f"{g:>12.0f}" for g in a.gammas))
            for sd in a.seeds:
                row = f"       {sd}        "
                for g in a.gammas:
                    r = single_run(kind, g, sd, pos, T, use_filter=flt,
                                   emag=a.emag, vx=a.vx, flat=a.flat,
                                   terrain_only=a.terrain)
                    res[(flt, kind, g, sd)] = r
                    if r["done"]:
                        cell = "DONE"
                    elif r["fell"]:
                        cell = f"FELL {r['x']:.1f}"
                    else:
                        cell = f"STALL {r['x']:.1f}"
                    row += f"{cell:>12s}"
                print(row)

    # --- summary ----------------------------------------------------------
    n = len(a.gammas) * len(a.seeds)
    print("\n" + "=" * 74)
    print("SUMMARY -- completion rate")
    print("=" * 74)
    print("  method   completed   mean over completed runs")
    print("                        |pitch|    roll std   pred. error")
    print("  " + "-" * 64)
    for flt in arms:
        tag = ("  [C(s) ON]" if flt else "  [C(s) OFF]") if a.compare else ""
        for kind in a.methods:
            ok = [res[(flt, kind, g, s)] for g in a.gammas for s in a.seeds
                  if res[(flt, kind, g, s)]["done"]]
            p = np.mean([r["pitch"] for r in ok]) if ok else np.nan
            rr = np.mean([r["roll"] for r in ok]) if ok else np.nan
            pe = np.mean([r["pred"] for r in ok]) if ok else np.nan
            print(f"  {kind:5s}      {len(ok):2d}/{n}       "
                  f"{p:7.4f}     {rr:7.4f}   {pe:7.4f}{tag}")

    # per gain: are the failures concentrated at one gain?
    print("\n  COMPLETION PER GAIN (tuning or architecture?)")
    for flt in arms:
        if a.compare:
            print(f"\n  C(s) {'ON' if flt else 'OFF'}")
        print("  method " + "".join(f"{g:>10.0f}" for g in a.gammas))
        print("  " + "-" * (7 + 10 * len(a.gammas)))
        for kind in a.methods:
            row = f"  {kind:5s} "
            for g in a.gammas:
                c = sum(1 for s in a.seeds
                        if res[(flt, kind, g, s)]["done"])
                row += f"{c:>7d}/{len(a.seeds)}"
            print(row)
    print("\n  Failures at ONE gain point to tuning; at all gains to the architecture.")

    if a.compare:
        print("\n" + "=" * 74)
        print("C(s) DECISION")
        print("=" * 74)
        print("  gain     C(s) OFF      C(s) ON      (all methods x seeds)")
        print("  " + "-" * 52)
        nn = len(a.methods) * len(a.seeds)
        for g in a.gammas:
            off = sum(1 for k in a.methods for s in a.seeds
                      if res[(False, k, g, s)]["done"])
            on = sum(1 for k in a.methods for s in a.seeds
                     if res[(True, k, g, s)]["done"])
            print(f"  {g:7.0f}   {off:7d}/{nn}     {on:7d}/{nn}")

    # raw grid: single source for later tables
    import csv
    with open(odir / "grid.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filter", "method", "gamma", "seed", "done", "stalled",
                    "fell", "x_final", "pitch_abs", "roll_std", "pred_err"])
        for (flt, kind, g, sd), r in res.items():
            w.writerow([int(flt), kind, g, sd, int(r["done"]),
                        int(r["stalled"]), int(r["fell"]), f"{r['x']:.4f}",
                        f"{r['pitch']:.6f}", f"{r['roll']:.6f}",
                        f"{r['pred']:.6f}"])
    print(f"\n{'='*74}")
    print(f"ALL OUTPUTS: {odir}")
    for f in sorted(odir.iterdir()):
        print(f"  {f.name}")
    sys.stdout = tee.std
    tee.close()


if __name__ == "__main__":
    main()
