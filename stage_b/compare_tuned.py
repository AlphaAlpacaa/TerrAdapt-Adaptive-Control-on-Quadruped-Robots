"""
Fair comparison: every method at its OWN best gain (ladder rungs of
sweep_terrain.py).

sweep_terrain.py ran all methods at one gain (1000), which was bad for M0
and M1. On L4a, prediction error:
    gamma=100   M0 0.2737  M1 0.2603  M2 0.1649   <- best for all three
    gamma=1000  M0 1.0689  M1 0.7195  M2 0.1861
    gamma=3000  M0 0.9334  M1 0.8666  M2 1.6375
At gamma=1000 M0 looked harmful on soft ground (roll std 0.1030 vs
baseline 0.0108); at gamma=100 it is better than the baseline (0.0088).
That was a gain artefact, not the architecture.

Two metrics:
1. best error       : the method's potential
2. gain robustness  : worst/best ratio over the window and the number of
                      gains at which it stays up (the right gain is not
                      known on a real robot)

Usage:
    python compare_tuned.py                          # whole ladder
    python compare_tuned.py --rungs L0 L4a
    python compare_tuned.py --gammas 30 100 300 1000
"""

import argparse
import numpy as np

from sweep_terrain import RUNGS, run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+",
                    default=["L0", "L1", "L3a", "L4a", "L4c"])
    ap.add_argument("--kinds", nargs="+", default=["M0", "M1", "M2"])
    ap.add_argument("--gammas", type=float, nargs="+",
                    default=[30.0, 100.0, 300.0, 1000.0, 3000.0])
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--T", type=float, default=8.0)
    args = ap.parse_args()

    print("FAIR COMPARISON: every method at its own best gain")
    print(f"trot, vx={args.vx} m/s, T={args.T}s")
    print(f"gain grid: {[int(g) for g in args.gammas]}")
    print(f"total runs: {len(args.rungs)} x {len(args.kinds)} x "
          f"{len(args.gammas)} + {len(args.rungs)} = "
          f"{len(args.rungs)*len(args.kinds)*len(args.gammas)+len(args.rungs)}\n")

    best = {}
    for rung in args.rungs:
        if rung not in RUNGS:
            print(f"!! unknown rung: {rung}")
            continue
        extra, desc = RUNGS[rung]
        print(f"\n{'='*78}")
        print(f"{rung}: {desc}")
        print(f"{'='*78}")

        # baseline (gain independent)
        b = run_one(extra, "none", 0.0, args.vx, args.T)
        bst = "STANDING" if b["ok"] else f"FELL {b['fell']:.1f}s"
        print(f"  baseline (no adaptation): {bst}  pitch {b['pitch']:+.4f}  "
              f"roll std {b['roll_std']:.4f}  tau {b['tau']:.2f}")

        best[rung] = {"none": b}
        print(f"\n  {'method':<6} {'gamma':>7} {'status':>11} {'pred.':>10} "
              f"{'roll std':>9} {'pitch':>9} {'tau':>7}")
        print("  " + "-" * 64)
        for kind in args.kinds:
            curve = []
            for g in args.gammas:
                r = run_one(extra, kind, g, args.vx, args.T)
                curve.append((g, r))
                st = "STANDING" if r["ok"] else f"FELL {r['fell']:.1f}s"
                pv = lambda v, f: (f"{v:{f}}" if v is not None else "    -")
                print(f"  {kind:<6} {g:>7.0f} {st:>11} "
                      f"{pv(r['pred'],'10.4f')} {pv(r['roll_std'],'9.4f')} "
                      f"{pv(r['pitch'],'+9.4f')} {pv(r['tau'],'7.2f')}")
            # best: smallest prediction error among runs that stayed up
            ok = [(g, r) for g, r in curve
                  if r["ok"] and r.get("pred") is not None]
            if ok:
                g_b, r_b = min(ok, key=lambda t: t[1]["pred"])
                worst = max(r["pred"] for _, r in ok)
                best[rung][kind] = dict(gamma=g_b, pred=r_b["pred"],
                                        roll=r_b["roll_std"],
                                        pitch=r_b["pitch"], tau=r_b["tau"],
                                        n_ok=len(ok), n_tot=len(curve),
                                        spread=worst / r_b["pred"])
                print(f"  {'':<6} {'-> best':>7} gamma={g_b:.0f}, "
                      f"error={r_b['pred']:.4f}, "
                      f"up at {len(ok)}/{len(curve)} gains, "
                      f"window ratio {worst/r_b['pred']:.2f}x")
            else:
                best[rung][kind] = None
                print(f"  {'':<6} {'-> best':>7} fell at every gain")
            print()

    # ------------------------------------------------------------------
    print(f"\n{'='*78}")
    print("SUMMARY 1: BEST ERROR (each method at its own best gain)")
    print(f"{'='*78}")
    kinds = args.kinds
    print(f"{'rung':<7} " + " ".join(f"{k:>16}" for k in kinds)
          + f" {'best':>8} {'ratio':>7}")
    print(f"{'':7} " + " ".join(f"{'error @ gamma':>16}" for k in kinds))
    print("-" * 78)
    for rung in best:
        cells, vals = [], {}
        for k in kinds:
            e = best[rung].get(k)
            if e:
                vals[k] = e["pred"]
                cells.append(f"{e['pred']:>9.4f} @{e['gamma']:>5.0f}")
            else:
                cells.append(f"{'fell':>16}")
        if vals:
            win = min(vals, key=vals.get)
            others = [v for k, v in vals.items() if k != win]
            ratio = (min(others) / vals[win]) if others else 1.0
            print(f"{rung:<7} " + " ".join(cells)
                  + f" {win:>8} {ratio:>6.2f}x")
        else:
            print(f"{rung:<7} " + " ".join(cells) + f" {'-':>8} {'-':>7}")

    print(f"\n{'='*78}")
    print("SUMMARY 2: GAIN ROBUSTNESS")
    print(f"{'='*78}")
    print(f"{'rung':<7} " + " ".join(f"{k:>18}" for k in kinds))
    print(f"{'':7} " + " ".join(f"{'up / window':>18}" for k in kinds))
    print("-" * 78)
    for rung in best:
        cells = []
        for k in kinds:
            e = best[rung].get(k)
            cells.append(f"{e['n_ok']}/{e['n_tot']}  {e['spread']:>6.2f}x"
                         if e else f"{'0/' + str(len(args.gammas)):>18}")
        print(f"{rung:<7} " + " ".join(f"{c:>18}" for c in cells))
    print("\n  'up'    : number of gains completed without falling")
    print("  'window': worst/best error ratio among those runs (small = good)")


if __name__ == "__main__":
    main()
