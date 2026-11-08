"""
Uncertainty ladder: M0 / M1 / M2 on fixed-condition rungs (early stage of
the project, run_walk.py based; the main results use scenario.py).

Rungs and the structure of the uncertainty
------------------------------------------
L0  flat ground, no payload: the SRB model's own error; constant mean plus
    a within-cycle oscillation (swing-leg momentum). Partly known structure.
L1  unknown payload (--payload): theta_z = -g dm/m, constant, one parameter.
    Fully known structure; M0 should suffice.
L3a slope (--slope): gravity rotates in the body frame; constant, known.
L3b slippery ground (--mu): the MPC friction cone exceeds reality; slip
    depends on the contact state.
L4  soft ground (--soft): contact force depends on penetration depth and
    rate, which the SRB model does not contain; no clean regressor.

Rung values were calibrated with verify_terrain.py. The first values
(--mu 0.45, --soft 0.12 on the floor only) changed nothing measurable:
  1) MuJoCo blends solref; the stiffer geom dominates, so softness must be
     applied to the feet too (penetration 10.1 -> 30.5 mm at 0.12);
  2) the required friction coefficient is ~0.29 on average, so any mu
     above that has no effect; slip only grows at mu = 0.15;
  3) the reference ground is not hard: 10 mm penetration with a 20 mm foot
     radius, and stiffer is not possible (tau >= 2*timestep).

Usage
-----
    python sweep_terrain.py                  # whole ladder
    python sweep_terrain.py --rungs L0 L4a   # two rungs only
    python sweep_terrain.py --gamma 1000 --T 10
"""

import argparse
import subprocess
import sys
import re
import numpy as np

RUNGS = {
    "L0":  ([], "reference: MuJoCo default (10 mm penetration)"),
    "L1":  (["--payload", "3"], "unknown payload 3 kg (24%)"),
    "L3a": (["--slope", "8"], "slope 8 deg"),
    "L3b": (["--mu", "0.15"], "slippery mu=0.15 (slip 110->170 mm/s)"),
    "L4a": (["--soft", "0.12"], "soft: penetration 10->31 mm"),
    "L4b": (["--soft", "0.30"], "very soft: penetration 10->79 mm"),
    "L4c": (["--soft", "0.12", "--mu", "0.15"], "soft AND slippery"),
}

# Patterns that parse run_walk.py's console output (must match its print
# statements exactly).
PATTERNS = {
    "pitch": r"pitch\s+mean\s+([-+]?\d+\.\d+)",
    "roll_std": r"roll\s+mean\s+[-+]?\d+\.\d+\s+std\s+(\d+\.\d+)",
    "z": r"z mean\s+(\d+\.\d+)",
    "vx": r"vx\s+mean\s+([-+]?\d+\.\d+)",
    "tau": r"tau max\s+(\d+\.\d+)",
    "x": r"distance\s+([-+]?\d+\.\d+)",
    "swing": r"swing error\s+mean\s+(\d+\.\d+)",
    "pred": r"prediction error\s+:\s+mean\s+(\d+\.\d+)",
    "fell": r"!! t=(\d+\.\d+)s: robot fell",
}


def run_one(extra, adaptive, gamma, vx, T):
    cmd = [sys.executable, "run_walk.py", "--gait", "trot",
           "--vx", str(vx), "--T", str(T)] + list(extra)
    if adaptive != "none":
        cmd += ["--adaptive", adaptive, "--gamma", str(gamma)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout
    res = {}
    for k, pat in PATTERNS.items():
        mm = re.search(pat, out)
        res[k] = float(mm.group(1)) if mm else None
    res["ok"] = res["fell"] is None
    if p.returncode != 0:
        res["err"] = p.stderr.strip().splitlines()[-1] if p.stderr else "?"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", default=list(RUNGS.keys()))
    ap.add_argument("--kinds", nargs="+",
                    default=["none", "M0", "M1", "M2"])
    ap.add_argument("--gamma", type=float, default=1000.0)
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--T", type=float, default=8.0)
    args = ap.parse_args()

    print("UNCERTAINTY LADDER")
    print(f"trot, vx={args.vx} m/s, T={args.T}s, gamma={args.gamma:.0f}\n")

    table = {}
    for rung in args.rungs:
        if rung not in RUNGS:
            print(f"!! unknown rung: {rung}")
            continue
        extra, desc = RUNGS[rung]
        print(f"\n{'='*72}")
        print(f"{rung}: {desc}    ({' '.join(extra) if extra else 'no extra arguments'})")
        print(f"{'='*72}")
        print(f"{'layer':<8} {'status':>11} {'pitch':>9} {'roll std':>9} "
              f"{'z':>8} {'vx':>7} {'tau':>7} {'pred.':>9}")
        print("-" * 72)
        table[rung] = {}
        for kind in args.kinds:
            r = run_one(extra, kind, args.gamma, args.vx, args.T)
            table[rung][kind] = r
            if "err" in r:
                print(f"{kind:<8} ERROR: {r['err'][:50]}")
                continue
            st = "STANDING" if r["ok"] else f"FELL {r['fell']:.1f}s"
            pv = lambda v, f: (f"{v:{f}}" if v is not None else "    -")
            print(f"{kind:<8} {st:>11} {pv(r['pitch'],'+9.4f')} "
                  f"{pv(r['roll_std'],'9.4f')} {pv(r['z'],'8.4f')} "
                  f"{pv(r['vx'],'7.3f')} {pv(r['tau'],'7.2f')} "
                  f"{pv(r['pred'],'9.4f')}")

    # ---------------- validity check ----------------
    # A rung that does not measurably change the baseline would only
    # re-measure L0.
    print(f"\n\n{'='*72}")
    print("VALIDITY: did the rung actually change the baseline?")
    print(f"{'='*72}")
    if "none" in args.kinds and "L0" in table:
        ref = table["L0"]["none"]
        print(f"{'rung':<8} {'d pitch':>9} {'d z (mm)':>10} "
              f"{'d tau':>8} {'':>9}  {'verdict':>12}")
        print("-" * 62)
        for rung in table:
            r = table[rung]["none"]
            if r.get("pitch") is None or ref.get("pitch") is None:
                continue
            dp = r["pitch"] - ref["pitch"]
            dz = (r["z"] - ref["z"]) * 1000
            dt_ = (r["tau"] - ref["tau"]) if r.get("tau") else 0.0
            # noise band: pitch 0.005 rad, z 3 mm, tau 1 Nm
            bite = (abs(dp) > 0.005) or (abs(dz) > 3.0) or (abs(dt_) > 1.0)
            verdict = "EFFECTIVE" if bite else "!! NO EFFECT"
            if rung == "L0":
                verdict = "reference"
            print(f"{rung:<8} {dp:>+9.4f} {dz:>+10.1f} {dt_:>+8.2f} "
                  f"{'':>9}  {verdict:>12}")
        print("\n  Comparisons on rungs marked NO EFFECT are INVALID: the")
        print("  perturbation is below the noise, so L0 is re-measured.")
    else:
        print("  (skipped: baseline or L0 not run)")

    print(f"\n\n{'='*72}")
    print("SUMMARY: PREDICTION ERROR (lower is better), rung x approximator")
    print(f"{'='*72}")
    kinds = [k for k in args.kinds if k != "none"]
    print(f"{'rung':<8} " + " ".join(f"{k:>10}" for k in kinds)
          + f"  {'best':>10}")
    print("-" * 60)
    for rung in table:
        row, vals = [], {}
        for k in kinds:
            v = table[rung][k].get("pred")
            vals[k] = v
            row.append(f"{v:>10.4f}" if v is not None else f"{'-':>10}")
        good = {k: v for k, v in vals.items() if v is not None}
        win = min(good, key=good.get) if good else "-"
        print(f"{rung:<8} " + " ".join(row) + f"  {win:>10}")

    print(f"\n{'='*72}")
    print("SUMMARY: STAYING UP")
    print(f"{'='*72}")
    print(f"{'rung':<8} " + " ".join(f"{k:>8}" for k in args.kinds))
    print("-" * 48)
    for rung in table:
        row = []
        for k in args.kinds:
            r = table[rung][k]
            row.append("up" if r.get("ok") else
                       (f"F{r['fell']:.1f}" if r.get("fell") else "error"))
        print(f"{rung:<8} " + " ".join(f"{v:>8}" for v in row))

    print("\nREADING GUIDE")
    print("  Read the staying-up column first: the prediction error of a")
    print("  run that fell is meaningless.")


if __name__ == "__main__":
    main()
