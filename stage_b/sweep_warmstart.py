"""
Warm start -- do the approximators retain what they adapted? (8 Nov)

Rationale: every run starts from fresh weights. The one claim a universal
approximator has over a classical regressor is that it can RETAIN a
state-dependent map; M0 can only retain a constant (beta).

Setup
-----
Terrain course (scene_terrain.xml, finish 8.0 m, T = 35 s), gamma = 300,
seeds 0/1/2, methods M0I M0 M0F M1 M2. Each cell runs twice:
  pass 1: COLD (normal); the final weights are saved as .npy
  pass 2: WARM, starting from pass 1's final weights
The event schedule is the same in both passes (payload at 6 s). Predictor,
filter and MPC restart in both passes; only the approximator weights are
carried over.

Known design effect (written before running): pass 1 ends LOADED, pass 2
starts without the payload for 6 s; the payload is not directly visible in
the features. The decision rule therefore looks at the LOADED segment.
M1's sigma-modification also forgets unexcited weights (~7 s).

Metrics (prediction error ||e~||)
  steady_load : second half of the loaded segment (6 s - finish)
  last_quarter: last 25% of the run (duration = finish time)
  unloaded    : second half of the 0-6 s segment
  drop        : pass1 - pass2 (positive = warm is better)

Decision rule -- written before running, not changed afterwards
---------------------------------------------------------------
M_k (every method except M0) RETAINS if, in the same seed:
 (a) steady_load drop > 0 AND > M0's steady_load drop,  in >= 2/3 seeds;
 (b) last_quarter drop > 0 AND > M0's last_quarter drop, in >= 2/3 seeds.
Without (b), "retained" cannot be told apart from "shorter transient".
For M0: is the drop > 0 (does it retain beta)?
Note added afterwards: the rule had no minimum effect size, which turned
out to be a flaw (M1 passed with a ~0.1% effect while M0 collapsed).

Checks (before pass 2; the script stops if any fails)
 V1: a loaded weight file re-saves BIT-IDENTICALLY;
 V2: with gamma = 0 a warm start gives theta_hat != 0 at the first step
     (the weights are really applied, reset did not overwrite them);
 V3: a repeated COLD pass reproduces pass 1 to 6 decimals.

    python sweep_warmstart.py
"""
import contextlib
import csv
import io
import sys
from pathlib import Path
import numpy as np

from scenario import run_course, segment_stats, Tee, RESULTS_ROOT
from course_def import terrain_course

METHODS = ["M0I", "M0", "M0F", "M1", "M2"]
SEEDS = [0, 1, 2]
G, T = 300.0, 35.0
POS = np.array([0.12, 0.05, 0.06])


def run(kind, seed, gamma=G, Tm=T, load=None, save=None):
    with contextlib.redirect_stdout(io.StringIO()):
        L, _ = run_course(kind, gamma, 0.3, Tm, terrain_course(), seed, False,
                          payload_pos=POS, terrain=True, obstacles=False,
                          load_weights=load, save_weights=save)
    return L


def metrics(L):
    course = terrain_course()
    t, pe = L["t"], L["pred_err"]
    t_end = float(t[-1])
    seg = segment_stats(L, course, t_end)
    last = (t >= 0.75 * t_end)
    st = ("FELL" if L["fell"] is not None else "STALLED" if L["stuck"]
          is not None else "COMPLETED" if L["done"] is not None else "TIME_UP")
    return dict(outcome=st, x=float(L["x_final"]), t_end=t_end,
                unloaded=seg[0]["pred"], steady_load=seg[1]["pred"],
                last_quarter=float(np.mean(pe[last])),
                pitch_load=abs(seg[1]["pitch"]))


def main():
    kok = Path(__file__).resolve().parent
    odir = kok / RESULTS_ROOT / "warmstart_terrain_g300_s0-1-2"
    if odir.exists() and any(odir.iterdir()):
        sys.exit(f"!! {odir} is not empty -- not overwriting.")
    odir.mkdir(parents=True, exist_ok=True)
    tee = Tee(odir / "summary.txt")
    sys.stdout = tee
    print("WARM START -- terrain, gamma=300, seeds 0/1/2, 2 passes")

    # ---------------- pass 1 (cold) ------------------------------------
    res = {}
    for k in METHODS:
        for sd in SEEDS:
            f = odir / f"weights_{k}_s{sd}.npy"
            L = run(k, sd, save=f)
            res[(k, sd, 1)] = metrics(L)
            print(f"  pass1 {k:4s} s{sd} {res[(k, sd, 1)]}", flush=True)

    # ---------------- checks -------------------------------------------
    print("\nCHECKS")
    ok = True
    for k in METHODS:
        f = odir / f"weights_{k}_s0.npy"
        g = odir / f"_v1_{k}.npy"
        np.save(g, np.load(f))
        v1 = f.read_bytes() == g.read_bytes()
        g.unlink()
        L0 = run(k, 0, gamma=0.0, Tm=0.05, load=f)
        th0 = L0["theta_raw"][0]
        v2 = bool(np.linalg.norm(th0) > 1e-9)
        Lc = run(k, 0)
        m = metrics(Lc)
        v3 = (abs(m["steady_load"] - res[(k, 0, 1)]["steady_load"]) < 1e-6
              and abs(m["x"] - res[(k, 0, 1)]["x"]) < 1e-6)
        print(f"  {k:4s} V1 bit-identical {v1}  V2 ||theta(0)||={np.linalg.norm(th0):.4f} "
              f"{v2}  V3 cold repeat {v3}", flush=True)
        ok &= v1 and v2 and v3
    if not ok:
        print("!! CHECKS FAILED -- pass 2 NOT run")
        sys.stdout = tee.std
        tee.close()
        sys.exit(1)

    # ---------------- pass 2 (warm) ------------------------------------
    print()
    for k in METHODS:
        for sd in SEEDS:
            L = run(k, sd, load=odir / f"weights_{k}_s{sd}.npy")
            res[(k, sd, 2)] = metrics(L)
            print(f"  pass2 {k:4s} s{sd} {res[(k, sd, 2)]}", flush=True)

    # ---------------- decision -----------------------------------------
    drop = lambda k, sd, m: res[(k, sd, 1)][m] - res[(k, sd, 2)][m]
    print("\n" + "=" * 74)
    print("DECISION (rule written before the run)")
    print("=" * 74)
    print("  drop = pass1 - pass2 (positive: warm is better)")
    print("  method  seed  unloaded  steady_load  last_quarter  pitch_load p1->p2")
    for k in METHODS:
        for sd in SEEDS:
            print(f"  {k:5s}   {sd}    {drop(k, sd, 'unloaded'):+.4f}   "
                  f"{drop(k, sd, 'steady_load'):+.4f}     {drop(k, sd, 'last_quarter'):+.4f}"
                  f"      {res[(k, sd, 1)]['pitch_load']:.4f}->{res[(k, sd, 2)]['pitch_load']:.4f}"
                  f"  [{res[(k, sd, 1)]['outcome'][:4]}/{res[(k, sd, 2)]['outcome'][:4]}]")
    print()
    m0 = sum(drop("M0", sd, "steady_load") > 0 for sd in SEEDS)
    print(f"  M0 retains beta (steady_load drop > 0): {m0}/3")
    for k in [k for k in METHODS if k != "M0"]:
        a = sum(drop(k, sd, "steady_load") > 0 and
                drop(k, sd, "steady_load") > drop("M0", sd, "steady_load") for sd in SEEDS)
        b = sum(drop(k, sd, "last_quarter") > 0 and
                drop(k, sd, "last_quarter") > drop("M0", sd, "last_quarter") for sd in SEEDS)
        print(f"  {k:4s}: (a) {a}/3  (b) {b}/3  -> "
              f"{'RETAINS (more than M0)' if a >= 2 and b >= 2 else 'does not retain more than M0'}")

    with open(odir / "grid.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "seed", "pass", "outcome", "x", "t_end", "unloaded",
                    "steady_load", "last_quarter", "pitch_load"])
        for (k, sd, gc), r in res.items():
            w.writerow([k, sd, gc, r["outcome"], f"{r['x']:.4f}", f"{r['t_end']:.2f}",
                        f"{r['unloaded']:.6f}", f"{r['steady_load']:.6f}",
                        f"{r['last_quarter']:.6f}", f"{r['pitch_load']:.6f}"])
    print(f"\nALL OUTPUTS: {odir}")
    sys.stdout = tee.std
    tee.close()


if __name__ == "__main__":
    main()
