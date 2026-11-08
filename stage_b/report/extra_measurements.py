"""
Two extra measurements for the report (7 Nov).

  python report/extra_measurements.py timing      # cost of the adaptive layer
  python report/extra_measurements.py transient   # response to the payload step

TIMING: controller.AdaptiveLayer.step and controller.solve_mpc are wrapped
with perf_counter. Flat ground, gamma = 300, seed 0, T = 10 s. The first
1 s (500 step calls, 50 MPC calls) is discarded as warm-up. Run the
methods one after another in the same process, with no other load.

TRANSIENT: flat ground (scene_course.xml) -- on the combined course the
robot is on the ramp at t = 6 s (x ~ 1.8 m) and the step would mix with
the ground; here only the payload acts.
  - pre       : mean over [5.0, 6.0) s of the 0.4 s (one gait period)
                moving-average pitch
  - peak      : max |pitch - pre| for t in [6, 7] s
  - final     : mean of the moving average over [10, 12] s
  - settling  : last exit of the moving average from a +-0.01 rad band
                around the final value, minus 6 s. A first version used
                +-0.005 rad, which is the size of the steady-state wander
                itself (every method gave ~2.7 s); the band is now checked
                to be >= 3 * std(steady) in every run.
Output: report/data/timing.csv, report/data/transient.csv
"""
import sys
import io
import csv
import time
import contextlib
from pathlib import Path
import numpy as np

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK))
import controller                     # noqa: E402
from scenario import run_course       # noqa: E402
from course_def import terrain_course  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)
POS = np.array([0.12, 0.05, 0.06])
METHODS = ["none", "M0I", "M0", "M0F", "M1", "M2"]


def run(kind, seed, T):
    with contextlib.redirect_stdout(io.StringIO()):
        L, _ = run_course(kind, None if kind == "none" else 300.0, 0.3, T,
                          terrain_course(), seed, False, payload_pos=POS,
                          terrain=False, obstacles=False)
    return L


def timing():
    t_step, t_mpc = [], []
    orig_step = controller.AdaptiveLayer.step
    orig_mpc = controller.solve_mpc

    def step_wrap(self, *a, **k):
        t0 = time.perf_counter()
        r = orig_step(self, *a, **k)
        t_step.append(time.perf_counter() - t0)
        return r

    def mpc_wrap(*a, **k):
        t0 = time.perf_counter()
        r = orig_mpc(*a, **k)
        t_mpc.append(time.perf_counter() - t0)
        return r

    controller.AdaptiveLayer.step = step_wrap
    controller.solve_mpc = mpc_wrap
    rows = []
    try:
        for k in METHODS:
            t_step.clear()
            t_mpc.clear()
            run(k, 0, 10.0)
            s = np.array(t_step[500:]) * 1e6       # us, warm-up discarded
            m = np.array(t_mpc[50:]) * 1e6
            r = dict(method=k, n_step=len(s),
                     step_median_us=float(np.median(s)) if len(s) else 0.0,
                     step_mean_us=float(np.mean(s)) if len(s) else 0.0,
                     step_p95_us=float(np.percentile(s, 95)) if len(s) else 0.0,
                     mpc_median_us=float(np.median(m)), mpc_mean_us=float(np.mean(m)))
            r["budget_percent"] = 100 * r["step_mean_us"] / 2000.0
            r["mpc_amortised_us"] = r["mpc_mean_us"] / 10.0   # once per 20 ms
            rows.append(r)
            print(f"{k:5s} step median {r['step_median_us']:7.1f} us  mean "
                  f"{r['step_mean_us']:7.1f}  p95 {r['step_p95_us']:7.1f}  "
                  f"-> {r['budget_percent']:.2f}% of 2 ms   | MPC median "
                  f"{r['mpc_median_us']/1000:.2f} ms ({r['mpc_amortised_us']:.0f} us "
                  f"per 2 ms)", flush=True)
    finally:
        controller.AdaptiveLayer.step = orig_step
        controller.solve_mpc = orig_mpc
    with open(DATA / "timing.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def transient():
    rows = []
    for k in METHODS:
        for sd in [0, 1, 2]:
            L = run(k, sd, 12.0)
            t, p = L["t"], L["pitch"]
            n = int(round(0.4 / (t[1] - t[0])))
            ps = np.convolve(p, np.ones(n) / n, mode="same")
            pre = float(np.mean(ps[(t >= 5.0) & (t < 6.0)]))
            w1 = (t >= 6.0) & (t <= 7.0)
            peak = float(np.max(np.abs(p[w1] - pre)))
            final = float(np.mean(ps[(t >= 10.0) & (t <= 12.0)]))
            BAND = 0.01
            sig = float(np.std(ps[(t >= 9.0) & (t <= 11.8)]))
            idx = np.where((t >= 6.0) & (t <= 11.8) & (np.abs(ps - final) > BAND))[0]
            ts = float(t[idx[-1]] - 6.0) if len(idx) else 0.0
            np.savez_compressed(DATA / f"transient_{k}_s{sd}.npz", t=t, pitch=p)
            standing = L["fell"] is None and L["stuck"] is None
            rows.append(dict(method=k, seed=sd, standing=int(standing),
                             pre=pre, peak=peak, final=final,
                             steady_offset=final - pre, settling_s=ts,
                             sigma_steady=sig, band_valid=int(BAND >= 3 * sig)))
            print(f"{k:5s} s{sd}  pre {pre:+.4f}  peak |dp| {peak:.4f} rad  "
                  f"final {final:+.4f}  offset {final-pre:+.4f}  settling {ts:.2f} s  "
                  f"sig {sig:.4f} {'band OK' if BAND >= 3*sig else 'BAND TOO NARROW!'}"
                  f"  {'ok' if standing else 'BROKE'}", flush=True)
    with open(DATA / "transient.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    {"timing": timing, "transient": transient}[sys.argv[1]]()
