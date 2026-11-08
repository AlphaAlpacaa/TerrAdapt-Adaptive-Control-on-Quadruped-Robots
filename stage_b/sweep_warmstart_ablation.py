"""
Warm-start ablation -- is the carried alpha_hat what breaks M0? (8 Nov)

Warm M0 failed 0/3 (theta_hat hit the bound of 80 on the rough section)
while warm M0F completed 3/3. Diagnosis: M0 carries the gait-locked pitch
disturbance in alpha_hat (alpha_my 3.9, M0F -0.1), enters the rough
section with a large alpha_hat, and the alpha_hat <-> ||e|| loop runs away
through the plant. This is a correlation; the ablation tests causality.

Setup: same as pass 2 of the warm-start sweep (terrain, gamma = 300,
T = 35, payload at 6 s, seeds 0/1/2). Only difference: in the loaded
weights alpha is set to ZERO and beta is carried at its end-of-pass-1
value. Source weights (read only):
sonuclar/sicakbaslat_arazi_g300_t0-1-2/agirlik_M0_t{seed}.npy

Decision rule -- written before running
---------------------------------------
 - COMPLETED and steady loaded ||e~|| < 0.25 in >= 2 of 3 seeds
   (cold 0.197, full warm 1.107) -> "the carried alpha_hat is the cause".
 - Otherwise beta_hat is involved too and the reading must be revised.

Check: alpha block exactly zero, beta block bit-identical to the file
(asserted before running).

    python sweep_warmstart_ablation.py
"""
import sys
import csv
from pathlib import Path
import numpy as np

from scenario import Tee, RESULTS_ROOT
from sweep_warmstart import run, metrics, SEEDS

N = 6


def main():
    kok = Path(__file__).resolve().parent
    # weights recorded by the original (8 Nov) run, kept under their names
    source = kok / RESULTS_ROOT / "sicakbaslat_arazi_g300_t0-1-2"
    odir = kok / RESULTS_ROOT / "warmstart_ablation_M0_alpha0_terrain_g300_s0-1-2"
    if odir.exists() and any(odir.iterdir()):
        sys.exit(f"!! {odir} is not empty -- not overwriting.")
    odir.mkdir(parents=True, exist_ok=True)
    tee = Tee(odir / "summary.txt")
    sys.stdout = tee
    print("ABLATION: warm M0, alpha = 0, beta carried -- terrain, gamma = 300")
    rows = []
    for sd in SEEDS:
        w = np.load(source / f"agirlik_M0_t{sd}.npy")
        w2 = w.copy()
        w2[:N] = 0.0
        assert np.all(w2[:N] == 0.0) and w2[N:].tobytes() == w[N:].tobytes()
        f = odir / f"weights_M0_alpha0_s{sd}.npy"
        np.save(f, w2)
        m = metrics(run("M0", sd, load=f))
        rows.append(dict(seed=sd, beta_norm=float(np.linalg.norm(w2[N:])), **m))
        print(f"  s{sd} |beta|={np.linalg.norm(w2[N:]):.3f}  {m}", flush=True)
    ok = sum(r["outcome"] == "COMPLETED" and r["steady_load"] < 0.25 for r in rows)
    print(f"\nDECISION: {ok}/3 -> "
          f"{'CARRIED ALPHA IS THE CAUSE' if ok >= 2 else 'beta involved too -- revise the reading'}")
    with open(odir / "grid.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    sys.stdout = tee.std
    tee.close()


if __name__ == "__main__":
    main()
