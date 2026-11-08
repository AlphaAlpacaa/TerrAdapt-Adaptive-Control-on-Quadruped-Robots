"""
Asama A ablasyonu: gurbuzlestirici terimin varligi/yoklugu.

Patil & Dixon (2022), Bolum IV:
  "For the simulations in Tab. I, the robustifying term ks*sgn(e) is removed
   to better quantitatively compare the effects of the DNN term."

Yani makalenin karsilastirma tablosu gurbuzlestirici terim OLMADAN uretilmis.
Terim varken hatayi kendi basina bastiriyor ve yaklasiklayicilar arasindaki
fark goruunmez hale geliyor -- asagida ikisi de kosulup gosteriliyor.
"""

import numpy as np
from run_stage_a import simulate, rms
from approximators import M0_LinearRegressor, M1_RBF, M2_LbDNN


def build(seed=1):
    return [
        M0_LinearRegressor(gamma=50.0, seed=seed),
        M1_RBF(n_centers_per_dim=5, width=1.0, gamma=30.0, sigma=0.01, seed=seed),
        M2_LbDNN(n_hidden=2, width=6, activation="lrelu", gamma=10.0, seed=seed),
    ]


def run_case(use_robust, T=10.0, dt=1e-3, label=""):
    print(f"\n=== {label} (use_robust={use_robust}) ===")
    print(f"{'Model':<26} {'par':>5} {'eRMS':>9} {'eRMS_ss':>10} "
          f"{'fRMS_ss':>10} {'uRMS':>8}")
    print("-" * 72)
    out = {}
    for m in build():
        log = simulate(m, T=T, dt=dt, use_robust=use_robust)
        r = dict(
            n=m.n_weights,
            e_rms=rms(log["e_norm"], log["t"]),
            e_ss=rms(log["e_norm"], log["t"], t0=5.0),
            f_ss=rms(log["f_err"], log["t"], t0=5.0),
            u_rms=rms(log["u_norm"], log["t"]),
        )
        out[m.name] = r
        print(f"{m.name:<26} {r['n']:>5} {r['e_rms']:>9.4f} "
              f"{r['e_ss']:>10.3e} {r['f_ss']:>10.4f} {r['u_rms']:>8.3f}")
    return out


if __name__ == "__main__":
    a = run_case(True, label="Gurbuzlestirici terim ACIK")
    b = run_case(False, label="Gurbuzlestirici terim KAPALI (makalenin tablo kosulu)")

    print("\n--- Fonksiyon kestirim hatasinda M0'a gore iyilesme (terim kapali) ---")
    base = b["M0 (alpha*||e|| + beta)"]["f_ss"]
    for k, v in b.items():
        imp = 100.0 * (base - v["f_ss"]) / base
        print(f"{k:<26} fRMS_ss={v['f_ss']:.4f}   iyilesme={imp:+6.1f}%")
