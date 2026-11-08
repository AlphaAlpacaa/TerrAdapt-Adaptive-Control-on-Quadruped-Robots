"""
Asama A kosucusu: uc yaklasiklayiciyi ayni tesis uzerinde kosturur,
izleme hatasini, fonksiyon kestirim hatasini ve Lyapunov fonksiyonunu kaydeder.

Kontrol yasasi (Patil & Dixon 2022, denklem 6):
    u = xd_dot - rho(||e||) e - k1 e - ks sgn(e) - Phi_hat

Lyapunov adayi (denklem 20, uyarlanmis):
    V = 0.5 e^T e + 0.5 * (agirlik parcasi)

Teorem 1 ks > eps_bar + c kosulunu istiyor. Simulasyonda eps_bar bilinmedigi
icin ks taranabilir; makale karsilastirma tablosunda ks terimini kaldiriyor
(DNN teriminin etkisini yalitmak icin). Burada ikisi de kosulabiliyor.
"""

import numpy as np
import json
import os

from plant import f_true, x_desired, rho_gain, rk4_step
from approximators import M0_LinearRegressor, M1_RBF, M2_LbDNN


def simulate(approx, T=10.0, dt=1e-3, k1=20.0, ks=1.0, x0=None,
             use_robust=True, use_rho=True):
    """
    Tek bir yaklasiklayici ile kapali cevrim simulasyonu.

    use_robust : ks*sgn(e) terimi. Makale karsilastirma tablosunda BUNU kaldiriyor.
    use_rho    : rho(||e||)*e terimi. Bu KALIR -- f(x)-f(xd) farkini karsilayan
                 terim bu; kaldirilirsa yaklasiklayicidan bagimsiz bir hata tabani
                 olusur ve kiyas anlamsizlasir.
    """
    if x0 is None:
        x0 = np.array([1.0, 2.0])          # makalenin baslangic kosulu

    n = len(x0)
    nw = approx.n_weights
    state = np.concatenate([x0, approx.pack()])
    n_steps = int(round(T / dt))

    log = {k: [] for k in ["t", "e_norm", "f_err", "u_norm", "V", "w_norm"]}

    def deriv(t, s):
        x = s[:n]
        approx.unpack(s[n:])
        xd, xd_dot = x_desired(t)
        e = x - xd
        Phi_hat = approx.estimate(x, xd, e)

        u = xd_dot - k1 * e - Phi_hat
        if use_rho:
            u = u - rho_gain(np.linalg.norm(e)) * e
        if use_robust:
            u = u - ks * np.sign(e)

        xdot = f_true(x) + u
        wdot = approx.adapt_rate(x, xd, e)
        return np.concatenate([xdot, wdot])

    t = 0.0
    for i in range(n_steps):
        # kayit
        x = state[:n]
        approx.unpack(state[n:])
        xd, _ = x_desired(t)
        e = x - xd
        Phi_hat = approx.estimate(x, xd, e)
        f_err = f_true(x) - Phi_hat
        w = approx.pack()

        log["t"].append(t)
        log["e_norm"].append(float(np.linalg.norm(e)))
        log["f_err"].append(float(np.linalg.norm(f_err)))
        log["V"].append(0.5 * float(e @ e) + 0.5 * approx.lyapunov_weight_term())
        log["w_norm"].append(float(np.linalg.norm(w)))
        _, xd_dot = x_desired(t)
        u = xd_dot - k1 * e - Phi_hat
        if use_rho:
            u = u - rho_gain(np.linalg.norm(e)) * e
        if use_robust:
            u = u - ks * np.sign(e)
        log["u_norm"].append(float(np.linalg.norm(u)))

        state = rk4_step(deriv, state, t, dt)
        t += dt

        if not np.all(np.isfinite(state)):
            print(f"  !! {approx.name}: t={t:.3f} s'de iraksadi")
            break

    return {k: np.array(v) for k, v in log.items()}


def rms(sig, t, t0=None, t1=None):
    m = np.ones_like(t, dtype=bool)
    if t0 is not None:
        m &= t >= t0
    if t1 is not None:
        m &= t <= t1
    return float(np.sqrt(np.mean(sig[m] ** 2)))


def main():
    T, dt = 10.0, 1e-3
    out_dir = os.path.dirname(os.path.abspath(__file__))

    models = [
        M0_LinearRegressor(gamma=50.0, seed=1),
        M1_RBF(n_centers_per_dim=5, width=1.0, gamma=30.0, sigma=0.01, seed=1),
        M2_LbDNN(n_hidden=2, width=6, activation="lrelu", gamma=10.0, seed=1),
    ]

    results = {}
    print(f"Asama A: T={T}s, dt={dt}s, k1=20, ks=1\n")
    print(f"{'Model':<26} {'par':>6} {'eRMS':>9} {'eRMS_ss':>9} "
          f"{'fRMS_ss':>9} {'uRMS':>9} {'V son':>10}")
    print("-" * 84)

    for m in models:
        log = simulate(m, T=T, dt=dt)
        results[m.name] = {
            "n_weights": m.n_weights,
            "e_rms": rms(log["e_norm"], log["t"]),
            "e_rms_ss": rms(log["e_norm"], log["t"], t0=5.0),
            "f_rms_ss": rms(log["f_err"], log["t"], t0=5.0),
            "u_rms": rms(log["u_norm"], log["t"]),
            "V_final": float(log["V"][-1]),
            "V_max": float(np.max(log["V"])),
            "w_norm_final": float(log["w_norm"][-1]),
        }
        r = results[m.name]
        print(f"{m.name:<26} {m.n_weights:>6} {r['e_rms']:>9.4f} "
              f"{r['e_rms_ss']:>9.2e} {r['f_rms_ss']:>9.4f} "
              f"{r['u_rms']:>9.3f} {r['V_final']:>10.4f}")
        np.savez(os.path.join(out_dir, f"log_{m.name.split()[0]}.npz"), **log)

    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    print("\nSutunlar: eRMS = tum araliktaki izleme hatasi RMS'i")
    print("          eRMS_ss = kalici durum (t>5s) izleme hatasi")
    print("          fRMS_ss = kalici durum fonksiyon kestirim hatasi ||f - Phi_hat||")
    print("          V son = Lyapunov fonksiyonunun son degeri")


if __name__ == "__main__":
    main()
