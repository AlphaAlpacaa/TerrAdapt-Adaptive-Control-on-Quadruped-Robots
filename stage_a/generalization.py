"""
Asama A - AYIRT EDICI TEST: genelleme.

Gozlem: yuksek uyarlama kazanciyla M0 (alpha*||e|| + beta) da cok iyi fonksiyon
kestirimi veriyor. Sebebi su -- beta_hat hizli uyarlandiginda bir FONKSIYON
ogrenmiyor, f(t)'yi anlik olarak KOVALIYOR. Yani yuksek kazancli bir bozucu
gozleyicisi gibi davraniyor.

Bu, "sinir agi daha iyi yaklasiklar" iddiasinin tek basina yeterli olmadigini
gosteriyor. Ayrim su iki kosulda ortaya cikmali:

  (A) Belirsizlik, uyarlamanin kovalayabileceginden HIZLI degisiyorsa
  (B) Ogrenileni GORULMEMIS bir kosula tasimak gerekiyorsa (genelleme)

Bu dosya (B)'yi test eder:
  1. Ag/parametreler bir yorunge uzerinde egitilir (uyarlama acik)
  2. Agirliklar DONDURULUR
  3. FARKLI bir referans yorunge verilir
  4. Fonksiyon kestirim hatasi olculur

M0 nokta-kestirimi oldugu icin genellemez. M1/M2 bir fonksiyon kodladigi icin
genellemelidir. Proje tezinin cekirdegi bu.
"""

import numpy as np
from plant import f_true, rho_gain, rk4_step
from approximators import M0_LinearRegressor, M1_RBF, M2_LbDNN


# --- referans yorungeler -----------------------------------------------------

def traj_train(t):
    """Egitim yorungesi: makalenin kendi referansi."""
    return (np.array([np.sin(2.0 * t), -np.cos(t)]),
            np.array([2.0 * np.cos(2.0 * t), np.sin(t)]))


def traj_test(t):
    """Test yorungesi: farkli frekans, farkli genlik, farkli faz.
    Ayni durum bolgesini tarar ama ayni egri degildir."""
    return (np.array([1.3 * np.sin(0.7 * t + 0.9), 1.1 * np.cos(1.6 * t)]),
            np.array([1.3 * 0.7 * np.cos(0.7 * t + 0.9), -1.1 * 1.6 * np.sin(1.6 * t)]))


def run(approx, traj, T, dt=1e-3, k1=20.0, freeze=False, x0=None):
    n = 2
    if x0 is None:
        xd0, _ = traj(0.0)
        x0 = xd0.copy()
    state = np.concatenate([x0, approx.pack()])

    def deriv(t, s):
        x = s[:n]
        approx.unpack(s[n:])
        xd, xd_dot = traj(t)
        e = x - xd
        Phi = approx.estimate(x, xd, e)
        u = xd_dot - rho_gain(np.linalg.norm(e)) * e - k1 * e - Phi
        xdot = f_true(x) + u
        wdot = (np.zeros_like(s[n:]) if freeze
                else approx.adapt_rate(x, xd, e))
        return np.concatenate([xdot, wdot])

    ts, fe, ee = [], [], []
    t = 0.0
    for _ in range(int(round(T / dt))):
        x = state[:n]
        approx.unpack(state[n:])
        xd, _ = traj(t)
        e = x - xd
        Phi = approx.estimate(x, xd, e)
        ts.append(t)
        ee.append(np.linalg.norm(e))
        fe.append(np.linalg.norm(f_true(x) - Phi))
        state = rk4_step(deriv, state, t, dt)
        t += dt
        if not np.all(np.isfinite(state)):
            break
    return np.array(ts), np.array(ee), np.array(fe)


def rms_tail(sig, ts, t0):
    m = ts >= t0
    return float(np.sqrt(np.mean(sig[m] ** 2))) if m.any() else float("nan")


def main():
    T_train, T_test = 20.0, 10.0

    models = [
        ("M0 (alpha||e||+beta)",
         lambda: M0_LinearRegressor(gamma=800.0, seed=1)),
        ("M1 (RBF 7x7)",
         lambda: M1_RBF(n_centers_per_dim=7, width=0.8, gamma=600.0,
                        sigma=0.001, seed=1)),
        ("M2 (Lb-DNN 2x6)",
         lambda: M2_LbDNN(n_hidden=2, width=6, activation="lrelu",
                          gamma=200.0, switch_gain=True, seed=1)),
    ]

    print("Genelleme testi")
    print(f"  1) Egitim yorungesinde {T_train:.0f}s, uyarlama ACIK")
    print(f"  2) Agirliklar DONDURULUR")
    print(f"  3) Test yorungesinde {T_test:.0f}s, uyarlama KAPALI\n")

    print(f"{'Model':<22} {'par':>5} {'fRMS egitim':>12} {'fRMS test':>11} "
          f"{'bozulma':>9}")
    print("-" * 64)

    for name, ctor in models:
        m = ctor()
        # 1) egitim
        ts, ee, fe = run(m, traj_train, T_train)
        f_train = rms_tail(fe, ts, T_train * 0.5)
        # 2-3) agirliklar donduruldu, yeni yorunge
        ts2, ee2, fe2 = run(m, traj_test, T_test, freeze=True)
        f_test = rms_tail(fe2, ts2, T_test * 0.3)
        ratio = f_test / f_train if f_train > 0 else float("nan")
        print(f"{name:<22} {m.n_weights:>5} {f_train:>12.4f} "
              f"{f_test:>11.4f} {ratio:>8.1f}x")

    print("\nbozulma = test hatasinin egitim hatasina orani.")
    print("Buyuk oran = ogrenilen sey o yorungeye ozel, genellemiyor.")


if __name__ == "__main__":
    main()
