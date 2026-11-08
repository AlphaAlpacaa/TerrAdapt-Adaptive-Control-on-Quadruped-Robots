"""
Lb-DNN uyarlama kuralinin DOGRULUK testi.

Patil & Dixon'in uyarlama kurali (denklem 7-8) su Jacobian'i kullanir:

    Lambda_j = Omega_j * (I_{L_{j+1}} kron v_j^T)

ve Lyapunov analizinde bu terimin
    d Phi / d vec(V_j)
ile ayni olmasi gerekir -- cunku Lyapunov turevindeki capraz terim
    e^T * (d Phi / d vec(V_j)) * vec(V_j_tilde)
uyarlama kuraliyla tam olarak gotrulur.

Bu dosya Lambda_j'yi SONLU FARKLARLA hesaplanan gercek Jacobian ile karsilastirir.
Eslesmiyorsa kural yanlis kodlanmistir (en sik hata: vec(.) satir/sutun
onceligi ya da Kronecker carpiminin sirasi).

Bu test, agi quadruped'e tasimadan once calistirilmasi gereken tek testtir:
quadruped icinde bir isaret hatasini bulmak gunler alir, burada saniyeler.
"""

import numpy as np
from approximators import M2_LbDNN


def numeric_jacobian(net, xd, j, eps=1e-6):
    """d Phi / d vec(V_j), satir-oncelikli vec, merkezi fark."""
    V0 = net.V[j].copy()
    rows, cols = V0.shape
    J = np.zeros((net.n, rows * cols))
    flat = V0.ravel().copy()          # satir-oncelikli (C duzeni)
    for idx in range(rows * cols):
        up = flat.copy(); up[idx] += eps
        dn = flat.copy(); dn[idx] -= eps
        net.V[j] = up.reshape(rows, cols)
        f_up, *_ = net._forward(xd)
        net.V[j] = dn.reshape(rows, cols)
        f_dn, *_ = net._forward(xd)
        J[:, idx] = (f_up - f_dn) / (2.0 * eps)
    net.V[j] = V0
    return J


def analytic_jacobian(net, xd, j):
    """Makalenin Lambda_j'si, satir-oncelikli vec duzenine cevrilmis."""
    _, _, phi_list, dphi_list, xda = net._forward(xd)
    Omega = net._omega(j, phi_list, dphi_list)      # (n, L_{j+1})
    v = xda if j == 0 else phi_list[j - 1]          # R^{L_j}
    Lam = np.kron(Omega, v.reshape(1, -1))          # (n, L_{j+1}*L_j)

    # Lam sutunlari (L_{j+1}, L_j) sirasinda; V_j ise (L_j, L_{j+1}).
    # Satir-oncelikli vec(V_j) ile eslestirmek icin sutunlari yeniden sirala.
    n = net.n
    Lj, Ljp1 = net.L[j], net.L[j + 1]
    Lam = Lam.reshape(n, Ljp1, Lj).transpose(0, 2, 1).reshape(n, Lj * Ljp1)
    return Lam


def main():
    print("Lb-DNN Jacobian dogrulamasi (analitik Lambda_j  vs  sonlu fark)\n")
    rng = np.random.default_rng(7)

    for activation in ["tanh", "lrelu"]:
        print(f"--- aktivasyon: {activation} ---")
        for k, width in [(1, 4), (2, 6), (3, 5)]:
            net = M2_LbDNN(n=2, n_hidden=k, width=width,
                           activation=activation, seed=3)
            ok_all = True
            for _ in range(5):
                xd = rng.uniform(-2.0, 2.0, size=2)
                for j in range(net.k + 1):
                    Ja = analytic_jacobian(net, xd, j)
                    Jn = numeric_jacobian(net, xd, j)
                    err = np.max(np.abs(Ja - Jn))
                    scale = max(np.max(np.abs(Jn)), 1e-9)
                    if err / scale > 1e-5:
                        ok_all = False
                        print(f"  k={k} w={width} katman j={j}: "
                              f"BAGIL HATA {err/scale:.2e}  <-- UYUSMUYOR")
            status = "gecti" if ok_all else "KALDI"
            print(f"  k={k} gizli katman, genislik {width}: {status}")
        print()


if __name__ == "__main__":
    main()
