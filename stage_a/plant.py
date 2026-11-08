"""
Asama A - Test tesisi ve ortak altyapi.

Tesis: Patil, Le, Greene & Dixon, IEEE L-CSS 2022, Bolum IV (Simulations).
    xdot = f(x) + u,   x in R^2
    f(x) = [ x1*x2*tanh(x2) + sech^2(x1) ,  sech^2(x1+x2) - sech^2(x2) ]^T
    xd(t) = [ sin(2t), -cos(t) ]^T
    x(0) = [1, 2]^T

Makaledeki kontrol yasasi (denklem 6):
    u = xd_dot - rho(||e||)*e - k1*e - ks*sgn(e) - Phi_hat
burada e = x - xd ve Phi_hat bilinmeyen f'nin kestirimi.

Integrator: klasik 4. mertebe Runge-Kutta, elle yazildi (hazir ODE cozucu yok).
"""

import numpy as np

# ----------------------------------------------------------------------------
# Tesis dinamigi
# ----------------------------------------------------------------------------

def sech(z):
    """Hiperbolik sekant. 1/cosh(z), buyuk |z| icin tasma korumali."""
    return 1.0 / np.cosh(np.clip(z, -300.0, 300.0))


def f_true(x):
    """Bilinmeyen tesis fonksiyonu f(x). Kontrolcu bunu BILMEZ."""
    x1, x2 = x[0], x[1]
    return np.array([
        x1 * x2 * np.tanh(x2) + sech(x1) ** 2,
        sech(x1 + x2) ** 2 - sech(x2) ** 2,
    ])


def x_desired(t):
    """Istenen yorunge xd(t) ve turevi xd_dot(t)."""
    xd = np.array([np.sin(2.0 * t), -np.cos(t)])
    xd_dot = np.array([2.0 * np.cos(2.0 * t), np.sin(t)])
    return xd, xd_dot


# ----------------------------------------------------------------------------
# Kontrol yasasi yardimcilari
# ----------------------------------------------------------------------------

def rho_gain(e_norm):
    """
    Durum bagimli kazanc rho(||e||).

    Makale rho'yu "bilinen kesin artan fonksiyon" olarak birakiyor; fe = f(x)-f(xd)
    terimini ||fe|| <= rho(||e||)*||e|| seklinde sinirlamak icin kullaniliyor
    (Kamalapurkar ve dig., RISE destek lemmalari).

    Bu tesis icin f'nin Jacobian'i polinomsal buyudugunden rho(s) = c0 + c1*s + c2*s^2
    makul bir ustur. Katsayilar asagida sayisal olarak dogrulanir.
    """
    return 1.0 + 2.0 * e_norm + 1.0 * e_norm ** 2


def rk4_step(deriv, state, t, dt):
    """Tek bir klasik RK4 adimi. deriv(t, state) -> dstate."""
    k1 = deriv(t, state)
    k2 = deriv(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = deriv(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = deriv(t + dt, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def projection(theta, y, bound):
    """
    Izdusum operatoru (Lavretsky & Gibson 2011, Ek E).

    Basitlestirilmis kure izdusumu: ||theta|| <= bound kumesini degismez tutar.
    Sinirda ve disari dogru giden bilesen varsa o bilesen atilir.
    Bu, Lyapunov analizinde kullanilan
        (theta_hat - theta*)^T (Proj(theta_hat, y) - y) <= 0
    ozelligini saglar.
    """
    n = np.linalg.norm(theta)
    if n < bound:
        return y
    outward = float(theta @ y)
    if outward <= 0.0:
        return y
    return y - outward * theta / (n ** 2)
