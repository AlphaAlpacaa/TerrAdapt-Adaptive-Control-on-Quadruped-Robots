"""
Verification of the adaptive layer -- no MuJoCo needed.

Tests:
  1  Lyapunov equation              residual of A_c^T P + P A_c + Q_p = 0
  2  Lb-DNN Jacobian                Lambda_j vs central finite differences
  3  Observability                  is E^T P (A-Lambda)^-1 E non-singular?
  4  Parameter convergence          inject a known constant theta*, check
                                    whether theta_hat converges to it
  4b sigma sweep for M1
  4c phase-dependent disturbance (M0's structural limit)
  5  Vdot <= 0                      along the trajectory
  6  Projection bound               ||theta_hat|| <= theta_max
  7  Yaw robustness of P            P is solved at psi = 0 only

Plant: the SRB dynamics (srb_dynamics.continuous_matrices) integrated with
a hand-written RK4. No controller, no MPC -- only predictor and adaptation.
"""

import numpy as np

from a1_params import MASS_TOTAL, I_COMPOSITE, Z_COM_NOMINAL, GRAVITY
from srb_dynamics import (continuous_matrices, disturbance_matrix,
                          wrench_from_accel, N_STATE, N_INPUT)
from adaptive import (AdaptiveLayer, solve_lyapunov, lyapunov_residual,
                      features, FEATURE_SCALE, N_FEATURE)
import approximators_b as ap

FEET = np.array([[0.183, -0.132, 0.0], [0.183, 0.132, 0.0],
                 [-0.183, -0.132, 0.0], [-0.183, 0.132, 0.0]])
COM0 = np.array([0.0, 0.0, Z_COM_NOMINAL])


def static_force():
    """Weight split equally over four feet in static equilibrium."""
    f = np.zeros(N_INPUT)
    for i in range(4):
        f[3*i + 2] = MASS_TOTAL * GRAVITY / 4.0
    return f


# ---------------------------------------------------------------------------
# Stabilised test plant
# ---------------------------------------------------------------------------
#
# With a constant u the body drifts freely (||e_tilde|| = 3270 after 20 s)
# and the features saturate, which says nothing about the approximators. In
# the real loop the MPC keeps the body near the reference, so here an
# unconstrained force distribution does the same: the desired body wrench
# is mapped to foot forces with a pseudo-inverse,
#
#     wrench = W u,   W = [[I I I I], [ [r_1]x [r_2]x [r_3]x [r_4]x ]]
#     u = pinv(W) wrench
#
# u is also given to the predictor (a known input); stabilising through E
# would expose the predictor to an unknown input that theta_hat would absorb.

def wrench_map(foot_pos, com):
    """W, (6, 12): foot forces to body wrench."""
    W = np.zeros((6, N_INPUT))
    for i in range(4):
        r = np.asarray(foot_pos[i], float) - np.asarray(com, float)
        W[0:3, 3*i:3*i+3] = np.eye(3)
        W[3:6, 3*i:3*i+3] = np.array([[0.0, -r[2], r[1]],
                                      [r[2], 0.0, -r[0]],
                                      [-r[1], r[0], 0.0]])
    return W


def stabilizing_u(x, x_ref, I_w, kp=np.array([80.0, 80.0, 400.0]),
                  kd=np.array([30.0, 30.0, 80.0]),
                  kp_ang=np.array([120.0, 120.0, 120.0]),
                  kd_ang=np.array([25.0, 25.0, 25.0])):
    """
    Desired wrench from a PD, distributed to the feet with a pseudo-inverse.
    No constraints (friction cone, f_min/f_max): a TEST plant, not a controller.
    """
    p, Th, v, w = x[0:3], x[3:6], x[6:9], x[9:12]
    a_des = kp * (x_ref[0:3] - p) + kd * (x_ref[6:9] - v)
    al_des = kp_ang * (x_ref[3:6] - Th) + kd_ang * (x_ref[9:12] - w)
    f_des = MASS_TOTAL * (a_des + np.array([0.0, 0.0, GRAVITY]))
    m_des = I_w @ al_des
    W = wrench_map(FEET, p)
    return np.linalg.pinv(W) @ np.concatenate([f_des, m_des])


def x0_nominal():
    x = np.zeros(N_STATE)
    x[2] = Z_COM_NOMINAL
    x[12] = GRAVITY
    return x


# ---------------------------------------------------------------------------

def test_lyapunov():
    print("1) LYAPUNOV EQUATION  A_c^T P + P A_c = -Q_p")
    L = AdaptiveLayer(kind="M0")
    res = lyapunov_residual(L.A_c, L.P, L.Q_p)
    ev_ac = np.max(np.linalg.eigvals(L.A_c).real)
    ev_p = np.linalg.eigvalsh(L.P)
    print(f"   residual ||A_c^T P + P A_c + Q_p|| = {res:.2e} "
          f"({'PASS' if res < 1e-10 else 'FAIL'})")
    print(f"   largest real part of eig(A_c) = {ev_ac:+.3f} "
          f"({'PASS' if ev_ac < 0 else 'FAIL'} -- predictor stable)")
    print(f"   eig(P) in [{ev_p.min():.2e}, {ev_p.max():.2e}] "
          f"({'PASS' if ev_p.min() > 0 else 'FAIL'} -- P > 0)")
    return L


def test_dnn_jacobian():
    print("\n2) Lb-DNN JACOBIAN  Lambda_j vs central finite differences")
    print("   (input and output sizes differ, so the code is tested with")
    print("    asymmetric dimensions)")
    rng = np.random.default_rng(3)
    print(f"   {'activation':<10} {'hidden':>6} {'layers':>7} "
          f"{'largest relative error':>23}")
    print("   " + "-" * 50)
    worst_all = 0.0
    for act in ["tanh", "lrelu"]:
        for k in [1, 2, 3]:
            net = ap.M2_LbDNN(n_in=N_FEATURE, n_out=ap.N_THETA, n_hidden=k,
                              width=7, activation=act, seed=5,
                              scale=FEATURE_SCALE)
            # input away from the LReLU kink (gradient discontinuity)
            xi = rng.normal(0.0, 0.6, N_FEATURE) * FEATURE_SCALE
            worst_layers = []
            for j in range(k + 1):
                Lam = net.jacobian_block(xi, j)
                m_cols = Lam.shape[1]
                h = 1e-6
                worst = 0.0
                cols = rng.choice(m_cols, size=min(m_cols, 25), replace=False)
                for m in cols:
                    Vt = net.V[j].T.copy()
                    flat = Vt.ravel().copy()
                    up, dn = flat.copy(), flat.copy()
                    up[m] += h
                    dn[m] -= h
                    net.V[j] = up.reshape(Vt.shape).T.copy()
                    th_u = net.estimate(xi)
                    net.V[j] = dn.reshape(Vt.shape).T.copy()
                    th_d = net.estimate(xi)
                    net.V[j] = flat.reshape(Vt.shape).T.copy()
                    num = (th_u - th_d) / (2 * h)
                    den = max(np.max(np.abs(num)), 1e-9)
                    worst = max(worst, np.max(np.abs(num - Lam[:, m])) / den)
                worst_layers.append(worst)
            wl = max(worst_layers)
            worst_all = max(worst_all, wl)
            print(f"   {act:<10} {k:>6} {k+1:>7} {wl:>22.2e}  "
                  f"{'PASS' if wl < 1e-4 else 'FAIL'}")
    print(f"   largest overall: {worst_all:.2e} "
          f"({'PASS' if worst_all < 1e-4 else 'FAIL'})")


def test_observability():
    print("\n3) OBSERVABILITY  E^T P (A - Lambda)^-1 E")
    print("   In steady state s = -E^T P e_tilde = 0 with")
    print("   e_tilde_ss = -(A-Lambda)^-1 E Delta, so if this 6x6 matrix is")
    print("   non-singular the only solution is Delta = 0 (theta_hat -> theta*).")
    L = AdaptiveLayer(kind="M0")
    Minv = np.linalg.inv(L.A_c)
    G = L.E.T @ L.P @ Minv @ L.E
    sv = np.linalg.svd(G, compute_uv=False)
    print(f"   singular values : {np.round(sv, 6).tolist()}")
    print(f"   condition number: {sv.max()/sv.min():.1f}")
    print(f"   smallest singular value {sv.min():.3e} "
          f"({'PASS' if sv.min() > 1e-9 else 'FAIL'} -- non-singular)")


def run_plant(kind, theta_star_fn, T=20.0, dt=0.002, excite=False, seed=0):
    """
    Stabilised plant + adaptive layer. Returns (final theta_hat,
    mean ||theta_hat - theta*||, final ||e_tilde||, max vdot, layer).

    excite=True moves the reference with slow sinusoids so the features
    vary (persistent excitation); with a constant reference the features
    are constant and RBF/DNN only fit one point.
    """
    I_w = I_COMPOSITE
    E = disturbance_matrix(MASS_TOTAL, I_COMPOSITE, 0.0)
    L = AdaptiveLayer(kind=kind, dt=dt, seed=seed)
    x = x0_nominal()
    L.reset(x)
    n = int(T / dt)
    errs, vds = [], []
    u = static_force()
    for i in range(n):
        t = i * dt
        x_ref = x0_nominal()
        if excite:
            x_ref[0] = 0.08 * np.sin(2*np.pi*0.35*t)
            x_ref[1] = 0.05 * np.sin(2*np.pi*0.23*t + 1.0)
            x_ref[5] = 0.10 * np.sin(2*np.pi*0.17*t)
            x_ref[6] = 0.08 * 2*np.pi*0.35 * np.cos(2*np.pi*0.35*t)
            x_ref[7] = 0.05 * 2*np.pi*0.23 * np.cos(2*np.pi*0.23*t + 1.0)
            x_ref[11] = 0.10 * 2*np.pi*0.17 * np.cos(2*np.pi*0.17*t)

        th_true = theta_star_fn(t)
        # the predictor gets the u applied in the PREVIOUS step
        L.step(x, u, FEET, dt, phase=(t / 0.4) % 1.0)
        u = stabilizing_u(x, x_ref, I_w)

        A, B = continuous_matrices(x[5], FEET, x[0:3], MASS_TOTAL, I_COMPOSITE)

        def deriv(xx):
            d = A @ xx + B @ u + E @ th_true
            d[12] = 0.0
            return d

        k1 = deriv(x)
        k2 = deriv(x + 0.5*dt*k1)
        k3 = deriv(x + 0.5*dt*k2)
        k4 = deriv(x + dt*k3)
        x = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        if i > n // 2:
            errs.append(np.linalg.norm(L.theta - th_true))
            vds.append(L.vdot)
    return (L.theta.copy(), float(np.mean(errs)),
            float(np.linalg.norm(L.e_tilde)), float(np.max(vds)), L)


def test_convergence():
    print("\n4) PARAMETER CONVERGENCE")
    print("   Plant: xdot = A x + B u + E theta*, RK4, STABILISED")
    print("   (u from the pseudo-inverse wrench map, also given to the predictor)")
    theta_star = np.array([0.8, -0.5, 1.2, 0.4, -0.6, 0.3])
    print(f"\n   theta* = {theta_star.tolist()}  (m/s^2 and rad/s^2)")
    w = wrench_from_accel(theta_star, MASS_TOTAL, I_COMPOSITE, 0.0)
    print(f"   as a wrench: f = {np.round(w[0:3],2).tolist()} N, "
          f"m = {np.round(w[3:6],2).tolist()} N m")

    results = {}
    for label, excite in [("CONSTANT reference", False),
                          ("EXCITED reference (sinusoids)", True)]:
        print(f"\n   --- {label} ---")
        print(f"   {'approximator':<24} {'||theta_hat-theta*||':>21} "
              f"{'||e_tilde||':>12} {'parameters':>10}")
        print("   " + "-" * 70)
        for kind in ["M0", "M1", "M2"]:
            th, de, et, vmax, L = run_plant(kind, lambda t: theta_star,
                                            excite=excite)
            if not excite:
                results[kind] = (de, et, th, vmax)
            print(f"   {L.name:<24} {de:>20.4f} {et:>12.2e} "
                  f"{L.n_weights:>10}")
            if kind == "M0" and not excite:
                print(f"      final estimate: {np.round(th,3).tolist()}")
                print(f"      theta*        : {theta_star.tolist()}")
    print("\n   M0's regressor contains a constant term, so beta_hat captures a")
    print("   constant disturbance exactly. M1's sigma-modification leaves a")
    print("   small bias by design. M2 has no parameter-convergence guarantee,")
    print("   only boundedness.")
    return results


def test_sigma():
    print("\n4b) M1 sigma-MODIFICATION: BIAS / DRIFT TRADE-OFF")
    print("   W_dot = Gamma phi s^T - sigma Gamma W pulls W toward zero, leaving")
    print("   a bias that grows with sigma; in exchange the weights do not drift")
    print("   without persistent excitation. sigma is swept, not chosen ad hoc.\n")
    theta_star = np.array([0.8, -0.5, 1.2, 0.4, -0.6, 0.3])
    print(f"   {'sigma':>8} {'||theta_hat-theta*||':>21} {'||W||':>10} "
          f"{'||e_tilde||':>12}")
    print("   " + "-" * 54)
    best, best_s = np.inf, None
    for sg in [0.0, 0.0005, 0.002, 0.01, 0.05]:
        L2 = AdaptiveLayer(kind="M1", dt=0.002, seed=0, sigma=sg)
        th, de, et, _, L2 = _run_with(L2, theta_star, excite=True)
        wn = float(np.linalg.norm(L2.approx.W))
        print(f"   {sg:>8.4f} {de:>20.4f} {wn:>10.3f} {et:>12.2e}")
        if de < best:
            best, best_s = de, sg
    print(f"\n   best bias: sigma = {best_s} -> {best:.4f}")
    print("   sigma = 0 removes the bias but also the drift protection.")


def _run_with(L, theta_star, excite=False, T=20.0, dt=0.002,
              theta_fn=None, T_gait=0.4):
    """
    run_plant with a ready layer (sigma sweep, phase-dependent test).

    theta_fn : t -> theta*(t); None uses the constant theta_star
    T_gait   : phase given to the layer, t/T_gait mod 1
    """
    E = disturbance_matrix(MASS_TOTAL, I_COMPOSITE, 0.0)
    x = x0_nominal()
    L.reset(x)
    n = int(T / dt)
    errs, vds = [], []
    u = static_force()
    for i in range(n):
        t = i * dt
        x_ref = x0_nominal()
        if excite:
            x_ref[0] = 0.08 * np.sin(2*np.pi*0.35*t)
            x_ref[1] = 0.05 * np.sin(2*np.pi*0.23*t + 1.0)
            x_ref[5] = 0.10 * np.sin(2*np.pi*0.17*t)
            x_ref[6] = 0.08 * 2*np.pi*0.35 * np.cos(2*np.pi*0.35*t)
            x_ref[7] = 0.05 * 2*np.pi*0.23 * np.cos(2*np.pi*0.23*t + 1.0)
            x_ref[11] = 0.10 * 2*np.pi*0.17 * np.cos(2*np.pi*0.17*t)
        th_true = theta_star if theta_fn is None else theta_fn(t)
        L.step(x, u, FEET, dt, phase=(t / T_gait) % 1.0)
        u = stabilizing_u(x, x_ref, I_COMPOSITE)
        A, B = continuous_matrices(x[5], FEET, x[0:3], MASS_TOTAL, I_COMPOSITE)

        def deriv(xx):
            d = A @ xx + B @ u + E @ th_true
            d[12] = 0.0
            return d
        k1 = deriv(x); k2 = deriv(x + 0.5*dt*k1)
        k3 = deriv(x + 0.5*dt*k2); k4 = deriv(x + dt*k3)
        x = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        if i > n // 2:
            errs.append(np.linalg.norm(L.theta - th_true))
            vds.append(L.vdot)
    return (L.theta.copy(), float(np.mean(errs)),
            float(np.linalg.norm(L.e_tilde)), float(np.max(vds)), L)


def test_phase_dependent():
    print("\n4c) PHASE-DEPENDENT DISTURBANCE -- M0'S STRUCTURAL LIMIT")
    print("   The real disturbance is not constant: theta_x averages -1.63 m/s^2")
    print("   and oscillates by +-1.4 m/s^2 within the gait cycle.")
    print("       theta*(t) = theta_0 + A sin(2 pi t / T_gait)")
    print("   M0's regressor has no phase; M1/M2 receive the phase.\n")
    T_gait = 0.4
    theta_0 = np.array([-1.6, 0.0, 0.6, 0.0, 14.0, 0.0])
    amp = np.array([1.4, 0.0, 0.3, 0.0, 3.0, 0.0])

    def th_fn(t):
        return theta_0 + amp * np.sin(2*np.pi*t/T_gait)

    print(f"   theta_0   = {theta_0.tolist()}")
    print(f"   amplitude = {amp.tolist()}   period = {T_gait} s")
    print(f"\n   {'approximator':<24} {'mean |error|':>12} "
          f"{'||e_tilde||':>12} {'parameters':>10}")
    print("   " + "-" * 62)
    for kind in ["M0", "M1", "M2"]:
        L = AdaptiveLayer(kind=kind, dt=0.002, seed=0)
        th, de, et, _, L = _run_with(L, None, excite=True, T=20.0,
                                     theta_fn=th_fn, T_gait=T_gait)
        print(f"   {L.name:<24} {de:>12.4f} {et:>12.2e} "
              f"{L.n_weights:>10}")
    print("\n   If M0 gets the mean right and misses the oscillation, its error")
    print("   stays near the RMS of the amplitude: ||amp||/sqrt(2) = "
          f"{np.linalg.norm(amp)/np.sqrt(2):.3f}")


def test_vdot(results):
    print("\n5) Vdot <= 0  (guaranteed part: -e_tilde^T Q_p e_tilde)")
    for kind, (_, _, _, vmax) in results.items():
        print(f"   {kind}: largest Vdot along the trajectory = {vmax:+.3e} "
              f"({'PASS' if vmax <= 1e-12 else 'FAIL'})")
    print("   NOTE: derivative of the ERROR part of V; with a time-varying")
    print("   theta the result is UUB rather than asymptotic.")


def test_projection():
    print("\n6) PROJECTION BOUND  ||theta_hat|| <= theta_max")
    big = np.array([60.0, -45.0, 90.0, 30.0, -25.0, 15.0])
    print(f"   very large theta* in the plant: ||theta*|| = "
          f"{np.linalg.norm(big):.1f}  (theta_max = 30)")
    for kind in ["M0", "M1", "M2"]:
        L = AdaptiveLayer(kind=kind, dt=0.002, theta_max=30.0, seed=0)
        x = x0_nominal()
        L.reset(x)
        E = disturbance_matrix(MASS_TOTAL, I_COMPOSITE, 0.0)
        u = static_force()
        worst = 0.0
        for i in range(3000):
            L.step(x, u, FEET, 0.002)
            worst = max(worst, float(np.linalg.norm(L.theta)))
            u = stabilizing_u(x, x0_nominal(), I_COMPOSITE)
            A, B = continuous_matrices(x[5], FEET, x[0:3], MASS_TOTAL,
                                       I_COMPOSITE)
            d = A @ x + B @ u + E @ big
            d[12] = 0.0
            x = x + 0.002 * d
        print(f"   {kind}: largest ||theta_hat|| = {worst:7.2f} "
              f"({'PASS' if worst <= 30.0 + 1e-6 else 'FAIL'})")


def test_psi_robustness():
    print("\n7) YAW ROBUSTNESS OF P")
    print("   P is solved at psi = 0; is A_c(psi)^T P + P A_c(psi) < 0")
    print("   over the whole yaw range?")
    L = AdaptiveLayer(kind="M0")
    worst, worst_psi = -np.inf, 0.0
    for psi in np.linspace(-np.pi, np.pi, 73):
        A, _ = continuous_matrices(psi, FEET, COM0, MASS_TOTAL, I_COMPOSITE)
        A_c = A - L.lam * np.eye(N_STATE)
        M = A_c.T @ L.P + L.P @ A_c
        ev = float(np.max(np.linalg.eigvalsh(0.5*(M + M.T))))
        if ev > worst:
            worst, worst_psi = ev, psi
    print(f"   worst eigenvalue {worst:+.4e} (psi = {worst_psi:+.3f} rad)")
    print(f"   {'PASS' if worst < 0 else 'FAIL'}: negative definite over the "
          f"whole yaw range")


def main():
    print("ADAPTIVE LAYER VERIFICATION (body acceleration residual)\n")
    test_lyapunov()
    test_dnn_jacobian()
    test_observability()
    res = test_convergence()
    test_sigma()
    test_phase_dependent()
    test_vdot(res)
    test_projection()
    test_psi_robustness()
    print("\nPARAMETER COUNTS (cost axis)")
    for kind in ["M0I", "M0", "M0F", "M1", "M2"]:
        L = AdaptiveLayer(kind=kind)
        print(f"   {L.name:<34} {L.n_weights:>5}")


if __name__ == "__main__":
    main()
