"""
Independent check of the swing control law -- no MuJoCo needed.

In run_walk.py the swing error mixes with body motion, MPC, contacts and
gait planning. Here the leg is attached to a fixed body and only the swing
law is tested, which separates the PD's structural lag cleanly.

Plant -- full 3-DOF leg dynamics:

    M(q) qddot + c(q, qdot) + g(q) = tau

    M : leg_mass_matrix(), from the kinetic energy, checked independently
    g : leg_gravity_torque(), from the potential energy
    c : Christoffel symbols from finite differences of M:
        c_i = sum_{j,k} Gamma_ijk qdot_j qdot_k
        Gamma_ijk = 1/2 ( dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i )

The controller does NOT know the Coriolis term (controller.py uses only
M qddot + g); the plant contains it, so this test is not a tautology.

Controller (same law as controller.py, Di Carlo et al. Eqs. 2 and 16):

    tau = J^T [ kp (p_d - p) + kd (v_d - v) ]        <- PD
        + G(q)                                       <- gravity compensation
        + M(q) J^-1 (a_d - Jdot qdot)                <- acceleration feed-forward

Integration: hand-written RK4.
"""

import numpy as np

from a1_params import Q_NOMINAL, TAU_MAX
from a1_kinematics import (foot_position, foot_jacobian, jacobian_dot,
                           leg_mass_matrix, leg_gravity_torque,
                           swing_feedforward_torque, joint_friction_torque,
                           inverse_kinematics, link_com_positions)
from gait import bezier_swing

LEG = "FR"
MASSES = (0.696, 1.013, 0.226)
INERTIAS = {
    "hip":   np.diag([0.000469, 0.000808, 0.000552]),
    "thigh": np.diag([0.005529, 0.005139, 0.001367]),
    "calf":  np.diag([0.002997, 0.003014, 0.000032]),
}
COMS = {"hip": np.array([-0.003311, 0.000635, 0.000031]),
        "thigh": np.array([-0.003237, 0.022327, -0.027326]),
        "calf": np.array([0.006435, 0.0, -0.107388])}
R_EYE = np.eye(3)

# Joint parameters read from the model (check_actuation.py). Without them
# the isolated test gave 0.3 mm against MuJoCo's 43 mm:
#   armature     -> M diagonal 1.49x / 1.52x / 2.36x, Lambda_zz 2.97x
#   damping      -> 3.62 Nm at 1.81 rad/s (swing torque budget 4.9 Nm)
#   frictionloss -> speed-independent error
ARMATURE = np.array([0.01, 0.01, 0.01])        # kg m^2
DAMPING = np.array([1.0, 2.0, 2.0])            # N m s / rad
FRICTIONLOSS = np.array([0.2, 0.2, 0.2])       # N m
FRIC_EPS = 0.05                                # rad/s, tanh transition band


# ---------------------------------------------------------------------------
# Plant: full leg dynamics
# ---------------------------------------------------------------------------

def coriolis(q, dq, eps=1e-6):
    """
    Coriolis/centrifugal vector c(q, qdot) (3,) from the Christoffel
    symbols, with dM/dq by finite differences (M is analytic).
    """
    q = np.asarray(q, float)
    dM = np.zeros((3, 3, 3))              # dM[i,j,k] = dM_ij / dq_k
    for k in range(3):
        up, dn = q.copy(), q.copy()
        up[k] += eps
        dn[k] -= eps
        # armature omitted: a constant diagonal does not change dM/dq
        Mu = leg_mass_matrix(LEG, up, MASSES, INERTIAS, COMS)
        Md = leg_mass_matrix(LEG, dn, MASSES, INERTIAS, COMS)
        dM[:, :, k] = (Mu - Md) / (2 * eps)

    c = np.zeros(3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                G = 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i])
                c[i] += G * dq[j] * dq[k]
    return c


LOSSES = True       # switched off in the energy-conservation test


def accel(q, dq, tau):
    """
    qddot = M^-1 (tau - c - g)

    Sign: U(q) = -sum_i m_i g^T p_ci with g = [0, 0, -9.81];
    leg_gravity_torque() = -sum_i m_i g^T dp_ci/dq_j = +dU/dq_j, which is
    exactly the g(q) of M qddot + c + g = tau, so a controller adding it to
    tau cancels it.
    """
    M = leg_mass_matrix(LEG, q, MASSES, INERTIAS, COMS, ARMATURE)
    c = coriolis(q, dq)
    g = leg_gravity_torque(LEG, q, R_EYE, MASSES, COMS)
    # viscous + Coulomb losses, as in the MuJoCo joint model
    d_loss = (DAMPING * dq + FRICTIONLOSS * np.tanh(dq / FRIC_EPS)) \
        if LOSSES else 0.0
    return np.linalg.solve(M, tau - c - g - d_loss)


def rk4_step(q, dq, tau, dt):
    """Hand-written RK4 for the second-order system."""
    def f(y):
        qq, dd = y[:3], y[3:]
        return np.concatenate([dd, accel(qq, dd, tau)])
    y = np.concatenate([q, dq])
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y[:3], y[3:]


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

def swing_law(q, dq, p_d, v_d, a_d, kp, kd, grav=True, ff=True,
              fric=True, armature=True):
    """
    Same swing law as controller.py.

    armature=False : omit the rotor inertia in the feed-forward
    fric=False     : omit the friction compensation
    """
    J = foot_jacobian(LEG, q)
    p = foot_position(LEG, q)
    v = J @ dq
    tau = J.T @ (kp * (p_d - p) + kd * (v_d - v))
    if grav:
        tau = tau + leg_gravity_torque(LEG, q, R_EYE, MASSES, COMS)
    if ff:
        tau = tau + swing_feedforward_torque(
            LEG, q, dq, a_d, MASSES, INERTIAS, COMS,
            armature=(ARMATURE if armature else None))
    if fric:
        tau = tau + joint_friction_torque(dq, DAMPING, FRICTIONLOSS, FRIC_EPS)
    return np.clip(tau, -TAU_MAX, TAU_MAX)


def run_swing(kp=700.0, kd=25.0, height=0.07, T_sw=0.20, step=0.06,
              grav=True, ff=True, fric=True, armature=True, dt=0.002):
    """
    One swing from the nominal foot position, step forward.
    Returns (mean error, max error, touchdown error, max torque) in m / Nm.
    """
    p0 = foot_position(LEG, Q_NOMINAL)
    pf = p0 + np.array([step, 0.0, 0.0])

    q = np.array(Q_NOMINAL, float)
    dq = np.zeros(3)
    n = int(round(T_sw / dt))
    errs, taus = [], []
    for i in range(n):
        s = i / n
        p_d, v_ds, a_dss = bezier_swing(p0, pf, s, height=height)
        v_d = v_ds / T_sw
        a_d = a_dss / (T_sw * T_sw)
        tau = swing_law(q, dq, p_d, v_d, a_d, kp, kd, grav, ff,
                        fric, armature)
        errs.append(np.linalg.norm(p_d - foot_position(LEG, q)))
        taus.append(np.max(np.abs(tau)))
        q, dq = rk4_step(q, dq, tau, dt)

    td_err = np.linalg.norm(pf - foot_position(LEG, q))
    return np.mean(errs), np.max(errs), td_err, np.max(taus)


# ---------------------------------------------------------------------------

def main():
    print("Swing control law check (no MuJoCo, full leg dynamics)\n")

    # 1) plant consistency: with zero torque the leg swings like a pendulum
    print("1) Plant consistency (losses OFF -- energy must be conserved)")
    global LOSSES
    LOSSES = False
    q, dq = np.array(Q_NOMINAL, float), np.zeros(3)
    E0 = None
    for i in range(500):
        M = leg_mass_matrix(LEG, q, MASSES, INERTIAS, COMS, ARMATURE)
        # total energy T + U, U = -sum m_i g^T p_ci
        U = -sum(m * float(np.array([0, 0, -9.81]) @ p)
                 for m, p in zip(MASSES, link_com_positions(LEG, q, COMS)))
        E = 0.5 * float(dq @ M @ dq) + U
        if E0 is None:
            E0 = E
        q, dq = rk4_step(q, dq, np.zeros(3), 0.001)
    drift = abs(E - E0) / max(abs(E0), 1e-12)
    print(f"   free pendulum, 0.5 s, relative energy drift: "
          f"{drift:.2e} ({'PASS' if drift < 1e-4 else 'FAIL'})")
    print("   (a wrong Coriolis term would not conserve energy)")
    LOSSES = True
    print("   losses ON again: damping=[1,2,2] frictionloss=0.2\n")

    # 2) feed-forward ablation
    print("2) TERM-BY-TERM ABLATION  (kp=700, kd=25, h=0.07 m, T=0.20 s)")
    print("   Plant as in MuJoCo: armature + damping + frictionloss ON.")
    print("   Controller terms are switched on one by one.\n")
    print(f"   {'controller contains':<34} {'mean':>8} {'max':>8} "
          f"{'tdown':>8} {'torque':>8}")
    print("   " + "-" * 70)
    cases = [
        ("pure PD",                         dict(grav=0, ff=0, fric=0)),
        ("+ gravity G(q)",                  dict(grav=1, ff=0, fric=0)),
        ("+ accel. (no armature)",          dict(grav=1, ff=1, fric=0, armature=0)),
        ("+ accel. (with armature)",        dict(grav=1, ff=1, fric=1-1, armature=1)),
        ("+ friction (full inverse dyn.)",  dict(grav=1, ff=1, fric=1, armature=1)),
    ]
    res = []
    for name, kw in cases:
        kw = {k: bool(v) for k, v in kw.items()}
        e_m, e_x, e_td, t_x = run_swing(**kw)
        res.append((name, e_m, e_x, e_td, t_x))
        print(f"   {name:<34} {e_m*1000:>7.1f} {e_x*1000:>7.1f} "
              f"{e_td*1000:>7.1f} {t_x:>7.2f}")
    print("   (errors in mm, torque in Nm)")
    print("\n   Measured in MuJoCo (same gains, at the apex): 42.8 mm")
    print(f"   'pure PD' row here: {res[0][1]*1000:.1f} mm")
    print(f"\n   share of the armature in the feed-forward: "
          f"{res[2][1]*1000:.1f} -> {res[3][1]*1000:.1f} mm")
    print(f"   share of the friction compensation: "
          f"{res[3][1]*1000:.1f} -> {res[4][1]*1000:.1f} mm")
    print(f"   TOTAL: {res[1][1]*1000:.1f} -> {res[4][1]*1000:.1f} mm "
          f"({100*(1-res[4][1]/res[1][1]):.0f}% reduction)")

    # 3) gain sweep
    print("\n3) GAIN SWEEP (mean error, mm)")
    print(f"   {'kp':>6} {'PD+gravity':>12} {'full ID':>10}")
    print("   " + "-" * 30)
    for kp in [200, 400, 700, 1200, 2000]:
        a = run_swing(kp=kp, kd=kp/28.0, grav=True, ff=False, fric=False)[0]
        b = run_swing(kp=kp, kd=kp/28.0, grav=True, ff=True, fric=True)[0]
        print(f"   {kp:>6} {a*1000:>11.1f} {b*1000:>9.1f}")
    print("   (with feed-forward a lower kp suffices -> softer touchdown)")

    # 4) sensitivity to the trajectory demand
    print("\n4) SENSITIVITY TO THE TRAJECTORY DEMAND (mean error, mm)")
    print(f"   {'h (m)':>7} {'T (s)':>7} {'a_tdown':>9} "
          f"{'PD+gravity':>12} {'full ID':>8}")
    print("   " + "-" * 48)
    for h, T in [(0.05, 0.20), (0.07, 0.20), (0.09, 0.20),
                 (0.07, 0.25), (0.07, 0.30)]:
        a_td = 32.0 * h / T**2
        x = run_swing(height=h, T_sw=T, grav=True, ff=False, fric=False)[0]
        y = run_swing(height=h, T_sw=T, grav=True, ff=True, fric=True)[0]
        print(f"   {h:>7.2f} {T:>7.2f} {a_td:>8.1f} "
              f"{x*1000:>11.1f} {y*1000:>7.1f}")
    print("   (the pure-PD error is proportional to the demand -- a structural")
    print("    lag; with feed-forward it is almost independent of it)")


if __name__ == "__main__":
    main()
