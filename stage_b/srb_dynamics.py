"""
Single-rigid-body (SRB) dynamics and discretisation.

Sources: Di Carlo, Wensing, Katz, Bledt & Kim, IROS 2018, Sec. III;
         Sombolestan & Nguyen, IEEE T-RO 2024, Eqs. (10)-(11).

Sign issue between the two papers
---------------------------------
Di Carlo Eq. (9): omega = W(psi) Theta_dot with, for small roll/pitch,
W(psi) = R_z(psi), hence Theta_dot = W^-1 omega = R_z(psi)^T omega
(Eqs. 11-12). Sombolestan Eq. (2) and its D^c matrix write
Theta_dot = R_z(psi) omega (no transpose). Checked numerically:
W^-1 == R_z^T. This module uses R_z^T, with omega in the WORLD frame;
verify_srb_mujoco.py checks the choice against MuJoCo.

State (Sombolestan order, 13):
    X = [ p_c (3), Theta (3), p_c_dot (3), omega (3), ||g|| (1) ]
Input: U = [F_1; F_2; F_3; F_4] in R^12, ground reaction forces, world frame.
"""

import numpy as np
from scipy.linalg import expm

from a1_params import MASS_TOTAL, I_COMPOSITE, GRAVITY

N_STATE = 13
N_INPUT = 12
N_LEGS = 4


def rot_z(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def skew(v):
    """[v]_x : skew(v) @ w == cross(v, w)"""
    return np.array([[0.0, -v[2],  v[1]],
                     [v[2], 0.0,  -v[0]],
                     [-v[1], v[0], 0.0]])


def world_inertia(psi, I_body=None):
    """
    Inertia in the world frame. Di Carlo Eq. (15), small roll/pitch:
    I ~= R_z(psi) I_B R_z(psi)^T
    """
    if I_body is None:
        I_body = I_COMPOSITE
    R = rot_z(psi)
    return R @ I_body @ R.T


def continuous_matrices(psi, foot_pos, com_pos, mass=MASS_TOTAL, I_body=None):
    """
    Continuous-time A(psi) and B(r_i, psi).

    psi      : yaw (rad)
    foot_pos : (4,3) foot positions, world frame
    com_pos  : (3,)  centre of mass, world frame
    """
    foot_pos = np.asarray(foot_pos, float).reshape(N_LEGS, 3)
    com_pos = np.asarray(com_pos, float)

    A = np.zeros((N_STATE, N_STATE))
    B = np.zeros((N_STATE, N_INPUT))

    # p_c_dot = p_c_dot        (rows 0:3, cols 6:9)
    A[0:3, 6:9] = np.eye(3)

    # Theta_dot = R_z^T(psi) omega   (rows 3:6, cols 9:12)   <-- transpose
    A[3:6, 9:12] = rot_z(psi).T

    # gravity in p_c_ddot: unit vector * ||g||   (rows 6:9, col 12)
    A[6:9, 12] = np.array([0.0, 0.0, -1.0])

    # force -> acceleration and moment
    I_w = world_inertia(psi, I_body)
    I_inv = np.linalg.inv(I_w)
    for i in range(N_LEGS):
        r = foot_pos[i] - com_pos
        B[6:9, 3 * i:3 * i + 3] = np.eye(3) / mass          # p_ddot
        B[9:12, 3 * i:3 * i + 3] = I_inv @ skew(r)          # omega_dot
    return A, B


def disturbance_matrix(mass=MASS_TOTAL, I_body=None, psi=0.0):
    """
    Input matrix E (13, 6) of the unknown body-level ACCELERATION residual:

        theta = [a_x, a_y, a_z, alpha_x, alpha_y, alpha_z]   (m/s^2, rad/s^2)

        pddot    = (1/m) sum f_i + g      + a_dist        -> rows 6:9
        omegadot = I_w^-1 sum r_i x f_i   + alpha_dist    -> rows 9:12

    so E = [[0_(6x6)], [I_3, 0], [0, I_3], [0]] and xdot = A x + B u + E theta.

    Why acceleration and not a wrench: with a wrench, E = [(1/m) I; I_w^-1]
    and the singular values of the observability matrix E^T P (A-Lambda)^-1 E
    are [0.63, 0.093, 0.078, 8.1e-5, 8.1e-5, 8.1e-5] (condition number
    7832): one gain adapts the force channels ~8000x faster than the moment
    channels. In acceleration units both blocks of E are identity and the
    condition number is ~1. wrench_from_accel() converts back for reporting.
    """
    E = np.zeros((N_STATE, 6))
    E[6:9, 0:3] = np.eye(3)
    E[9:12, 3:6] = np.eye(3)
    return E


def wrench_from_accel(theta, mass=MASS_TOTAL, I_body=None, psi=0.0):
    """
    Acceleration residual -> body wrench (for reporting):
        f = m * a  (N),   m_wrench = I_w * alpha  (N m)
    """
    theta = np.asarray(theta, float)
    f = mass * theta[0:3]
    tau = world_inertia(psi, I_body) @ theta[3:6]
    return np.concatenate([f, tau])


def discretize(A, B, dt, E=None):
    """
    Zero-order-hold discretisation (Di Carlo Eq. 25):

        expm( [[A, B], [0, 0]] * dt ) = [[Ad, Bd], [0, I]]

    If E is given, [B, E] is treated as one input matrix so that Bd and Ed
    come from the same ZOH (approximating Ed by E*dt accumulates error over
    the horizon). Returns (Ad, Bd) or (Ad, Bd, Ed).
    """
    n, m = A.shape[0], B.shape[1]
    if E is None:
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A
        M[:n, n:] = B
        Ex = expm(M * dt)
        return Ex[:n, :n].copy(), Ex[:n, n:].copy()

    p = E.shape[1]
    M = np.zeros((n + m + p, n + m + p))
    M[:n, :n] = A
    M[:n, n:n+m] = B
    M[:n, n+m:] = E
    Ex = expm(M * dt)
    return (Ex[:n, :n].copy(), Ex[:n, n:n+m].copy(), Ex[:n, n+m:].copy())


def continuous_derivative(X, U, foot_pos, mass=MASS_TOTAL, I_body=None):
    """
    State derivative written directly from the physics, independent of the
    matrices -- reference for checking the matrix form.
    """
    X = np.asarray(X, float)
    U = np.asarray(U, float).reshape(N_LEGS, 3)
    p, Theta, pdot, omega, gnorm = (X[0:3], X[3:6], X[6:9], X[9:12], X[12])
    psi = Theta[2]

    F_sum = U.sum(axis=0)
    p_ddot = F_sum / mass + np.array([0.0, 0.0, -gnorm])

    I_w = world_inertia(psi, I_body)
    tau = np.zeros(3)
    for i in range(N_LEGS):
        r = np.asarray(foot_pos[i], float) - p
        tau += np.cross(r, U[i])
    omega_dot = np.linalg.solve(I_w, tau)

    Theta_dot = rot_z(psi).T @ omega

    return np.concatenate([pdot, Theta_dot, p_ddot, omega_dot, [0.0]])


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _nominal_setup():
    from a1_params import Z_NOMINAL, Q_NOMINAL, LEGS
    from a1_kinematics import foot_position
    com = np.array([0.0, 0.0, Z_NOMINAL])
    feet = np.array([foot_position(l, Q_NOMINAL) + np.array([0, 0, Z_NOMINAL])
                     for l in LEGS])
    return com, feet


def self_test():
    print("SRB dynamics checks\n")
    com, feet = _nominal_setup()

    # --- 1) static equilibrium: total force carries the weight -----------
    print("1) Static equilibrium")
    Fz = MASS_TOTAL * GRAVITY / 4.0
    U = np.tile(np.array([0.0, 0.0, Fz]), 4)
    X = np.zeros(N_STATE)
    X[0:3] = com
    X[12] = GRAVITY
    dX = continuous_derivative(X, U, feet)
    print(f"   Fz per foot          = {Fz:.2f} N")
    print(f"   linear acceleration  = {np.round(dX[6:9], 8).tolist()} m/s^2")
    print(f"   angular acceleration = {np.round(dX[9:12], 8).tolist()} rad/s^2")
    ok = np.allclose(dX[6:12], 0.0, atol=1e-6)
    print(f"   {'PASS' if ok else 'FAIL'}: zero acceleration in equilibrium\n")

    # --- 2) matrix form vs direct physics ---------------------------------
    print("2) A,B matrices vs direct physics (200 random states/inputs)")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        X = np.zeros(N_STATE)
        X[0:3] = com + rng.normal(0, 0.05, 3)
        X[3:6] = rng.normal(0, 0.15, 3)
        X[6:9] = rng.normal(0, 0.4, 3)
        X[9:12] = rng.normal(0, 0.4, 3)
        X[12] = GRAVITY
        f = feet + rng.normal(0, 0.02, feet.shape)
        U = rng.normal(0, 40.0, N_INPUT)
        A, B = continuous_matrices(X[5], f, X[0:3])
        d_mat = A @ X + B @ U
        d_phy = continuous_derivative(X, U, f)
        worst = max(worst, np.max(np.abs(d_mat - d_phy)))
    print(f"   largest difference: {worst:.2e} "
          f"({'PASS' if worst < 1e-9 else 'FAIL'})\n")

    # --- 3) ZOH discretisation vs fine RK4 ---------------------------------
    print("3) ZOH discretisation vs fine-step RK4")
    X0 = np.zeros(N_STATE)
    X0[0:3] = com
    X0[3:6] = [0.03, -0.02, 0.25]
    X0[6:9] = [0.3, -0.1, 0.05]
    X0[9:12] = [0.1, 0.2, -0.15]
    X0[12] = GRAVITY
    U = np.tile([5.0, -3.0, Fz + 8.0], 4)
    A, B = continuous_matrices(X0[5], feet, X0[0:3])

    for dt in [0.01, 0.02, 0.03]:
        Ad, Bd = discretize(A, B, dt)
        X_zoh = Ad @ X0 + Bd @ U
        # reference: same A,B with a very fine RK4
        Xr, n_sub = X0.copy(), 4000
        h = dt / n_sub
        for _ in range(n_sub):
            k1 = A @ Xr + B @ U
            k2 = A @ (Xr + 0.5 * h * k1) + B @ U
            k3 = A @ (Xr + 0.5 * h * k2) + B @ U
            k4 = A @ (Xr + h * k3) + B @ U
            Xr = Xr + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        err = np.max(np.abs(X_zoh - Xr))
        print(f"   dt={dt:.3f}: largest difference {err:.2e} "
              f"({'PASS' if err < 1e-10 else 'FAIL'})")
    print()

    # --- 4) structural checks ----------------------------------------------
    print("4) Structural checks")
    A, B = continuous_matrices(0.0, feet, com)
    print(f"   A shape {A.shape}, B shape {B.shape}")
    print(f"   gravity row A[6:9,12] = {A[6:9,12].tolist()} "
          f"({'PASS' if np.allclose(A[6:9,12], [0,0,-1]) else 'FAIL'})")
    ok = np.allclose(A[3:6, 9:12], np.eye(3))
    print(f"   Theta_dot = omega at psi=0 : {'PASS' if ok else 'FAIL'}")
    ok = np.allclose(B[8, 2::3], 1.0 / MASS_TOTAL)
    print(f"   B[8, z columns] = 1/m      : {'PASS' if ok else 'FAIL'}")
    # symmetric feet with equal vertical forces -> no yaw acceleration
    A2, B2 = continuous_matrices(0.0, feet, com)
    tau_yaw = (B2[11, 2::3]).sum()
    print(f"   sum of yaw coefficients with equal vertical forces = "
          f"{tau_yaw:.2e} ({'PASS' if abs(tau_yaw) < 1e-12 else 'FAIL'})")

    # --- 5) yaw dependence -------------------------------------------------
    print("\n5) Yaw dependence (inertia rotated to the world frame)")
    for psi in [0.0, np.pi / 4, np.pi / 2]:
        Iw = world_inertia(psi)
        print(f"   psi={psi:.3f}: I_w diagonal = {np.round(np.diag(Iw), 4).tolist()}")
    print("   (at psi=pi/2 Ixx and Iyy should swap)")


if __name__ == "__main__":
    self_test()
