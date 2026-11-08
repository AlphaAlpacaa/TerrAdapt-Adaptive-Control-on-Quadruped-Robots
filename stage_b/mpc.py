"""
Convex MPC: ground reaction force optimisation.

Sources: Di Carlo et al., IROS 2018, Sec. IV;
         Sombolestan & Nguyen, IEEE T-RO 2024, Eq. (12).

Problem (Eqs. 18-21):

    min_{x,u}  sum_{i=0}^{k-1}  ||x_{i+1} - x_{i+1,ref}||_Q + ||u_i||_R
    s.t.       x_{i+1} = A_i x_i + B_i u_i
               c_ <= C u_i <= c^      (friction pyramid + force limits)
               D_i u_i = 0            (swing legs carry no force)

Condensed form (Eqs. 27-32): the dynamics are eliminated,

    X = A_qp x_0 + B_qp U
    A_qp = [ A_0 ; A_1 A_0 ; A_2 A_1 A_0 ; ... ]                   (13k x 13)
    B_qp[i,j] = (A_i ... A_{j+1}) B_j   for j <= i, else 0         (13k x 12k)

    J(U) = 1/2 U^T H U + U^T g + const,
    H = 2 (B_qp^T L B_qp + K),   g = 2 B_qp^T L (A_qp x_0 - X_ref),

so H is 12k x 12k regardless of the state size.

Friction pyramid (Eqs. 22-24), per foot:
    f_min <= f_z <= f_max,  -mu f_z <= f_x, f_y <= mu f_z
"""

import numpy as np
from srb_dynamics import (continuous_matrices, discretize,
                          disturbance_matrix, N_STATE, N_INPUT, N_LEGS)
from a1_params import (MASS_TOTAL, GRAVITY, Q_MPC, R_MPC, MPC_HORIZON,
                       F_MIN, F_MAX, MU_EFFECTIVE)


# ---------------------------------------------------------------------------
# Condensed dynamics
# ---------------------------------------------------------------------------

def build_condensed(Ad_list, Bd_list):
    """
    Build A_qp and B_qp.
    Ad_list, Bd_list : discrete matrices over the horizon, length k
    """
    k = len(Ad_list)
    n, m = N_STATE, N_INPUT
    A_qp = np.zeros((n * k, n))
    B_qp = np.zeros((n * k, m * k))

    # cumulative product: P_i = A_i A_{i-1} ... A_0
    P = np.eye(n)
    for i in range(k):
        P = Ad_list[i] @ P
        A_qp[n*i:n*(i+1), :] = P

    for i in range(k):
        for j in range(i + 1):
            # (A_i A_{i-1} ... A_{j+1}) B_j
            M = np.eye(n)
            for t in range(i, j, -1):
                M = M @ Ad_list[t]
            B_qp[n*i:n*(i+1), m*j:m*(j+1)] = M @ Bd_list[j]
    return A_qp, B_qp


def build_disturbance(Ad_list, Ed_list):
    """
    Condensed affine disturbance term C_qp, (13k x 6).

    theta_hat is held CONSTANT over the horizon (adaptation is slower than
    the horizon):

        x_{i+1} = Ad_i x_i + Bd_i u_i + Ed_i theta_hat
        X = A_qp x_0 + B_qp U + C_qp theta_hat
        C_qp[i] = sum_{j=0..i} (A_i A_{i-1} ... A_{j+1}) Ed_j

    Same product structure as B_qp, but summed over j (one theta, not one
    per step).
    """
    k = len(Ad_list)
    n = N_STATE
    p = Ed_list[0].shape[1]
    C_qp = np.zeros((n * k, p))
    for i in range(k):
        acc = np.zeros((n, p))
        for j in range(i + 1):
            M = np.eye(n)
            for t in range(i, j, -1):
                M = M @ Ad_list[t]
            acc += M @ Ed_list[j]
        C_qp[n*i:n*(i+1), :] = acc
    return C_qp


def rollout(Ad_list, Bd_list, x0, U, Ed_list=None, theta=None):
    """Step-by-step rollout -- to check the condensed form."""
    k = len(Ad_list)
    X = np.zeros(N_STATE * k)
    x = np.asarray(x0, float).copy()
    for i in range(k):
        u = U[N_INPUT*i:N_INPUT*(i+1)]
        x = Ad_list[i] @ x + Bd_list[i] @ u
        if Ed_list is not None and theta is not None:
            x = x + Ed_list[i] @ theta
        X[N_STATE*i:N_STATE*(i+1)] = x
    return X


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def friction_block(mu=MU_EFFECTIVE):
    """
    5x3 inequality matrix and bounds for one foot.
    Rows: f_min <= fz <= f_max, |fx| <= mu fz (2 rows), |fy| <= mu fz (2 rows),
    written as fx - mu fz <= 0, -fx - mu fz <= 0, and likewise for fy.
    """
    C = np.array([
        [0.0, 0.0, 1.0],     # fz
        [1.0, 0.0, -mu],     # fx - mu fz <= 0
        [-1.0, 0.0, -mu],    # -fx - mu fz <= 0
        [0.0, 1.0, -mu],     # fy - mu fz <= 0
        [0.0, -1.0, -mu],    # -fy - mu fz <= 0
    ])
    lo = np.array([F_MIN, -np.inf, -np.inf, -np.inf, -np.inf])
    hi = np.array([F_MAX, 0.0, 0.0, 0.0, 0.0])
    return C, lo, hi


def build_constraints(contact_schedule, mu=MU_EFFECTIVE):
    """
    All constraints over the horizon.
    contact_schedule : (k, 4) bool  -- True = stance, False = swing
    Swing legs are forced to zero force (lo = hi = 0).
    """
    k = contact_schedule.shape[0]
    Cf, lof, hif = friction_block(mu)
    rows_per_leg = Cf.shape[0]

    blocks, lo_all, hi_all = [], [], []
    for i in range(k):
        for leg in range(N_LEGS):
            row = np.zeros((rows_per_leg, N_INPUT * k))
            col = N_INPUT * i + 3 * leg
            row[:, col:col+3] = Cf
            blocks.append(row)
            if contact_schedule[i, leg]:
                lo_all.append(lof)
                hi_all.append(hif)
            else:
                # swing: fz = 0 and the pyramid forces fx = fy = 0
                lo_all.append(np.array([0.0, -np.inf, -np.inf, -np.inf, -np.inf]))
                hi_all.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    C = np.vstack(blocks)
    return C, np.concatenate(lo_all), np.concatenate(hi_all)


# ---------------------------------------------------------------------------
# QP setup
# ---------------------------------------------------------------------------

def build_qp(x0, X_ref, Ad_list, Bd_list, contact_schedule,
             Q=None, R=R_MPC, mu=MU_EFFECTIVE, Ed_list=None, theta=None):
    """
    Returns H, g, C, lo, hi:  min 1/2 U^T H U + U^T g,  lo <= C U <= hi.

    With theta the affine disturbance enters the PREDICTION:

        J(U) = || A_qp x_0 + C_qp theta + B_qp U - X_ref ||_L + ||U||_K
        g    = 2 B_qp^T L (A_qp x_0 + C_qp theta - X_ref)

    H is unchanged. The correction enters the MPC dynamics instead of
    being added to the solution afterwards, so the MPC distributes it
    inside the friction cone and force limits.
    """
    k = len(Ad_list)
    Q = Q_MPC if Q is None else Q
    A_qp, B_qp = build_condensed(Ad_list, Bd_list)
    x_aff = A_qp @ x0
    if Ed_list is not None and theta is not None:
        x_aff = x_aff + build_disturbance(Ad_list, Ed_list) @ np.asarray(theta, float)

    # L: block-diagonal state weight (13k x 13k); Q is 12x12, so the
    # gravity state gets zero weight.
    Qfull = np.zeros((N_STATE, N_STATE))
    Qfull[:12, :12] = Q
    L = np.kron(np.eye(k), Qfull)
    K = R * np.eye(N_INPUT * k)

    H = 2.0 * (B_qp.T @ L @ B_qp + K)
    g = 2.0 * B_qp.T @ L @ (x_aff - X_ref)
    H = 0.5 * (H + H.T)                      # enforce symmetry

    C, lo, hi = build_constraints(contact_schedule, mu)
    return H, g, C, lo, hi, A_qp, B_qp


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def solve_osqp(H, g, C, lo, hi, warm=None, **kw):
    """Solver used in practice."""
    import osqp
    from scipy import sparse
    prob = osqp.OSQP()
    prob.setup(P=sparse.csc_matrix(H), q=g, A=sparse.csc_matrix(C),
               l=lo, u=hi, verbose=False, eps_abs=1e-6, eps_rel=1e-6,
               max_iter=4000, **kw)
    if warm is not None:
        prob.warm_start(x=warm)
    res = prob.solve()
    return np.asarray(res.x, float), res.info.status


def solve_scipy(H, g, C, lo, hi, warm=None, **kw):
    """
    Fallback solver (if osqp is missing): scipy SLSQP. Slow; for checks
    and small horizons only.
    """
    from scipy.optimize import minimize

    n = H.shape[0]
    x0 = np.zeros(n) if warm is None else np.asarray(warm, float)

    def f(u):
        return 0.5 * u @ H @ u + g @ u

    def fp(u):
        return H @ u + g

    cons = []
    fin_lo = np.isfinite(lo)
    fin_hi = np.isfinite(hi)
    if fin_lo.any():
        Cl, ll = C[fin_lo], lo[fin_lo]
        cons.append({"type": "ineq", "fun": lambda u: Cl @ u - ll,
                     "jac": lambda u: Cl})
    if fin_hi.any():
        Ch, hh = C[fin_hi], hi[fin_hi]
        cons.append({"type": "ineq", "fun": lambda u: hh - Ch @ u,
                     "jac": lambda u: -Ch})

    res = minimize(f, x0, jac=fp, constraints=cons, method="SLSQP",
                   options={"maxiter": 400, "ftol": 1e-10})
    return res.x, ("solved" if res.success else res.message)


def solve_qp(H, g, C, lo, hi, warm=None):
    try:
        return solve_osqp(H, g, C, lo, hi, warm)
    except ImportError:
        return solve_scipy(H, g, C, lo, hi, warm)


# ---------------------------------------------------------------------------
# Top-level interface
# ---------------------------------------------------------------------------

def solve_mpc(x0, x_ref_traj, foot_pos, contact_schedule, dt_mpc,
              mass=MASS_TOTAL, I_body=None, mu=MU_EFFECTIVE, warm=None,
              Q=None, theta=None):
    """
    x0               : (13,) current state
    x_ref_traj       : (k, 13) reference trajectory (i = 1..k)
    foot_pos         : (k, 4, 3) foot positions over the horizon (world)
    contact_schedule : (k, 4) bool
    dt_mpc           : horizon step
    theta            : estimated disturbance (enters the prediction model)
    """
    k = contact_schedule.shape[0]
    psi = x0[5]
    Ed = None if theta is None else disturbance_matrix(mass, I_body, psi)
    Ad_list, Bd_list, Ed_list = [], [], []
    for i in range(k):
        A, B = continuous_matrices(psi, foot_pos[i], x0[0:3], mass, I_body)
        if Ed is None:
            Ad, Bd = discretize(A, B, dt_mpc)
        else:
            Ad, Bd, Edd = discretize(A, B, dt_mpc, Ed)
            Ed_list.append(Edd)
        Ad_list.append(Ad)
        Bd_list.append(Bd)

    X_ref = np.asarray(x_ref_traj, float).ravel()
    H, g, C, lo, hi, A_qp, B_qp = build_qp(
        x0, X_ref, Ad_list, Bd_list, contact_schedule, Q=Q, mu=mu,
        Ed_list=(Ed_list if Ed is not None else None), theta=theta)
    U, status = solve_qp(H, g, C, lo, hi, warm)
    return U[:N_INPUT], U, status, (Ad_list, Bd_list, A_qp, B_qp,
                                    Ed_list if Ed is not None else None)
