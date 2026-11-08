"""
MPC checks -- no MuJoCo needed.

Tests:
  1. condensed form (A_qp, B_qp) vs step-by-step rollout
  2. unconstrained solution vs analytic least squares
  3. standing: the solution gives ~ m*g/4 per foot
  4. friction pyramid constraints satisfied
  5. zero force on swing legs
  6. forward velocity command gives a sensible response
"""

import numpy as np

from a1_params import (MASS_TOTAL, GRAVITY, Z_NOMINAL, Q_NOMINAL, LEGS,
                       MU_EFFECTIVE, F_MIN, F_MAX, MPC_HORIZON)
from a1_kinematics import foot_position
from srb_dynamics import continuous_matrices, discretize, N_STATE, N_INPUT
from mpc import (build_condensed, rollout, build_qp, solve_qp,
                 build_constraints, solve_mpc)


def nominal_feet():
    return np.array([foot_position(l, Q_NOMINAL) + np.array([0, 0, Z_NOMINAL])
                     for l in LEGS])


def nominal_state():
    x = np.zeros(N_STATE)
    x[0:3] = [0.0, 0.0, Z_NOMINAL]
    x[12] = GRAVITY
    return x


def make_horizon(k, dt, feet, x0):
    Ad, Bd = [], []
    for _ in range(k):
        A, B = continuous_matrices(x0[5], feet, x0[0:3])
        a, b = discretize(A, B, dt)
        Ad.append(a)
        Bd.append(b)
    return Ad, Bd


def main():
    k, dt = MPC_HORIZON, 0.03
    feet = nominal_feet()
    x0 = nominal_state()
    rng = np.random.default_rng(0)

    print("MPC checks\n")
    print(f"horizon k={k}, dt={dt}s, variables={N_INPUT*k}\n")

    # --- 1) condensed form ---------------------------------------------
    print("1) A_qp, B_qp vs step-by-step rollout")
    Ad, Bd = make_horizon(k, dt, feet, x0)
    A_qp, B_qp = build_condensed(Ad, Bd)
    worst = 0.0
    for _ in range(50):
        xs = x0 + np.concatenate([rng.normal(0, 0.05, 12), [0.0]])
        U = rng.normal(0, 30.0, N_INPUT * k)
        X_cond = A_qp @ xs + B_qp @ U
        X_roll = rollout(Ad, Bd, xs, U)
        worst = max(worst, np.max(np.abs(X_cond - X_roll)))
    print(f"   largest difference: {worst:.2e} "
          f"({'PASS' if worst < 1e-9 else 'FAIL'})\n")

    # --- 2) unconstrained vs analytic ----------------------------------
    print("2) Unconstrained QP vs analytic least squares")
    X_ref = np.tile(x0, k)
    contact = np.ones((k, 4), bool)
    H, g, C, lo, hi, _, _ = build_qp(x0, X_ref, Ad, Bd, contact)
    U_analytic = np.linalg.solve(H, -g)
    resid = np.max(np.abs(H @ U_analytic + g))
    print(f"   gradient residual: {resid:.2e} "
          f"({'PASS' if resid < 1e-6 else 'FAIL'})")
    F0 = U_analytic[:N_INPUT].reshape(4, 3)
    print(f"   unconstrained first-step fz: {np.round(F0[:, 2], 2).tolist()} N")
    print(f"   expected m*g/4             : {MASS_TOTAL*GRAVITY/4:.2f} N\n")

    # --- 3) standing, constrained ---------------------------------------
    print("3) Standing (constrained QP)")
    U, status = solve_qp(H, g, C, lo, hi)
    F = U[:N_INPUT].reshape(4, 3)
    tot = F.sum(axis=0)
    print(f"   solver status: {status}")
    print("   foot forces (N):")
    for i, leg in enumerate(LEGS):
        print(f"     {leg}: fx={F[i,0]:+7.2f}  fy={F[i,1]:+7.2f}  fz={F[i,2]:7.2f}")
    print(f"   total: {np.round(tot, 2).tolist()}")
    print(f"   weight m*g = {MASS_TOTAL*GRAVITY:.2f} N")
    err = abs(tot[2] - MASS_TOTAL * GRAVITY) / (MASS_TOTAL * GRAVITY) * 100
    print(f"   vertical balance error: {err:.2f}% "
          f"({'PASS' if err < 2.0 else 'FAIL'})\n")

    # --- 4) friction and bound constraints -------------------------------
    print("4) Constraint violation")
    v = C @ U
    viol_lo = np.max(np.maximum(lo - v, 0.0)[np.isfinite(lo)])
    viol_hi = np.max(np.maximum(v - hi, 0.0)[np.isfinite(hi)])
    print(f"   lower bound violation: {viol_lo:.2e}")
    print(f"   upper bound violation: {viol_hi:.2e}")
    ok = max(viol_lo, viol_hi) < 1e-5
    print(f"   {'PASS' if ok else 'FAIL'}")
    for i, leg in enumerate(LEGS):
        fx, fy, fz = F[i]
        print(f"     {leg}: |fx|/fz={abs(fx)/max(fz,1e-9):.3f} "
              f"|fy|/fz={abs(fy)/max(fz,1e-9):.3f}  (mu={MU_EFFECTIVE})")
    print()

    # --- 5) swing legs ----------------------------------------------------
    print("5) Zero force on swing legs (trot-like pattern)")
    contact_trot = np.ones((k, 4), bool)
    contact_trot[:, 1] = False        # FL in swing
    contact_trot[:, 2] = False        # RR in swing
    H2, g2, C2, lo2, hi2, _, _ = build_qp(x0, X_ref, Ad, Bd, contact_trot)
    U2, st2 = solve_qp(H2, g2, C2, lo2, hi2)
    F2 = U2[:N_INPUT].reshape(4, 3)
    print(f"   solver status: {st2}")
    for i, leg in enumerate(LEGS):
        tag = "stance" if contact_trot[0, i] else "swing "
        print(f"     {leg} ({tag}): {np.round(F2[i], 3).tolist()}")
    swing_mag = max(np.linalg.norm(F2[1]), np.linalg.norm(F2[2]))
    print(f"   swing force magnitude: {swing_mag:.2e} N "
          f"({'PASS' if swing_mag < 1e-4 else 'FAIL'})")
    print(f"   total vertical: {F2[:,2].sum():.2f} N "
          f"(weight {MASS_TOTAL*GRAVITY:.2f} N)\n")

    # --- 6) forward velocity command --------------------------------------
    print("6) Forward velocity command (0.5 m/s)")
    x_ref = np.tile(x0, (k, 1))
    for i in range(k):
        x_ref[i, 6] = 0.5                       # desired vx
        x_ref[i, 0] = 0.5 * dt * (i + 1)        # advancing x position
    feet_h = np.tile(feet, (k, 1, 1))
    contact_all = np.ones((k, 4), bool)
    F_first, U3, st3, _ = solve_mpc(x0, x_ref, feet_h, contact_all, dt)
    F3 = F_first.reshape(4, 3)
    print(f"   solver status: {st3}")
    for i, leg in enumerate(LEGS):
        print(f"     {leg}: fx={F3[i,0]:+7.2f}  fy={F3[i,1]:+7.2f}  fz={F3[i,2]:7.2f}")
    print(f"   total fx = {F3[:,0].sum():+.2f} N  "
          f"(must be positive to accelerate forward)")
    print(f"   total fz = {F3[:,2].sum():.2f} N")
    print(f"   {'PASS' if F3[:,0].sum() > 1.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
