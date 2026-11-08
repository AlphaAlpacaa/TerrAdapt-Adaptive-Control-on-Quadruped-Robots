"""
A1 leg kinematics: forward kinematics and analytic Jacobian.

No library is used; the chain is derived by hand. Body tree read from the
model:

    trunk
      +-- {leg}_hip     pos = [+-0.183, +-0.047, 0]   joint axis x  (abduction)
            +-- {leg}_thigh  pos = [0, +-0.085, 0]    joint axis y  (thigh)
                  +-- {leg}_calf  pos = [0, 0, -0.2]  joint axis y  (knee)
                        +-- foot sphere  pos = [0, 0, -0.2],  r = 0.02

Foot position in the body frame:

    p = p_hip + R_x(q1) * ( [0, s*d, 0]
                          + R_y(q2)      * [0, 0, -l1]
                          + R_y(q2+q3)   * [0, 0, -l2] )

Expanded (c2 = cos q2, c23 = cos(q2+q3), L = l1*c2 + l2*c23):

    px = hx - l1*s2 - l2*s23
    py = hy + s*d*c1 + L*s1
    pz =      s*d*s1 - L*c1

Check: nominal stance (q = [0, 0.9, -1.8], FR) gives
p = [0.183, -0.132, -0.24864]; with the 0.27 m body height the world z is
0.02136. MuJoCo gives [0.183, -0.132, 0.0214].
"""

import numpy as np
from a1_params import (HIP_POS, LEG_SIGN, D_ABD, L_THIGH, L_CALF,
                       Z_NOMINAL, Q_NOMINAL, LEGS)


def foot_position(leg, q):
    """
    Foot centre position in the body frame.
    leg : "FR" | "FL" | "RR" | "RL"
    q   : [q1 abduction, q2 thigh, q3 knee]  (rad)
    """
    hx, hy, _ = HIP_POS[leg]
    s = LEG_SIGN[leg]
    q1, q2, q3 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s23, c23 = np.sin(q2 + q3), np.cos(q2 + q3)

    L = L_THIGH * c2 + L_CALF * c23           # vertical extent in the leg plane
    return np.array([
        hx - L_THIGH * s2 - L_CALF * s23,
        hy + s * D_ABD * c1 + L * s1,
        s * D_ABD * s1 - L * c1,
    ])


def foot_jacobian(leg, q):
    """
    Analytic Jacobian J = d p_foot / d q (3x3), body frame.
    Rows [px, py, pz], columns [q1, q2, q3]; partial derivatives of the
    closed form above.
    """
    s = LEG_SIGN[leg]
    q1, q2, q3 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s23, c23 = np.sin(q2 + q3), np.cos(q2 + q3)

    L = L_THIGH * c2 + L_CALF * c23
    dL_dq2 = -L_THIGH * s2 - L_CALF * s23     # dL/dq2
    dL_dq3 = -L_CALF * s23                    # dL/dq3

    J = np.zeros((3, 3))

    # px = hx - l1*s2 - l2*s23      (independent of q1)
    J[0, 0] = 0.0
    J[0, 1] = -L_THIGH * c2 - L_CALF * c23
    J[0, 2] = -L_CALF * c23

    # py = hy + s*d*c1 + L*s1
    J[1, 0] = -s * D_ABD * s1 + L * c1
    J[1, 1] = dL_dq2 * s1
    J[1, 2] = dL_dq3 * s1

    # pz = s*d*s1 - L*c1
    J[2, 0] = s * D_ABD * c1 + L * s1
    J[2, 1] = -dL_dq2 * c1
    J[2, 2] = -dL_dq3 * c1

    return J


def numeric_jacobian(leg, q, eps=1e-7):
    """Finite-difference Jacobian -- to check the analytic one."""
    J = np.zeros((3, 3))
    for i in range(3):
        up, dn = np.array(q, float), np.array(q, float)
        up[i] += eps
        dn[i] -= eps
        J[:, i] = (foot_position(leg, up) - foot_position(leg, dn)) / (2 * eps)
    return J


def inverse_kinematics(leg, p_target, q_guess=None, iters=100, tol=1e-12,
                       max_step=0.3):
    """
    Damped Newton-Raphson inverse kinematics with step limiting, using the
    analytic Jacobian. Warm-started from the previous solution, so it
    converges quickly. The chain has multiple solutions (knee forward or
    backward); Newton goes to the nearest one, which is the desired
    behaviour for locomotion.
    """
    q = np.array(Q_NOMINAL if q_guess is None else q_guess, float)
    p_target = np.asarray(p_target, float)
    lam = 1e-8
    err_prev = np.inf
    for _ in range(iters):
        err_vec = p_target - foot_position(leg, q)
        err = np.linalg.norm(err_vec)
        if err < tol:
            break
        # increase damping when not converging (Levenberg-Marquardt style)
        lam = lam * 0.5 if err < err_prev else min(lam * 10.0, 1e-2)
        err_prev = err
        J = foot_jacobian(leg, q)
        dq = np.linalg.solve(J.T @ J + lam * np.eye(3), J.T @ err_vec)
        n = np.linalg.norm(dq)
        if n > max_step:                      # step limit
            dq *= max_step / n
        q = q + dq
    return q


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    print("A1 kinematics checks\n")

    # 1) nominal stance vs MuJoCo foot positions
    #    (world = body + [0, 0, Z_NOMINAL]; the body is level)
    mujoco_feet = {
        "FR": np.array([ 0.183, -0.132, 0.0214]),
        "FL": np.array([ 0.183,  0.132, 0.0214]),
        "RR": np.array([-0.183, -0.132, 0.0214]),
        "RL": np.array([-0.183,  0.132, 0.0214]),
    }
    print("1) Nominal stance, analytic FK vs MuJoCo")
    worst = 0.0
    for leg in LEGS:
        p_body = foot_position(leg, Q_NOMINAL)
        p_world = p_body + np.array([0.0, 0.0, Z_NOMINAL])
        d = np.max(np.abs(p_world - mujoco_feet[leg]))
        worst = max(worst, d)
        print(f"   {leg}: analytic={np.round(p_world,5).tolist()}  "
              f"mujoco={mujoco_feet[leg].tolist()}  diff={d:.2e}")
    print(f"   largest difference: {worst:.2e} m "
          f"({'PASS' if worst < 1e-4 else 'FAIL'})\n")

    # 2) analytic Jacobian vs finite differences
    print("2) Analytic Jacobian vs finite differences (200 random configurations)")
    rng = np.random.default_rng(0)
    from a1_params import Q_MIN, Q_MAX
    worst = 0.0
    for _ in range(200):
        leg = LEGS[rng.integers(0, 4)]
        q = rng.uniform(Q_MIN, Q_MAX)
        Ja, Jn = foot_jacobian(leg, q), numeric_jacobian(leg, q)
        rel = np.max(np.abs(Ja - Jn)) / max(np.max(np.abs(Jn)), 1e-9)
        worst = max(worst, rel)
    print(f"   largest relative error: {worst:.2e} "
          f"({'PASS' if worst < 1e-5 else 'FAIL'})\n")

    # 3) inverse kinematics: FK -> IK -> FK, in the locomotion workspace
    #    (abduction +-0.35, thigh 0.3..1.7, knee -2.5..-0.95)
    print("3) Inverse kinematics loop (500 targets, locomotion workspace)")
    lo = np.array([-0.35, 0.30, -2.50])
    hi = np.array([ 0.35, 1.70, -0.95])
    worst, fails = 0.0, 0
    for _ in range(500):
        leg = LEGS[rng.integers(0, 4)]
        q_true = rng.uniform(lo, hi)
        p = foot_position(leg, q_true)
        q_sol = inverse_kinematics(leg, p)          # no warm start: hardest case
        d = np.linalg.norm(foot_position(leg, q_sol) - p)
        worst = max(worst, d)
        fails += (d > 1e-8)
    print(f"   largest position error: {worst:.2e} m, failures: {fails}/500 "
          f"({'PASS' if fails == 0 else 'FAIL'})")

    # 3b) warm-started, as used along a trajectory
    print("3b) Warm-started IK (along a foot trajectory)")
    q = np.array(Q_NOMINAL, float)
    worst = 0.0
    for i in range(400):
        th = 2 * np.pi * i / 400.0
        # 6 cm circle around the nominal foot position
        p = foot_position("FR", Q_NOMINAL) + np.array(
            [0.06 * np.cos(th), 0.0, 0.06 * np.sin(th)])
        q = inverse_kinematics("FR", p, q_guess=q)
        worst = max(worst, np.linalg.norm(foot_position("FR", q) - p))
    print(f"   largest position error: {worst:.2e} m "
          f"({'PASS' if worst < 1e-9 else 'FAIL'})\n")

    # 4) conditioning near the nominal stance
    print("4) Jacobian condition number (around the nominal stance)")
    for leg in ["FR"]:
        J = foot_jacobian(leg, Q_NOMINAL)
        print(f"   {leg} nominal: det={np.linalg.det(J):+.5f}  "
              f"cond={np.linalg.cond(J):.1f}")
        # approaching the singularity as the knee straightens
        for q3 in [-2.6, -2.0, -1.5, -1.0, -0.95]:
            J = foot_jacobian(leg, [0.0, 0.9, q3])
            print(f"     knee={q3:+.2f}: det={np.linalg.det(J):+.6f}  "
                  f"cond={np.linalg.cond(J):8.1f}")


# ---------------------------------------------------------------------------
# Leg gravity compensation
# ---------------------------------------------------------------------------
#
# With PD only, the swing leg cannot carry its own weight (1.935 kg, ~19 N;
# kp=180 at 50 mm error gives only 9 N) and drags. The gravity term of the
# inverse dynamics is added as feed-forward (G_i in Di Carlo Eq. 2):
#
#   U(q)   = -sum_i m_i * g^T p_i(q)          potential energy
#   tau_j  = dU/dq_j = -sum_i m_i * g^T (dp_i/dq_j)
#
# Link COM positions from the chain, Jacobians by finite differences.

def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def link_com_positions(leg, q, com_offsets=None):
    """
    Centres of mass of the three links (hip, thigh, calf), body frame.

    com_offsets : {"hip":(3,), "thigh":(3,), "calf":(3,)} COM offset of each
                  link in its own frame. None assumes the link midpoint
                  (approximate); run_walk.py passes the values read from
                  the MuJoCo model (body_ipos).
    """
    if com_offsets is None:
        com_offsets = {"hip": np.zeros(3),
                       "thigh": np.array([0.0, 0.0, -0.5 * L_THIGH]),
                       "calf": np.array([0.0, 0.0, -0.5 * L_CALF])}

    hx, hy, hz = HIP_POS[leg]
    s = LEG_SIGN[leg]
    q1, q2, q3 = q

    R1 = _rot_x(q1)
    R2 = R1 @ _rot_y(q2)
    R3 = R1 @ _rot_y(q2 + q3)

    o_hip = np.array([hx, hy, hz])
    o_thigh = o_hip + R1 @ np.array([0.0, s * D_ABD, 0.0])
    o_calf = o_thigh + R2 @ np.array([0.0, 0.0, -L_THIGH])

    return [
        o_hip + R1 @ np.asarray(com_offsets["hip"], float),
        o_thigh + R2 @ np.asarray(com_offsets["thigh"], float),
        o_calf + R3 @ np.asarray(com_offsets["calf"], float),
    ]


def leg_gravity_torque(leg, q, R_world, masses, com_offsets=None,
                       gravity=9.81, eps=1e-6):
    """
    Joint torque (3,) needed to hold the leg's own weight.

    R_world : body -> world rotation (to express gravity in the body frame)
    masses  : (m_hip, m_thigh, m_calf)
    """
    g_body = R_world.T @ np.array([0.0, 0.0, -gravity])
    tau = np.zeros(3)
    q = np.asarray(q, float)

    for j in range(3):
        up, dn = q.copy(), q.copy()
        up[j] += eps
        dn[j] -= eps
        p_up = link_com_positions(leg, up, com_offsets)
        p_dn = link_com_positions(leg, dn, com_offsets)
        for i, m_i in enumerate(masses):
            dpi = (p_up[i] - p_dn[i]) / (2.0 * eps)
            tau[j] -= m_i * float(g_body @ dpi)
    return tau


def link_rotations(leg, q):
    """
    Rotation of each link relative to the body (link -> body):
    hip = R_x(q1), thigh = R_x(q1) R_y(q2), calf = R_x(q1) R_y(q2+q3).
    """
    q1, q2, q3 = q
    R1 = _rot_x(q1)
    return [R1, R1 @ _rot_y(q2), R1 @ _rot_y(q2 + q3)]


def link_angular_jacobians(leg, q):
    """
    Angular-velocity Jacobian Jw_i of each link, omega_i = Jw_i qdot
    (body frame, 3x3, columns [q1, q2, q3]). Joint axes: q1 along e_x,
    q2 and q3 along R_x(q1) e_y:
      hip   : [e_x, 0,        0      ]
      thigh : [e_x, R1 e_y,   0      ]
      calf  : [e_x, R1 e_y,   R1 e_y ]
    """
    q1 = q[0]
    ex = np.array([1.0, 0.0, 0.0])
    ey_rot = _rot_x(q1) @ np.array([0.0, 1.0, 0.0])
    Jw_hip = np.column_stack([ex, np.zeros(3), np.zeros(3)])
    Jw_thigh = np.column_stack([ex, ey_rot, np.zeros(3)])
    Jw_calf = np.column_stack([ex, ey_rot, ey_rot])
    return [Jw_hip, Jw_thigh, Jw_calf]


def link_com_jacobians(leg, q, com_offsets=None, eps=1e-6):
    """
    COM Jacobian of each link, Jc_i = d p_ci / d q (3x3), by finite
    differences (same derivative as leg_gravity_torque).
    """
    q = np.asarray(q, float)
    Js = [np.zeros((3, 3)) for _ in range(3)]
    for j in range(3):
        up, dn = q.copy(), q.copy()
        up[j] += eps
        dn[j] -= eps
        p_up = link_com_positions(leg, up, com_offsets)
        p_dn = link_com_positions(leg, dn, com_offsets)
        for i in range(3):
            Js[i][:, j] = (p_up[i] - p_dn[i]) / (2.0 * eps)
    return Js


def leg_mass_matrix(leg, q, masses, inertias=None, com_offsets=None,
                    armature=None):
    """
    Joint-space mass matrix M(q) of the leg (3x3):

        M(q) = sum_i [ m_i Jc_i^T Jc_i + Jw_i^T I_i(q) Jw_i ],
        I_i(q) = R_i I_i^link R_i^T

    from the kinetic energy T_i = 1/2 m_i ||v_ci||^2 + 1/2 omega_i^T I_i omega_i.

    inertias : {"hip", "thigh", "calf"} -> (3,3) inertia of each link about
               its own COM in its own frame. None treats the links as point
               masses (approximate).
    armature : (3,) reflected rotor inertia (MuJoCo dof_armature), added as
               a constant diagonal: M_true(q) = M(q) + diag(armature). A
               rotor geared by n contributes 1/2 (n^2 I_r) qdot^2, which is
               configuration independent. On the A1 it is not negligible:
               armature = 0.01 kg m^2 against a diagonal of
               [0.020, 0.019, 0.007] makes the knee 2.4x heavier and
               Lambda_zz 3x larger.

    The inertial coupling from trunk motion is neglected: the body
    acceleration (< 1 m/s^2) is a few percent of what the foot trajectory
    demands.
    """
    Jc = link_com_jacobians(leg, q, com_offsets)
    Jw = link_angular_jacobians(leg, q)
    R = link_rotations(leg, q)
    keys = ["hip", "thigh", "calf"]

    M = np.zeros((3, 3))
    for i, m_i in enumerate(masses):
        M += m_i * (Jc[i].T @ Jc[i])
        if inertias is not None:
            I_link = np.asarray(inertias[keys[i]], float)
            I_body = R[i] @ I_link @ R[i].T
            M += Jw[i].T @ I_body @ Jw[i]
    if armature is not None:
        M = M + np.diag(np.asarray(armature, float))
    return M


def joint_friction_torque(dq, damping=None, frictionloss=None, eps=0.05):
    """
    Feed-forward torque (3,) that cancels joint friction. MuJoCo joints
    have two loss terms:

        viscous  : tau = -damping * qdot                 (dof_damping)
        Coulomb  : tau = -frictionloss * sign(qdot)      (dof_frictionloss)

    Both are compensated with the opposite sign. On the A1 damping is
    [1.0, 2.0, 2.0] N m s/rad: at the swing peak the thigh turns at
    1.81 rad/s -> 3.62 Nm out of a total swing torque of 4.9 Nm.

    tanh(qdot/eps) replaces sign() to avoid chattering around zero speed
    (eps = 0.05 rad/s). The coefficients are read exactly from the model
    here (a perfect-model assumption).
    """
    dq = np.asarray(dq, float)
    tau = np.zeros(3)
    if damping is not None:
        tau = tau + np.asarray(damping, float) * dq
    if frictionloss is not None:
        tau = tau + np.asarray(frictionloss, float) * np.tanh(dq / eps)
    return tau


def jacobian_dot(leg, q, dq, eps=1e-6):
    """
    Jdot = dJ/dt = sum_k (dJ/dq_k) qdot_k (3x3), by finite differences of
    the analytic Jacobian; used in a_foot = J qddot + Jdot qdot.
    """
    q = np.asarray(q, float)
    dq = np.asarray(dq, float)
    Jd = np.zeros((3, 3))
    for k in range(3):
        up, dn = q.copy(), q.copy()
        up[k] += eps
        dn[k] -= eps
        dJ = (foot_jacobian(leg, up) - foot_jacobian(leg, dn)) / (2.0 * eps)
        Jd += dJ * dq[k]
    return Jd


def swing_feedforward_torque(leg, q, dq, a_body, masses, inertias=None,
                             com_offsets=None, lam=1e-6, armature=None):
    """
    Acceleration feed-forward for the swing leg (3,) Nm.

    Di Carlo et al., IROS 2018, Eq. (2):
        tau_i = J(q_i)^T [ Lambda_i (a_ref - Jdot qdot) + ... ] + G_i
    For a square, non-singular J this equals, in joint space,
        qddot_ref = J^{-1} (a_body - Jdot qdot),  tau_ff = M(q) qddot_ref,
    which avoids inverting Lambda = (J M^{-1} J^T)^{-1}.

    a_body   : desired foot acceleration, BODY frame (m/s^2)
    lam      : damping for the solve near the straight-knee singularity
    armature : rotor inertia added to M(q); required on the A1.
    Gravity (leg_gravity_torque) and friction (joint_friction_torque) are
    separate; Coriolis is neglected (~0.1 Nm vs ~2 Nm at trot speeds).
    """
    J = foot_jacobian(leg, q)
    Jd = jacobian_dot(leg, q, dq)
    rhs = np.asarray(a_body, float) - Jd @ np.asarray(dq, float)
    # damped solve: do not blow up the torque near the singularity
    qddot = np.linalg.solve(J.T @ J + lam * np.eye(3), J.T @ rhs)
    M = leg_mass_matrix(leg, q, masses, inertias, com_offsets, armature)
    return M @ qddot


def _gravity_self_test():
    """Consistency check of the gravity compensation."""
    print("\n5) Leg gravity compensation")
    masses = (0.696, 1.013, 0.226)
    R = np.eye(3)

    # chain consistency: calf COM + R3 @ [0,0,-l2 + offset] gives the foot
    q = np.array(Q_NOMINAL, float)
    coms = link_com_positions("FR", q)
    p_foot = foot_position("FR", q)
    o_calf = coms[2] - _rot_x(q[0]) @ _rot_y(q[1] + q[2]) @ \
             np.array([0.0, 0.0, -0.5 * L_CALF])
    p_from_chain = o_calf + _rot_x(q[0]) @ _rot_y(q[1] + q[2]) @ \
                   np.array([0.0, 0.0, -L_CALF])
    err = np.max(np.abs(p_from_chain - p_foot))
    print(f"   chain consistency (COM -> foot): {err:.2e} "
          f"({'PASS' if err < 1e-12 else 'FAIL'})")

    tau = leg_gravity_torque("FR", q, R, masses)
    print(f"   gravity torque in the nominal stance: {np.round(tau, 3).tolist()} Nm")
    total_w = sum(masses) * 9.81
    print(f"   leg weight: {total_w:.2f} N")

    # leg hanging straight down: pitch joint torques ~0
    q_hang = np.array([0.0, 0.0, 0.0])
    tau_h = leg_gravity_torque("FR", q_hang, R, masses)
    print(f"   torque with the leg hanging straight: {np.round(tau_h, 4).tolist()} Nm "
          f"({'PASS' if np.max(np.abs(tau_h[1:])) < 1e-9 else 'FAIL'})")
    print("   (with q2 = q3 = 0 the leg is vertical, zero pitch-joint torque)")


# ---------------------------------------------------------------------------
# Leg dynamics checks (mass matrix, Jdot, feed-forward)
# ---------------------------------------------------------------------------

def _skew_to_vec(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def _dynamics_self_test():
    print("\n6) Leg mass matrix M(q)")
    rng = np.random.default_rng(1)
    masses = (0.696, 1.013, 0.226)
    # representative inertias (order of the A1 MJCF); run_walk.py passes
    # the values read from body_inertia/body_iquat
    inertias = {
        "hip":   np.diag([0.000469, 0.000808, 0.000552]),
        "thigh": np.diag([0.005529, 0.005139, 0.001367]),
        "calf":  np.diag([0.002997, 0.003014, 0.000032]),
    }
    coms = {"hip": np.array([-0.003311, 0.000635, 0.000031]),
            "thigh": np.array([-0.003237, 0.022327, -0.027326]),
            "calf": np.array([0.006435, 0.0, -0.107388])}

    # 6a) symmetry and positive definiteness
    worst_asym, min_eig = 0.0, np.inf
    for _ in range(300):
        leg = LEGS[rng.integers(0, 4)]
        q = rng.uniform([-0.6, 0.0, -2.6], [0.6, 2.0, -0.6])
        M = leg_mass_matrix(leg, q, masses, inertias, coms)
        worst_asym = max(worst_asym, np.max(np.abs(M - M.T)))
        min_eig = min(min_eig, np.linalg.eigvalsh(M).min())
    print(f"   asymmetry          : {worst_asym:.2e} "
          f"({'PASS' if worst_asym < 1e-9 else 'FAIL'})")
    print(f"   smallest eigenvalue: {min_eig:.3e} "
          f"({'PASS' if min_eig > 0 else 'FAIL'})")

    # 6b) independent check with the kinetic energy T = 1/2 qdot^T M qdot.
    #     The left side is built WITHOUT the Jacobians, from finite-difference
    #     velocities (v_ci from positions, omega = vec(Rdot R^T)), which
    #     checks Jw independently.
    worst_rel = 0.0
    h = 1e-6
    for _ in range(200):
        leg = LEGS[rng.integers(0, 4)]
        q = rng.uniform([-0.6, 0.1, -2.5], [0.6, 1.9, -0.7])
        dq = rng.normal(0.0, 2.0, 3)

        p_up = link_com_positions(leg, q + h * dq, coms)
        p_dn = link_com_positions(leg, q - h * dq, coms)
        R_up = link_rotations(leg, q + h * dq)
        R_dn = link_rotations(leg, q - h * dq)
        R_mid = link_rotations(leg, q)

        T_num = 0.0
        keys = ["hip", "thigh", "calf"]
        for i, m_i in enumerate(masses):
            v = (p_up[i] - p_dn[i]) / (2 * h)
            Rdot = (R_up[i] - R_dn[i]) / (2 * h)
            w = _skew_to_vec(Rdot @ R_mid[i].T)
            I_body = R_mid[i] @ inertias[keys[i]] @ R_mid[i].T
            T_num += 0.5 * m_i * float(v @ v) + 0.5 * float(w @ I_body @ w)

        M = leg_mass_matrix(leg, q, masses, inertias, coms)
        T_mat = 0.5 * float(dq @ M @ dq)
        worst_rel = max(worst_rel, abs(T_num - T_mat) / max(T_num, 1e-12))
    print(f"   kinetic energy relative error (200 samples): {worst_rel:.2e} "
          f"({'PASS' if worst_rel < 1e-6 else 'FAIL'})")

    M_nom = leg_mass_matrix("FR", Q_NOMINAL, masses, inertias, coms)
    print(f"   nominal stance M(q) diagonal: "
          f"{np.round(np.diag(M_nom), 5).tolist()} kg m^2")

    # 6c) Jdot identity: a_foot = J qddot + Jdot qdot
    print("\n7) Jdot and the acceleration identity")
    worst = 0.0
    for _ in range(200):
        leg = LEGS[rng.integers(0, 4)]
        q0 = rng.uniform([-0.6, 0.1, -2.5], [0.6, 1.9, -0.7])
        dq = rng.normal(0.0, 2.0, 3)
        ddq = rng.normal(0.0, 10.0, 3)
        hh = 1e-5

        def pos(t):
            return foot_position(leg, q0 + dq * t + 0.5 * ddq * t * t)

        a_num = (pos(hh) - 2 * pos(0.0) + pos(-hh)) / (hh * hh)
        a_ana = foot_jacobian(leg, q0) @ ddq + jacobian_dot(leg, q0, dq) @ dq
        worst = max(worst, np.max(np.abs(a_num - a_ana)) /
                    max(np.max(np.abs(a_num)), 1e-9))
    print(f"   largest relative error: {worst:.2e} "
          f"({'PASS' if worst < 1e-4 else 'FAIL'})")

    # 6d) feed-forward reproduces the desired acceleration
    print("\n8) Swing feed-forward")
    q = np.array(Q_NOMINAL, float)
    dq = np.zeros(3)
    a_des = np.array([0.0, 0.0, 25.6])          # vertical acceleration at the apex
    tau_ff = swing_feedforward_torque("FR", q, dq, a_des, masses,
                                      inertias, coms)
    print(f"   a_des = {a_des.tolist()} m/s^2 ({np.linalg.norm(a_des)/9.81:.1f}g)")
    print(f"   tau_ff = {np.round(tau_ff, 3).tolist()} Nm "
          f"(|tau| = {np.linalg.norm(tau_ff):.2f}, limit {33.5})")
    tau_g = leg_gravity_torque("FR", q, np.eye(3), masses, coms)
    print(f"   tau_g  = {np.round(tau_g, 3).tolist()} Nm  (for comparison)")

    # closed loop: tau_ff -> qddot -> a_foot must return a_des
    M = leg_mass_matrix("FR", q, masses, inertias, coms)
    qdd = np.linalg.solve(M, tau_ff)
    a_back = foot_jacobian("FR", q) @ qdd + jacobian_dot("FR", q, dq) @ dq
    rel = np.max(np.abs(a_back - a_des)) / np.linalg.norm(a_des)
    print(f"   round-trip relative error: {rel:.2e} "
          f"({'PASS' if rel < 1e-4 else 'FAIL'})")
    print("   (the remaining deviation comes from the lam=1e-6 damping)")

    # 8b) operational-space mass Lambda = (J M^-1 J^T)^-1
    J_nom = foot_jacobian("FR", q)
    ARM = np.array([0.01, 0.01, 0.01])      # A1 model (dof_armature)
    DAMP = np.array([1.0, 2.0, 2.0])        # dof_damping
    FRIC = np.array([0.2, 0.2, 0.2])        # dof_frictionloss
    M_arm = leg_mass_matrix("FR", q, masses, inertias, coms, ARM)
    Lam = np.linalg.inv(J_nom @ np.linalg.inv(M) @ J_nom.T)
    Lam_a = np.linalg.inv(J_nom @ np.linalg.inv(M_arm) @ J_nom.T)
    print(f"\n   Lambda diagonal, without armature : "
          f"{np.round(np.diag(Lam), 4).tolist()} kg")
    print(f"   Lambda diagonal, with armature    : "
          f"{np.round(np.diag(Lam_a), 4).tolist()} kg "
          f"({Lam_a[2,2]/Lam[2,2]:.2f}x)")

    print("\n9) Joint friction compensation")
    print(f"   {'qdot (rad/s)':>14} {'viscous (Nm)':>26} {'Coulomb (Nm)':>26}")
    print("   " + "-" * 68)
    for v in [0.5, 1.81, 4.0]:
        dqv = np.array([0.0, -v, 0.0])
        tv = joint_friction_torque(dqv, DAMP, None)
        tc = joint_friction_torque(dqv, None, FRIC)
        print(f"   {-v:>14.2f} {str(np.round(tv,3).tolist()):>26} "
              f"{str(np.round(tc,3).tolist()):>26}")
    print("   thigh at the swing peak ~1.81 rad/s -> 3.62 Nm viscous torque,")
    print("   out of 4.9 Nm total swing torque (MuJoCo measurement).")
    dq0 = np.zeros(3)
    t0 = joint_friction_torque(dq0, DAMP, FRIC)
    print(f"   compensation at zero speed: {np.round(t0,6).tolist()} Nm "
          f"({'PASS' if np.allclose(t0, 0) else 'FAIL'} -- tanh is continuous)")


if __name__ == "__main__":
    self_test()
    _gravity_self_test()
    _dynamics_self_test()
