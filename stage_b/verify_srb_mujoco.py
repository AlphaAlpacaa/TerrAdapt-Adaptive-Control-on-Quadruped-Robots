"""
SRB model vs MuJoCo's multibody physics.

The SRB model rests on three assumptions (Di Carlo 2018):
  1. negligible leg inertia      <-- legs are 62% of the A1's mass
  2. small roll/pitch -> R ~= R_z(psi)
  3. small angular velocity -> omega x (I omega) negligible
This script measures the deviation numerically.

Pitfalls found and fixed in this script:
1. mj_forward does not fill d.subtree_linvel / d.subtree_angmom;
   mj_subtreeVel(m, d) must be called.
2. d.qvel[3:6] of the free joint is the angular velocity in the BODY
   frame; in the world frame w_world = R @ w_body.
3. Settling the robot in the air with gravity on lets it fall through the
   floor and corrupts the contact forces; settling is done with gravity OFF.
4. d.qfrc_applied persists and must be cleared between tests.

Theta_dot = R_z^T omega was also checked analytically (Z-Y-X Euler,
R_dot = [omega]x R, zero roll/pitch):
    ||omega - R_z   Theta_dot|| = 1.4e-07
    ||omega - R_z^T Theta_dot|| = 3.3
i.e. omega = R_z Theta_dot  =>  Theta_dot = R_z^T omega (Di Carlo).
"""

import numpy as np
from pathlib import Path
import mujoco

from a1_params import MASS_TOTAL, GRAVITY, LEGS, Q_NOMINAL
from srb_dynamics import continuous_derivative, rot_z, world_inertia, N_STATE

BASE_HEIGHT = 1.0          # robot held in the air during the measurement


def find_scene():
    here = Path(__file__).resolve().parent
    for base in [here] + list(here.parents)[:4]:
        hit = list(base.glob("**/unitree_a1/scene_torque.xml"))
        if hit:
            return str(hit[0])
    raise FileNotFoundError("scene_torque.xml missing -- run make_torque_model.py first")


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def euler_zyx(q):
    w, x, y, z = q
    return np.array([
        np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)),
        np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0)),
        np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)),
    ])


def lock_torques(m, d, gain=400.0, damp=15.0):
    q, v = d.qpos[7:], d.qvel[6:]
    tau = gain * (np.tile(Q_NOMINAL, 4) - q) - damp * v
    lim = float(m.actuator_ctrlrange[0][1])
    return np.clip(tau, -lim, lim)


def settle(m, d, n=800):
    """Settle the legs in the nominal stance with gravity OFF."""
    g_save = m.opt.gravity.copy()
    m.opt.gravity[:] = 0.0
    d.qfrc_applied[:] = 0.0
    if m.nkey > 0:
        d.qpos[:] = m.key_qpos[0]
    d.qpos[2] = BASE_HEIGHT
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    for _ in range(n):
        d.ctrl[:] = lock_torques(m, d)
        mujoco.mj_step(m, d)
    # put the base back exactly, zero the velocities
    d.qpos[0:3] = [0.0, 0.0, BASE_HEIGHT]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qvel[:] = 0.0
    m.opt.gravity[:] = g_save
    d.qfrc_applied[:] = 0.0
    mujoco.mj_forward(m, d)
    q_err = np.max(np.abs(d.qpos[7:] - np.tile(Q_NOMINAL, 4)))
    return q_err


def foot_geom_ids(m):
    ids = []
    for leg in LEGS:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
        g_best = None
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == bid and int(m.geom_type[g]) == 2:
                g_best = g
        ids.append(g_best)
    return ids


def test1(m, d):
    print("=== TEST 1: SRB prediction vs MuJoCo ===")
    print("Robot in the air (z=1.0), joints locked by PD, gravity on, known")
    print("forces applied at the feet, acceleration measured over one step.\n")

    gids = foot_geom_ids(m)
    bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
            for leg in LEGS]
    Fz = MASS_TOTAL * GRAVITY / 4.0

    cases = {
        "weight support": np.tile([0.0, 0.0, Fz], (4, 1)),
        "forward push":   np.tile([12.0, 0.0, Fz], (4, 1)),
        "lateral push":   np.tile([0.0, 10.0, Fz], (4, 1)),
        "pitch moment":   np.array([[0, 0, Fz+25], [0, 0, Fz+25],
                                    [0, 0, Fz-25], [0, 0, Fz-25]], float),
        "roll moment":    np.array([[0, 0, Fz+20], [0, 0, Fz-20],
                                    [0, 0, Fz+20], [0, 0, Fz-20]], float),
        "yaw moment":     np.array([[ 8, 0, Fz], [-8, 0, Fz],
                                    [-8, 0, Fz], [ 8, 0, Fz]], float),
    }

    q_err = settle(m, d)
    print(f"largest joint deviation after settling: {q_err:.2e} rad\n")

    for name, F in cases.items():
        settle(m, d)

        def apply():
            d.qfrc_applied[:] = 0.0
            feet = np.array([d.geom_xpos[g].copy() for g in gids])
            for i in range(4):
                mujoco.mj_applyFT(m, d, F[i], np.zeros(3), feet[i],
                                  bids[i], d.qfrc_applied)
            d.ctrl[:] = lock_torques(m, d)
            mujoco.mj_forward(m, d)
            mujoco.mj_subtreeVel(m, d)
            return feet

        feet0 = apply()
        com0 = d.subtree_com[1].copy()
        lin0 = d.subtree_linvel[1].copy()
        ang0 = d.subtree_angmom[1].copy()
        quat0 = d.qpos[3:7].copy()
        w_world0 = quat_to_R(quat0) @ d.qvel[3:6].copy()

        mujoco.mj_step(m, d)
        apply()
        lin1 = d.subtree_linvel[1].copy()
        ang1 = d.subtree_angmom[1].copy()

        dt = m.opt.timestep
        acc_mj = (lin1 - lin0) / dt
        dL_mj = (ang1 - ang0) / dt

        # check of the measurement itself: external force balance
        F_ext = F.sum(axis=0) + MASS_TOTAL * np.array([0, 0, -GRAVITY])
        acc_expected = F_ext / MASS_TOTAL

        # SRB prediction
        X = np.zeros(N_STATE)
        X[0:3] = com0
        X[3:6] = euler_zyx(quat0)
        X[9:12] = w_world0
        X[12] = GRAVITY
        dX = continuous_derivative(X, F.ravel(), feet0)
        acc_srb = dX[6:9]
        dL_srb = world_inertia(X[5]) @ dX[9:12]

        print(f"--- {name} ---")
        print(f"  {'':12s} {'x':>9} {'y':>9} {'z':>9}")
        print(f"  {'acc MuJoCo':12s} {acc_mj[0]:>9.4f} {acc_mj[1]:>9.4f} {acc_mj[2]:>9.4f}")
        print(f"  {'acc Newton':12s} {acc_expected[0]:>9.4f} {acc_expected[1]:>9.4f} {acc_expected[2]:>9.4f}"
              f"   <- measurement check")
        print(f"  {'acc SRB':12s} {acc_srb[0]:>9.4f} {acc_srb[1]:>9.4f} {acc_srb[2]:>9.4f}")
        print(f"  {'dL MuJoCo':12s} {dL_mj[0]:>9.4f} {dL_mj[1]:>9.4f} {dL_mj[2]:>9.4f}")
        print(f"  {'dL SRB':12s} {dL_srb[0]:>9.4f} {dL_srb[1]:>9.4f} {dL_srb[2]:>9.4f}")

        n_dL = np.linalg.norm(dL_srb)
        rel = np.linalg.norm(dL_mj - dL_srb) / n_dL * 100 if n_dL > 1e-9 else float('nan')
        print(f"  acceleration measurement error : {np.max(np.abs(acc_mj - acc_expected)):.2e} m/s^2")
        print(f"  dL relative deviation          : {rel:.1f}%\n")


def test2(m, d):
    print("=== TEST 2: Theta_dot = R_z^T omega ===")
    print("Gravity off, no forces, joints locked, known angular velocity.\n")

    settle(m, d)
    g_save = m.opt.gravity.copy()
    m.opt.gravity[:] = 0.0
    d.qfrc_applied[:] = 0.0          # clear forces of the previous test

    psi0 = 0.6
    d.qpos[0:3] = [0.0, 0.0, BASE_HEIGHT]
    d.qpos[3:7] = [np.cos(psi0/2), 0.0, 0.0, np.sin(psi0/2)]
    d.qvel[:] = 0.0
    w_body = np.array([0.30, -0.20, 0.10])
    d.qvel[3:6] = w_body
    mujoco.mj_forward(m, d)

    R0 = quat_to_R(d.qpos[3:7])
    w_world = R0 @ w_body
    e0 = euler_zyx(d.qpos[3:7])

    n = 3
    for _ in range(n):
        d.ctrl[:] = lock_torques(m, d)
        d.qfrc_applied[:] = 0.0
        mujoco.mj_step(m, d)
    e1 = euler_zyx(d.qpos[3:7])
    Theta_dot = (e1 - e0) / (n * m.opt.timestep)

    pred_T = rot_z(psi0).T @ w_world
    pred_N = rot_z(psi0) @ w_world
    eT = np.max(np.abs(Theta_dot - pred_T))
    eN = np.max(np.abs(Theta_dot - pred_N))

    print(f"omega (body, qvel[3:6])  : {np.round(w_body, 4).tolist()}")
    print(f"omega (world, R@w_body)  : {np.round(w_world, 4).tolist()}")
    print(f"measured Theta_dot       : {np.round(Theta_dot, 4).tolist()}")
    print(f"R_z^T omega_world        : {np.round(pred_T, 4).tolist()}   error={eT:.4f}")
    print(f"R_z   omega_world        : {np.round(pred_N, 4).tolist()}   error={eN:.4f}")
    print(f"\nBETTER: "
          f"{'R_z^T (Di Carlo, used in the code)' if eT < eN else 'R_z (Sombolestan)'}")

    m.opt.gravity[:] = g_save


def main():
    path = find_scene()
    print("model:", path, "\n")
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    test1(m, d)
    test2(m, d)


if __name__ == "__main__":
    main()
