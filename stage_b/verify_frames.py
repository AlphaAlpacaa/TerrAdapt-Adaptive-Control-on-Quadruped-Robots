"""
Frame transformation checks -- no MuJoCo needed.

Three frames circulate in the control loop:
  * world        : MPC forces, footstep targets, body state
  * body (trunk) : output of foot_position() and foot_jacobian()
  * COM          : SRB state X[0:3]

The trunk origin and the COM differ by a constant 19.6 mm. Missing this
offset once sent the swing foot to the wrong place and the robot sank a
little more at every step; this file catches such errors.
"""

import numpy as np

from a1_params import (COM_OFFSET, Z_NOMINAL, Z_COM_NOMINAL, Q_NOMINAL,
                       LEGS, R_FOOT)
from a1_kinematics import foot_position, foot_jacobian
from gait import footstep_target, GaitScheduler, bezier_swing


def Rz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def main():
    print("Frame transformation checks\n")

    # --- 1) COM / trunk relation -----------------------------------------
    print("1) COM and trunk origin")
    print(f"   Z_NOMINAL (trunk)  = {Z_NOMINAL}")
    print(f"   Z_COM_NOMINAL      = {Z_COM_NOMINAL:.4f}")
    print(f"   difference         = {Z_NOMINAL - Z_COM_NOMINAL:.4f} m "
          f"(= -COM_OFFSET[2] = {-COM_OFFSET[2]:.4f})")
    ok = abs((Z_NOMINAL + COM_OFFSET[2]) - Z_COM_NOMINAL) < 1e-12
    print(f"   {'PASS' if ok else 'FAIL'}\n")

    # --- 2) feet touch the ground in the nominal stance ------------------
    print("2) Do the feet touch the ground in the nominal stance?")
    p_trunk = np.array([0.0, 0.0, Z_NOMINAL])
    worst = 0.0
    for leg in LEGS:
        p_body = foot_position(leg, Q_NOMINAL)       # relative to the trunk
        p_world = p_trunk + p_body                   # yaw = roll = pitch = 0
        # lowest point of the foot sphere = centre - R_FOOT
        contact_z = p_world[2] - R_FOOT
        worst = max(worst, abs(contact_z))
        print(f"   {leg}: centre z={p_world[2]:.5f}  contact z={contact_z:+.5f}")
    print(f"   largest deviation: {worst*1000:.2f} mm "
          f"({'PASS' if worst < 2e-3 else 'FAIL'})")
    print("   (the keyframe may not give exact contact; 2 mm tolerance)\n")

    # --- 3) footstep target height ----------------------------------------
    print("3) footstep_target() height")
    g = GaitScheduler("trot")
    p_com = np.array([0.0, 0.0, Z_COM_NOMINAL])
    for leg in LEGS:
        p_f = footstep_target(leg, p_com, 0.0, np.zeros(3), np.zeros(3),
                              g.stance_time)
        print(f"   {leg}: target={np.round(p_f, 4).tolist()}  "
              f"contact z={p_f[2]-R_FOOT:+.4f}")
    ok = abs(p_f[2] - R_FOOT) < 1e-12
    print(f"   target centre = ground + R_FOOT: {'PASS' if ok else 'FAIL'}\n")

    # --- 4) world <-> body round trip -------------------------------------
    print("4) World/body round trip (yawed body)")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        psi = rng.uniform(-np.pi, np.pi)
        R = Rz(psi)
        p_trunk_w = np.array([rng.normal(0, 0.3), rng.normal(0, 0.3),
                              Z_NOMINAL])
        p_com_w = p_trunk_w + R @ COM_OFFSET
        leg = LEGS[rng.integers(0, 4)]
        q = Q_NOMINAL + rng.normal(0, 0.1, 3)

        p_body = foot_position(leg, q)          # body frame, relative to trunk
        p_world = p_trunk_w + R @ p_body        # world

        # back: world -> body (relative to the trunk)
        back = R.T @ (p_world - p_trunk_w)
        worst = max(worst, np.max(np.abs(back - p_body)))

        # WRONG path: treating the COM as the trunk origin
        wrong = R.T @ (p_world - p_com_w)
        err_wrong = np.max(np.abs(wrong - p_body))

    print(f"   correct round-trip error       : {worst:.2e} "
          f"({'PASS' if worst < 1e-12 else 'FAIL'})")
    print(f"   error of using the COM as trunk: {err_wrong*1000:.1f} mm "
          f"<- size of the old bug\n")

    # --- 5) foot velocity: the omega x r term ----------------------------
    print("5) Contribution of omega x r to the foot world velocity")
    omega = np.array([0.0, 0.5, 0.0])        # 0.5 rad/s pitch rate
    for leg in ["FR", "RR"]:
        p_body = foot_position(leg, Q_NOMINAL)
        contrib = np.cross(omega, p_body)
        print(f"   {leg}: omega x r = {np.round(contrib, 4).tolist()} m/s")
    print("   (without it the swing PD sees a wrong velocity error)\n")

    # --- 6) swing trajectory touchdown --------------------------------------
    print("6) Does the foot land exactly at the end of the swing?")
    p0 = np.array([0.183, -0.132, R_FOOT])
    pf = np.array([0.253, -0.132, R_FOOT])
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p, v, a = bezier_swing(p0, pf, s, height=0.09)
        print(f"   s={s:.2f}: z={p[2]:.4f} (contact {p[2]-R_FOOT:+.4f})  "
              f"vz={v[2]:+.4f}")
    p1, v1, a1 = bezier_swing(p0, pf, 1.0, height=0.09)
    ok = abs(p1[2] - R_FOOT) < 1e-12 and abs(v1[2]) < 1e-12
    print(f"   {'PASS' if ok else 'FAIL'}: touchdown z = R_FOOT and vz = 0")


if __name__ == "__main__":
    main()
