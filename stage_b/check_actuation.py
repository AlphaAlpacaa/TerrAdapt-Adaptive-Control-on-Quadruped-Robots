"""
Which model parameters am I not reading? -- actuator and joint audit.

verify_swing.py isolates the leg and gets 0.3 mm tracking error with the
same swing law; MuJoCo gave 43 mm. Torque was not saturating, the
trajectory was reachable and a correctly sized force feed-forward changed
nothing, so the joint inertia/damping in my M(q) had to be smaller than
MuJoCo's.

Candidates and their signatures:
  armature (dof_armature)
      reflected rotor inertia, a constant diagonal added to M(q); error
      grows with the acceleration demand, most at the knee.
  damping (dof_damping)
      viscous, tau = -b qdot; error grows with the velocity demand.
  frictionloss (dof_frictionloss)
      Coulomb, tau = -f sign(qdot); independent of speed (lengthening
      T_swing barely changed the error: 35.2 -> 36.3 mm).
  gear / ctrlrange / forcerange
      gear != 1 means d.ctrl is not the joint torque; ctrl/force limits
      mean MuJoCo clips the command, invisible in run_walk.py's "tau max"
      (which logs the commanded torque).

No simulation step is taken: the script reads the model, recomputes
Lambda with the armature and compares the expected PD lag with the
measured 43 mm.

Usage:  python check_actuation.py
"""

import numpy as np
import mujoco

from a1_params import LEGS, Q_NOMINAL, TAU_MAX
from a1_kinematics import foot_jacobian, leg_mass_matrix
from run_walk import find_scene, quat_to_R

# measured with run_walk.py (trot in place, kp=700, h=0.07, T_sw=0.2)
MEASURED_MID_SWING_ERR = 0.0428      # m, ez over s = 0.4-0.5
KP = 700.0
H_SWING = 0.07
T_SWING = 0.20


def main():
    m = mujoco.MjModel.from_xml_path(find_scene())
    print(f"model: {find_scene()}\n")

    # -----------------------------------------------------------------
    print("1) JOINT PARAMETERS (dof_*)")
    print(f"   {'joint':<16} {'armature':>10} {'damping':>10} "
          f"{'frictionloss':>13} {'stiffness':>10}")
    print("   " + "-" * 64)
    dof_of_leg = {}
    for leg in LEGS:
        dofs = []
        for part, jname in [("hip", f"{leg}_hip_joint"),
                            ("thigh", f"{leg}_thigh_joint"),
                            ("calf", f"{leg}_calf_joint")]:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                # some models use names without "_joint"
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                        f"{leg}_{part}")
            if jid < 0:
                print(f"   !! joint not found: {jname}")
                continue
            dadr = m.jnt_dofadr[jid]
            dofs.append(dadr)
            if leg == "FR":
                print(f"   {jname:<16} {m.dof_armature[dadr]:>10.5f} "
                      f"{m.dof_damping[dadr]:>10.5f} "
                      f"{m.dof_frictionloss[dadr]:>13.5f} "
                      f"{m.jnt_stiffness[jid]:>10.5f}")
        dof_of_leg[leg] = dofs

    arm = np.array([m.dof_armature[d] for d in dof_of_leg["FR"]])
    damp = np.array([m.dof_damping[d] for d in dof_of_leg["FR"]])
    fric = np.array([m.dof_frictionloss[d] for d in dof_of_leg["FR"]])
    print(f"\n   FR leg:  armature={arm.tolist()}")
    print(f"            damping={damp.tolist()}")
    print(f"            frictionloss={fric.tolist()}")

    # -----------------------------------------------------------------
    print("\n2) ACTUATORS (is the make_torque_model.py conversion right?)")
    print(f"   {'actuator':<18} {'type':>5} {'gear':>8} {'ctrllim':>8} "
          f"{'ctrlrange':>20} {'forcelim':>9} {'forcerange':>20}")
    print("   " + "-" * 94)
    bad = []
    for a in range(min(m.nu, 4)):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"#{a}"
        g = m.actuator_gear[a, 0]
        cl = bool(m.actuator_ctrllimited[a])
        cr = m.actuator_ctrlrange[a]
        fl = bool(m.actuator_forcelimited[a])
        fr = m.actuator_forcerange[a]
        print(f"   {name:<18} {int(m.actuator_trntype[a]):>5} {g:>8.3f} "
              f"{str(cl):>8} {str(np.round(cr,3).tolist()):>20} "
              f"{str(fl):>9} {str(np.round(fr,3).tolist()):>20}")
        if abs(g - 1.0) > 1e-9:
            bad.append(f"{name}: gear={g}")
        if cl and (cr[1] < TAU_MAX - 1e-6):
            bad.append(f"{name}: ctrlrange upper limit {cr[1]} < {TAU_MAX}")
    print(f"   ({m.nu} actuators in total, first 4 shown)")
    if bad:
        print("\n   !! PROBLEM:")
        for b in bad:
            print(f"      {b}")
        print("      d.ctrl is not the joint torque or is clipped; the clipping")
        print("      does not show in run_walk.py's 'tau max'.")
    else:
        print("\n   gear=1 and no clipping -> d.ctrl is the joint torque. PASS")

    # -----------------------------------------------------------------
    print("\n3) MASS MATRIX: my M(q) vs the true one with armature")
    q = np.array(Q_NOMINAL, float)
    masses, coms, inert = [], {}, {}
    for key in ["hip", "thigh", "calf"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"FR_{key}")
        masses.append(float(m.body_mass[bid]))
        coms[key] = m.body_ipos[bid].copy()
        Riq = quat_to_R(m.body_iquat[bid])
        inert[key] = Riq @ np.diag(m.body_inertia[bid]) @ Riq.T

    M_mine = leg_mass_matrix("FR", q, tuple(masses), inert, coms)
    M_true = M_mine + np.diag(arm)
    print(f"   my M(q) diagonal         : {np.round(np.diag(M_mine),5).tolist()}")
    print(f"   + armature               : {np.round(arm,5).tolist()}")
    print(f"   true M diagonal          : {np.round(np.diag(M_true),5).tolist()}")
    ratio = np.diag(M_true) / np.diag(M_mine)
    print(f"   ratio (true / mine)      : {np.round(ratio,2).tolist()}")

    J = foot_jacobian("FR", q)
    Lam_mine = np.linalg.inv(J @ np.linalg.inv(M_mine) @ J.T)
    Lam_true = np.linalg.inv(J @ np.linalg.inv(M_true) @ J.T)
    print(f"\n   Lambda_zz mine           : {Lam_mine[2,2]:.4f} kg")
    print(f"   Lambda_zz with armature  : {Lam_true[2,2]:.4f} kg "
          f"({Lam_true[2,2]/Lam_mine[2,2]:.2f}x)")

    # -----------------------------------------------------------------
    print("\n4) EXPECTED PD LAG vs MEASURED 42.8 mm")
    a_mid = 16.0 * H_SWING / T_SWING**2      # |a_z| at the apex
    # at the apex the vertical joint speed is ~0, so inertia dominates
    print(f"   acceleration at the apex : {a_mid:.1f} m/s^2 "
          f"({a_mid/9.81:.1f}g)")
    for name, Lam in [("with my M(q)", Lam_mine), ("with armature", Lam_true)]:
        f_need = Lam[2, 2] * a_mid
        lag = f_need / KP
        print(f"   {name:<24}: f={f_need:5.2f} N -> lag "
              f"{lag*1000:5.1f} mm")
    print(f"   MEASURED                 : {MEASURED_MID_SWING_ERR*1000:5.1f} mm")

    # damping contribution: horizontal speed is largest at the apex
    v_horiz = 1.5 * 0.06 / T_SWING
    dq_typ = np.linalg.solve(J.T @ J + 1e-9*np.eye(3),
                             J.T @ np.array([v_horiz, 0.0, 0.0]))
    tau_damp = damp * dq_typ
    tau_fric = fric * np.sign(dq_typ)
    f_damp = np.linalg.solve(J.T, tau_damp)
    f_fric = np.linalg.solve(J.T, tau_fric)
    print(f"\n   typical joint speed at apex : {np.round(dq_typ,2).tolist()} rad/s")
    print(f"   damping torque (b qdot)     : {np.round(tau_damp,3).tolist()} Nm "
          f"-> at the foot {np.linalg.norm(f_damp):.2f} N "
          f"({np.linalg.norm(f_damp)/KP*1000:.1f} mm)")
    print(f"   Coulomb torque              : {np.round(tau_fric,3).tolist()} Nm "
          f"-> at the foot {np.linalg.norm(f_fric):.2f} N "
          f"({np.linalg.norm(f_fric)/KP*1000:.1f} mm)")

    tot = (Lam_true[2, 2] * a_mid + np.linalg.norm(f_damp)
           + np.linalg.norm(f_fric)) / KP
    print(f"\n   TOTAL expected lag       : {tot*1000:.1f} mm "
          f"(measured {MEASURED_MID_SWING_ERR*1000:.1f} mm)")
    if abs(tot - MEASURED_MID_SWING_ERR) < 0.012:
        print("   -> MATCHES. The error comes from unread model parameters:")
        print("      add the armature to M(q), damping and Coulomb to the")
        print("      feed-forward.")
    else:
        print("   -> DOES NOT MATCH. Something else; see the checks below.")

    # -----------------------------------------------------------------
    print("\n5) OTHER CHECKS")
    print(f"   timestep        : {m.opt.timestep} s")
    print(f"   integrator      : {int(m.opt.integrator)} "
          f"(0=Euler 1=RK4 2=implicit 3=implicitfast)")
    print(f"   solver          : {int(m.opt.solver)} iterations={m.opt.iterations}")
    print(f"   geom margin/gap : "
          f"margin={m.geom_margin.max():.4f} gap={m.geom_gap.max():.4f}")
    print("   (margin > 0 makes d.ncon count contacts before real contact)")
    nlim = int((m.jnt_limited[:] == 1).sum())
    print(f"   limited joints  : {nlim}/{m.njnt}")


if __name__ == "__main__":
    main()
