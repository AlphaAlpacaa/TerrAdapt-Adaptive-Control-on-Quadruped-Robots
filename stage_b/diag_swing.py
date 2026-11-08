"""
Swing error broken down by phase -- where does the 25 mm come from?

run_walk.py reports one number (mean swing error ~25 mm) and it did not
react to the acceleration feed-forward or to freezing the target. If a
correctly sized force correction does not change the error, the error is
not a force deficit. This script shows:
  - in which axis (x / y / z)
  - at which point of the swing (s = 0 lift-off ... s = 1 touchdown)
  - whether the commanded position is kinematically reachable
  - whether the joint torque saturates
  - whether the foot touches the ground during swing (dragging)
and measures the touchdown error at two instants:
  (a) when the gait schedule switches to stance (what run_walk.py measures)
  (b) when MuJoCo reports the real contact (what happens physically).

Usage:
    python diag_swing.py                   # trot in place, 3 s
    python diag_swing.py --vx 0.3 --T 6
    python diag_swing.py --noff --hold 1.0 # ablation
"""

import argparse
import numpy as np
import mujoco

from a1_params import LEGS, Q_NOMINAL, TAU_MAX, Z_NOMINAL, R_FOOT
from a1_kinematics import foot_position, inverse_kinematics
from controller import Controller
from gait import bezier_swing
from run_walk import (find_scene, quat_to_R, euler_zyx, read_state,
                      foot_geom_ids, read_joint_params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gait", default="trot")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--period", type=float, default=None)
    ap.add_argument("--kp", type=float, default=None)
    ap.add_argument("--height", type=float, default=None)
    ap.add_argument("--noff", action="store_true")
    ap.add_argument("--nofric", action="store_true")
    ap.add_argument("--fricscale", type=float, default=1.0)
    ap.add_argument("--noarm", action="store_true")
    ap.add_argument("--hold", type=float, default=0.5)
    ap.add_argument("--ramp", type=float, default=1.0)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(find_scene())
    d = mujoco.MjData(m)
    if m.nkey > 0:
        d.qpos[:] = m.key_qpos[0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    gids = foot_geom_ids(m)

    def leg_link_data(leg):
        ms, coms, inert = [], {}, {}
        for key in ["hip", "thigh", "calf"]:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_{key}")
            ms.append(float(m.body_mass[bid]))
            coms[key] = m.body_ipos[bid].copy()
            Riq = quat_to_R(m.body_iquat[bid])
            inert[key] = Riq @ np.diag(m.body_inertia[bid]) @ Riq.T
        return tuple(ms), coms, inert

    masses, coms, inertias = leg_link_data("FR")
    armature, damping, frictionloss = read_joint_params(m)
    print(f"joint armature={armature.tolist()} damping={damping.tolist()} "
          f"frictionloss={frictionloss.tolist()}")
    ctrl = Controller(gait=args.gait, dt_control=m.opt.timestep,
                      gait_period=args.period,
                      link_masses=masses, link_coms=coms,
                      link_inertias=inertias,
                      joint_armature=(None if args.noarm else armature),
                      joint_damping=damping,
                      joint_frictionloss=frictionloss,
                      accel_ff=not args.noff,
                      friction_comp=not args.nofric,
                      friction_scale=args.fricscale,
                      target_hold=args.hold)
    if args.kp is not None:
        ctrl.kp_swing = args.kp
        ctrl.kd_swing = args.kp / 28.0
    if args.height is not None:
        ctrl.swing_height = args.height

    dt = m.opt.timestep
    n = int(round(args.T / dt))

    rows = []                          # one row per swing step
    td_sched = {l: [] for l in LEGS}   # touchdown error at the schedule switch
    td_real = {l: [] for l in LEGS}    # touchdown error at real contact
    drag = {l: 0 for l in LEGS}        # swing steps in contact
    swing_steps = {l: 0 for l in LEGS}
    prev_mj_contact = {l: False for l in LEGS}
    pending = {l: None for l in LEGS}  # target waiting for real contact

    gid_of = dict(zip(LEGS, gids))

    for i in range(n):
        t = i * dt
        w = min(1.0, t / max(args.ramp, 1e-9))
        ctrl.set_command(args.vx * w, 0.0, 0.0)

        X, foot_w, foot_b, q, dq, R = read_state(m, d, gids)
        tau = ctrl.compute(t, X, foot_w, foot_b, q, dq, R)
        contact = ctrl.last_contact
        swing_s = ctrl.gait.swing_phase(t)

        # --- rebuild the commanded position (same path as the controller)
        for j, leg in enumerate(LEGS):
            if contact[j] or ctrl._swing_start[leg] is None:
                continue
            T_sw = max(ctrl.gait.swing_time, 1e-6)
            p_d, _, _ = bezier_swing(ctrl._swing_start[leg],
                                      ctrl._target[leg], swing_s[j],
                                      ctrl.swing_height)
            e = p_d - foot_w[j]

            # Reachability: the commanded world position is expressed in
            # the body frame and tried with IK; a large residual means the
            # leg physically cannot get there. swing_steps is counted at
            # every step (drag is too), IK only every second step.
            swing_steps[leg] += 1
            if i % 2:
                continue
            p_trunk = d.qpos[0:3].copy()
            p_d_body = R.T @ (p_d - p_trunk)
            q_sol = inverse_kinematics(leg, p_d_body,
                                       q_guess=q[3*j:3*j+3])
            reach = np.linalg.norm(foot_position(leg, q_sol) - p_d_body)

            rows.append((leg, swing_s[j], e[0], e[1], e[2],
                         np.linalg.norm(e), reach,
                         np.max(np.abs(tau[3*j:3*j+3])),
                         float(d.qpos[2]), float(euler_zyx(d.qpos[3:7])[1])))

        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        # --- real MuJoCo contact
        X2, foot_w2, _, _, _, _ = read_state(m, d, gids)
        now_contact = {l: False for l in LEGS}
        for ci in range(d.ncon):
            g1, g2 = d.contact[ci].geom1, d.contact[ci].geom2
            for leg in LEGS:
                if g1 == gid_of[leg] or g2 == gid_of[leg]:
                    now_contact[leg] = True

        for j, leg in enumerate(LEGS):
            # dragging: schedule says swing but the foot is on the ground
            if not contact[j]:
                if now_contact[leg]:
                    drag[leg] += 1
            # error at the schedule switch (recorded by the controller)
            if ctrl.td_error[leg] is not None:
                td_sched[leg].append(ctrl.td_error[leg].copy())
                # the controller clears _target in the stance branch of the
                # same call; recover the target from the error:
                #   td_error = foot_world - target  ->  target = foot - err
                pending[leg] = foot_w[j] - ctrl.td_error[leg]
                ctrl.td_error[leg] = None
            ctrl.touchdown[leg] = None
            # moment real contact starts
            if now_contact[leg] and not prev_mj_contact[leg]:
                if pending[leg] is not None:
                    td_real[leg].append(foot_w2[j] - pending[leg])
                    pending[leg] = None
            prev_mj_contact[leg] = now_contact[leg]

    # -----------------------------------------------------------------
    print("\nSWING ERROR DIAGNOSIS")
    print(f"gait={args.gait} vx={args.vx} T={args.T}s  "
          f"kp={ctrl.kp_swing:.0f} h={ctrl.swing_height:.3f} "
          f"T_sw={ctrl.gait.swing_time:.3f}s")
    print(f"accel FF={'ON' if ctrl.accel_ff else 'OFF'}  "
          f"friction FF={'ON x%.2f' % ctrl.friction_scale if ctrl.friction_comp else 'OFF'}  "
          f"target freeze s>{ctrl.target_hold:.2f}")
    print(f"final state: z={d.qpos[2]:.4f} pitch={euler_zyx(d.qpos[3:7])[1]:+.4f}")

    if not rows:
        print("no swing step recorded")
        return

    A = np.array([r[1:] for r in rows], float)
    s_col = A[:, 0]

    print(f"\n1) ERROR BY SWING PHASE  ({len(rows)} swing steps)")
    print(f"   {'s range':>10} {'n':>5} {'ex':>8} {'ey':>8} {'ez':>8} "
          f"{'|e|':>8} {'IK resid':>9} {'tau':>7} {'body z':>8} {'pitch':>8}")
    print("   " + "-" * 86)
    edges = np.linspace(0.0, 1.0, 11)
    for a, b in zip(edges[:-1], edges[1:]):
        msk = (s_col >= a) & (s_col < b + (1e-9 if b == 1.0 else 0.0))
        if msk.sum() == 0:
            continue
        S = A[msk]
        print(f"   {a:.1f}-{b:.1f}".rjust(13) + f" {msk.sum():>5} "
              f"{S[:,1].mean()*1000:>7.1f} {S[:,2].mean()*1000:>7.1f} "
              f"{S[:,3].mean()*1000:>7.1f} {S[:,4].mean()*1000:>7.1f} "
              f"{S[:,5].mean()*1000:>8.2f} {S[:,6].mean():>6.2f} "
              f"{S[:,7].mean():>7.4f} {S[:,8].mean():>+7.4f}")
    print("   (ex, ey, ez, |e| and IK residual in mm; tau in Nm)")

    k = int(np.argmax(A[:, 4]))
    print(f"\n   largest |e|  : {A[k,4]*1000:.1f} mm, s={A[k,0]:.3f}, "
          f"components ex={A[k,1]*1000:+.1f} ey={A[k,2]*1000:+.1f} "
          f"ez={A[k,3]*1000:+.1f} mm")
    print(f"   IK residual  : mean {A[:,5].mean()*1000:.3f} mm  "
          f"max {A[:,5].max()*1000:.3f} mm")
    if A[:, 5].max() > 0.002:
        print("   !! large IK residual -> the COMMANDED TRAJECTORY IS UNREACHABLE")
        print("      (kinematic cause, not a force deficit)")
    else:
        print("   IK residual negligible -> trajectory reachable; tracking error")
    print(f"   max joint torque in swing: {A[:,6].max():.2f} Nm "
          f"(limit {TAU_MAX})"
          + ("  !! SATURATION" if A[:, 6].max() > 0.98 * TAU_MAX else ""))

    print("\n2) DRAGGING: schedule says swing, foot on the ground")
    for leg in LEGS:
        nd, ns = drag[leg], max(swing_steps[leg], 1)
        print(f"   {leg}: {nd:>5}/{ns:<5} swing steps in contact "
              f"({100.0*nd/ns:5.1f}%)")

    print("\n3) TOUCHDOWN ERROR: schedule instant vs REAL contact (mm)")
    print(f"   {'leg':<6} {'n':>4} | {'sched. dx':>10} {'dz':>8} | "
          f"{'real dx':>10} {'dz':>8} | {'delay':>9}")
    print("   " + "-" * 68)
    for leg in LEGS:
        a_ = np.array(td_sched[leg][1:]) * 1000 if len(td_sched[leg]) > 1 else None
        b_ = np.array(td_real[leg][1:]) * 1000 if len(td_real[leg]) > 1 else None
        if a_ is None or b_ is None:
            print(f"   {leg:<6} (not enough data)")
            continue
        print(f"   {leg:<6} {len(b_):>4} | {a_[:,0].mean():>10.1f} "
              f"{a_[:,2].mean():>8.1f} | {b_[:,0].mean():>10.1f} "
              f"{b_[:,2].mean():>8.1f} | "
              f"{a_[:,2].mean()-b_[:,2].mean():>8.1f}")
    print("   real dz ~ 0: the foot lands correctly, only later than the")
    print("   schedule (timing, not position). Large real dz: wrong placement.")


if __name__ == "__main__":
    main()
