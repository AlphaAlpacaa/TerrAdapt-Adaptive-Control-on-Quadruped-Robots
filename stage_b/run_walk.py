"""
Basic closed-loop walk in MuJoCo, plus the state-reading helpers used by
scenario.py.

Usage:
    python run_walk.py                 # no viewer, measurement only
    python run_walk.py --view          # with the MuJoCo viewer
    python run_walk.py --gait stand    # standing test first (recommended)
    python run_walk.py --vx 0.3 --T 5  # 0.3 m/s forward, 5 s

The state is read directly from the simulator (ground truth). A real robot
would need an IMU + joint-encoder estimator; this is a stated limitation.
"""

import argparse
import numpy as np
from pathlib import Path
import mujoco

from a1_params import (MASS_TOTAL, GRAVITY, Z_NOMINAL, Q_NOMINAL, LEGS,
                       TAU_MAX)
from a1_kinematics import foot_position, foot_jacobian
from srb_dynamics import N_STATE
from controller import Controller


def find_scene():
    here = Path(__file__).resolve().parent
    for base in [here] + list(here.parents)[:4]:
        hit = list(base.glob("**/unitree_a1/scene_torque.xml"))
        if hit:
            return str(hit[0])
    raise FileNotFoundError("scene_torque.xml missing")


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


def read_state(m, d, foot_gids):
    """SRB state and leg data from MuJoCo."""
    R = quat_to_R(d.qpos[3:7])
    com = d.subtree_com[1].copy()              # computed by mj_forward
    v_world = d.qvel[0:3].copy()               # world frame
    w_world = R @ d.qvel[3:6].copy()           # qvel[3:6] is in the BODY frame

    X = np.zeros(N_STATE)
    X[0:3] = com
    X[3:6] = euler_zyx(d.qpos[3:7])
    X[6:9] = v_world
    X[9:12] = w_world
    X[12] = GRAVITY

    foot_w = np.array([d.geom_xpos[g].copy() for g in foot_gids])
    q = d.qpos[7:].copy()
    dq = d.qvel[6:].copy()
    foot_b = np.array([foot_position(l, q[3*i:3*i+3]) for i, l in enumerate(LEGS)])
    return X, foot_w, foot_b, q, dq, R


def read_joint_params(m, leg="FR"):
    """
    Joint parameters of one leg: armature, damping, frictionloss.

    These were initially not read and explained almost all of the swing
    tracking error (42.8 mm in MuJoCo): armature 0.01 kg m^2 triples the
    operational-space mass, damping [1, 2, 2] gives 3.6 Nm out of a 4.9 Nm
    swing torque budget, frictionloss 0.2 Nm a speed-independent error.
    """
    arm, damp, fric = [], [], []
    for part in ["hip", "thigh", "calf"]:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{leg}_{part}_joint")
        if jid < 0:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{leg}_{part}")
        if jid < 0:
            raise RuntimeError(f"joint not found: {leg}_{part}")
        dadr = m.jnt_dofadr[jid]
        arm.append(float(m.dof_armature[dadr]))
        damp.append(float(m.dof_damping[dadr]))
        fric.append(float(m.dof_frictionloss[dadr]))
    return np.array(arm), np.array(damp), np.array(fric)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gait", default="trot")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--period", type=float, default=None,
                    help="gait period (s); trot default 0.4")
    ap.add_argument("--kp", type=float, default=None, help="swing kp")
    ap.add_argument("--height", type=float, default=None,
                    help="swing height (m)")
    ap.add_argument("--nograv", action="store_true",
                    help="turn OFF leg gravity compensation")
    ap.add_argument("--noff", action="store_true",
                    help="turn OFF acceleration feed-forward (ablation)")
    ap.add_argument("--ffscale", type=float, default=1.0,
                    help="feed-forward scale 0..1 (ablation sweep)")
    ap.add_argument("--nofric", action="store_true",
                    help="turn OFF joint friction compensation (ablation)")
    ap.add_argument("--fricscale", type=float, default=1.0,
                    help="friction compensation scale; values other than 1 "
                         "represent parameter uncertainty")
    ap.add_argument("--noarm", action="store_true",
                    help="omit armature in the feed-forward (ablation)")
    ap.add_argument("--stancefric", action="store_true",
                    help="apply friction compensation in stance too (default "
                         "off: pitch 2.5x worse and +11 Nm torque)")
    ap.add_argument("--soft", type=float, default=None,
                    help="ground contact time constant (solref[0], s). "
                         "Default ~0.02 (hard); 0.05-0.20 soft ground.")
    ap.add_argument("--damprat", type=float, default=None,
                    help="ground contact damping ratio (solref[1]). <1 bouncy, "
                         ">1 overdamped")
    ap.add_argument("--mu", type=float, default=None,
                    help="ground sliding friction. Default 1.0; "
                         "0.3-0.5 slippery.")
    ap.add_argument("--slope", type=float, default=0.0,
                    help="slope (deg), applied by rotating gravity; "
                         "mathematically equivalent to tilting the ground.")
    ap.add_argument("--payload", type=float, default=0.0,
                    help="mass added to the trunk (kg), unknown to the "
                         "controller; theta_hat_z should change by -g*dm/m")
    ap.add_argument("--gamma", type=float, default=None,
                    help="adaptation gain (defaults: M0=50 M1=30 M2=10)")
    ap.add_argument("--lam", type=float, default=None,
                    help="state predictor gain Lambda (1/s), default 20")
    ap.add_argument("--adaptive", default="none",
                    choices=["none", "M0I", "M0", "M0F", "M1", "M2"],
                    help="adaptive layer (body acceleration residual)")
    ap.add_argument("--hold", type=float, default=0.5,
                    help="footstep target frozen for s > hold; 1.0 = never")
    ap.add_argument("--wrp", type=float, default=None,
                    help="roll/pitch MPC weight (default: a1_params.Q_MPC)")
    ap.add_argument("--ramp", type=float, default=1.0,
                    help="ramp-up time of the velocity command (s)")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(find_scene())

    # --- unknown payload -------------------------------------------------
    # A point mass at the trunk COM: body_inertia is unchanged, so this is
    # a clean one-variable experiment. The controller keeps 12.453 kg.
    theta_z_pred = 0.0
    if args.payload != 0.0:
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        if tid < 0:
            tid = 1
        m0 = float(m.body_mass[tid])
        m.body_mass[tid] = m0 + args.payload
        mujoco.mj_setConst(m, mujoco.MjData(m))
        theta_z_pred = -GRAVITY * args.payload / MASS_TOTAL
        print(f"PAYLOAD: trunk mass {m0:.3f} -> {m0+args.payload:.3f} kg "
              f"(+{args.payload} kg)")
        print(f"     true total {MASS_TOTAL + args.payload:.3f} kg, "
              f"known to the controller {MASS_TOTAL:.3f} kg")
        print(f"     derived expectation: theta_z = -g*dm/m = "
              f"{theta_z_pred:+.3f} m/s^2")

    # --- ground properties -------------------------------------------------
    # floor geom: attached to the world body, plane (type 0) or hfield (1)
    floor_gid = None
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 and int(m.geom_type[g]) in (0, 1):
            floor_gid = g
            break
    if floor_gid is None:
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == 0:
                floor_gid = g
                break
    if floor_gid is not None:
        if args.mu is not None:
            # MuJoCo takes the element-wise MAXIMUM of the two geoms'
            # friction, so lowering only the floor does nothing; the feet
            # must be lowered too.
            fg = foot_geom_ids(m)
            old_f = float(m.geom_friction[floor_gid, 0])
            old_t = float(m.geom_friction[fg[0], 0])
            m.geom_friction[floor_gid, 0] = args.mu
            for g in fg:
                m.geom_friction[g, 0] = args.mu
            print(f"GROUND: friction floor {old_f:.2f} -> {args.mu:.2f}, "
                  f"feet {old_t:.2f} -> {args.mu:.2f}")
            print("       (the MPC still uses mu = 1.0 in the friction cone,")
            print("        on purpose: this mismatch is part of the uncertainty)")
        if args.soft is not None or args.damprat is not None:
            sr = m.geom_solref[floor_gid].copy()
            new = sr.copy()
            if args.soft is not None:
                new[0] = args.soft
            if args.damprat is not None:
                new[1] = args.damprat
            m.geom_solref[floor_gid] = new
            # MuJoCo mixes solref between the two geoms; apply to the feet
            # too (softness is a property of the foot-ground contact).
            for g in foot_geom_ids(m):
                m.geom_solref[g] = new
            print(f"GROUND: floor solref (time constant, damping ratio) "
                  f"{np.round(sr,4).tolist()} -> {np.round(new,4).tolist()}")
    if args.slope != 0.0:
        th = np.deg2rad(args.slope)
        m.opt.gravity[:] = [GRAVITY*np.sin(th), 0.0, -GRAVITY*np.cos(th)]
        print(f"GROUND: slope {args.slope:+.1f} deg "
              f"(gravity {np.round(m.opt.gravity,3).tolist()})")

    d = mujoco.MjData(m)
    if m.nkey > 0:
        d.qpos[:] = m.key_qpos[0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    gids = foot_geom_ids(m)
    Qw = None
    if args.wrp is not None:
        Qw = [2.5, 2.5, 20.0, args.wrp, args.wrp, 1.5,
              0.2, 0.2, 0.2, 0.1, 0.1, 0.3]

    def leg_link_data(leg):
        """
        Link masses, COM offsets and inertia tensors from the model.
        MuJoCo stores a diagonal inertia (body_inertia) in the principal
        frame given by body_iquat; in the link frame
            I_link = R(iquat) diag(body_inertia) R(iquat)^T
        """
        ms, coms, inert = [], {}, {}
        warn = []
        for key in ["hip", "thigh", "calf"]:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_{key}")
            ms.append(float(m.body_mass[bid]))
            coms[key] = m.body_ipos[bid].copy()
            Riq = quat_to_R(m.body_iquat[bid])
            inert[key] = Riq @ np.diag(m.body_inertia[bid]) @ Riq.T
            # the kinematic chain assumes link frames aligned with the body
            if np.max(np.abs(m.body_quat[bid] - np.array([1.0, 0, 0, 0]))) > 1e-9:
                warn.append(key)
        return tuple(ms), coms, inert, warn

    masses, coms, inertias, quat_warn = leg_link_data("FR")
    print(f"leg link masses     : {np.round(masses,4).tolist()} kg "
          f"(total {sum(masses):.3f})")
    print(f"link COM offsets    : "
          f"hip={np.round(coms['hip'],4).tolist()} "
          f"thigh={np.round(coms['thigh'],4).tolist()} "
          f"calf={np.round(coms['calf'],4).tolist()}")
    print(f"link inertia traces : " + "  ".join(
        f"{k}={np.trace(v):.5f}" for k, v in inertias.items()) + " kg m^2")
    armature, damping, frictionloss = read_joint_params(m)
    print(f"joint armature      : {armature.tolist()} kg m^2")
    print(f"joint damping       : {damping.tolist()} N m s/rad")
    print(f"joint frictionloss  : {frictionloss.tolist()} N m")
    if quat_warn:
        print(f"!! WARNING: non-identity quat on {quat_warn}; "
              f"recheck the chain derivation")

    ctrl = Controller(gait=args.gait, dt_control=m.opt.timestep, Q_weights=Qw,
                      gait_period=args.period,
                      link_masses=masses, link_coms=coms,
                      link_inertias=inertias,
                      joint_armature=(None if args.noarm else armature),
                      joint_damping=damping,
                      joint_frictionloss=frictionloss,
                      gravity_comp=not args.nograv,
                      accel_ff=not args.noff, ff_scale=args.ffscale,
                      friction_comp=not args.nofric,
                      friction_scale=args.fricscale,
                      friction_in_stance=args.stancefric,
                      target_hold=args.hold,
                      adaptive=args.adaptive,
                      adaptive_kw=({k: v for k, v in
                                    [("gamma", args.gamma),
                                     ("lam", args.lam)] if v is not None}))
    if args.kp is not None:
        ctrl.kp_swing = args.kp
        ctrl.kd_swing = args.kp / 28.0
    if args.height is not None:
        ctrl.swing_height = args.height

    n_steps = int(round(args.T / m.opt.timestep))
    log = {k: [] for k in ["t", "z", "roll", "pitch", "yaw",
                           "vx", "vy", "wz", "tau_max", "n_contact",
                           "swing_err", "n_mj_contact", "ff_tau",
                           "fric_tau", "target_move",
                           "theta_lin", "theta_ang", "pred_err"]}
    log_theta = []

    print(f"gait={args.gait}  vx={args.vx}  T={args.T}s  "
          f"dt={m.opt.timestep}s  steps={n_steps}")
    print(f"MPC: horizon={ctrl.k} x {ctrl.dt_mpc}s = {ctrl.k*ctrl.dt_mpc:.2f}s, "
          f"update every {ctrl.mpc_every}s")
    T_sw = max(ctrl.gait.swing_time, 1e-9)
    a_td = ctrl.swing_height * 32.0 / T_sw**2      # s=0 and s=1 (lift-off/touchdown)
    a_mid = ctrl.swing_height * 16.0 / T_sw**2     # s=0.5 (apex)
    # operational-space mass in the nominal stance: Lambda = (J M^-1 J^T)^-1
    from a1_kinematics import leg_mass_matrix, foot_jacobian as _fj
    _J = _fj("FR", Q_NOMINAL)
    _M = leg_mass_matrix("FR", Q_NOMINAL, masses, inertias, coms, armature)
    Lam = np.linalg.inv(_J @ np.linalg.inv(_M) @ _J.T)
    print(f"swing: kp={ctrl.kp_swing:.0f} kd={ctrl.kd_swing:.0f} "
          f"h={ctrl.swing_height:.3f}m T={T_sw:.3f}s")
    print(f"  required vertical acceleration: touchdown {a_td:.1f} "
          f"({a_td/9.81:.1f}g), apex {a_mid:.1f} m/s^2")
    print(f"  Lambda_zz={Lam[2,2]:.3f} kg -> touchdown force "
          f"{Lam[2,2]*a_td:.1f} N; pure PD would lag by "
          f"{Lam[2,2]*a_td/ctrl.kp_swing*1000:.1f} mm to produce it")
    print(f"  gravity compensation {'ON' if ctrl.gravity_comp else 'OFF'}, "
          f"acceleration feed-forward "
          f"{'ON (scale %.2f)' % ctrl.ff_scale if ctrl.accel_ff else 'OFF'}"
          f"{'' if not args.noarm else ' [armature omitted]'}")
    print(f"  friction compensation "
          f"{'ON (scale %.2f)' % ctrl.friction_scale if ctrl.friction_comp else 'OFF'}"
          f"{', swing only' if not ctrl.friction_in_stance else ', swing+stance'}"
          f",  target freeze s>{ctrl.target_hold:.2f}")
    print(f"adaptive layer: {ctrl.layer.name}", end="")
    if ctrl.layer.approx is not None:
        g = getattr(ctrl.layer.approx, "gamma", 0.0)
        EtPE = ctrl.layer.E.T @ ctrl.layer.P @ ctrl.layer.E
        sv = float(np.linalg.svd(EtPE, compute_uv=False).max())
        w_ad = np.sqrt(g * sv)              # adaptation natural frequency
        f_gait = 1.0 / ctrl.gait.period     # what must be tracked
        print(f"  gamma={g:.0f}  Lambda={ctrl.layer.lam:.0f}/s  "
              f"theta_max={ctrl.layer.theta_max:.0f}")
        print(f"  adaptation bandwidth: sqrt(gamma*||E^T P E||) = {w_ad:.1f} rad/s "
              f"= {w_ad/(2*np.pi):.2f} Hz")
        print(f"  gait frequency: {f_gait:.2f} Hz "
              f"({2*np.pi*f_gait:.1f} rad/s)")
        if w_ad < 2*np.pi*f_gait:
            need = (2*np.pi*f_gait)**2 / sv
            print("  !! bandwidth too low: the within-cycle disturbance cannot "
                  "be tracked.")
            print(f"     gamma ~ {need:.0f} would be needed "
                  f"({need/max(g,1e-9):.0f}x); sweep with --gamma.")
        else:
            print("  bandwidth sufficient for within-cycle variation")
    print()

    viewer = None
    if args.view:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, d)

    td_log = {l: [] for l in LEGS}
    tde_log = {l: [] for l in LEGS}
    fell = False
    try:
        for i in range(n_steps):
            t = i * m.opt.timestep
            w = min(1.0, t / max(args.ramp, 1e-9))
            ctrl.set_command(args.vx * w, args.vy * w, args.yaw * w)

            X, foot_w, foot_b, q, dq, R = read_state(m, d, gids)
            tau = ctrl.compute(t, X, foot_w, foot_b, q, dq, R)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            for leg in LEGS:
                if ctrl.touchdown[leg] is not None:
                    td_log[leg].append(ctrl.touchdown[leg].copy())
                    ctrl.touchdown[leg] = None
                if ctrl.td_error[leg] is not None:
                    tde_log[leg].append(ctrl.td_error[leg].copy())
                    ctrl.td_error[leg] = None

            if i % 5 == 0:
                e = euler_zyx(d.qpos[3:7])
                log["t"].append(t)
                log["z"].append(float(d.qpos[2]))
                log["roll"].append(float(e[0]))
                log["pitch"].append(float(e[1]))
                log["yaw"].append(float(e[2]))
                log["vx"].append(float(d.qvel[0]))
                log["vy"].append(float(d.qvel[1]))
                log["wz"].append(float(d.qvel[5]))
                log["tau_max"].append(float(np.max(np.abs(tau))))
                log["n_contact"].append(int(ctrl.gait.contact(t).sum()))
                log["swing_err"].append(float(np.max(ctrl.swing_err)))
                log["ff_tau"].append(float(np.max(ctrl.ff_tau)))
                log["fric_tau"].append(float(np.max(ctrl.fric_tau)))
                log["target_move"].append(float(np.max(ctrl.target_move)))
                log["theta_lin"].append(float(np.linalg.norm(ctrl.theta[0:3])))
                log["theta_ang"].append(float(np.linalg.norm(ctrl.theta[3:6])))
                log["pred_err"].append(float(np.linalg.norm(ctrl.layer.e_tilde)))
                log_theta.append(ctrl.theta.copy())
                # actual foot contacts in MuJoCo
                nc = 0
                for ci in range(d.ncon):
                    g1, g2 = d.contact[ci].geom1, d.contact[ci].geom2
                    if g1 in gids or g2 in gids:
                        nc += 1
                log["n_mj_contact"].append(nc)

            if viewer is not None:
                viewer.sync()

            if d.qpos[2] < 0.12 or abs(euler_zyx(d.qpos[3:7])[0]) > 1.0:
                print(f"!! t={t:.3f}s: robot fell "
                      f"(z={d.qpos[2]:.3f}, roll={euler_zyx(d.qpos[3:7])[0]:.3f})")
                fell = True
                break
    finally:
        if viewer is not None:
            viewer.close()

    L = {k: np.array(v) for k, v in log.items()}
    print(f"\n{'duration':<10} {len(L['t'])*5*m.opt.timestep:.2f} s "
          f"({'FELL' if fell else 'STANDING'})")
    if len(L["t"]) > 10:
        half = len(L["t"]) // 2
        print(f"{'z mean':<10} {L['z'][half:].mean():.4f} m "
              f"(target {Z_NOMINAL})   std {L['z'][half:].std():.4f}")
        print(f"{'roll':<10} mean {L['roll'][half:].mean():+.4f}  "
              f"std {L['roll'][half:].std():.4f} rad")
        print(f"{'pitch':<10} mean {L['pitch'][half:].mean():+.4f}  "
              f"std {L['pitch'][half:].std():.4f} rad")
        print(f"{'vx':<10} mean {L['vx'][half:].mean():+.4f} m/s "
              f"(command {args.vx})   std {L['vx'][half:].std():.4f}")
        print(f"{'vy':<10} mean {L['vy'][half:].mean():+.4f} m/s")
        print(f"{'tau max':<10} {L['tau_max'].max():.2f} Nm "
              f"(limit {TAU_MAX})")
        print(f"{'distance':<10} {float(d.qpos[0]):.3f} m")
        print(f"{'swing error':<10} mean {L['swing_err'][half:].mean()*1000:.1f} mm "
              f"max {L['swing_err'].max()*1000:.1f} mm")
        print(f"{'contacts (plan)':<10} mean {L['n_contact'][half:].mean():.2f} legs")
        print(f"{'contacts (real)':<10} mean {L['n_mj_contact'][half:].mean():.2f} legs"
              f"   <- much lower than planned means feet in the air")
        print(f"{'accel FF':<10} max {L['ff_tau'].max():.2f} Nm "
              f"mean {L['ff_tau'][half:].mean():.2f} Nm")
        print(f"{'friction FF':<10} max {L['fric_tau'].max():.2f} Nm "
              f"mean {L['fric_tau'][half:].mean():.2f} Nm")
        print(f"{'target jitter':<10} max {L['target_move'].max()*1000:.2f} mm/step")
        if ctrl.layer.approx is not None:
            print(f"\nADAPTIVE LAYER: {ctrl.layer.name}  "
                  f"({ctrl.layer.n_weights} parameters)")
            print(f"  theta_hat linear  : mean {L['theta_lin'][half:].mean():.3f} "
                  f"max {L['theta_lin'].max():.3f} m/s^2")
            print(f"  theta_hat angular : mean {L['theta_ang'][half:].mean():.3f} "
                  f"max {L['theta_ang'].max():.3f} rad/s^2")
            print(f"  prediction error  : mean {L['pred_err'][half:].mean():.4f} "
                  f"max {L['pred_err'].max():.4f}")
            TH = np.array(log_theta)
            thm = TH[len(TH)//2:].mean(axis=0)
            print("  theta_hat components (mean of the second half):")
            print(f"    acceleration a = {np.round(thm[0:3],4).tolist()} m/s^2")
            print(f"    angular alpha  = {np.round(thm[3:6],4).tolist()} rad/s^2")
            from srb_dynamics import wrench_from_accel
            w = wrench_from_accel(thm)
            print(f"    wrench  f = {np.round(w[0:3],2).tolist()} N   "
                  f"m = {np.round(w[3:6],3).tolist()} N m")
            if args.payload != 0.0:
                print("\n  PAYLOAD NOTE")
                print(f"    derived CHANGE: theta_z(dm)-theta_z(0) = "
                      f"{theta_z_pred:+.3f} m/s^2")
                print(f"    theta_z in this run = {thm[2]:+.3f} m/s^2")
                print("    Compare differences, not absolute values: without a")
                print("    payload theta_z is already ~+0.66 (the SRB model misses")
                print("    leg motion). test_payload.py measures the difference.")

    # --- touchdown error in the world frame ------------------------------
    # The body-frame touchdown position mixes in the pitch, so an apparent
    # "foot slides forward" is often a measurement artefact.
    print("\nTOUCHDOWN ERROR (world frame, relative to the target)")
    print(f"{'leg':<6} {'n':>6} {'dx (mm)':>10} {'dy (mm)':>10} "
          f"{'dz (mm)':>10} {'|d| (mm)':>10}")
    print("-" * 56)
    allerr = []
    for leg in LEGS:
        if len(tde_log[leg]) >= 2:
            arr = np.array(tde_log[leg][1:]) * 1000.0   # skip the first touchdown
            allerr.append(arr)
            print(f"{leg:<6} {len(arr):>6} {arr[:,0].mean():>10.1f} "
                  f"{arr[:,1].mean():>10.1f} {arr[:,2].mean():>10.1f} "
                  f"{np.linalg.norm(arr,axis=1).mean():>10.1f}")
        else:
            print(f"{leg:<6} {len(tde_log[leg]):>6}   (not enough data)")
    if allerr:
        A = np.vstack(allerr)
        print(f"{'ALL':<6} {len(A):>6} {A[:,0].mean():>10.1f} "
              f"{A[:,1].mean():>10.1f} {A[:,2].mean():>10.1f} "
              f"{np.linalg.norm(A,axis=1).mean():>10.1f}")
        print("   dx<0: the foot lands behind the target -> support behind "
              "the COM -> nose-up pitch moment")

    inward = []
    print("\nTOUCHDOWN POSITION (body frame, relative to nominal)")
    print(f"{'leg':<6} {'touchdowns':>12} {'x (mm)':>18} {'y (mm)':>18}")
    print(f"{'':6} {'':12} {'nominal / meas.':>18} {'nominal / meas.':>18}")
    print("-" * 58)
    for leg in LEGS:
        nom = foot_position(leg, Q_NOMINAL)
        if len(td_log[leg]) >= 2:
            arr = np.array(td_log[leg][1:])     # skip the first touchdown
            mx, my = arr[:, 0].mean(), arr[:, 1].mean()
            dy_in = abs(my) - abs(nom[1])      # <0 inward, >0 outward
            flag = ("IN" if dy_in < -0.004 else
                    "OUT" if dy_in > 0.004 else "")
            inward.append(dy_in)
            print(f"{leg:<6} {len(arr):>12} "
                  f"{nom[0]*1000:7.1f} /{mx*1000:8.1f} "
                  f"{nom[1]*1000:7.1f} /{my*1000:8.1f}  "
                  f"{dy_in*1000:+6.1f} mm {flag}")
        else:
            print(f"{leg:<6} {len(td_log[leg]):>12}  (not enough data)")
    if inward:
        mi = float(np.mean(inward))
        print(f"   mean lateral deviation: {mi*1000:+.1f} mm "
              f"({'inward' if mi < -0.004 else 'outward' if mi > 0.004 else 'nominal, within 4 mm'})")

    np.savez("walk_log.npz", **L)
    print("\nwalk_log.npz written")


if __name__ == "__main__":
    main()
