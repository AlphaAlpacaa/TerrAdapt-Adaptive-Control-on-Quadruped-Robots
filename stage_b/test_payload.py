"""
Unknown payload -- a verifiable test of the adaptive layer.

A plausible-looking theta_hat is not enough; adding a known mass gives a
derived reference.

Derivation
----------
A point mass dm is added at the trunk COM; the controller keeps using
MASS_TOTAL = 12.453 kg.

    real  : pddot = sum_f/(m+dm) + g
    model : pddot = sum_f/m + g + theta_a
    theta_a = sum_f (1/(m+dm) - 1/m) = -sum_f dm / (m(m+dm))

In quasi-steady state sum_f_z = (m+dm) g, so theta_z = -g dm/m.
body_inertia is unchanged (a point mass at the COM adds no inertia about
its own axes): a clean one-variable experiment.

The criterion is a DIFFERENCE: theta_z(dm) - theta_z(0) =? -g dm/m.
Without a payload theta_z is already ~+0.66 m/s^2 (the SRB model misses
leg motion), so comparing absolute values is wrong. dm is swept to check
linearity: the change must be proportional to dm with slope -g/m.

Usage:
    python test_payload.py                      # dm = 1,2,3,4 kg
    python test_payload.py --dm 2 --T 10
    python test_payload.py --kinds M0 M2
"""

import argparse
import numpy as np
import mujoco

from a1_params import MASS_TOTAL, GRAVITY, LEGS
from controller import Controller
from run_walk import (find_scene, quat_to_R, euler_zyx, read_state,
                      foot_geom_ids, read_joint_params)


def load_model(payload=0.0):
    """Load the model, optionally with a point mass at the trunk COM."""
    m = mujoco.MjModel.from_xml_path(find_scene())
    if payload != 0.0:
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        if tid < 0:
            tid = 1
        m.body_mass[tid] = float(m.body_mass[tid]) + payload
        mujoco.mj_setConst(m, mujoco.MjData(m))
    return m


def leg_data(m):
    masses, coms, inert = [], {}, {}
    for key in ["hip", "thigh", "calf"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"FR_{key}")
        masses.append(float(m.body_mass[bid]))
        coms[key] = m.body_ipos[bid].copy()
        Riq = quat_to_R(m.body_iquat[bid])
        inert[key] = Riq @ np.diag(m.body_inertia[bid]) @ Riq.T
    return tuple(masses), coms, inert


def run(kind, payload, vx=0.3, T=8.0, gamma=None, ramp=1.0):
    """One run. Returns a dict (mean theta, pitch, z, fell)."""
    m = load_model(payload)
    d = mujoco.MjData(m)
    if m.nkey > 0:
        d.qpos[:] = m.key_qpos[0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    gids = foot_geom_ids(m)
    masses, coms, inert = leg_data(m)
    armature, damping, frictionloss = read_joint_params(m)

    kw = {} if gamma is None else {"gamma": gamma}
    ctrl = Controller(gait="trot", dt_control=m.opt.timestep,
                      link_masses=masses, link_coms=coms,
                      link_inertias=inert, joint_armature=armature,
                      joint_damping=damping, joint_frictionloss=frictionloss,
                      adaptive=kind, adaptive_kw=kw)

    n = int(round(T / m.opt.timestep))
    th, pit, zs = [], [], []
    fell = None
    for i in range(n):
        t = i * m.opt.timestep
        w = min(1.0, t / max(ramp, 1e-9))
        ctrl.set_command(vx * w, 0.0, 0.0)
        X, foot_w, foot_b, q, dq, R = read_state(m, d, gids)
        tau = ctrl.compute(t, X, foot_w, foot_b, q, dq, R)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        if i > n // 2 and i % 5 == 0:
            th.append(ctrl.theta.copy())
            pit.append(float(euler_zyx(d.qpos[3:7])[1]))
            zs.append(float(d.qpos[2]))
        if d.qpos[2] < 0.12 or abs(euler_zyx(d.qpos[3:7])[0]) > 1.0:
            fell = t
            break
    if not th:
        return dict(theta=np.zeros(6), pitch=np.nan, z=np.nan, fell=fell,
                    x=float(d.qpos[0]))
    return dict(theta=np.mean(th, axis=0), pitch=float(np.mean(pit)),
                z=float(np.mean(zs)), fell=fell, x=float(d.qpos[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dm", type=float, nargs="+",
                    default=[1.0, 2.0, 3.0, 4.0])
    ap.add_argument("--kinds", nargs="+", default=["M0", "M1", "M2"])
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--T", type=float, default=8.0)
    ap.add_argument("--gamma", type=float, default=None)
    args = ap.parse_args()

    print("UNKNOWN PAYLOAD TEST")
    print(f"trot, vx={args.vx} m/s, T={args.T}s, "
          f"mass known to the controller {MASS_TOTAL} kg\n")
    print("Derivation:  theta_z(dm) - theta_z(0) = -g dm/m  (a difference)\n")

    # --- baseline: no adaptive layer, with/without payload -----------
    print("A) WITHOUT ADAPTIVE LAYER: what does the payload do?")
    print(f"   {'dm (kg)':>8} {'pitch':>9} {'z':>8} {'distance':>9} {'status':>10}")
    print("   " + "-" * 48)
    for dm in [0.0] + list(args.dm):
        r = run("none", dm, args.vx, args.T)
        st = "STANDING" if r["fell"] is None else f"FELL {r['fell']:.1f}s"
        print(f"   {dm:>8.1f} {r['pitch']:>+9.4f} {r['z']:>8.4f} "
              f"{r['x']:>9.3f} {st:>10}")

    # --- approximators ---------------------------------------------------
    for kind in args.kinds:
        print(f"\nB) {kind}")
        base = run(kind, 0.0, args.vx, args.T, args.gamma)
        print(f"   theta_hat without payload: a = "
              f"{np.round(base['theta'][0:3],4).tolist()} m/s^2")
        print(f"                              alpha = "
              f"{np.round(base['theta'][3:6],4).tolist()} rad/s^2")
        print(f"   pitch without payload={base['pitch']:+.4f}  z={base['z']:.4f}\n")
        print(f"   {'dm':>6} {'expected':>10} {'measured':>10} {'error':>8} "
              f"{'pitch':>9} {'z':>8} {'status':>8}")
        print("   " + "-" * 64)
        rows = []
        for dm in args.dm:
            r = run(kind, dm, args.vx, args.T, args.gamma)
            d_meas = r["theta"][2] - base["theta"][2]
            d_pred = -GRAVITY * dm / MASS_TOTAL
            err = 100.0 * abs(d_meas - d_pred) / abs(d_pred)
            rows.append((dm, d_pred, d_meas, err))
            st = "up" if r["fell"] is None else f"F{r['fell']:.1f}"
            print(f"   {dm:>6.1f} {d_pred:>+10.3f} {d_meas:>+10.3f} "
                  f"{err:>7.1f}% {r['pitch']:>+9.4f} {r['z']:>8.4f} "
                  f"{st:>8}")
        R = np.array(rows)
        # linearity: least-squares slope through the origin, expected -g/m
        slope = float(np.sum(R[:, 0] * R[:, 2]) / np.sum(R[:, 0] ** 2))
        slope_pred = -GRAVITY / MASS_TOTAL
        resid = R[:, 2] - slope * R[:, 0]
        r2 = 1.0 - np.sum(resid**2) / np.sum((R[:, 2] - R[:, 2].mean())**2)
        print("\n   LINEARITY (least squares through the origin)")
        print(f"     measured slope : {slope:+.4f} m/s^2 / kg")
        print(f"     expected       : -g/m = {slope_pred:+.4f}")
        print(f"     slope error    : {100*abs(slope-slope_pred)/abs(slope_pred):.1f}%")
        print(f"     R^2            : {r2:.4f}")
        ok = (abs(slope - slope_pred) / abs(slope_pred) < 0.15) and r2 > 0.9
        print(f"     {'PASS' if ok else 'FAIL'}: estimate proportional to the "
              f"payload with the right slope")

    print("\nNOTE: theta_hat_x is ~ -1.6 m/s^2 (-20 N) in every run and does")
    print("not depend on the payload: the forward momentum of the swing legs,")
    print("which the SRB model ignores (Di Carlo Assumption 1).")


if __name__ == "__main__":
    main()
