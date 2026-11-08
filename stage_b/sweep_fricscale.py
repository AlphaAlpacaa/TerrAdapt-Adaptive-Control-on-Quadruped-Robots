"""
Friction-compensation parameter uncertainty -- stability limit sweep.

Finding: OVER-estimating joint friction destabilises the swing leg.

Derivation
----------
Swing-leg joint dynamics (MuJoCo joint model):

    M(q) qddot + b qdot + f sign(qdot) + g(q) = tau

The controller compensates friction with a scale k_f (= fricscale):

    tau = J^T [kp (p_d - p) + kd (v_d - v)] + g(q) + M J^-1 a_d
          + k_f [ b qdot + f tanh(qdot/eps) ]

Near small errors the qdot coefficient becomes the NET DAMPING

    M qddot + [ (1 - k_f) b + kd J^T J ] qdot + ... = kp J^T (p_d - p)

  k_f < 1 : under-compensation -> positive extra damping; tracking worse,
            stability kept
  k_f = 1 : full compensation; only the PD's kd remains
  k_f > 1 : over-compensation -> (1-k_f) b negative; beyond a point kd
            cannot overcome it, the net damping turns negative and the
            leg's joint speed feeds itself.

Sufficient condition: D(k_f, q) = (1 - k_f) B_eff + kd J(q)^T J(q) > 0,
with B_eff = diag(b + f/eps) (slope of tanh at zero speed included -- the
joint speed passes through zero at every lift-off and touchdown), and the
smallest eigenvalue (not the diagonal) over the locomotion workspace.

Corrected estimate (eps = 0.05): nominal stance k_f* = 1.067, worst
configuration k_f* = 1.023. Measured: 1.20 standing, 1.30 fell. The
estimate lies BELOW the measurement, as a sufficient condition should
(the kp term still restores, the worst configuration is not held
indefinitely, and the analysis is a linearisation).

Why it matters: "over-compensate to be safe" is unstable; the uncertain
parameter must be estimated adaptively. Its structure is known
(b qdot + f sign(qdot)), so a classical regressor can capture it.

Usage:
    python sweep_fricscale.py                      # trot in place
    python sweep_fricscale.py --vx 0.3 --T 6
    python sweep_fricscale.py --lo 0.4 --hi 1.7 --n 14
"""

import argparse
import numpy as np
import mujoco

from a1_params import LEGS, Q_NOMINAL, TAU_MAX
from a1_kinematics import foot_jacobian
from controller import Controller
from run_walk import (find_scene, quat_to_R, euler_zyx, read_state,
                      foot_geom_ids, read_joint_params)


def kcrit_at(q, kd, damping, frictionloss, eps):
    """
    Bisection for lambda_min[(1-k) B_eff + kd J(q)^T J(q)] = 0, with
    B_eff = diag(b + f/eps); the full matrix is used, not a scalar bound.
    """
    J = foot_jacobian("FR", np.asarray(q, float))
    D = kd * (J.T @ J)
    B_eff = np.diag(np.asarray(damping, float) +
                    np.asarray(frictionloss, float) / eps)
    lo, hi = 1.0, 8.0
    if np.linalg.eigvalsh((1 - hi) * B_eff + D).min() > 0:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.linalg.eigvalsh((1 - mid) * B_eff + D).min() > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def predicted_critical(kd, damping, frictionloss, eps=0.05, n=3000, seed=0):
    """
    Returns (nominal, worst configuration, worst q); the worst case is
    searched over the locomotion workspace.
    """
    k_nom = kcrit_at(Q_NOMINAL, kd, damping, frictionloss, eps)
    lo = np.array([-0.35, 0.30, -2.50])
    hi = np.array([0.35, 1.70, -0.95])
    rng = np.random.default_rng(seed)
    k_worst, q_worst = np.inf, None
    for _ in range(n):
        q = rng.uniform(lo, hi)
        k = kcrit_at(q, kd, damping, frictionloss, eps)
        if k < k_worst:
            k_worst, q_worst = k, q
    return k_nom, k_worst, q_worst


def run_once(m, gids, masses, coms, inertias, armature, damping, frictionloss,
             fricscale, gait, vx, T, ramp=1.0, kp=None):
    """One run. Returns a dict."""
    d = mujoco.MjData(m)
    if m.nkey > 0:
        d.qpos[:] = m.key_qpos[0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    ctrl = Controller(gait=gait, dt_control=m.opt.timestep,
                      link_masses=masses, link_coms=coms,
                      link_inertias=inertias,
                      joint_armature=armature, joint_damping=damping,
                      joint_frictionloss=frictionloss,
                      friction_scale=fricscale)
    if kp is not None:
        ctrl.kp_swing = kp
        ctrl.kd_swing = kp / 28.0

    dt = m.opt.timestep
    n = int(round(T / dt))
    errs, taus, fts, pitches, rolls, zs = [], [], [], [], [], []
    fell_at = None
    for i in range(n):
        t = i * dt
        w = min(1.0, t / max(ramp, 1e-9))
        ctrl.set_command(vx * w, 0.0, 0.0)
        X, foot_w, foot_b, q, dq, R = read_state(m, d, gids)
        tau = ctrl.compute(t, X, foot_w, foot_b, q, dq, R)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        if i % 5 == 0:
            e = euler_zyx(d.qpos[3:7])
            errs.append(float(np.max(ctrl.swing_err)))
            taus.append(float(np.max(np.abs(tau))))
            fts.append(float(np.max(ctrl.fric_tau)))
            rolls.append(float(e[0]))
            pitches.append(float(e[1]))
            zs.append(float(d.qpos[2]))

        if d.qpos[2] < 0.12 or abs(euler_zyx(d.qpos[3:7])[0]) > 1.0:
            fell_at = t
            break

    half = max(len(errs) // 2, 1)
    return dict(
        fell=fell_at, T_survived=len(errs) * 5 * dt,
        err=float(np.mean(errs[half:])) if errs else np.nan,
        err_max=float(np.max(errs)) if errs else np.nan,
        tau_max=float(np.max(taus)) if taus else np.nan,
        fric_max=float(np.max(fts)) if fts else np.nan,
        roll=float(np.mean(rolls[half:])) if rolls else np.nan,
        roll_std=float(np.std(rolls[half:])) if rolls else np.nan,
        pitch=float(np.mean(pitches[half:])) if pitches else np.nan,
        z=float(np.mean(zs[half:])) if zs else np.nan,
        x=float(d.qpos[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gait", default="trot")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--lo", type=float, default=0.4)
    ap.add_argument("--hi", type=float, default=1.6)
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--kp", type=float, default=None)
    ap.add_argument("--eps", type=float, default=0.05,
                    help="tanh transition band of the Coulomb compensation "
                         "(rad/s); larger widens the stable band but weakens "
                         "the compensation near zero speed")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(find_scene())
    gids = foot_geom_ids(m)

    masses, coms, inertias = [], {}, {}
    for key in ["hip", "thigh", "calf"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"FR_{key}")
        masses.append(float(m.body_mass[bid]))
        coms[key] = m.body_ipos[bid].copy()
        Riq = quat_to_R(m.body_iquat[bid])
        inertias[key] = Riq @ np.diag(m.body_inertia[bid]) @ Riq.T
    masses = tuple(masses)
    armature, damping, frictionloss = read_joint_params(m)

    kd = (args.kp / 28.0) if args.kp is not None else 25.0
    eps = args.eps
    k_nom, k_worst, q_worst = predicted_critical(kd, damping, frictionloss, eps)
    print("FRICTION COMPENSATION STABILITY LIMIT")
    print(f"gait={args.gait} vx={args.vx} T={args.T}s kd={kd:.1f} "
          f"eps={eps}\n")
    print(f"model: damping={damping.tolist()} "
          f"frictionloss={frictionloss.tolist()}")
    print("\nSUFFICIENT LIMIT:  lambda_min[(1-k) B_eff + kd J^T J] = 0")
    print(f"   B_eff = diag(b + f/eps) = "
          f"{np.round(damping + frictionloss/eps, 2).tolist()}")
    print(f"   nominal stance      : k_f* = {k_nom:.3f}")
    print(f"   worst configuration : k_f* = {k_worst:.3f}  "
          f"(q = {np.round(q_worst,3).tolist()})")
    print("   Sufficient condition: stability guaranteed below it; the")
    print("   measured limit is expected to lie above these values.")

    print(f"\n{'k_f':>6} {'status':>12} {'swing':>9} {'tau max':>9} "
          f"{'fric FF':>9} {'roll std':>9} {'pitch':>8} {'z':>7} {'x':>7}")
    print("-" * 84)
    scales = np.linspace(args.lo, args.hi, args.n)
    last_ok = None
    first_bad = None
    for ks in scales:
        r = run_once(m, gids, masses, coms, inertias, armature, damping,
                     frictionloss, float(ks), args.gait, args.vx, args.T,
                     kp=args.kp)
        if r["fell"] is None:
            state = "STANDING"
            last_ok = ks if (last_ok is None or ks > last_ok) else last_ok
        else:
            state = f"FELL {r['fell']:.2f}s"
            if first_bad is None or ks < first_bad:
                first_bad = ks
        print(f"{ks:>6.2f} {state:>12} {r['err']*1000:>8.1f}mm "
              f"{r['tau_max']:>8.1f} {r['fric_max']:>8.1f} "
              f"{r['roll_std']:>9.4f} {r['pitch']:>+8.4f} "
              f"{r['z']:>7.4f} {r['x']:>7.3f}")

    print("\nRESULT")
    if last_ok is not None and first_bad is not None:
        print(f"   measured limit     : {last_ok:.2f} standing, {first_bad:.2f} fell")
        print(f"   sufficient limit   : nominal {k_nom:.3f}, "
              f"worst configuration {k_worst:.3f}")
        if k_worst <= last_ok:
            print(f"   PASS: the theoretical limit ({k_worst:.3f}) lies BELOW the "
                  f"measured one ({last_ok:.2f}) -> conservative, as expected.")
            print(f"   margin: {last_ok/k_worst:.2f}x")
        else:
            print("   FAIL: the theoretical limit lies ABOVE the measured one;")
            print("   the sufficient condition is violated -- derivation error.")
    elif first_bad is None:
        print(f"   no fall in the swept range ({args.lo}..{args.hi}); "
              f"increase --hi")
    else:
        print("   never stayed up in the swept range; decrease --lo")
    print("\n   k_f < 1: under-compensation -> tracking degrades, stays stable")
    print("   k_f > 1: over-compensation -> net damping turns negative, FALLS")


if __name__ == "__main__":
    main()
