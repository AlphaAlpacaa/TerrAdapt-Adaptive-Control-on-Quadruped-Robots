"""
Do the ground settings actually act? -- direct physics measurement.

With --mu 0.45 and --soft 0.12 in sweep_terrain.py the baseline numbers
did not change (pitch 0.0850 -> 0.0858 / 0.0809, z 0.2636 -> 0.2638 /
0.2633): either the settings were not applied or their effect was below
the noise, and in both cases those rungs still measured flat ground.

This script looks at behaviour, not at model parameters:
  penetration      d.contact[i].dist (negative = interpenetration);
                   must grow on soft ground, otherwise solref has no effect
  required mu      sqrt(ft1^2 + ft2^2) / fn from mj_contactForce;
                   any mu above it has no effect
  actual slip      horizontal foot displacement while in contact
  contact count    deviation from the plan

Usage:
    python verify_terrain.py
    python verify_terrain.py --T 4
"""

import argparse
import numpy as np
import mujoco

from a1_params import LEGS, GRAVITY, MASS_TOTAL
from controller import Controller
from run_walk import (find_scene, quat_to_R, euler_zyx, read_state,
                      foot_geom_ids, read_joint_params)


CASES = [
    ("reference (hard, grip)", {}),
    ("--soft 0.05",            {"soft": 0.05}),
    ("--soft 0.12",            {"soft": 0.12}),
    ("--soft 0.30",            {"soft": 0.30}),
    ("--soft 0.12 incl. feet", {"soft": 0.12, "soft_feet": True}),
    ("--soft 0.30 incl. feet", {"soft": 0.30, "soft_feet": True}),
    ("--mu 0.45",              {"mu": 0.45}),
    ("--mu 0.25",              {"mu": 0.25}),
    ("--mu 0.15",              {"mu": 0.15}),
    ("--mu 0.08",              {"mu": 0.08}),
]


def floor_geom(m):
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 and int(m.geom_type[g]) in (0, 1):
            return g
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:
            return g
    return None


def build(cfg):
    m = mujoco.MjModel.from_xml_path(find_scene())
    fg = floor_geom(m)
    feet = foot_geom_ids(m)
    info = {}
    if fg is None:
        raise RuntimeError("floor geom not found")
    info["floor_type"] = int(m.geom_type[fg])
    info["solref0"] = m.geom_solref[fg].copy()
    info["solref_foot0"] = m.geom_solref[feet[0]].copy()
    info["solmix_floor"] = float(m.geom_solmix[fg])
    info["solmix_foot"] = float(m.geom_solmix[feet[0]])
    if "mu" in cfg:
        m.geom_friction[fg, 0] = cfg["mu"]
        for g in feet:
            m.geom_friction[g, 0] = cfg["mu"]
    if "soft" in cfg:
        m.geom_solref[fg, 0] = cfg["soft"]
        if cfg.get("soft_feet"):
            for g in feet:
                m.geom_solref[g, 0] = cfg["soft"]
    info["solref"] = m.geom_solref[fg].copy()
    info["mu_floor"] = float(m.geom_friction[fg, 0])
    info["mu_foot"] = float(m.geom_friction[feet[0], 0])
    return m, fg, feet, info


def measure(cfg, vx=0.3, T=6.0, adaptive="none"):
    m, fg, feet, info = build(cfg)
    d = mujoco.MjData(m)
    if m.nkey > 0:
        d.qpos[:] = m.key_qpos[0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    gids = feet

    masses, coms, inert = [], {}, {}
    for key in ["hip", "thigh", "calf"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"FR_{key}")
        masses.append(float(m.body_mass[bid]))
        coms[key] = m.body_ipos[bid].copy()
        Riq = quat_to_R(m.body_iquat[bid])
        inert[key] = Riq @ np.diag(m.body_inertia[bid]) @ Riq.T
    armature, damping, frictionloss = read_joint_params(m)

    ctrl = Controller(gait="trot", dt_control=m.opt.timestep,
                      link_masses=tuple(masses), link_coms=coms,
                      link_inertias=inert, joint_armature=armature,
                      joint_damping=damping, joint_frictionloss=frictionloss,
                      adaptive=adaptive)

    n = int(round(T / m.opt.timestep))
    pen, mu_req, ncon = [], [], []
    slip = {l: 0.0 for l in LEGS}
    prev_xy = {l: None for l in LEGS}
    pit, zs = [], []
    fell = None
    gid_of = dict(zip(LEGS, gids))
    frc = np.zeros(6)

    for i in range(n):
        t = i * m.opt.timestep
        w = min(1.0, t / 1.0)
        ctrl.set_command(vx * w, 0.0, 0.0)
        X, foot_w, foot_b, q, dq, R = read_state(m, d, gids)
        tau = ctrl.compute(t, X, foot_w, foot_b, q, dq, R)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        if t > 2.0:                     # after the ramp-up
            nc = 0
            touching = set()
            for ci in range(d.ncon):
                c = d.contact[ci]
                leg = None
                for L in LEGS:
                    if c.geom1 == gid_of[L] or c.geom2 == gid_of[L]:
                        leg = L
                if leg is None:
                    continue
                nc += 1
                touching.add(leg)
                pen.append(-float(c.dist))         # positive = penetration
                mujoco.mj_contactForce(m, d, ci, frc)
                fn = abs(frc[0])
                ft = np.hypot(frc[1], frc[2])
                if fn > 1.0:
                    mu_req.append(ft / fn)
            ncon.append(nc)
            # slip: horizontal displacement of a foot in contact
            for j, L in enumerate(LEGS):
                xy = d.geom_xpos[gid_of[L]][:2].copy()
                if L in touching and prev_xy[L] is not None:
                    slip[L] += float(np.linalg.norm(xy - prev_xy[L]))
                prev_xy[L] = xy if L in touching else None
            pit.append(float(euler_zyx(d.qpos[3:7])[1]))
            zs.append(float(d.qpos[2]))

        if d.qpos[2] < 0.12 or abs(euler_zyx(d.qpos[3:7])[0]) > 1.0:
            fell = t
            break

    dur = max(T - 2.0, 1e-9)
    return dict(
        info=info, fell=fell,
        pen_mean=float(np.mean(pen)) if pen else np.nan,
        pen_max=float(np.max(pen)) if pen else np.nan,
        mu_mean=float(np.mean(mu_req)) if mu_req else np.nan,
        mu_p95=float(np.percentile(mu_req, 95)) if mu_req else np.nan,
        mu_max=float(np.max(mu_req)) if mu_req else np.nan,
        ncon=float(np.mean(ncon)) if ncon else np.nan,
        slip=sum(slip.values()) / dur,
        pitch=float(np.mean(pit)) if pit else np.nan,
        z=float(np.mean(zs)) if zs else np.nan,
        x=float(d.qpos[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=6.0)
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--adaptive", default="none")
    args = ap.parse_args()

    print("GROUND SETTINGS CHECK -- behaviour, not parameters")
    print(f"trot, vx={args.vx}, T={args.T}s, layer={args.adaptive}")
    print("(first 2 s ramp-up, measured afterwards)\n")

    base = None
    print(f"{'setting':<24} {'penetr.(mm)':>11} {'required mu':>18} "
          f"{'slip':>9} {'contacts':>8} {'pitch':>9} {'z':>8} {'status':>8}")
    print(f"{'':24} {'mean / max':>11} {'mean / p95 / max':>18} "
          f"{'mm/s':>9}")
    print("-" * 100)
    for name, cfg in CASES:
        try:
            r = measure(cfg, args.vx, args.T, args.adaptive)
        except Exception as e:
            print(f"{name:<24} ERROR: {e}")
            continue
        if base is None:
            base = r
        st = "up" if r["fell"] is None else f"F{r['fell']:.1f}"
        print(f"{name:<24} {r['pen_mean']*1000:5.2f} /{r['pen_max']*1000:5.2f} "
              f"{r['mu_mean']:5.2f} /{r['mu_p95']:5.2f} /{r['mu_max']:5.2f} "
              f"{r['slip']*1000:8.1f} {r['ncon']:8.2f} "
              f"{r['pitch']:>+9.4f} {r['z']:>8.4f} {st:>8}")

    i = base["info"]
    print("\nMODEL INFO")
    print(f"  floor geom type   : {i['floor_type']} (0=plane 1=hfield)")
    print(f"  floor solref      : {np.round(i['solref0'],4).tolist()}")
    print(f"  foot solref       : {np.round(i['solref_foot0'],4).tolist()}")
    print(f"  solmix floor/foot : {i['solmix_floor']:.2f} / "
          f"{i['solmix_foot']:.2f}")
    print("  MuJoCo mixes solref between the two geoms with the solmix")
    print("  weights; softening only the floor halves the effect (the")
    print("  'incl. feet' rows test this).")

    print("\nHOW TO READ")
    print("  penetration: if it does not grow vs the reference, solref has no effect.")
    print("  required mu: any mu above it has no effect; for slip, mu must go")
    print("     below the mean (occasional slip below p95).")
    print("  slip: ~0 with grip; grows when mu actually binds.")
    print("  A setting that changes none of these measurements leaves the rung")
    print("  equal to flat ground, and comparisons on it are INVALID.")


if __name__ == "__main__":
    main()
