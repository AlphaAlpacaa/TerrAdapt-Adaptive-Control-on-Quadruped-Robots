"""
Course run: one long run in which the ground and the payload change while
the robot walks.

Adaptive control matters when conditions CHANGE: the adaptation has to
track the change, and how fast each method recovers can be measured.

Default course (time-triggered events, cumulative):
   0- 4 s  flat hard ground, no payload
   4- 8 s  + off-centre payload           -- force AND moment disturbance
   8-12 s  + ground softens               -- contact-dependent disturbance
  12-16 s  + ground becomes slippery      -- friction cone violated
  16-20 s  + slope                        -- gravity rotates
With --terrain / --obstacles the ground changes with position instead
(course_def.terrain_course) and only the payload is time-triggered.

The payload is off-centre so that it excites force and moment components
together. It is a visible body in the model (make_course_model.py); at the
event its mass and colour change together.

Usage:
    python make_course_model.py            # once, builds the model
    python scenario.py --adaptive M2 --view
    python scenario.py --adaptive none     # baseline
    python scenario.py --compare           # none/M0/M1/M2 in sequence
"""

import argparse
import contextlib
from pathlib import Path
import sys
import time
import numpy as np
import mujoco

from a1_params import LEGS, GRAVITY, MASS_TOTAL
from controller import Controller
from run_walk import (quat_to_R, euler_zyx, read_state, foot_geom_ids,
                      read_joint_params)
from make_course_model import (PAYLOAD_BODY, PAYLOAD_GEOM, box_inertia,
                               find_model_dir)
from course_def import (default_course, random_payload_pos,
                        terrain_course, TERRAIN_ZONES)


# Root folder for run outputs. The name is kept from the original project
# so that recorded measurements stay where they are.
RESULTS_ROOT = "sonuclar"


def run_label(args):
    """
    Folder name that identifies a run, so that plots, logs and console
    output are never overwritten by the next run.

    examples:  terrain_g300_y+0.12+0.05_s0
               obstacles-fixed_gvars_yseed_s2
    """
    course = ("obstacles" if args.obstacles
              else ("terrain" if args.terrain else "flat"))
    if args.obstacles:
        try:
            import json
            zf = Path(__file__).resolve().parent / "obstacle_zone.json"
            z = json.loads(zf.read_text(encoding="utf-8"))
            course += "-loose" if z["loose"] else "-fixed"
            if z["base"] == "flat":
                course += "-flatbase"
        except Exception:
            pass
    g = f"g{args.gamma:g}" if args.gamma else "gvars"
    if args.ppos is not None:
        y = f"y{args.ppos[0]:+.2f}{args.ppos[1]:+.2f}".replace(" ", "")
    else:
        y = "yseed"
    if getattr(args, "pendulum", None) is not None:
        course = f"pendulum{args.pendulum:g}"
    parts = [course, g, y, f"s{args.seed}"]
    # Filtered and unfiltered runs must not share a folder.
    if getattr(args, "filter", False):
        parts.append("filtered")
    if getattr(args, "emag", "tracking") != "tracking":
        parts.append("emag-prediction")
    if getattr(args, "tag", None):
        parts.append(args.tag)
    return "_".join(parts)


class Tee:
    """Writes to the console and to a file at the same time."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.std = sys.stdout

    def write(self, s):
        self.std.write(s)
        self.f.write(s)

    def flush(self):
        self.std.flush()
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def zone_list(terrain, obstacles):
    """
    Zone labels for the live panel. plot_run.py is not imported here: it
    pins matplotlib to the non-interactive Agg backend.
    """
    if not terrain:
        return []
    z = list(TERRAIN_ZONES)
    if obstacles:
        import json
        f = Path(__file__).resolve().parent / "obstacle_zone.json"
        if f.exists():
            o = json.loads(f.read_text(encoding="utf-8"))
            if o["base"] == "flat":
                return [(0.0, o["x0"], "flat"),
                        (o["x0"], o["x1"], f"OBSTACLES ({o['n']})")]
            z = [t for t in z if t[0] < o["x0"]]
            z.append((o["x0"], o["x1"], f"OBSTACLES ({o['n']})"))
    return z


def course_end(terrain, obstacles):
    """
    x (m) of the finish line. Without it the robot kept walking and fell
    off the edge of the height map (x = 12.0), and the fall leaked into the
    segment statistics.
    """
    if not (terrain or obstacles):
        return None
    x_end = 7.4                      # end of the terrain-course descent
    if obstacles:
        import json
        f = Path(__file__).resolve().parent / "obstacle_zone.json"
        if f.exists():
            x_end = float(json.loads(f.read_text(encoding="utf-8"))["x1"])
    return x_end + 0.6               # fully out of the zone


def find_scene_course(terrain=False, obstacles=False):
    mdir = find_model_dir()
    if obstacles:
        name, need = "scene_obstacles.xml", "make_obstacle_model.py"
    elif terrain:
        name, need = "scene_terrain.xml", "make_terrain_model.py"
    else:
        name, need = "scene_course.xml", "make_course_model.py"
    p = mdir / name
    if not p.exists():
        raise FileNotFoundError(f"{name} missing -- run 'python {need}' first")
    return str(p)


def load_hfield(m):
    """
    Load the height-field data from .npy. The XML heightfield carries only
    the grid size and scale, not the heights; without this the terrain is
    silently flat.
    """
    hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_HFIELD, "course_hf")
    if hid < 0:
        # obstacle scene built on a flat base: no heightfield, not an error
        return None
    mdir = find_model_dir()
    f = mdir / "course_hfield.npy"
    if not f.exists():
        raise FileNotFoundError("course_hfield.npy missing -- make_terrain_model.py")
    H = np.load(f)
    z_top = float(m.hfield_size[hid, 2])
    m.hfield_data[:] = (H / z_top).ravel().astype(np.float32)
    return float(H.max())


def set_const_safe(m, d):
    """
    Call mj_setConst without corrupting the state. mj_setConst uses d as
    scratch and resets qpos to qpos0, which teleported the robot back to
    the start when the payload was attached. Save, call, restore, forward.
    """
    qpos = d.qpos.copy()
    qvel = d.qvel.copy()
    act = d.act.copy() if d.act.size else None
    ctrl = d.ctrl.copy()
    tt = d.time
    mujoco.mj_setConst(m, d)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    if act is not None:
        d.act[:] = act
    d.ctrl[:] = ctrl
    d.time = tt
    mujoco.mj_forward(m, d)


def floor_geom(m):
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 and int(m.geom_type[g]) in (0, 1):
            return g
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:
            return g
    return None


def leg_data(m):
    masses, coms, inert = [], {}, {}
    for key in ["hip", "thigh", "calf"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"FR_{key}")
        masses.append(float(m.body_mass[bid]))
        coms[key] = m.body_ipos[bid].copy()
        Riq = quat_to_R(m.body_iquat[bid])
        inert[key] = Riq @ np.diag(m.body_inertia[bid]) @ Riq.T
    return tuple(masses), coms, inert


def run_course(adaptive="none", gamma=None, vx=0.3, T=20.0, course=None,
               seed=0, view=False, ramp=1.0, payload_pos=None,
               terrain=False, obstacles=False,
               live=False, speed=0.0, live_hz=15.0,
               use_filter=False, emag_source="tracking", pendulum=None,
               load_weights=None, save_weights=None):
    """Run the course once. Returns (log dict, event list).

    pendulum: None or {"L": m, "axis": "x"|"y", "damping": N m s/rad}.
    If given, the payload hangs from a pivot at payload_pos on a pendulum of
    length L (make_pendulum_model.py). Flat ground only. The pendulum angle
    is logged in L["pendulum"].
    load_weights / save_weights: warm start (.npy path or array) / save the
    final approximator weights.
    """
    if course is None:
        course = terrain_course() if (terrain or obstacles) else default_course()
    rng = np.random.default_rng(seed)
    if payload_pos is None:
        payload_pos = random_payload_pos(rng)

    if pendulum is not None:
        if terrain or obstacles:
            raise ValueError("pendulum only on flat ground")
        m = mujoco.MjModel.from_xml_path(
            str(find_model_dir() / "scene_pendulum.xml"))
    else:
        m = mujoco.MjModel.from_xml_path(find_scene_course(terrain, obstacles))
    # The obstacle scene can be built on terrain or on a flat base; the
    # presence of a heightfield tells which.
    hmax = load_hfield(m) if (terrain or obstacles) else None
    rel_z = hmax is not None      # fall criterion relative to the ground
    d = mujoco.MjData(m)
    if m.nkey > 0:
        d.qpos[:] = m.key_qpos[0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    gids = foot_geom_ids(m)
    fg = floor_geom(m)
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PAYLOAD_BODY)
    pgid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, PAYLOAD_GEOM)
    # payload position for this run (overrides the XML default)
    pjq = None
    if pendulum is None:
        m.body_pos[pid] = payload_pos
    else:
        # pivot at the payload position, mass L below the pivot (body frame)
        vid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pendulum_pivot")
        pj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "pendulum_hinge")
        assert vid >= 0 and pj >= 0, "run first: python make_pendulum_model.py"
        m.body_pos[vid] = payload_pos
        m.body_pos[pid] = [0.0, 0.0, -float(pendulum["L"])]
        m.jnt_axis[pj] = ([1.0, 0, 0] if pendulum.get("axis", "x") == "x"
                          else [0, 1.0, 0])
        if "damping" in pendulum:
            m.dof_damping[m.jnt_dofadr[pj]] = float(pendulum["damping"])
        pjq = int(m.jnt_qposadr[pj])
        mujoco.mj_forward(m, d)

    masses, coms, inert = leg_data(m)
    armature, damping, frictionloss = read_joint_params(m)
    kw = {} if gamma is None else {"gamma": gamma}
    kw["seed"] = seed
    kw["use_filter"] = bool(use_filter)
    kw["emag_source"] = emag_source
    ctrl = Controller(gait="trot", dt_control=m.opt.timestep,
                      link_masses=masses, link_coms=coms, link_inertias=inert,
                      joint_armature=armature, joint_damping=damping,
                      joint_frictionloss=frictionloss,
                      adaptive=adaptive, adaptive_kw=kw)
    if load_weights is not None:
        w0 = (np.load(load_weights) if isinstance(load_weights, (str, Path))
              else np.asarray(load_weights, float))
        ctrl.layer.load_weights(w0)       # applied after the first reset()

    print(f"\n{'='*70}")
    print(f"COURSE  layer={ctrl.layer.name}"
          + (f"  gamma={gamma:.0f}" if gamma else "")
          + f"  seed={seed}")
    print(f"{'='*70}")
    print(f"payload position (body frame): "
          f"{np.round(payload_pos, 4).tolist()} m")
    print(f"  horizontal distance from centre: "
          f"{np.linalg.norm(payload_pos[:2])*1000:.0f} mm -> creates a MOMENT")
    print("\nevents:")
    print(f"  {0.0:5.1f} s  flat hard ground, no payload (baseline)")
    for t_e, kind, val, label in course:
        print(f"  {t_e:5.1f} s  {label}")

    # Viewer on its OWN MjData copy. The passive viewer's render thread
    # works on the MjData it is given and changed the constraint-solver
    # work buffers (efc_*, efm_*) of the simulation data, which changed the
    # physics: M0 gamma=1000 on the combined course completed headless but
    # stalled with --view. The simulation now never shares its MjData with
    # the viewer; only qpos/qvel are copied into d_vis at each refresh.
    viewer = None
    d_vis = None
    if view:
        from mujoco import viewer as mj_viewer
        d_vis = mujoco.MjData(m)
        d_vis.qpos[:] = d.qpos
        d_vis.qvel[:] = d.qvel
        mujoco.mj_forward(m, d_vis)
        viewer = mj_viewer.launch_passive(m, d_vis)

    # Live plot. live_plot must be imported before plot_run, which pins
    # matplotlib to the non-interactive Agg backend.
    panel = None
    if live:
        from live_plot import LivePanel
        panel = LivePanel(adaptive, T, course=course, hz=live_hz,
                          zones=zone_list(obstacles or terrain, obstacles))

    # Real-time playback: speed=0 -> as fast as possible (measurement),
    # 1 -> real time, 3 -> three times faster (video).
    t_wall0 = time.perf_counter()
    last_sync = 0.0
    view_hz = 60.0

    n = int(round(T / m.opt.timestep))
    keys = ["t", "z", "roll", "pitch", "yaw", "vx", "vy", "tau_max",
            "swing_err", "pred_err", "n_mj_contact", "pen", "x",
            "terrain_h", "pendulum"]
    log = {k: [] for k in keys}
    log_theta = []
    log_theta_raw = []
    fired = [False] * len(course)
    events = []
    fell = None
    # Stall detector: the robot can stop without falling (it once sat at
    # 0.18 m above the ground, above the 0.12 m fall threshold, and was
    # reported as "standing" for twenty runs). Less than 5 cm of progress
    # in 4 s ends the run.
    stuck = None
    done = None
    x_finish = course_end(terrain, obstacles)
    STALL_WIN, STALL_MIN = 4.0, 0.05
    t_chk, x_chk = 0.0, float(d.qpos[0])

    try:
        for i in range(n):
            t = i * m.opt.timestep

            # --- fire events ---------------------------------------------
            for j, (t_e, kind, val, label) in enumerate(course):
                if fired[j] or t < t_e:
                    continue
                fired[j] = True
                # the model is shared with the viewer's render thread:
                # change it under the viewer lock
                with (viewer.lock() if viewer is not None
                      else contextlib.nullcontext()):
                    if kind == "payload":
                        m.body_mass[pid] = val
                        m.body_inertia[pid] = box_inertia(val)
                        m.geom_rgba[pgid, 3] = 0.85      # becomes visible
                        set_const_safe(m, d)             # without corrupting state
                    elif kind == "soft":
                        m.geom_solref[fg, 0] = val
                        for g in gids:
                            m.geom_solref[g, 0] = val
                    elif kind == "mu":
                        m.geom_friction[fg, 0] = val
                        for g in gids:
                            m.geom_friction[g, 0] = val
                    elif kind == "slope":
                        th = np.deg2rad(val)
                        m.opt.gravity[:] = [GRAVITY*np.sin(th), 0.0,
                                            -GRAVITY*np.cos(th)]
                events.append((t, label))
                print(f"  >> t={t:5.2f}s  {label}")

            w = min(1.0, t / max(ramp, 1e-9))
            ctrl.set_command(vx * w, 0.0, 0.0)
            X, foot_w, foot_b, q, dq, R = read_state(m, d, gids)
            tau = ctrl.compute(t, X, foot_w, foot_b, q, dq, R)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            if i % 5 == 0:
                e = euler_zyx(d.qpos[3:7])
                nc, pens = 0, []
                for ci in range(d.ncon):
                    c = d.contact[ci]
                    if c.geom1 in gids or c.geom2 in gids:
                        nc += 1
                        pens.append(-float(c.dist))
                log["t"].append(t)
                log["z"].append(float(d.qpos[2]))
                log["roll"].append(float(e[0]))
                log["pitch"].append(float(e[1]))
                log["yaw"].append(float(e[2]))
                log["vx"].append(float(d.qvel[0]))
                log["vy"].append(float(d.qvel[1]))
                log["tau_max"].append(float(np.max(np.abs(tau))))
                log["swing_err"].append(float(np.max(ctrl.swing_err)))
                log["pred_err"].append(
                    float(np.linalg.norm(ctrl.layer.e_tilde)))
                log["n_mj_contact"].append(nc)
                log["pen"].append(float(np.mean(pens)) if pens else 0.0)
                log["x"].append(float(d.qpos[0]))
                log["terrain_h"].append(float(ctrl.terrain_h))
                log["pendulum"].append(float(d.qpos[pjq]) if pjq is not None
                                       else 0.0)
                # ctrl.theta is what the MPC sees (after C(s) if enabled);
                # the raw estimate is logged separately for estimation metrics.
                log_theta.append(ctrl.theta.copy())
                log_theta_raw.append(ctrl.layer.theta.copy())
                if panel is not None:
                    panel.update(
                        t, float(d.qpos[0]), float(e[1]), float(e[0]),
                        float(d.qpos[2]), float(ctrl.terrain_h),
                        float(np.linalg.norm(ctrl.theta)),
                        float(np.max(np.abs(tau))))

            # Refresh the viewer at 60 Hz: copy the state into the viewer's
            # own MjData; the simulation data are never touched.
            if viewer is not None:
                now_w = time.perf_counter()
                if now_w - last_sync >= 1.0 / view_hz:
                    with viewer.lock():
                        d_vis.qpos[:] = d.qpos
                        d_vis.qvel[:] = d.qvel
                        d_vis.time = d.time
                        mujoco.mj_forward(m, d_vis)
                    viewer.sync()
                    last_sync = now_w

            # wait for wall-clock time (only with --speed)
            if speed > 0.0:
                lag = (t_wall0 + t / speed) - time.perf_counter()
                if lag > 5e-4:
                    time.sleep(lag)

            # Fall criterion relative to the ground (on terrain an absolute
            # z threshold is wrong).
            z_rel = d.qpos[2] - (ctrl.terrain_h if rel_z else 0.0)
            if z_rel < 0.12 or abs(euler_zyx(d.qpos[3:7])[0]) > 1.0:
                fell = t
                print(f"  !! t={t:.2f}s ROBOT FELL "
                      f"(z={d.qpos[2]:.3f}, roll={euler_zyx(d.qpos[3:7])[0]:+.3f})")
                break

            # finish line -- checked before the stall detector
            if x_finish is not None and d.qpos[0] > x_finish:
                done = t
                print(f"  ** t={t:.2f}s COURSE COMPLETED "
                      f"(x={d.qpos[0]:.2f} m > {x_finish:.2f} m)")
                break

            # progress check (stall without falling)
            if t - t_chk >= STALL_WIN:
                progress = float(d.qpos[0]) - x_chk
                if t > 2.0 and progress < STALL_MIN:
                    stuck = t
                    print(f"  !! t={t:.2f}s ROBOT STALLED -- only "
                          f"{progress*1000:+.0f} mm in the last "
                          f"{STALL_WIN:.0f} s (x={d.qpos[0]:.2f} m, ground "
                          f"{ctrl.terrain_h:.3f} m). Not fallen, cannot walk.")
                    break
                t_chk, x_chk = t, float(d.qpos[0])
    finally:
        if viewer is not None:
            viewer.close()
        if panel is not None:
            st = ("FELL %.1f s" % fell) if fell is not None else (
                ("STALLED %.1f s" % stuck) if stuck is not None else
                ("COMPLETED %.1f s" % done) if done is not None
                else "TIME UP")
            panel.finish(f"{st}   --   distance {d.qpos[0]:.2f} m")

    L = {k: np.array(v) for k, v in log.items()}
    L["theta"] = np.array(log_theta)            # what the MPC used
    L["theta_raw"] = np.array(log_theta_raw)    # raw estimate
    L["x_final"] = float(d.qpos[0])
    L["fell"] = fell
    L["stuck"] = stuck
    L["done"] = done
    L["payload_pos"] = payload_pos
    L["weights"] = (None if ctrl.layer.approx is None
                    else ctrl.layer.approx.get_weights())
    if save_weights is not None and L["weights"] is not None:
        np.save(save_weights, L["weights"])
    return L, events


# ---------------------------------------------------------------------------
# Segment analysis
# ---------------------------------------------------------------------------

def segment_stats(L, course, T):
    """Mean behaviour per event interval; first half = transient,
    second half = steady state."""
    bounds = [0.0] + [c[0] for c in course] + [T]
    labels = ["baseline (flat, no payload)"] + [c[3] for c in course]
    t = L["t"]
    rows = []
    for k in range(len(labels)):
        t0, t1 = bounds[k], bounds[k+1]
        msk = (t >= t0) & (t < t1)
        if msk.sum() < 4:
            continue
        # if the run ended early, split at the last available sample
        t_end = min(t1, float(t[msk].max()))
        mid = t0 + 0.5 * (t_end - t0)
        ss = (t >= mid) & (t <= t_end)   # steady state
        tr = (t >= t0) & (t < mid)       # transient
        if ss.sum() < 2 or tr.sum() < 2:
            ss = tr = msk                # too short: do not split
        rows.append(dict(
            label=labels[k], t0=t0, t1=min(t1, t_end if t_end > t0 else t1),
            pitch=float(np.mean(L["pitch"][ss])),
            roll_std=float(np.std(L["roll"][ss])),
            z=float(np.mean(L["z"][ss])),
            vx=float(np.mean(L["vx"][ss])),
            tau=float(np.max(L["tau_max"][msk])),
            swing=float(np.mean(L["swing_err"][ss])),
            pred=float(np.mean(L["pred_err"][ss])),
            pred_tr=float(np.mean(L["pred_err"][tr])),
            pen=float(np.mean(L["pen"][ss])),
        ))
    return rows


def print_segments(rows):
    print(f"\n{'segment':<34} {'pitch':>8} {'roll std':>9} {'z':>7} "
          f"{'vx':>6} {'tau':>6} {'swing':>8} {'pred. error':>18}")
    print(f"{'':34} {'':8} {'':9} {'':7} {'':6} {'':6} {'mm':>8} "
          f"{'transient / steady':>18}")
    print("-" * 104)
    for r in rows:
        print(f"{r['t0']:4.0f}-{r['t1']:<3.0f} {r['label']:<25} "
              f"{r['pitch']:>+8.4f} {r['roll_std']:>9.4f} {r['z']:>7.4f} "
              f"{r['vx']:>6.3f} {r['tau']:>6.2f} {r['swing']*1000:>8.1f} "
              f"{r['pred_tr']:>8.4f} /{r['pred']:>8.4f}")
    print("\n  'transient' = first half of the segment (right after the event)")
    print("  'steady'    = second half (after the adaptation has recovered)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptive", default="none",
                    choices=["none", "M0I", "M0", "M0F", "M1", "M2"])
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--payload-pos", dest="ppos", type=float, nargs=3,
                    default=None, metavar=("X", "Y", "Z"),
                    help="payload position in the body frame (m). The "
                         "seeded default has y ~ 0, which leaves the lateral "
                         "axis unexcited.")
    ap.add_argument("--filter", action="store_true",
                    help="turn ON the L1 low-pass filter C(s) (paper Eqs. "
                         "33/45, wn=60, zeta=0.7). Off by default: worse at "
                         "every gain in a 54-run comparison.")
    ap.add_argument("--emag", default="tracking",
                    choices=["tracking", "prediction"],
                    help="regressor magnitude: 'tracking' = ||e|| (paper "
                         "Eqs. 27/30, default), 'prediction' = ||e_tilde|| "
                         "(first version; quadratic feedback loop).")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--live-hz", dest="live_hz", type=float, default=15.0,
                    help="live plot refresh rate (Hz); lower runs faster")
    ap.add_argument("--live", action="store_true",
                    help="open a live plot window next to the viewer")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="playback speed relative to real time. 0 = as fast "
                         "as possible (measurement), 1 = real time, "
                         "3 = three times faster (video)")
    ap.add_argument("--compare", action="store_true",
                    help="none/M0/M1/M2 in sequence on the same course")
    ap.add_argument("--terrain", action="store_true",
                    help="terrain course (scene_terrain.xml)")
    ap.add_argument("--obstacles", action="store_true",
                    help="obstacle field (scene_obstacles.xml, "
                         "make_obstacle_model.py)")
    ap.add_argument("--pendulum", type=float, default=None, metavar="L",
                    help="hang the payload on a pendulum of length L (m), "
                         "flat ground; run make_pendulum_model.py first. "
                         "Measured setting: 0.04")
    ap.add_argument("--save-weights", dest="save_weights", default=None,
                    help="save the final approximator weights (.npy)")
    ap.add_argument("--load-weights", dest="load_weights", default=None,
                    help="warm start from .npy weights")
    ap.add_argument("--tag", default=None,
                    help="free text appended to the output folder name")
    ap.add_argument("--outdir", default=None,
                    help=f"output folder (default {RESULTS_ROOT}/<label>)")
    ap.add_argument("--plot", action="store_true", default=True)
    ap.add_argument("--noplot", dest="plot", action="store_false")
    args = ap.parse_args()

    kok = Path(__file__).resolve().parent
    odir = Path(args.outdir) if args.outdir else (kok / RESULTS_ROOT / run_label(args))
    odir.mkdir(parents=True, exist_ok=True)
    tee = Tee(odir / "summary.txt")
    sys.stdout = tee
    print(f"OUTPUT FOLDER: {odir}")

    course = (terrain_course() if (args.terrain or args.obstacles
                                   or args.pendulum is not None)
              else default_course())
    if args.terrain and not args.obstacles and args.T < 25.0:
        args.T = 35.0      # finish at 8.0 m; ~27 s at the measured 0.297 m/s
    if args.obstacles:
        # The duration comes from where the obstacle field is; it is read
        # from the file make_obstacle_model.py writes, not guessed. Too
        # short a T silently turns the run into an obstacle-free one.
        import json
        zf = find_model_dir() / "obstacle_zone.json"
        if zf.exists():
            z = json.loads(zf.read_text(encoding="utf-8"))
            if args.T < z["T"]:
                args.T = float(z["T"])
            print(f"\nOBSTACLE FIELD: x = {z['x0']:.2f} .. {z['x1']:.2f} m, "
                  f"{z['n']} obstacles ({z['n_block']} above swing height)"
                  + ("  [LOOSE]" if z["loose"] else "  [FIXED]"))
            print(f"  base: {z['base']},  duration T={args.T:.0f} s")
            if z["loose"] and args.compare:
                print("\n  !! WARNING: obstacles are LOOSE and --compare is set.")
                print("     Each method kicks the boxes elsewhere; after the")
                print("     first collision the runs no longer see the same")
                print("     course. For measurements:")
                print("       python make_obstacle_model.py --fixed")
        else:
            print("\n!! obstacle_zone.json missing -- cannot infer the "
                  f"duration, using --T {args.T:.0f}")
    # All runs use the same payload position (fair comparison)
    pos = (np.array(args.ppos, float) if args.ppos is not None
           else random_payload_pos(np.random.default_rng(args.seed)))
    if args.ppos is not None:
        mx, my = 3.0 * 9.81 * abs(pos[0]), 3.0 * 9.81 * abs(pos[1])
        print(f"\nPAYLOAD POSITION GIVEN: {np.round(pos, 4).tolist()} m")
        print(f"  pitch moment {mx:.2f} N.m,  roll moment {my:.2f} N.m")
        if abs(pos[1]) < 0.01:
            print("  !! y ~ 0 -> lateral axis not excited, disturbance stays constant")
        else:
            print("  the two diagonals are loaded differently -> the "
                  "disturbance oscillates at the gait frequency")

    kinds = (["none", "M0", "M1", "M2"] if args.compare else [args.adaptive])
    results = {}
    for kind in kinds:
        L, ev = run_course(kind, args.gamma, args.vx, args.T, course,
                           args.seed, args.view,
                           payload_pos=pos, terrain=args.terrain,
                           obstacles=args.obstacles,
                           live=args.live, speed=args.speed,
                           live_hz=args.live_hz,
                           use_filter=args.filter, emag_source=args.emag,
                           pendulum=(None if args.pendulum is None else
                                     {"L": args.pendulum, "axis": "x"}),
                           load_weights=args.load_weights,
                           save_weights=args.save_weights)
        rows = segment_stats(L, course, args.T)
        print_segments(rows)
        st = ("FELL %.2fs" % L["fell"]) if L["fell"] is not None else (
            ("STALLED %.2fs" % L["stuck"]) if L["stuck"] is not None else
            ("COURSE COMPLETED %.1fs" % L["done"]) if L["done"] is not None
            else "TIME UP, standing")
        print(f"\n  RESULT: {st}, distance {L['x_final']:.3f} m")
        results[kind] = (L, rows)
        np.savez(odir / f"course_{kind}.npz",
                 **{k: v for k, v in L.items()
                    if k not in ("fell", "stuck", "done")})
        print(f"  course_{kind}.npz written")

    if args.compare:
        print(f"\n\n{'='*78}")
        print("COMPARISON: prediction error per segment (steady state)")
        print(f"{'='*78}")
        labels = [r["label"] for r in results[kinds[-1]][1]]
        print(f"{'segment':<28} " + " ".join(f"{k:>10}" for k in kinds[1:]))
        print("-" * 62)
        for i, lab in enumerate(labels):
            cells = []
            for k in kinds[1:]:
                rows = results[k][1]
                cells.append(f"{rows[i]['pred']:>10.4f}"
                             if i < len(rows) else f"{'-':>10}")
            print(f"{lab:<28} " + " ".join(cells))
        print(f"\n{'outcome':<28}")
        print("-" * 62)
        for k in kinds:
            L = results[k][0]
            st = ("FELL %.1fs" % L["fell"]) if L["fell"] is not None else (
                ("STALLED %.1fs" % L["stuck"]) if L["stuck"] is not None else
                ("COMPLETED %.1fs" % L["done"]) if L["done"] is not None
                else "time up")
            print(f"  {k:<8} {st:<16} distance {L['x_final']:.2f} m")

    if args.plot:
        try:
            from plot_run import plot_course
            name = ("comparison.png" if args.compare
                    else f"{args.adaptive}.png")
            out = plot_course(results, course, args.T, out=str(odir / name),
                              terrain=(args.terrain or args.obstacles),
                              obstacles=args.obstacles)
            print(f"\nplot written: {out}")
        except ImportError as e:
            print(f"\n(plot skipped: {e})")

    print(f"\n{'='*78}")
    print(f"ALL OUTPUTS: {odir}")
    for f in sorted(odir.iterdir()):
        print(f"  {f.name}")
    sys.stdout = tee.std
    tee.close()


if __name__ == "__main__":
    main()
