"""
Obstacle field: cubes, planks and logs.

Why, in addition to the height map: the height map (make_terrain_model.py)
is a continuous, smooth surface. The edge of a hard box is discontinuous:
1 mm one way the foot lands 5 cm lower, 1 mm the other way it lands on the
box. Neither the contact time nor the contact normal can be predicted.
This is the least structured disturbance built in the project.

Sizes are derived from the gait
-------------------------------
Trot: period 0.40 s, duty 0.50 -> T_stance = T_swing = 0.20 s. With
Raibert foot placement at v = 0.3 m/s:

    step length = T_stance * v = 0.20 * 0.3 = 0.060 m

Swing apex profile 16 s^2 (1-s)^2 with peak 0.08 m; its mean height is
16 * B(3,3) * 0.08 = 0.5333 * 0.08 = 0.0427 m. Consequences:
  1. The step (6 cm) is shorter than the obstacles, so the robot steps ON
     them rather than over them: a sudden-height-change test.
  2. Obstacles below ~4.3 cm are mostly cleared, taller ones mostly hit.
Obstacles are placed on both sides of this threshold:
    small cube   ~3.0 cm   below          -> mostly cleared
    plank         3.0 cm   wide, stepped on
    medium cube  ~5.0 cm   at the limit
    thin log      5.0 cm   at the limit

Loose (default) or fixed
------------------------
Loose: every obstacle is a body with its own freejoint and gets kicked
around; the keyframe qpos/qvel are extended accordingly (otherwise MuJoCo
fails with "keyframe size mismatch"). With --fixed they are plain geoms in
the worldbody. USE --fixed FOR COMPARISONS: loose obstacles are kicked to
different places by different methods, so after the first collision the
runs no longer see the same course.

Logs use condim=6 (sliding + torsional + rolling friction); with condim=3
MuJoCo has no rolling resistance and a log would roll away forever. The
rolling coefficient (--roll-friction) is a lever arm in metres:
0 = rolls forever, 0.004 = a few turns (default), 0.02 = stops almost at
once. Cylinders are oriented with zaxis rather than euler, so their axis
is unambiguous.

Usage:
    python make_obstacle_model.py                   # after the terrain, loose
    python make_obstacle_model.py --fixed           # fixed (COMPARISON)
    python make_obstacle_model.py --base flat       # on flat ground, quick test
    python make_obstacle_model.py --cube-density 2  # twice as many cubes
    python make_obstacle_model.py --roll-friction 0 # logs never stop
    python make_obstacle_model.py --scale 0.7       # shrink everything
"""

import argparse
import json
import re
from pathlib import Path
import numpy as np
import mujoco

from make_terrain_model import find_model_dir, terrain_height, X1

# Thresholds derived from the gait (see above)
STEP_LEN = 0.060          # m, T_stance * v
SWING_MEAN = 0.0427       # m, mean of the swing profile
SWING_PEAK = 0.080        # m, peak of the swing profile

C_CUBE = "0.72 0.52 0.30 1"
C_PLANK = "0.62 0.45 0.27 1"
C_LOG = "0.45 0.33 0.20 1"

RHO_WOOD = 600.0          # kg/m^3, pine
CORRIDOR = 0.55           # half width of the foot corridor (m):
                          # feet at +-0.132 plus lateral drift


# ---------------------------------------------------------------------------
# Obstacle list
# ---------------------------------------------------------------------------

def build_obstacles(x0, zone_len, scale=1.0, seed=0, nx=11, ny=7,
                    gap_prob=0.12):
    """
    Four bands:
      A  small-cube field     (below the threshold)
      B  planks               (wide, stepped on)
      C  logs                 (at / above the threshold)
      D  medium cubes + tilted log

    Obstacles with anchor=True are not moved when resolving overlaps:
    planks and logs are the skeleton of the course.
    """
    rng = np.random.default_rng(seed)
    obs = []

    def box(kind, xy, half, yaw=0.0, anchor=False):
        obs.append(dict(kind=kind, type="box", xy=np.array(xy, float),
                        size=np.array(half, float) * scale, rgba=C_CUBE
                        if "cube" in kind else C_PLANK,
                        yaw=float(yaw), axis=None, anchor=anchor))

    def log(kind, xy, r, half_len, yaw=0.0, anchor=True):
        # yaw=0 -> axis along y (log lying across the path)
        ax = np.array([np.sin(yaw), np.cos(yaw), 0.0])
        obs.append(dict(kind=kind, type="cylinder", xy=np.array(xy, float),
                        size=np.array([r, half_len], float) * scale,
                        rgba=C_LOG, yaw=float(yaw), axis=ax, anchor=anchor))

    L = zone_len

    # --- band A: small-cube field ----------------------------------------
    # Grid + jitter; purely random placement clustered and sometimes the
    # robot met no obstacle at all.
    xa0, xa1 = x0 + 0.04, x0 + 0.34 * L
    for i in range(nx):
        for j in range(ny):
            if rng.random() < gap_prob:    # not too regular
                continue
            dx = (xa1 - xa0) / max(nx - 1, 1)
            dy = 1.00 / max(ny - 1, 1)
            xg = xa0 + dx * i + rng.uniform(-0.3, 0.3) * dx
            yg = -0.50 + dy * j + rng.uniform(-0.3, 0.3) * dy
            h = rng.uniform(0.013, 0.018)          # 2.6 - 3.6 cm
            box("small cube", (xg, yg), (h, h, h), rng.uniform(0, np.pi / 2))

    # --- band B: planks --------------------------------------------------
    xb = x0 + 0.42 * L
    box("plank", (xb, 0.0), (0.05, 0.38, 0.015), 0.0, anchor=True)
    box("plank", (xb + 0.16, -0.30), (0.045, 0.22, 0.015), 0.25, anchor=True)

    # --- band C: logs ----------------------------------------------------
    # A cylinder has no flat top, so a foot cannot stand on it, only step
    # over it, which needs diameter < step length (6.0 cm). Measured: a
    # 5.0 cm log was passed, a 7.0 cm log locked the robot at x = 9.14 m.
    # Thick logs are kept only outside the foot corridor (decorative).
    xc = x0 + 0.60 * L
    log("thin log", (xc, 0.05), 0.025, 0.30, 0.0)
    log("medium log", (xc + 0.22, -0.08), 0.028, 0.32, 0.15)
    # decorative, off the path
    log("edge log", (xc + 0.05, -1.00), 0.045, 0.35, 0.05)
    log("edge log", (xc + 0.75, 1.02), 0.040, 0.30, -0.08)

    # --- band D: medium cubes + tilted log -------------------------------
    xd0, xd1 = x0 + 0.78 * L, x0 + 0.97 * L
    for _ in range(6):
        h = rng.uniform(0.022, 0.028)              # 4.4 - 5.6 cm
        box("medium cube", (rng.uniform(xd0, xd1), rng.uniform(-0.40, 0.40)),
            (h, h, h), rng.uniform(0, np.pi / 2))
    log("tilted log", (xd1 + 0.10, 0.12), 0.028, 0.26, 0.9)

    return obs


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------

def aabb_half(o):
    """
    Axis-aligned half extents of the obstacle in the (x, y) plane. A
    circumscribed circle is not used: for a 76 cm plank it would push every
    cube out of the field.
    """
    c, s = abs(np.cos(o["yaw"])), abs(np.sin(o["yaw"]))
    if o["type"] == "box":
        a, b = float(o["size"][0]), float(o["size"][1])
    else:                       # lying cylinder: radius in x, half length in y
        a, b = float(o["size"][0]), float(o["size"][1])
    return np.array([a * c + b * s, a * s + b * c])


def resolve_overlaps(obs, gap=0.012, iters=300):
    """
    Two interpenetrating hard bodies produce a large separating force in
    the first frame and both fly off. Push apart along the axis of least
    overlap; anchored obstacles stay put.
    """
    half = np.array([aabb_half(o) for o in obs])
    p = np.array([o["xy"] for o in obs])
    fixed = np.array([bool(o["anchor"]) for o in obs])
    n = len(obs)
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                if fixed[i] and fixed[j]:
                    continue
                d = p[j] - p[i]
                need = half[i] + half[j] + gap
                ov = need - np.abs(d)
                if ov[0] <= 0 or ov[1] <= 0:
                    continue                      # separate
                k = int(np.argmin(ov))            # axis of least overlap
                sgn = 1.0 if d[k] >= 0 else -1.0
                push = ov[k]
                if fixed[i]:
                    p[j][k] += sgn * push
                elif fixed[j]:
                    p[i][k] -= sgn * push
                else:
                    p[i][k] -= sgn * push * 0.5
                    p[j][k] += sgn * push * 0.5
                moved = True
        if not moved:
            break
    for o, q in zip(obs, p):
        o["xy"] = q
    return obs


# ---------------------------------------------------------------------------

def height_of(o):
    """Height of the obstacle above the ground (m)."""
    if o["type"] == "box":
        return 2.0 * float(o["size"][2])
    return 2.0 * float(o["size"][0])         # lying cylinder -> diameter


def y_extent(o):
    """Half width of the obstacle in y (including yaw)."""
    return float(aabb_half(o)[1])


def in_corridor(o):
    """Is the obstacle in the strip the feet pass through?"""
    return abs(float(o["xy"][1])) - y_extent(o) < CORRIDOR


def passable(o):
    """
    Can the obstacle be crossed by walking? Returns (passable, reason).

    A box has a flat top: the foot can stand on it if its height is below
    the swing peak (8.0 cm). A cylinder has no flat top: the foot must step
    over it, which needs diameter < step length (6.0 cm).
    """
    h = height_of(o)
    if o["type"] == "cylinder":
        d = 2.0 * float(o["size"][0])
        if d >= STEP_LEN:
            return False, f"diameter {d*100:.1f} cm >= step {STEP_LEN*100:.1f} cm"
        return True, ""
    if h >= SWING_PEAK:
        return False, f"height {h*100:.1f} cm >= swing peak {SWING_PEAK*100:.1f} cm"
    return True, ""


def mass_of(o, rho=RHO_WOOD):
    """Mass from the wood density (kg); relevant only for loose obstacles."""
    s = o["size"]
    if o["type"] == "box":
        v = 8.0 * float(s[0]) * float(s[1]) * float(s[2])
    else:
        v = np.pi * float(s[0]) ** 2 * (2.0 * float(s[1]))
    return rho * v


def ground_z(o, use_terrain):
    """World z of the obstacle centre: ground height + half height."""
    x, y = o["xy"]
    h0 = float(terrain_height(x, y)) if use_terrain else 0.0
    return h0 + 0.5 * height_of(o)


def quat_of(o):
    """Quaternion (w, x, y, z) for the keyframe."""
    if o["type"] == "box":
        a = 0.5 * o["yaw"]
        return np.array([np.cos(a), 0.0, 0.0, np.sin(a)])
    # shortest rotation taking the z axis to the given axis
    u = np.asarray(o["axis"], float)
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    w = 1.0 + float(z @ u)
    if w < 1e-9:                      # exactly opposite
        return np.array([0.0, 1.0, 0.0, 0.0])
    v = np.cross(z, u)
    q = np.array([w, v[0], v[1], v[2]])
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# XML
# ---------------------------------------------------------------------------

def obstacle_xml(obs, use_terrain, loose, roll_fric=0.004,
                 rho=RHO_WOOD):
    parts = []
    for i, o in enumerate(obs):
        z = ground_z(o, use_terrain)
        x, y = o["xy"]
        sz = " ".join(f"{v:.4f}" for v in o["size"])
        nm = f"obs{i:02d}"
        if o["type"] == "box":
            orient = f'euler="0 0 {o["yaw"]:.4f}"'
        else:
            a = o["axis"]
            orient = f'zaxis="{a[0]:.4f} {a[1]:.4f} {a[2]:.4f}"'
        if o["type"] == "cylinder":
            # condim=6: sliding + torsional + rolling friction
            fr = f'1 0.01 {roll_fric:.5f}'
            common = (f'type="cylinder" size="{sz}" rgba="{o["rgba"]}" '
                      f'friction="{fr}" condim="6"')
        else:
            # same friction as the floor geom
            common = (f'type="box" size="{sz}" rgba="{o["rgba"]}" '
                      f'friction="1 0.005 0.0001" condim="3"')
        if loose:
            parts.append(
                f'\n    <body name="{nm}" pos="{x:.4f} {y:.4f} {z:.4f}" '
                f'{orient}>\n'
                f'      <freejoint/>\n'
                f'      <geom name="{nm}_g" {common} '
                f'density="{rho:.0f}"/>\n'
                f'    </body>')
        else:
            parts.append(f'\n    <geom name="{nm}" {common} '
                         f'pos="{x:.4f} {y:.4f} {z:.4f}" {orient}/>')
    return "".join(parts)


ROBOT_DST = "a1_obstacles.xml"


def patch_keyframe(stext, obs, use_terrain):
    """
    Loose bodies enlarge nq and nv; the keyframe must be extended to
    exactly nq. Each free body adds 7 to qpos (position + quaternion) and
    6 to qvel. Returns (new text, number of keys patched); zero is not an
    error, the keyframe may be in another file.
    """
    add_q = []
    for o in obs:
        x, y = o["xy"]
        q = quat_of(o)
        add_q += [x, y, ground_z(o, use_terrain), q[0], q[1], q[2], q[3]]
    add_v = [0.0] * (6 * len(obs))

    def fix(mk):
        body = mk.group(0)
        for attr, extra in (("qpos", add_q), ("qvel", add_v)):
            m2 = re.search(rf'{attr}\s*=\s*"([^"]*)"', body)
            if m2 is None:
                continue
            vals = m2.group(1).split() + [f"{v:.10g}" for v in extra]
            body = body[:m2.start(1)] + " ".join(vals) + body[m2.end(1):]
        return body

    # matches both <key .../> and <key ...></key>
    return re.subn(r'<key\b[^>]*>', fix, stext)


def redirect_keyframe(stext, mdir, obs, use_terrain):
    """
    Extend the keyframe in the file that contains it. scene_terrain.xml
    pulls the robot in with <include>, and the keyframe lives in that file
    (a1_course.xml). A patched copy (a1_obstacles.xml) is written and the
    scene points to it; the original files are untouched.

    In MJCF the <include> comes before the scene body, so the robot joints
    are defined first and the free joints of the obstacles land at the END
    of qpos; appending is therefore correct.

    Returns (new scene text, where it was patched).
    """
    for fn in re.findall(r'<include\s+file\s*=\s*"([^"]+)"', stext):
        p = mdir / fn
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        t2, n = patch_keyframe(t, obs, use_terrain)
        if n == 0:
            continue
        (mdir / ROBOT_DST).write_text(t2, encoding="utf-8")
        return stext.replace(f'"{fn}"', f'"{ROBOT_DST}"'), fn

    # the keyframe may be in the scene itself
    t2, n = patch_keyframe(stext, obs, use_terrain)
    if n:
        return t2, "(scene file)"
    return stext, None          # no keyframe at all -- fine


def patch_scene(stext, obs, use_terrain, loose, roll_fric=0.004,
                rho=RHO_WOOD):
    mw = re.search(r"<worldbody>", stext)
    if mw is None:
        raise RuntimeError("<worldbody> not found")
    i = mw.end()
    return (stext[:i] + obstacle_xml(obs, use_terrain, loose, roll_fric, rho)
            + stext[i:])


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=["terrain", "flat"], default="terrain",
                    help="terrain: after the terrain course (x0=7.7). "
                         "flat: early on flat ground (quick test)")
    ap.add_argument("--x0", type=float, default=None,
                    help="start of the field (m). default: terrain 7.7, flat 1.5")
    ap.add_argument("--len", dest="zone_len", type=float, default=2.2)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--fixed", action="store_true",
                    help="obstacles fixed (do not move). Use for comparisons: "
                         "loose obstacles are kicked to different places by "
                         "different methods.")
    ap.add_argument("--cube-density", dest="cube_density", type=float,
                    default=1.0,
                    help="density of the small-cube field (1.0 = 11x7 grid)")
    ap.add_argument("--density", type=float, default=RHO_WOOD,
                    help=f"obstacle density kg/m3 (default {RHO_WOOD:.0f} = "
                         "pine). Heavier obstacles are kicked less.")
    ap.add_argument("--roll-friction", dest="roll_fric", type=float,
                    default=0.004,
                    help="rolling resistance of the logs. 0 = never stops, "
                         "0.004 = a few turns, 0.02 = stops at once")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    loose = not a.fixed
    use_terrain = (a.base == "terrain")
    nx = max(2, int(round(11 * np.sqrt(a.cube_density))))
    ny = max(2, int(round(7 * np.sqrt(a.cube_density))))
    # 7.7: the descent ends at 7.4 m; an earlier start would put obstacles
    # on the ramp.
    x0 = a.x0 if a.x0 is not None else (7.7 if use_terrain else 1.5)

    mdir = find_model_dir()
    src = mdir / ("scene_terrain.xml" if use_terrain else "scene_course.xml")
    dst = mdir / "scene_obstacles.xml"
    if not src.exists():
        need = "make_terrain_model.py" if use_terrain else "make_course_model.py"
        raise FileNotFoundError(f"{src.name} missing -- run 'python {need}' first")

    # Delete old outputs first. If the script stops halfway, an old
    # scene_obstacles.xml would otherwise be loaded silently by scenario.py.
    for stale in (dst, mdir / ROBOT_DST, mdir / "obstacle_zone.json",
                  Path(__file__).resolve().parent / "obstacle_zone.json"):
        if stale.exists():
            stale.unlink()

    print(f"model folder : {mdir}")
    print(f"base scene   : {src.name}")
    print(f"field        : x = {x0:.2f} .. {x0 + a.zone_len:.2f} m"
          + ("   [LOOSE -- moves when hit]" if loose
             else "   [FIXED]"))

    if use_terrain and x0 + a.zone_len > X1 - 0.3:
        raise SystemExit(
            f"\nERROR: field ends at {x0 + a.zone_len:.2f} m, the height map "
            f"ends at {X1:.2f} m.\n"
            f"A robot stepping off the map falls into the void.\n"
            f"Fix: enlarge X1 in make_terrain_model.py and rebuild with\n"
            f"     python make_terrain_model.py")

    obs = resolve_overlaps(build_obstacles(x0, a.zone_len, a.scale,
                                           a.seed, nx, ny))

    stext = src.read_text(encoding="utf-8")
    kf_where = None
    if loose and obs:
        stext, kf_where = redirect_keyframe(stext, mdir, obs, use_terrain)
        if kf_where is None:
            print("keyframe     : none in any file -- nothing to extend")
        else:
            print(f"keyframe     : found in {kf_where}, "
                  f"extended copy written as {ROBOT_DST}")
    dst.write_text(patch_scene(stext, obs, use_terrain, loose,
                               a.roll_fric, a.density),
                   encoding="utf-8")
    print(f"written      : {dst.name}  ({len(obs)} obstacles)")

    # --- checks --------------------------------------------------------
    print("\n=== CHECKS ===")
    m = mujoco.MjModel.from_xml_path(str(dst))
    d = mujoco.MjData(m)

    if use_terrain:
        # The XML heightfield carries no height data; without loading it
        # the terrain is flat and obstacles on the ramp appear to float.
        H = np.load(mdir / "course_hfield.npy")
        hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_HFIELD, "course_hf")
        m.hfield_data[:] = (H / float(m.hfield_size[hid, 2])).ravel().astype(np.float32)
        print(f"terrain data : loaded (max {H.max():.3f} m)")

    if m.nkey > 0:
        d.qpos[:] = m.key_qpos[0]
    mujoco.mj_forward(m, d)
    print(f"compiled     : nq={m.nq} nv={m.nv} ngeom={m.ngeom}"
          + (", keyframe length consistent" if m.nkey else ""))

    # 1) classification against the gait thresholds
    print(f"\n  thresholds: step {STEP_LEN*100:.0f} cm, "
          f"mean swing {SWING_MEAN*100:.1f} cm, peak {SWING_PEAK*100:.1f} cm")
    print("\n  kind            count   height       mass   class")
    print("  " + "-" * 60)
    groups = {}
    for o in obs:
        groups.setdefault(o["kind"], []).append(
            (height_of(o), mass_of(o, a.density)))
    n_block = 0
    for k, v in groups.items():
        hm = float(np.mean([q[0] for q in v]))
        mm = float(np.mean([q[1] for q in v]))
        if hm < SWING_MEAN:
            cls = "cleared"
        elif hm < SWING_PEAK:
            cls = "AT LIMIT"
            n_block += len(v)
        else:
            cls = "BLOCKING"
            n_block += len(v)
        print(f"  {k:14s}  {len(v):3d}    {hm*100:5.1f} cm  "
              f"{mm*1000:7.0f} g   {cls}")
    print("  " + "-" * 60)
    print(f"  total {len(obs)} obstacles; {n_block} above the mean swing height")
    if loose:
        m_tot = sum(mass_of(o, a.density) for o in obs)
        m_max = max(mass_of(o, a.density) for o in obs)
        print(f"  LOOSE: total {m_tot:.2f} kg, heaviest obstacle "
              f"{m_max*1000:.0f} g (robot 12.45 kg)")
        if m_max > 3.0:
            print("     !! heaviest obstacle exceeds a quarter of the robot "
                  "mass -- behaves like a wall")
    if n_block == 0:
        print("  !! WARNING: none above the threshold -- the robot clears all")
        print("     of them and the experiment does not discriminate. Increase --scale.")

    # 1b) passability: an impassable obstacle on the path would lock the
    #     robot, a fault of the course, not of the controller
    on_path = [o for o in obs if in_corridor(o)]
    blocking = [(i, o, r) for i, o in enumerate(obs)
                if in_corridor(o) and not passable(o)[0]
                for r in [passable(o)[1]]]
    print(f"\n  in the foot corridor (|y| < {CORRIDOR:.2f} m): "
          f"{len(on_path)}/{len(obs)} obstacles")
    if blocking:
        print(f"  !! {len(blocking)} IMPASSABLE OBSTACLES ON THE PATH:")
        for i, o, r in blocking:
            print(f"     obs{i:02d} {o['kind']:12s} y={o['xy'][1]:+.2f}  {r}")
        print("     The robot will lock here. Shrink with --scale or move the")
        print("     obstacle outside |y| > %.2f." % CORRIDOR)
    else:
        print(f"  passability: all {len(on_path)} obstacles on the path "
              f"can be crossed (PASS)")
        print(f"     box      : height < {SWING_PEAK*100:.1f} cm "
              f"(flat top, can be stepped on)")
        print(f"     cylinder : diameter < {STEP_LEN*100:.1f} cm "
              f"(no flat top, must be stepped over)")
    off_path = [o for o in obs if not in_corridor(o)]
    if off_path:
        print(f"  off the path: {len(off_path)} obstacles (decorative, "
              f"thickest {max(height_of(o) for o in off_path)*100:.1f} cm)")

    # 2) initial interpenetration
    pen = []
    for c in range(d.ncon):
        con = d.contact[c]
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
        if ("obs" in g1 or "obs" in g2) and con.dist < -1e-4:
            pen.append((g1, g2, float(con.dist)))
    if pen:
        print(f"\n  !! {len(pen)} obstacle contacts interpenetrate:")
        for g1, g2, dd in pen[:5]:
            print(f"     {g1} <-> {g2}  {dd*1000:.1f} mm")
        print("     (reduce --scale or increase --len)")
    else:
        print("\n  interpenetration : NONE (PASS)")

    # 2b) loose mode: are the bodies where intended after the keyframe?
    #     If the appended 7-blocks were in a different order each obstacle
    #     would sit at another one's place without any error.
    if loose:
        worst_p, worst_b = 0.0, ""
        nfree = 0
        for i, o in enumerate(obs):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obs{i:02d}")
            if bid < 0:
                continue
            if m.body_jntnum[bid] == 1 and \
               m.jnt_type[m.body_jntadr[bid]] == mujoco.mjtJoint.mjJNT_FREE:
                nfree += 1
            target = np.array([o["xy"][0], o["xy"][1], ground_z(o, use_terrain)])
            e = float(np.max(np.abs(d.xpos[bid] - target)))
            if e > worst_p:
                worst_p, worst_b = e, f"obs{i:02d}"
        print(f"  free joints  : {nfree}/{len(obs)} bodies have a freejoint "
              f"({'PASS' if nfree == len(obs) else 'FAIL'})")
        print(f"  keyframe pos : largest deviation {worst_p*1e6:.1f} um "
              f"({'PASS' if worst_p < 1e-5 else 'FAIL ' + worst_b})")

    # 3) obstacles resting on the ground (terrain data correct?)
    worst, worst_nm = 0.0, ""
    for i, o in enumerate(obs):
        x, y = o["xy"]
        z_bottom = ground_z(o, use_terrain) - 0.5 * height_of(o)
        h0 = float(terrain_height(x, y)) if use_terrain else 0.0
        if abs(z_bottom - h0) > worst:
            worst, worst_nm = abs(z_bottom - h0), f"obs{i:02d}"
    print(f"  on the ground: largest gap {worst*1000:.2f} mm "
          f"({'PASS' if worst < 1e-6 else 'FAIL ' + worst_nm})")

    # 4) robot starts clear of the obstacles
    x_robot = float(d.qpos[0])
    x_first = min(float(o["xy"][0]) for o in obs)
    dist = x_first - x_robot
    print(f"  start        : robot x={x_robot:.2f}, first obstacle x={x_first:.2f} "
          f"-> {dist:.2f} m ({'PASS' if dist > 0.5 else 'FAIL, too close'})")

    # 5) duration
    v_mean = 0.20 if use_terrain else 0.28     # slower on terrain (measured)
    t_in = dist / v_mean
    t_out = (x0 + a.zone_len + 0.3 - x_robot) / v_mean
    T_sugg = int(np.ceil((t_out + 5) / 5) * 5)
    print(f"\n  estimate: first obstacle at ~{t_in:.0f} s, field ends at ~{t_out:.0f} s")

    # scenario.py reads the duration from this file.
    info = json.dumps(dict(
        base=a.base, x0=float(x0),
        x1=float(max(o["xy"][0] for o in obs) + 0.15),
        n=len(obs), n_block=int(n_block), loose=bool(loose), T=T_sugg))
    (mdir / "obstacle_zone.json").write_text(info, encoding="utf-8")
    # second copy next to the scripts: plot_run.py does not import MuJoCo
    # and cannot find the model folder
    (Path(__file__).resolve().parent / "obstacle_zone.json").write_text(
        info, encoding="utf-8")
    print(f"  obstacle_zone.json written (T={T_sugg} s, read by scenario.py)")

    print(f"\n{dst.name} ready.")
    print("  python scenario.py --obstacles --adaptive M2 --view")


if __name__ == "__main__":
    main()
