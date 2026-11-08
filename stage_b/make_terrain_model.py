"""
Terrain course: builds a height-field scene.

A slope applied by rotating the gravity vector is physically equivalent,
but the ground stays visibly flat. Here the ground is actually shaped:
ramp and bumps are visible and the robot walks over them.

Course (robot starts at x = 0 and walks in +x):

    x (m)        zone                      height
    ---------------------------------------------------------------
    -1.0 - 1.5   flat start                0
     1.5 - 3.2   RAMP UP                   0 -> h_top (steepest = --ramp)
     3.2 - 4.0   plateau                   h_top
     4.0 - 5.7   ROUGH                     h_top +- bumps
     5.7 - 7.4   RAMP DOWN                 h_top -> 0 (same angle as up)
     7.4 +       flat finish               0

The rough section varies in y too, so left and right legs land at
different heights; lateral asymmetry is the real difficulty for a
quadruped.

Scales: bump wavelength 25 cm is larger than the ~6 cm step, otherwise
every step would hit the same phase of the bump; resolution 2 cm (foot
diameter 4 cm).

MuJoCo height-field data are normalised to [0, 1] and scaled by the
elevation in size; row-major data[r*ncol + c] with r -> y, c -> x, first
value at (xmin, ymin). This script checks that assumption by casting rays
with mj_ray.

Usage:
    python make_terrain_model.py
    python make_terrain_model.py --ramp 12 --bump 0.05
"""

import argparse
import re
from pathlib import Path
import numpy as np
import mujoco

# Course box (world frame). X1 = 12.0 leaves room for the obstacle field
# that make_obstacle_model.py places after the terrain (x >= 7.7).
X0, X1 = -1.0, 12.0         # x range
Y0, Y1 = -2.0, 2.0          # y range
DX, DY = 0.02, 0.04         # resolution
Z_TOP = 0.40                # height-field maximum height
Z_BASE = 0.60               # base thickness (visual)

HF_NAME = "course_hf"
HF_GEOM = "course_terrain"


def find_model_dir():
    here = Path(__file__).resolve().parent
    for base in [here] + list(here.parents)[:4]:
        hit = list(base.glob("**/unitree_a1/a1_course.xml"))
        if hit:
            return hit[0].parent
    raise FileNotFoundError("unitree_a1/a1_course.xml missing -- "
                            "run make_course_model.py first")


# ---------------------------------------------------------------------------
# Terrain profile
# ---------------------------------------------------------------------------

def terrain_height(x, y, ramp_deg=8.0, bump=0.035, seed=0):
    """
    Course height h(x, y), vectorised.

    The ramp angle refers to the STEEPEST point. With a smoothstep the
    slope at the middle is 1.5x the mean slope, hence
        h_top = L * tan(angle) / 1.5.
    A smoothstep is used instead of a sharp break, which would catch the
    foot. Both ramps have the same length, so they have the same angle.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    L_ramp = 1.7
    h_top = L_ramp * np.tan(np.deg2rad(ramp_deg)) / 1.5

    h = np.zeros(np.broadcast(x, y).shape)

    # 1.5 - 3.2 : ramp up (smoothstep)
    s = np.clip((x - 1.5) / L_ramp, 0.0, 1.0)
    ramp_up = h_top * (3 * s**2 - 2 * s**3)

    # 5.7 - 7.4 : ramp down (same length as up -> same angle)
    s2 = np.clip((x - 5.7) / L_ramp, 0.0, 1.0)
    ramp_dn = h_top * (1.0 - (3 * s2**2 - 2 * s2**3))

    h = np.where(x < 5.7, ramp_up, ramp_dn)
    h = np.where(x < 1.5, 0.0, h)
    h = np.where(x > 5.7 + L_ramp, 0.0, h)

    # 4.0 - 5.7 : rough section on the plateau. Terms odd in y (sin and an
    # x-wave whose phase shifts with y) make left and right differ; an even
    # cos(y) term gave only 5 mm left/right difference.
    m = (x >= 4.0) & (x <= 5.7)
    pen = np.clip((x - 4.0) / 0.3, 0.0, 1.0) * np.clip((5.7 - x) / 0.3, 0.0, 1.0)
    rough = (np.sin(2*np.pi*x/0.25 + 3.0*y)
             + 0.7*np.sin(2*np.pi*y/0.33 + 0.4)
             + 0.5*np.sin(2*np.pi*x/0.41 + 2*np.pi*y/0.55 + 1.3))
    rough = rough / 2.2 * bump
    h = h + np.where(m, rough * pen, 0.0)

    return np.maximum(h, 0.0)


def build_field(ramp_deg=8.0, bump=0.035):
    ncol = int(round((X1 - X0) / DX)) + 1      # x
    nrow = int(round((Y1 - Y0) / DY)) + 1      # y
    xs = np.linspace(X0, X1, ncol)
    ys = np.linspace(Y0, Y1, nrow)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")   # (nrow, ncol)
    H = terrain_height(XX, YY, ramp_deg, bump)
    return H, xs, ys, nrow, ncol


# ---------------------------------------------------------------------------
# XML
# ---------------------------------------------------------------------------

def patch_scene(stext, nrow, ncol):
    """Replace the plane with a height field and add the hfield asset."""
    cx, cy = 0.5 * (X0 + X1), 0.5 * (Y0 + Y1)
    rx, ry = 0.5 * (X1 - X0), 0.5 * (Y1 - Y0)

    # 1) find the plane geom
    mplane = re.search(r'<geom\b[^>]*type\s*=\s*"plane"[^>]*/>', stext)
    if mplane is None:
        raise RuntimeError('no type="plane" geom in the scene')
    plane = mplane.group(0)
    mat = re.search(r'material\s*=\s*"([^"]+)"', plane)
    mat_attr = f' material="{mat.group(1)}"' if mat else ' rgba="0.55 0.5 0.42 1"'

    hf_geom = (f'<geom name="{HF_GEOM}" type="hfield" hfield="{HF_NAME}" '
               f'pos="{cx} {cy} 0"{mat_attr} '
               f'friction="1 0.005 0.0001" condim="3"/>')
    stext = stext.replace(plane, hf_geom)

    # 2) hfield asset
    hf_asset = (f'\n    <hfield name="{HF_NAME}" nrow="{nrow}" ncol="{ncol}" '
                f'size="{rx} {ry} {Z_TOP} {Z_BASE}"/>')
    masset = re.search(r"<asset>", stext)
    if masset:
        i = masset.end()
        stext = stext[:i] + hf_asset + stext[i:]
    else:
        mw = re.search(r"<worldbody>", stext)
        stext = (stext[:mw.start()] + f"<asset>{hf_asset}\n  </asset>\n  "
                 + stext[mw.start():])
    return stext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ramp", type=float, default=10.0, help="ramp angle (deg)")
    ap.add_argument("--bump", type=float, default=0.035, help="bump amplitude (m)")
    a = ap.parse_args()

    mdir = find_model_dir()
    scene_src = mdir / "scene_course.xml"
    scene_dst = mdir / "scene_terrain.xml"
    if not scene_src.exists():
        raise FileNotFoundError("scene_course.xml missing -- make_course_model.py")

    H, xs, ys, nrow, ncol = build_field(a.ramp, a.bump)
    print(f"model folder : {mdir}")
    print(f"grid         : {nrow} rows (y) x {ncol} cols (x) = {nrow*ncol} points")
    print(f"resolution   : dx={DX*100:.0f} cm, dy={DY*100:.0f} cm")
    print(f"height       : min {H.min():.4f}, max {H.max():.4f} m "
          f"(Z_TOP={Z_TOP})")
    if H.max() > Z_TOP:
        raise RuntimeError(f"profile exceeds Z_TOP: {H.max():.3f} > {Z_TOP}")

    stext = scene_src.read_text(encoding="utf-8")
    scene_dst.write_text(patch_scene(stext, nrow, ncol), encoding="utf-8")
    print(f"written      : {scene_dst.name}")

    # --- load the model and fill the data ------------------------------
    m = mujoco.MjModel.from_xml_path(str(scene_dst))
    hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_HFIELD, HF_NAME)
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, HF_GEOM)
    assert hid >= 0 and gid >= 0, "hfield/geom not found"
    print(f"hfield id={hid}  geom id={gid}  "
          f"nrow={m.hfield_nrow[hid]} ncol={m.hfield_ncol[hid]}")

    # MuJoCo data are normalised to [0, 1] and scaled by Z_TOP
    m.hfield_data[:] = (H / Z_TOP).ravel().astype(np.float32)

    # --- check: measure the actual height by ray casting ---------------
    print("\n=== CHECK: ground height measured with mj_ray ===")
    d = mujoco.MjData(m)
    # Move the robot out of the way: mj_ray sees every geom, and a ray at
    # x = 0 would hit the robot instead of the terrain.
    if m.nkey > 0:
        d.qpos[:] = m.key_qpos[0]
    d.qpos[0] = 50.0
    mujoco.mj_forward(m, d)
    tgid = gid

    def ray_h(x, y):
        pnt = np.array([x, y, 3.0]); vec = np.array([0.0, 0.0, -1.0])
        gi = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(m, d, pnt, vec, None, 1, -1, gi)
        if dist < 0:
            return np.nan
        if int(gi[0]) != tgid:        # hit something other than the terrain
            return np.nan
        return 3.0 - dist

    print(f"  {'x':>6} {'y':>6} {'expected':>10} {'measured':>10} {'diff':>9}")
    print("  " + "-" * 46)
    tests = [(0.0, 0.0), (1.0, 0.0), (2.35, 0.0), (3.6, 0.0),
             (4.5, 0.0), (4.5, 0.13), (4.5, -0.13), (5.0, 0.0),
             (6.35, 0.0), (8.0, 0.0)]
    worst = 0.0
    for x, y in tests:
        exp = float(terrain_height(x, y, a.ramp, a.bump))
        got = ray_h(x, y)
        err = abs(got - exp)
        worst = max(worst, err)
        print(f"  {x:>6.2f} {y:>6.2f} {exp:>10.4f} {got:>10.4f} {err:>9.4f}")
    ok = worst < 0.01
    print(f"\n  largest difference: {worst:.4f} m ({'PASS' if ok else 'FAIL'})")
    if not ok:
        # axes swapped? test the transpose hypothesis
        exp_t = float(terrain_height(0.0, 2.35, a.ramp, a.bump))
        got_t = ray_h(0.0, 2.35)
        print("  !! AXES may be swapped. Transpose test:")
        print(f"     (0, 2.35) expected (flat)={exp_t:.4f} measured={got_t:.4f}")
        print("     if the measured value is a ramp value, write the data transposed:")
        print("     m.hfield_data[:] = (H.T / Z_TOP).ravel()")

    # --- ramp angles: both ramps, with a narrow chord (a wide chord
    #     averages and under-reports the steepest point) ----------------
    print("\n  ramp slopes (local, narrow chord):")
    steepest = {}
    for name, pts in (("UP    ", [(1.6, 1.8), (2.25, 2.45), (2.9, 3.1)]),
                      ("DOWN  ", [(5.8, 6.0), (6.45, 6.65), (7.1, 7.3)])):
        for xa, xb in pts:
            ha, hb = ray_h(xa, 0.0), ray_h(xb, 0.0)
            ang = np.rad2deg(np.arctan2(hb - ha, xb - xa))
            if abs(ang) > abs(steepest.get(name, 0.0)):
                steepest[name] = ang
            print(f"    {name} {xa:.2f}-{xb:.2f} m: {ang:+6.2f} deg")
    print(f"\n    steepest UP {steepest['UP    ']:+.2f}, "
          f"steepest DOWN {steepest['DOWN  ']:+.2f}  (target {a.ramp:.1f})")
    diff = abs(abs(steepest["UP    "]) - abs(steepest["DOWN  "]))
    print(f"    difference between ramps {diff:.2f} deg "
          f"({'PASS -- symmetric' if diff < 0.5 else 'FAIL -- ASYMMETRIC'})")
    print(f"  total rise: {ray_h(3.5, 0.0):.3f} m")

    # --- left/right difference on the rough section ------------------
    print("\n  lateral asymmetry on the rough section (feet at +-0.132 m):")
    for x in [4.3, 4.8, 5.3]:
        hl, hr = ray_h(x, 0.132), ray_h(x, -0.132)
        print(f"    x={x:.1f}: left {hl:.4f}  right {hr:.4f}  "
              f"diff {abs(hl-hr)*1000:5.1f} mm")

    # --- save the data: scenario.py refills it on every load -----------
    np.save(mdir / "course_hfield.npy", H.astype(np.float32))
    print("\n  height data saved: course_hfield.npy")
    print("  (the XML heightfield carries no data; scenario.py fills")
    print("   m.hfield_data from this file on every load)")

    print("\nscene_terrain.xml ready.")
    print("Usage:  python scenario.py --terrain --adaptive M2 --view")


if __name__ == "__main__":
    main()
