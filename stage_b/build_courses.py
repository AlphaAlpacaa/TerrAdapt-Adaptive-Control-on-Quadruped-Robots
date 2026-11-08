"""
Builds two SEPARATE courses (mainly for videos and quick checks).

Terrain and loose obstacles in one scene were heavy (84 loose obstacles:
nv = 522 instead of 18, solved at every step even before the robot gets
there). Separately:

  1) TERRAIN COURSE    ramp + plateau + rough + descent, no obstacles
                       nv = 18, ~8 m, ~30 s
  2) OBSTACLE COURSE   obstacle field on FLAT ground, right at the start
                       short (~4.5 m, ~16 s)

They are written to separate files (scene_terrain.xml / scene_obstacles.xml).

NOTE: this script always builds the obstacles on a FLAT base. The combined
course used for the main measurement (terrain + fixed obstacles, T = 60)
is built with:
    python make_obstacle_model.py --fixed

Usage:
    python build_courses.py                    # loose obstacles (video)
    python build_courses.py --fixed            # fixed obstacles, flat base
    python build_courses.py --cube-density 0.6 # fewer obstacles -> faster
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script, *flags):
    cmd = [sys.executable, str(HERE / script), *map(str, flags)]
    print("\n" + "=" * 74)
    print("$ " + " ".join([Path(cmd[1]).name] + cmd[2:]))
    print("=" * 74)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"\n!! {script} failed (code {r.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", action="store_true",
                    help="fixed obstacles (default loose, for video)")
    ap.add_argument("--cube-density", dest="cube_density", type=float,
                    default=1.0,
                    help="obstacle density; 0.6 gives roughly half the loose "
                         "bodies and is noticeably faster")
    ap.add_argument("--ramp", type=float, default=10.0)
    ap.add_argument("--gamma", type=float, default=300.0,
                    help="gain used in the suggested commands")
    a = ap.parse_args()

    # 1) terrain course -- no obstacles
    run("make_terrain_model.py", "--ramp", a.ramp)

    # 2) obstacle course -- flat ground, obstacles at the start
    obs = ["--base", "flat", "--cube-density", a.cube_density]
    if a.fixed:
        obs.append("--fixed")
    run("make_obstacle_model.py", *obs)

    # --- commands ------------------------------------------------------
    payload = "--payload-pos 0.12 0.05 0.06"
    g = f"--gamma {a.gamma:g}"
    common = f"{g} {payload}"
    v = "--view --live --live-hz 5 --speed 3"

    print("\n\n" + "=" * 74)
    print("TWO COURSES READY")
    print("=" * 74)
    print("  scene_terrain.xml    terrain   (no obstacles, nv=18)")
    print(f"  scene_obstacles.xml  obstacles (flat ground, "
          f"{'FIXED' if a.fixed else 'LOOSE'})")
    print(f"\n  payload {payload.split()[1:]} -> pitch AND roll moment")

    print("\n--- VIDEO: terrain course " + "-" * 47)
    for m in ["none", "M0", "M1", "M2"]:
        print(f"python scenario.py --terrain --adaptive {m:4s} {v} {common}")

    print("\n--- VIDEO: obstacle course " + "-" * 46)
    for m in ["none", "M0", "M1", "M2"]:
        print(f"python scenario.py --obstacles --adaptive {m:4s} {v} {common}")

    print("\n--- MEASUREMENT (no viewer, fastest) " + "-" * 36)
    print(f"python scenario.py --terrain   --compare {common}")
    print(f"python scenario.py --obstacles --compare {common}")

    print("\n--- MAIN MEASUREMENT (report table) " + "-" * 37)
    print("python make_obstacle_model.py --fixed     # combined course first!")
    print("python sweep_robust.py --baseline")
    print("  (this script's --fixed builds a FLAT-base obstacle course,")
    print("   not the combined course of the main measurement)")

    if not a.fixed and a.cube_density >= 1.0:
        print("\n  IF SLOW:  python build_courses.py --cube-density 0.6")


if __name__ == "__main__":
    main()
