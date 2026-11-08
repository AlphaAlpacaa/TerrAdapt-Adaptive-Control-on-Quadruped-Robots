"""
Payload-position sweep -- does a random placement break anything?

scenario.py draws a random position, but with one seed: seed 0 gave
[0.1201, -0.002, 0.0407], y ~ 0, so the roll axis was hardly excited.

Criterion: moment about the diagonal that normal forces cannot resist
--------------------------------------------------------------------
In a trot only two diagonal feet are on the ground. The moment component
along the line joining them has zero lever arm for the normal forces; only
foot friction and swing-leg reactions can resist it.

Payload at r = (x, y, z): tau = r x (-m g z_hat) = (-m g y, +m g x, 0).
Diagonal unit vector u = (-dx, dy, 0)/|.|, tau . u = m g (u_y x - u_x y);
the worse diagonal decides:
    M_unresisted = m g ( |u_y| |x| + |u_x| |y| )
For the A1 footprint (dx = 0.183, dy = 0.1308): u_y = 0.5815, u_x = 0.8135,
so per metre a lateral offset is 1.40x more harmful than a longitudinal one.

With y != 0 the two diagonals are loaded differently and the disturbance
oscillates at the gait frequency instead of being constant.

Usage:
    python sweep_payload_pos.py                      # 10 positions, 4 methods, flat
    python sweep_payload_pos.py --n 6 --terrain      # on terrain
    python sweep_payload_pos.py --pos 0.16 0.05 0.06 # one, the worst corner
    python sweep_payload_pos.py --table              # geometry table only
"""

import argparse
import numpy as np

from a1_params import HIP_X, HIP_Y, TAU_MAX, MASS_TOTAL

# lateral offset of the foot from the hip (abduction arm)
try:
    from a1_params import D_ABD
except ImportError:            # pragma: no cover
    D_ABD = 0.0838

FOOT_DX = HIP_X                      # 0.183
FOOT_DY = HIP_Y + D_ABD              # 0.047 + 0.0838 = 0.1308
G = 9.81


def unresistable_moment(pos, dm=3.0):
    """Moment along the diagonal support line (worse diagonal)."""
    x, y = float(pos[0]), float(pos[1])
    n = np.hypot(FOOT_DX, FOOT_DY)
    ux, uy = FOOT_DX / n, FOOT_DY / n
    return dm * G * (uy * abs(x) + ux * abs(y))


def moment_components(pos, dm=3.0):
    """(pitch moment, roll moment) in Nm, unsigned."""
    return dm * G * abs(float(pos[0])), dm * G * abs(float(pos[1]))


def geometry_table():
    """Without simulation: how demanding each position is."""
    from course_def import random_payload_pos
    print("\n" + "=" * 74)
    print("GEOMETRY: how demanding each position is (closed form, no simulation)")
    print("=" * 74)
    n = np.hypot(FOOT_DX, FOOT_DY)
    print(f"footprint     : dx=+-{FOOT_DX:.4f} m, dy=+-{FOOT_DY:.4f} m")
    print(f"diagonal unit : u=({FOOT_DX/n:.4f}, {FOOT_DY/n:.4f})")
    print(f"  -> x offset coefficient {FOOT_DY/n:.4f}")
    print(f"  -> y offset coefficient {FOOT_DX/n:.4f}  "
          f"({(FOOT_DX/FOOT_DY):.2f}x more harmful per metre)")
    print()
    print("  position (x, y)       pitch M   roll M   UNRESISTED      seed")
    print("  " + "-" * 66)
    rows = []
    # the same seeds the sweep uses (1000+k) plus scenario.py's 0
    for s in [0] + [1000 + k for k in range(12)]:
        p = random_payload_pos(np.random.default_rng(s))
        mp, mr = moment_components(p)
        rows.append((unresistable_moment(p), s, p, mp, mr))
    for mu, s, p, mp, mr in sorted(rows):
        mark = "  <- scenario.py default (y~0: roll axis unexcited)" if s == 0 else ""
        print(f"  ({p[0]:+.3f}, {p[1]:+.3f})      {mp:5.2f}    {mr:5.2f}"
              f"        {mu:5.2f} Nm      {s}{mark}")
    print("  " + "-" * 66)
    worst = np.array([0.16, 0.05, 0.06])
    mp, mr = moment_components(worst)
    print(f"  ({worst[0]:+.3f}, {worst[1]:+.3f})      {mp:5.2f}    {mr:5.2f}"
          f"        {unresistable_moment(worst):5.2f} Nm      (corner of the range)")
    mus = sorted(r[0] for r in rows)
    p0 = random_payload_pos(np.random.default_rng(0))
    print(f"\n  unresisted moment over the drawn positions: "
          f"{mus[0]:.2f} .. {mus[-1]:.2f} Nm")
    print(f"  seed 0                                    : "
          f"{unresistable_moment(p0):.2f} Nm")
    print(f"  corner of the range                       : "
          f"{unresistable_moment(worst):.2f} Nm "
          f"({unresistable_moment(worst)/unresistable_moment(p0):.2f}x)")


# ---------------------------------------------------------------------------

def parse_gammas(items):
    """--gammas M0=100 M1=300 M2=1000  ->  {'M0':100.0, ...}"""
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = float(v)
    return out


def scale_payload(course, dm):
    """Change the mass of the payload event (overrides 3.0 kg)."""
    if dm is None:
        return course
    return [(t, kind, (dm if kind == "payload" else val), lab)
            for t, kind, val, lab in course]


def run_one(kind, pos, args, gammas):
    """One (method, position) run. Returns a metric dict."""
    from scenario import run_course
    from course_def import default_course, terrain_course

    course = terrain_course() if args.terrain else default_course()
    course = scale_payload(course, args.dm)
    T = args.T if args.T else (30.0 if args.terrain else 14.0)
    # every method may run at its own gain
    g = gammas.get(kind, args.gamma)
    L, _ = run_course(kind, g, args.vx, T, course,
                      args.seed_ctrl, False, payload_pos=pos,
                      terrain=args.terrain)

    t = L["t"]
    t_evt = course[0][0]                       # payload time
    post = t >= t_evt + 1.0                    # skip the transient
    out = dict(
        fell=L["fell"], x_final=L["x_final"],
        roll_std=float(np.std(L["roll"][post])) if post.sum() > 2 else np.nan,
        pitch_abs=float(np.mean(np.abs(L["pitch"][post]))) if post.sum() > 2 else np.nan,
        sat=float(np.mean(L["tau_max"][post] > 0.995 * TAU_MAX)) if post.sum() > 2 else np.nan,
        pred=float(np.mean(L["pred_err"][post])) if post.sum() > 2 else np.nan,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of random positions")
    ap.add_argument("--methods", nargs="+",
                    default=["none", "M0", "M1", "M2"])
    ap.add_argument("--terrain", action="store_true")
    ap.add_argument("--pos", type=float, nargs=3, default=None,
                    help="force one position (x y z, body frame)")
    ap.add_argument("--gamma", type=float, default=None,
                    help="one gain for ALL methods")
    ap.add_argument("--gammas", nargs="+", default=None,
                    metavar="M0=100",
                    help="per-method gains")
    ap.add_argument("--dm", type=float, default=None,
                    help="payload mass (kg); default 3.0")
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--seed-ctrl", type=int, default=0,
                    help="controller/network seed, separate from the position")
    ap.add_argument("--table", action="store_true",
                    help="geometry table only, no simulation")
    args = ap.parse_args()

    gammas = parse_gammas(args.gammas)

    geometry_table()
    if args.dm is not None:
        print(f"\n  NOTE: payload mass set to {args.dm} kg "
              f"(table moments are for 3.0 kg; scale by {args.dm/3.0:.2f})")
    if gammas:
        print(f"\n  per-method gains: {gammas}")
    if args.table:
        return

    from course_def import random_payload_pos

    if args.pos is not None:
        positions = [np.array(args.pos)]
    else:
        # separate position seed: same controller seed, different positions
        positions = [random_payload_pos(np.random.default_rng(1000 + k))
                     for k in range(args.n)]
    positions.sort(key=unresistable_moment)     # easy to hard

    print("\n" + "=" * 74)
    print(f"SWEEP: {len(positions)} positions x {len(args.methods)} methods"
          + ("  [TERRAIN]" if args.terrain else "  [flat ground]"))
    print("=" * 74)

    res = {}
    for pi, pos in enumerate(positions):
        mu = unresistable_moment(pos)
        print(f"\n--- position {pi+1}/{len(positions)}  "
              f"({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f})  "
              f"unresisted moment {mu:.2f} Nm ---")
        for kind in args.methods:
            r = run_one(kind, pos, args, gammas)
            res[(pi, kind)] = r
            st = "STANDING" if r["fell"] is None else f"FELL {r['fell']:.1f}s"
            print(f"    {kind:5s} {st:12s} roll {r['roll_std']:.4f}  "
                  f"|pitch| {r['pitch_abs']:.4f}  saturation {100*r['sat']:.0f}%")

    # --- summary --------------------------------------------------------
    print("\n\n" + "=" * 78)
    print("SUMMARY -- sorted by unresisted moment (easy to hard)")
    print("=" * 78)
    hdr = "  M(Nm)   (x,y)          " + "".join(f"{k:>12s}" for k in args.methods)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for pi, pos in enumerate(positions):
        line = (f"  {unresistable_moment(pos):5.2f}  "
                f"({pos[0]:+.2f},{pos[1]:+.2f})  ")
        for kind in args.methods:
            r = res[(pi, kind)]
            cell = "FELL" if r["fell"] is not None else f"{r['roll_std']:.4f}"
            line += f"{cell:>12s}"
        print(line)
    print("  (number = roll std after the payload; FELL if the run fell)")

    print("\n  staying up:")
    for kind in args.methods:
        ok = sum(1 for pi in range(len(positions))
                 if res[(pi, kind)]["fell"] is None)
        rolls = [res[(pi, kind)]["roll_std"] for pi in range(len(positions))
                 if res[(pi, kind)]["fell"] is None]
        med = np.median(rolls) if rolls else np.nan
        print(f"    {kind:5s} {ok}/{len(positions)}   median roll std of "
              f"standing runs {med:.4f}")


if __name__ == "__main__":
    main()
