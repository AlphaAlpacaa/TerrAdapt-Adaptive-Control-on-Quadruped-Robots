"""
Gait scheduler, footstep planning and swing trajectory.

Sources:
  Bledt et al., IROS 2018 (MIT Cheetah 3)   -- per-leg phase variable
  Raibert, Legged Robots That Balance, 1986 -- footstep heuristic
  Pratt et al., Humanoids 2006              -- capture point
  Sombolestan & Nguyen, T-RO 2024, Eq. (14) -- linear combination of both

Footstep position (Eq. 14):

    p_f,i = p_h,i + (T_stance/2) * v_cmd + sqrt(z0/g) * (v - v_cmd)

    term 1: hip position (projected, world frame)
    term 2: Raibert heuristic -- advance by half the stance time
    term 3: capture-point feedback -- corrects the velocity error

Swing trajectory: smooth curve from lift-off to touchdown with a given apex
height.
"""

import numpy as np
from a1_params import (LEGS, HIP_POS, Z_NOMINAL, GRAVITY, D_ABD,
                       LEG_SIGN, R_FOOT, COM_OFFSET)

N_LEGS = 4


class GaitScheduler:
    """
    Finite-state machine with a per-leg phase variable.

    phi_i(t) = ((t / T_gait) + offset_i) mod 1
    phi_i < duty  -> stance (contact), otherwise swing

    Trot: diagonal pairs move together. For LEGS = ["FR","FL","RR","RL"]
    the offsets are [0, 0.5, 0.5, 0]: FR+RL one pair, FL+RR the other.
    """

    PRESETS = {
        #            period   duty   offsets (FR, FL, RR, RL)
        "stand":    (1.00,    1.00,  [0.0, 0.0,  0.0,  0.0]),
        "trot":     (0.40,    0.50,  [0.0, 0.5,  0.5,  0.0]),
        "walk":     (0.80,    0.75,  [0.0, 0.5,  0.25, 0.75]),
        "pace":     (0.40,    0.50,  [0.0, 0.5,  0.0,  0.5]),
        "bound":    (0.36,    0.45,  [0.0, 0.0,  0.5,  0.5]),
    }

    def __init__(self, name="trot", period=None, duty=None):
        if name not in self.PRESETS:
            raise ValueError(f"unknown gait: {name}")
        self.name = name
        p0, d0, off = self.PRESETS[name]
        # The swing time matters: the required vertical acceleration scales
        # with h * 32 / T_swing^2 (0.09 m / 0.2 s -> 72 m/s^2, beyond the
        # joint torque). A longer period helps directly.
        self.period = p0 if period is None else float(period)
        self.duty = d0 if duty is None else float(duty)
        self.offsets = np.array(off)

    def phase(self, t):
        """Phase variable of each leg, [0, 1)."""
        return np.mod(t / self.period + self.offsets, 1.0)

    def contact(self, t):
        """(4,) bool -- True = stance."""
        return self.phase(t) < self.duty

    def swing_phase(self, t):
        """
        Swing progress [0, 1] for swinging legs; 0 for stance legs.
        """
        ph = self.phase(t)
        sw = np.zeros(N_LEGS)
        m = ph >= self.duty
        if self.duty < 1.0:
            sw[m] = (ph[m] - self.duty) / (1.0 - self.duty)
        return sw

    @property
    def stance_time(self):
        return self.period * self.duty

    @property
    def swing_time(self):
        return self.period * (1.0 - self.duty)

    def contact_horizon(self, t, k, dt):
        """(k,4) bool -- contact schedule over the MPC horizon."""
        return np.array([self.contact(t + (i + 1) * dt) for i in range(k)])


def hip_world(leg, p_com, psi, R=None):
    """
    Projection of the hip joint in the world frame.

    HIP_POS is relative to the trunk origin, not the COM, so
        p_trunk = p_com - Rz @ COM_OFFSET
    first; otherwise every footstep target moves 11.3 mm forward and the
    body gets a constant pitch moment.
    """
    c, s = np.cos(psi), np.sin(psi)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    Ruse = Rz if R is None else np.asarray(R, float)
    offset = HIP_POS[leg].copy()
    offset[1] += LEG_SIGN[leg] * D_ABD      # nominal leg plane
    p_trunk = np.asarray(p_com, float) - Ruse @ COM_OFFSET
    p = p_trunk + Ruse @ offset
    p[2] = 0.0                              # projected onto the ground
    return p


def footstep_target(leg, p_com, psi, v_world, v_cmd, stance_time,
                    z0=Z_NOMINAL, terrain_height=None, R=None):
    """
    Sombolestan & Nguyen, Eq. (14).

    v_world : measured body velocity (world)
    v_cmd   : commanded velocity (world)
    """
    p_h = hip_world(leg, p_com, psi, R)
    raibert = 0.5 * stance_time * np.asarray(v_cmd, float)
    capture = np.sqrt(z0 / GRAVITY) * (np.asarray(v_world, float) -
                                       np.asarray(v_cmd, float))
    p_f = p_h + raibert + capture
    # The target is the CENTRE of the foot sphere, which sits R_FOOT above
    # the contact point; targeting the ground itself makes the swing PD push
    # the foot 2 cm into the floor (early impact, torque saturation).
    p_f[2] = (R_FOOT if terrain_height is None else terrain_height + R_FOOT)
    return p_f


def bezier_swing(p0, pf, s, height=0.08, td_vel=0.0):
    """
    Swing trajectory: cubic smoothing horizontally, single-apex profile
    vertically.

    s      : swing progress [0, 1]
    height : extra height at the apex (m)
    td_vel : desired vertical touchdown speed (m/s, positive = down);
             0 lands with zero vertical speed (no impact).

    Returns (position, d/ds, d2/ds2); divide by swing_time and swing_time^2
    to get velocity and acceleration in time.

    Vertical profile 16 s^2 (1-s)^2: 0 at s = 0 and 1, 1 at s = 0.5; its
    first derivative is zero at both ends (no lift-off/touchdown impact).
    Its second derivative 32 - 192 s + 192 s^2 is largest at lift-off and
    touchdown (+32, a_z = 32 h / T^2): for h = 0.07 m, T = 0.2 s that is
    56 m/s^2, which pure PD can only produce by lagging (~19 mm at kp=700),
    exactly when footstep accuracy matters. The acceleration feed-forward
    supplies this force instead.
    """
    s = float(np.clip(s, 0.0, 1.0))
    p0 = np.asarray(p0, float)
    pf = np.asarray(pf, float)

    # horizontal: 3s^2 - 2s^3 (zero velocity at both ends)
    a = 3.0 * s**2 - 2.0 * s**3
    da = 6.0 * s - 6.0 * s**2
    dda = 6.0 - 12.0 * s

    pos = p0 + (pf - p0) * a
    vel = (pf - p0) * da
    acc = (pf - p0) * dda

    # vertical: base interpolation + profile
    camber = 16.0 * s**2 * (1.0 - s)**2
    dcamber = 32.0 * s * (1.0 - s) * (1.0 - 2.0 * s)
    ddcamber = 32.0 - 192.0 * s + 192.0 * s**2

    pos[2] = (1 - a) * p0[2] + a * pf[2] + height * camber
    vel[2] = da * (pf[2] - p0[2]) + height * dcamber
    acc[2] = dda * (pf[2] - p0[2]) + height * ddcamber

    # optional touchdown speed: accelerate downward over the last 20%
    if td_vel > 0.0:
        w = np.clip((s - 0.8) / 0.2, 0.0, 1.0) ** 2
        vel[2] -= td_vel * w
        acc[2] -= td_vel * 2.0 * np.clip((s - 0.8) / 0.2, 0.0, 1.0) / 0.2

    return pos, vel, acc


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    print("Gait planner checks\n")

    # 1) trot: exactly two legs on the ground at all times
    print("1) Trot contact pattern")
    g = GaitScheduler("trot")
    counts = []
    for t in np.linspace(0, 2 * g.period, 401):
        counts.append(int(g.contact(t).sum()))
    counts = np.array(counts)
    print(f"   period={g.period}s, duty={g.duty}, "
          f"stance={g.stance_time}s, swing={g.swing_time}s")
    print(f"   legs on the ground: min={counts.min()} max={counts.max()} "
          f"mean={counts.mean():.2f}")
    ok = (counts == 2).all()
    print(f"   {'PASS' if ok else 'FAIL'}: exactly 2 legs on the ground in trot\n")

    # 2) diagonal pairs
    print("2) Diagonal pairing (FR-RL and FL-RR)")
    bad = 0
    for t in np.linspace(0, 2 * g.period, 401):
        c = g.contact(t)
        if c[0] != c[3] or c[1] != c[2]:
            bad += 1
    print(f"   mismatched instants: {bad}/401 "
          f"({'PASS' if bad == 0 else 'FAIL'})\n")

    # 3) swing trajectory
    print("3) Swing trajectory")
    p0 = np.array([0.183, -0.132, 0.0])
    pf = np.array([0.283, -0.132, 0.0])
    pos0, vel0, _ = bezier_swing(p0, pf, 0.0)
    pos1, vel1, _ = bezier_swing(p0, pf, 1.0)
    posm, _, _ = bezier_swing(p0, pf, 0.5)
    print(f"   s=0.0: pos={np.round(pos0,4).tolist()} vel={np.round(vel0,4).tolist()}")
    print(f"   s=0.5: pos={np.round(posm,4).tolist()}")
    print(f"   s=1.0: pos={np.round(pos1,4).tolist()} vel={np.round(vel1,4).tolist()}")
    ok = (np.allclose(pos0, p0) and np.allclose(pos1, pf)
          and np.allclose(vel0, 0, atol=1e-9) and np.allclose(vel1, 0, atol=1e-9))
    print(f"   correct end positions, zero end velocity: {'PASS' if ok else 'FAIL'}")
    print(f"   apex height: {posm[2]:.4f} m (expected 0.08)\n")

    # 4) analytic velocity/acceleration vs finite differences
    print("4) Analytic velocity/acceleration vs finite differences")
    worst_v, worst_a = 0.0, 0.0
    h = 1e-5
    for s in np.linspace(0.02, 0.98, 97):
        _, v, acc = bezier_swing(p0, pf, s)
        pu, vu, _ = bezier_swing(p0, pf, s + h)
        pd, vd, _ = bezier_swing(p0, pf, s - h)
        pm, _, _ = bezier_swing(p0, pf, s)
        worst_v = max(worst_v, np.max(np.abs(v - (pu - pd) / (2 * h))))
        worst_a = max(worst_a, np.max(np.abs(acc - (pu - 2*pm + pd) / h**2)))
    print(f"   velocity largest difference: {worst_v:.2e} "
          f"({'PASS' if worst_v < 1e-6 else 'FAIL'})")
    print(f"   accel.   largest difference: {worst_a:.2e} "
          f"({'PASS' if worst_a < 1e-3 else 'FAIL'})")

    # 4b) required physical acceleration: apex and touchdown
    print("   required vertical acceleration (a_z = acc_ds / T_swing^2):")
    for T_sw, hh in [(0.20, 0.07), (0.25, 0.05), (0.30, 0.07)]:
        _, _, acc_mid = bezier_swing(p0, pf, 0.5, height=hh)
        _, _, acc_td = bezier_swing(p0, pf, 1.0, height=hh)
        a_mid = abs(acc_mid[2]) / T_sw**2
        a_td = abs(acc_td[2]) / T_sw**2
        print(f"   T={T_sw:.2f}s h={hh:.2f}m -> apex {a_mid:5.1f} "
              f"({a_mid/9.81:.1f}g),  TOUCHDOWN {a_td:5.1f} m/s^2 "
              f"({a_td/9.81:.1f}g)  <- largest demand at touchdown")
    print()

    # 5) footstep planning
    print("5) Footstep target")
    p_com = np.array([0.0, 0.0, Z_NOMINAL])
    for v_cmd, v_act, label in [([0, 0, 0], [0, 0, 0], "standing"),
                                ([0.5, 0, 0], [0.5, 0, 0], "0.5 m/s, no error"),
                                ([0.5, 0, 0], [0.2, 0, 0], "0.5 m/s, lagging")]:
        p = footstep_target("FR", p_com, 0.0, np.array(v_act, float),
                            np.array(v_cmd, float), g.stance_time)
        print(f"   {label:22s}: {np.round(p, 4).tolist()}")
    print("   (under the hip when standing, ahead when moving, further back when lagging)")


if __name__ == "__main__":
    self_test()
