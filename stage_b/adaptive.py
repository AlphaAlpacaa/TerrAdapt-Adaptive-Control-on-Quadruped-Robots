"""
Adaptive layer: L1-type state predictor + body-level disturbance estimate.

Architecture (adaptive force control of Sombolestan & Nguyen):

    command -> reference trajectory ----+
                                        v
    state -------------------------> M P C <--- theta_hat (affine term)
      |                                 |
      |                                 v
      |                           foot forces F
      |                                 |
      +---> STATE PREDICTOR <-----------+
                 |
            e_tilde = xhat - x
                 |
            s = -E^T P e_tilde
                 |
          M0I / M0 / M0F / M1 / M2  ->  theta_hat  -> back to the MPC

Why the predictor error and not the tracking error: with the predictor the
adaptation is decoupled from tracking performance (the main idea of L1
adaptive control). e_tilde only says how far the model deviates from
reality; the tracking error also depends on the reference, MPC weights and
constraints.

Derivation
----------
    plant     : xdot     = A x + B u + E theta
    predictor : xhat_dot = A xhat + B u + E theta_hat - Lambda (xhat - x)

    e_tilde_dot = (A - Lambda) e_tilde + E Delta,     Delta = theta_hat - theta
    A_c = A - Lambda Hurwitz, P > 0 solves A_c^T P + P A_c = -Q_p

    V    = e_tilde^T P e_tilde + (1/gamma) Delta^T Delta
    Vdot = -e_tilde^T Q_p e_tilde + 2 Delta^T (E^T P e_tilde)
           + (2/gamma) Delta^T theta_hat_dot
    theta_hat_dot = -gamma E^T P e_tilde  ->  Vdot = -e_tilde^T Q_p e_tilde <= 0

Driving signal s = -E^T P e_tilde in R^6; each approximator distributes s
over its own parameters (approximators_b.py).

Limitations
-----------
1. theta is assumed constant. If it varies, Vdot gains -2/gamma Delta^T
   theta_dot and the result is uniform ultimate boundedness; the
   projection keeps the estimate bounded.
2. P is solved once at psi = 0. A depends on psi only through
   Theta_dot = R_z^T omega, which Lambda dominates; verify_adaptive.py
   checks P over a range of psi.
3. theta_hat collects everything the SRB model misses (including its
   small-angle error), not only the terrain.
"""

import numpy as np

from a1_params import MASS_TOTAL, I_COMPOSITE, GRAVITY, Z_COM_NOMINAL
from srb_dynamics import (continuous_matrices, disturbance_matrix,
                          N_STATE, N_INPUT)
import approximators_b as ap


# ---------------------------------------------------------------------------
# Lyapunov equation:  A_c^T P + P A_c = -Q_p
# ---------------------------------------------------------------------------

def solve_lyapunov(A_c, Q_p):
    """
    Solve the continuous Lyapunov equation explicitly.

    With column-major vec:
        vec(A_c^T P) = (I kron A_c^T) vec(P),  vec(P A_c) = (A_c^T kron I) vec(P)
    so (I kron A_c^T + A_c^T kron I) vec(P) = -vec(Q_p): a 169-unknown
    linear system for a 13x13 P.
    """
    n = A_c.shape[0]
    I = np.eye(n)
    M = np.kron(I, A_c.T) + np.kron(A_c.T, I)
    p = np.linalg.solve(M, -Q_p.ravel(order="F"))
    P = p.reshape((n, n), order="F")
    return 0.5 * (P + P.T)          # enforce symmetry


def lyapunov_residual(A_c, P, Q_p):
    """||A_c^T P + P A_c + Q_p|| -- to check the solution."""
    return float(np.linalg.norm(A_c.T @ P + P @ A_c + Q_p))


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

# Features: body velocity (3), angular velocity (3), roll/pitch (2), height
# relative to the estimated terrain (1), gait phase (2: sin and cos).
#
# Position (x, y) is deliberately excluded so the networks cannot memorise
# the course. The phase is included because the dominant disturbance
# oscillates within the gait cycle: the swing legs produce about
# +-1.4 m/s^2 of phase-dependent body acceleration around a mean of
# -1.63 m/s^2. Leaving the phase out would make the comparison structurally
# favour M0. sin/cos are used because the raw phase jumps from 1 to 0.
FEATURE_NAMES = ["vx", "vy", "vz", "wx", "wy", "wz", "roll", "pitch",
                 "dz", "sin_phi", "cos_phi"]
N_FEATURE = len(FEATURE_NAMES)

# Normalisation so every feature is O(1); required because the RBF centres
# are placed in the box [-1.5, 1.5].
FEATURE_SCALE = np.array([0.5, 0.3, 0.2,      # m/s
                          1.0, 1.0, 1.0,      # rad/s
                          0.15, 0.15,         # rad
                          0.02,               # m (height deviation)
                          1.0, 1.0])          # sin/cos already O(1)


def features(x, phase=0.0, terrain_h=0.0):
    """
    Feature vector (11,) from the SRB state and the gait phase.

    phase     : gait phase in [0, 1) (FR leg as reference)
    terrain_h : estimated local terrain height (m)

    The height feature must be relative to the terrain. An earlier version
    used the absolute body height; on the 0.19 m plateau that scales to
    9.2, far outside the RBF centre box, so all 64 Gaussians were ~1e-13
    and M1 reduced to its bias term on 55% of the course.
    """
    x = np.asarray(x, float)
    ph = 2.0 * np.pi * float(phase)
    return np.array([x[6], x[7], x[8],
                     x[9], x[10], x[11],
                     x[3], x[4],
                     x[2] - float(terrain_h) - Z_COM_NOMINAL,
                     np.sin(ph), np.cos(ph)])


# ---------------------------------------------------------------------------
# Adaptive layer
# ---------------------------------------------------------------------------

class AdaptiveLayer:
    """
    State predictor + approximator.

    kind        : "none" | "M0I" | "M0" | "M0F" | "M1" | "M2"
    lam         : predictor gain, Lambda = lam * I (1/s)
    q_vel       : weight of Q_p on the velocity / angular-velocity blocks
    theta_max   : norm bound on theta_hat
    use_filter  : low-pass filter C(s) on (paper Eqs. 33/45)
    omega_n     : C(s) natural frequency (paper Table I: 60 rad/s)
    zeta        : C(s) damping ratio (paper Table I: 0.7)
    emag_source : regressor magnitude: "tracking" = ||e|| (paper Eqs.
                  27/30, default) or "prediction" = ||e_tilde|| (first
                  version; creates a gamma*||e_tilde||^2 feedback loop)

    Low-pass filter C(s) -- off by default, by measurement
    -------------------------------------------------------
        u_a = -C(s) theta_hat,   C(s) = wn^2 / (s^2 + 2 zeta wn s + wn^2)

    In the paper's MPC version the filtered u_a also enters the prediction
    model as a constant residual (Eqs. 50-52), i.e. the same channel as
    here. Measured over 54 runs (sweep_robust.py --filter-compare):

        gain     C(s) off   C(s) on
         100       8/9        6/9
         300       9/9        6/9
        1000       3/9        3/9

    The filter never helps. The mechanism is not isolated; the candidate
    difference is the MPC rate (paper 300 Hz, here 50 Hz). When on, the
    filter is applied only to the copy sent to the MPC; the predictor sees
    the raw theta_hat, otherwise the Lyapunov cross term does not cancel.
    """

    def __init__(self, kind="M0", lam=20.0, q_pos=1.0, q_vel=10.0,
                 theta_max=80.0, mass=MASS_TOTAL, I_body=None,
                 dt=0.002, seed=0, use_filter=False, omega_n=60.0, zeta=0.7,
                 emag_source="tracking", **approx_kw):
        self.kind = kind
        self.mass = mass
        self.I_body = I_COMPOSITE if I_body is None else I_body
        self.lam = lam
        self.dt = dt
        self.theta_max = theta_max
        self.use_filter = bool(use_filter)
        self.omega_n = float(omega_n)
        self.zeta = float(zeta)
        if emag_source not in ("tracking", "prediction"):
            raise ValueError("emag_source: 'tracking' or 'prediction'")
        self.emag_source = emag_source

        # --- A_c and P, once at psi = 0 --------------------------------
        feet0 = np.array([[0.183, -0.132, 0.0], [0.183, 0.132, 0.0],
                          [-0.183, -0.132, 0.0], [-0.183, 0.132, 0.0]])
        A0, _ = continuous_matrices(0.0, feet0, np.array([0.0, 0.0, 0.25]),
                                    mass, self.I_body)
        self.A0 = A0
        self.A_c = A0 - lam * np.eye(N_STATE)
        q = np.full(N_STATE, q_pos)
        q[6:12] = q_vel          # E enters only these rows
        q[12] = 1e-3             # gravity state: conditioning only
        self.Q_p = np.diag(q)
        self.P = solve_lyapunov(self.A_c, self.Q_p)
        self.E = disturbance_matrix(mass, self.I_body, 0.0)
        self.EtP = self.E.T @ self.P          # (6, 13), s = -EtP e_tilde

        # --- approximator ----------------------------------------------
        if kind == "none":
            self.approx = None
        else:
            kw = dict(seed=seed)
            if kind in ("M1", "M2"):
                kw["scale"] = FEATURE_SCALE
            kw.update(approx_kw)
            self.approx = ap.make(kind, N_FEATURE, **kw)

        self.xhat = None
        self.theta = np.zeros(ap.N_THETA)       # raw estimate -> predictor
        self.theta_mpc = np.zeros(ap.N_THETA)   # (filtered) -> MPC
        self._cf_y = np.zeros(ap.N_THETA)       # C(s) state: output
        self._cf_v = np.zeros(ap.N_THETA)       # C(s) state: output derivative
        self.e_tilde = np.zeros(N_STATE)
        self.e_track = np.zeros(N_STATE)
        self.s = np.zeros(ap.N_THETA)
        self.vdot = 0.0
        self._pending = None

    # -----------------------------------------------------------------
    def reset(self, x0):
        self.xhat = np.asarray(x0, float).copy()
        self._cf_y[:] = 0.0
        self._cf_v[:] = 0.0
        # Warm start: weights are applied AFTER the reset so it cannot
        # overwrite them.
        if self._pending is not None and self.approx is not None:
            self.approx.set_weights(self._pending)
            self._pending = None

    def load_weights(self, w):
        """Load weights; applied at the first reset() if not started yet."""
        if self.approx is None:
            raise ValueError("no adaptation -- no weights to load")
        self._pending = np.asarray(w, dtype=np.float64).copy()
        if self.xhat is not None:          # already running: apply now
            self.approx.set_weights(self._pending)
            self._pending = None

    # -----------------------------------------------------------------
    def _apply_filter(self, th, dt):
        """
        C(s) = wn^2 / (s^2 + 2 zeta wn s + wn^2), per channel.

        State space y' = v, v' = wn^2 (th - y) - 2 zeta wn v, integrated
        with semi-implicit Euler (v first, then y); explicit Euler is close
        to its stability limit at wn*dt = 0.12.
        """
        wn, z = self.omega_n, self.zeta
        self._cf_v += dt * (wn * wn * (th - self._cf_y)
                            - 2.0 * z * wn * self._cf_v)
        self._cf_y += dt * self._cf_v
        return self._cf_y.copy()

    # -----------------------------------------------------------------
    def step(self, x, u, foot_pos, dt=None, phase=0.0, terrain_h=0.0,
             e_track=None):
        """
        One control step: advance the predictor, measure the error, adapt
        the weights, return the new theta_hat.

        x        : (13,) measured state
        u        : (12,) foot forces APPLIED in the previous step
        foot_pos : (4,3) foot positions in the world frame (for B)
        phase    : gait phase in [0, 1)
        e_track  : (13,) tracking error x - x_d (regressor magnitude)
        """
        dt = self.dt if dt is None else dt
        x = np.asarray(x, float)
        if self.approx is None:
            return np.zeros(ap.N_THETA)
        if self.xhat is None:
            self.reset(x)

        psi = x[5]
        A, B = continuous_matrices(psi, foot_pos, x[0:3], self.mass, self.I_body)

        # --- predictor error and driving signal ----------------------
        self.e_tilde = self.xhat - x
        self.s = -(self.EtP @ self.e_tilde)

        # Regressor magnitude. The paper uses two separate signals:
        # e_tilde gives the adaptation direction (Eq. 41, -B^T P e_tilde)
        # and ||e|| the regressor magnitude (Eqs. 27/30). Using ||e_tilde||
        # for both closes a quadratic loop gamma*||e_tilde||^2
        # (M0, gamma=1000: 0/3 with ||e_tilde||, 3/3 with ||e||).
        if e_track is not None:
            self.e_track = np.asarray(e_track, float)
        if self.emag_source == "tracking" and e_track is not None:
            emag = float(np.linalg.norm(self.e_track[0:12]))
        else:
            emag = float(np.linalg.norm(self.e_tilde))

        # --- weight adaptation (explicit Euler; dt = 2 ms) -------------
        w = self.approx.pack()
        xi = features(x, phase, terrain_h)
        wdot = self.approx.adapt_rate(xi, self.s, emag)
        self.approx.unpack(w + dt * wdot)

        # --- new estimate, norm bound ----------------------------------
        th = self.approx.estimate(xi, emag)
        nrm = np.linalg.norm(th)
        if nrm > self.theta_max:
            th = th * (self.theta_max / nrm)
        self.theta = th

        # --- C(s): only the copy sent to the MPC is filtered ----------
        self.theta_mpc = self._apply_filter(th, dt) if self.use_filter else th.copy()

        # --- advance the predictor (explicit Euler) --------------------
        # The predictor sees the RAW theta_hat, never the filtered one.
        xhat_dot = (A @ self.xhat + B @ np.asarray(u, float)
                    + self.E @ self.theta - self.lam * self.e_tilde)
        self.xhat = self.xhat + dt * xhat_dot
        self.xhat[12] = GRAVITY          # gravity state is constant

        # --- error part of V and the guaranteed part of Vdot ----------
        self.vdot = -float(self.e_tilde @ self.Q_p @ self.e_tilde)
        return self.theta_mpc

    # -----------------------------------------------------------------
    def lyapunov(self):
        """V = e_tilde^T P e_tilde + weight part."""
        ve = float(self.e_tilde @ self.P @ self.e_tilde)
        vw = 0.0 if self.approx is None else self.approx.lyapunov_weight_term()
        return ve + vw, ve, vw

    @property
    def n_weights(self):
        return 0 if self.approx is None else self.approx.n_weights

    @property
    def name(self):
        return "none (baseline)" if self.approx is None else self.approx.name
