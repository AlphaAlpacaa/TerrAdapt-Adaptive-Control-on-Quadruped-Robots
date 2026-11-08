"""
Full control loop: reference -> MPC -> swing control -> joint torques,
with the adaptive layer feeding theta_hat into the MPC prediction model.

Structure (Sombolestan & Nguyen, Fig. 2, baseline control structure):

    user command -> reference trajectory
                          |
              +-----------+-----------+
              v                       v
        gait planner            MPC (stance legs)
              |                       |
        swing trajectory          forces F_i
              |                       |
              +-----> low-level torque mapping -> robot

Torque mapping (paper Eqs. 15-16):
    stance: tau_i = -J(q_i)^T R^T F_i
    swing : tau_i =  J(q_i)^T R^T [ Kp (p_d - p) + Kd (v_d - v) ]
                    + gravity, inertia and friction feed-forward
"""

import numpy as np

from a1_params import (MASS_TOTAL, GRAVITY, Z_NOMINAL, Z_COM_NOMINAL,
                       Q_NOMINAL, LEGS, MPC_HORIZON, MU_EFFECTIVE,
                       TAU_MAX, R_FOOT)
from a1_kinematics import (foot_position, foot_jacobian, leg_gravity_torque,
                           swing_feedforward_torque, joint_friction_torque)
from srb_dynamics import N_STATE, N_INPUT, rot_z
from mpc import solve_mpc
from gait import GaitScheduler, footstep_target, bezier_swing
from adaptive import AdaptiveLayer


class Controller:
    def __init__(self, gait="trot", dt_control=0.002, dt_mpc=0.03,
                 gait_period=None, gait_duty=None,
                 mpc_every=0.02, horizon=MPC_HORIZON,
                 kp_swing=700.0, kd_swing=25.0, swing_height=0.07,
                 mu=MU_EFFECTIVE, Q_weights=None,
                 link_masses=(0.696, 1.013, 0.226), link_coms=None,
                 link_inertias=None, joint_armature=None,
                 joint_damping=None, joint_frictionloss=None,
                 gravity_comp=True, accel_ff=True, ff_scale=1.0,
                 friction_comp=True, friction_scale=1.0,
                 friction_in_stance=False, target_hold=0.5,
                 adaptive="none", adaptive_kw=None):
        self.gait = GaitScheduler(gait, gait_period, gait_duty)
        self.dt_control = dt_control
        self.dt_mpc = dt_mpc
        self.mpc_every = mpc_every
        self.k = horizon
        self.kp_swing = kp_swing
        self.kd_swing = kd_swing
        self.swing_height = swing_height
        self.mu = mu
        # Leg gravity compensation: the swing leg (1.935 kg, ~19 N) must also
        # carry its own weight, otherwise it drags.
        self.link_masses = link_masses
        self.link_coms = link_coms
        self.link_inertias = link_inertias
        self.gravity_comp = gravity_comp
        # Acceleration feed-forward (Lambda a_ref term of Di Carlo Eq. 2).
        # At touchdown the swing trajectory demands 32 h / T^2 = 56 m/s^2;
        # pure PD would lag by ~19 mm to produce that force.
        self.accel_ff = accel_ff
        self.ff_scale = float(ff_scale)      # 0..1, for ablation
        # Joint parameters read from the model (armature, damping,
        # frictionloss). Including them reduced the swing tracking error
        # from 24.7 to 4.5 mm.
        self.joint_armature = joint_armature
        self.joint_damping = joint_damping
        self.joint_frictionloss = joint_frictionloss
        self.friction_comp = friction_comp
        self.friction_scale = float(friction_scale)
        # Friction compensation in stance: off. Measured (trot in place,
        # 4 s): no tracking gain (4.5 vs 4.9 mm) but pitch 2.5x worse and
        # 11 Nm more torque, because it disturbs the forces the MPC allocates.
        self.friction_in_stance = friction_in_stance
        # Footstep target freeze (Bledt et al., IROS 2018): the capture-point
        # term makes the target jitter by +-5 mm per control step; it is
        # frozen in the second part of the swing.
        self.target_hold = float(target_hold)   # swing phase > this -> fixed
        self._target = {l: None for l in LEGS}

        # --- adaptive layer (body-level acceleration residual) ----------
        # theta_hat enters the MPC dynamics as an affine term, so the
        # correction is distributed inside the friction cone and force limits.
        self.adaptive_name = adaptive
        self.layer = AdaptiveLayer(kind=adaptive, dt=dt_control,
                                   I_body=None, **(adaptive_kw or {}))
        self.theta = np.zeros(6)
        # Predictor input: foot forces commanded in the previous step (swing
        # legs zero). The difference to the real ground reaction force is
        # part of what theta_hat absorbs.
        self._u_applied = np.zeros(N_INPUT)

        self.swing_err = np.zeros(4)
        self.ff_tau = np.zeros(4)
        self.fric_tau = np.zeros(4)
        self.target_move = np.zeros(4)
        # MPC state weights (default: a1_params.Q_MPC).
        self.Q = None if Q_weights is None else np.diag(np.asarray(Q_weights, float))

        self.v_cmd = np.zeros(3)         # desired velocity, world frame
        self.yaw_rate_cmd = 0.0

        # Integrated reference (Di Carlo Sec. IV-B): xy position and yaw are
        # integrals of the commanded velocities.
        self.p_ref = None                # initialised on the first call
        self.psi_ref = None
        self.ref_leash = 0.15            # m   maximum distance the reference
        self.yaw_leash = 0.30            # rad may run ahead (anti-windup)

        self._last_mpc_t = -np.inf
        self._F = np.zeros(N_INPUT)      # last MPC solution
        self._warm = None
        self._swing_start = {l: None for l in LEGS}   # lift-off position
        self._prev_contact = np.ones(4, bool)
        self._q_ik = {l: np.array(Q_NOMINAL, float) for l in LEGS}
        self._R = np.eye(3)
        self.touchdown = {l: None for l in LEGS}   # last touchdown, body frame
        self.td_error = {l: None for l in LEGS}    # last touchdown error, world
        self._terrain_h = None      # estimated terrain height under the body
        self.terrain_h = 0.0
        self.terrain_grad = np.zeros(2)   # (dz/dx, dz/dy) -- local slope
        self._td_buf = []           # recent touchdown points (world)
        self.td_buf_len = 8         # two gait cycles (swept)
        self.grad_max = 0.30        # slope limit (17 deg)
        self.dz_max = 0.04          # maximum height correction (m)
        self._body_xy = np.zeros(2)
        self.terrain_lpf = 0.02     # low-pass coefficient

        self.log = {}

    # -----------------------------------------------------------------
    def ground_z(self, x, y):
        """
        Height of the estimated terrain plane at (x, y). Footstep targets
        are evaluated here, not under the body: on the descent the front
        foot is 18 cm ahead and 4 cm lower.
        """
        d = np.array([float(x), float(y)]) - self._body_xy
        dz = float(np.clip(self.terrain_grad @ d, -self.dz_max, self.dz_max))
        return self.terrain_h + dz

    def set_command(self, vx=0.0, vy=0.0, yaw_rate=0.0):
        self.v_cmd = np.array([vx, vy, 0.0], float)
        self.yaw_rate_cmd = float(yaw_rate)

    # -----------------------------------------------------------------
    def advance_reference(self, x0, dt):
        """
        Integrate the reference position and yaw with the commanded
        velocities (Di Carlo Sec. IV-B), so the MPC position weight acts as
        real feedback. Anti-windup: the reference may not move further
        than ref_leash from the actual position.
        """
        if self.p_ref is None:
            self.p_ref = x0[0:3].copy()
            self.psi_ref = float(x0[5])

        self.p_ref = self.p_ref + self.v_cmd * dt
        self.psi_ref += self.yaw_rate_cmd * dt
        # Height relative to the terrain; an absolute height squeezes the
        # legs when the ground rises on the ramp.
        self.p_ref[2] = self.terrain_h + Z_COM_NOMINAL

        # leash
        d = self.p_ref[0:2] - x0[0:2]
        n = np.linalg.norm(d)
        if n > self.ref_leash:
            self.p_ref[0:2] = x0[0:2] + d * (self.ref_leash / n)
        dpsi = np.arctan2(np.sin(self.psi_ref - x0[5]),
                          np.cos(self.psi_ref - x0[5]))
        if abs(dpsi) > self.yaw_leash:
            self.psi_ref = x0[5] + np.sign(dpsi) * self.yaw_leash

    def reference_trajectory(self, x0, t):
        """
        Reference over the horizon (Di Carlo Sec. IV-B): only xy velocity,
        xy position, z, yaw and yaw rate are non-zero.
        """
        k, dt = self.k, self.dt_mpc
        X = np.zeros((k, N_STATE))
        for i in range(1, k + 1):
            tau = i * dt
            X[i-1, 0:3] = self.p_ref + self.v_cmd * tau
            X[i-1, 2] = self.terrain_h + Z_COM_NOMINAL  # relative to terrain
            X[i-1, 3:6] = [0.0, 0.0, self.psi_ref + self.yaw_rate_cmd * tau]
            X[i-1, 6:9] = self.v_cmd
            X[i-1, 9:12] = [0.0, 0.0, self.yaw_rate_cmd]
            X[i-1, 12] = GRAVITY
        return X

    # -----------------------------------------------------------------
    def predict_feet(self, foot_now, contact_now, x0, t):
        """
        Foot positions over the horizon: stance feet stay, swing feet use
        the planned footstep. One position set is repeated over the horizon.
        """
        k = self.k
        feet = np.zeros((k, 4, 3))
        psi = x0[5]
        v_act = x0[6:9]
        for i in range(k):
            for j, leg in enumerate(LEGS):
                if contact_now[j]:
                    feet[i, j] = foot_now[j]
                else:
                    p_t = footstep_target(
                        leg, x0[0:3], psi, v_act, self.v_cmd,
                        self.gait.stance_time, terrain_height=None)
                    # evaluate z at the target's own xy
                    p_t[2] = self.ground_z(p_t[0], p_t[1]) + R_FOOT
                    feet[i, j] = p_t
        return feet

    # -----------------------------------------------------------------
    def compute(self, t, state, foot_world, foot_body, q, dq, R):
        """
        state      : (13,) SRB state [p, Theta, pdot, omega, g]
        foot_world : (4,3) foot positions, world frame
        foot_body  : (4,3) foot positions, body frame
        q, dq      : (12,) joint angles and velocities
        R          : (3,3) body -> world rotation

        Returns: (12,) joint torques
        """
        contact = self.gait.contact(t)
        swing_s = self.gait.swing_phase(t)

        # --- terrain estimate (no perception; from the feet) ----------
        # (1) Height: low-pass-filtered mean of the stance feet. Feeds the
        #     MPC height reference.
        # (2) Slope: least-squares plane through the last K touchdowns.
        #     A single height for all feet misplaces footsteps by 41.8 mm
        #     on a 10 deg ramp. K=4 fits a spurious slope on the rough
        #     section (front and rear feet can be in anti-phase with the
        #     25 cm bumps), so K=8 (two gait cycles). Measured footstep
        #     error, ramp / rough: scalar 41.8 / 29.9 mm, K=4 9.0 / 50.6 mm,
        #     K=8 13.3 / 31.6 mm. The remaining ~30 mm on the rough section
        #     cannot be removed by any plane model; it is part of the
        #     uncertainty the adaptive layer must absorb.
        if contact.any():
            h_meas = float(np.mean(foot_world[contact, 2])) - R_FOOT
            if self._terrain_h is None:
                self._terrain_h = h_meas
            else:
                # low-pass: robust to a single bad contact
                a = self.terrain_lpf
                self._terrain_h = (1 - a) * self._terrain_h + a * h_meas
        self.terrain_h = 0.0 if self._terrain_h is None else self._terrain_h

        # Record only at touchdown, otherwise the buffer fills with copies
        # of the same point and K=8 looks back less than one cycle.
        for j_ in np.nonzero(contact & ~self._prev_contact)[0]:
            pt = foot_world[j_].copy()
            pt[2] -= R_FOOT
            self._td_buf.append(pt)
        if len(self._td_buf) > self.td_buf_len:
            del self._td_buf[:len(self._td_buf) - self.td_buf_len]

        xc, yc = float(state[0]), float(state[1])
        self._body_xy = np.array([xc, yc])
        if len(self._td_buf) >= 4:
            P = np.asarray(self._td_buf)
            A = np.column_stack([np.ones(len(P)), P[:, 0] - xc, P[:, 1] - yc])
            sol, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
            g_meas = np.clip(sol[1:3], -self.grad_max, self.grad_max)
            a = self.terrain_lpf
            self.terrain_grad = (1 - a) * self.terrain_grad + a * g_meas
        self.swing_err = np.zeros(4)
        self.ff_tau = np.zeros(4)
        self.fric_tau = np.zeros(4)
        self.target_move = np.zeros(4)
        self._R = R
        self.advance_reference(state, self.dt_control)

        # --- touchdown diagnostics -------------------------------------
        for j, leg in enumerate(LEGS):
            if contact[j] and not self._prev_contact[j]:
                self.touchdown[leg] = foot_body[j].copy()
                # error in the world frame (the body-frame value mixes in
                # the pitch deviation)
                if self._target[leg] is not None:
                    self.td_error[leg] = foot_world[j] - self._target[leg]

        # --- adaptive layer: every control step (faster than the MPC) --
        # The predictor time constant is 1/Lambda = 50 ms, so it must run
        # at the control rate (2 ms), not at the MPC rate (20 ms).
        if self.layer.approx is not None:
            # tracking error e = x - x_d (paper Eq. 13); current reference
            # point (tau = 0) at control rate
            x_d = np.zeros(N_STATE)
            x_d[0:3] = self.p_ref
            x_d[2] = self.terrain_h + Z_COM_NOMINAL
            x_d[5] = self.psi_ref
            x_d[6:9] = self.v_cmd
            x_d[11] = self.yaw_rate_cmd
            x_d[12] = GRAVITY
            e_tr = state - x_d
            e_tr[5] = np.arctan2(np.sin(e_tr[5]), np.cos(e_tr[5]))
            self.theta = self.layer.step(
                state, self._u_applied, foot_world, self.dt_control,
                phase=float(self.gait.phase(t)[0]),
                terrain_h=self.terrain_h, e_track=e_tr)

        # --- MPC, at its own rate ----------------------------------------
        if t - self._last_mpc_t >= self.mpc_every - 1e-12:
            X_ref = self.reference_trajectory(state, t)
            feet_h = self.predict_feet(foot_world, contact, state, t)
            sched = self.gait.contact_horizon(t, self.k, self.dt_mpc)
            F, U, status, _ = solve_mpc(
                state, X_ref, feet_h, sched, self.dt_mpc, mu=self.mu,
                warm=self._warm, Q=self.Q,
                theta=(self.theta if self.layer.approx is not None else None))
            if np.all(np.isfinite(F)):
                self._F = F
                self._warm = U
            self._last_mpc_t = t
            self._status = status

        F_legs = self._F.reshape(4, 3)

        # --- per-leg torques ---------------------------------------------
        tau = np.zeros(12)
        for j, leg in enumerate(LEGS):
            J = foot_jacobian(leg, q[3*j:3*j+3])

            if contact[j]:
                # stance: tau = -J^T R^T F   (paper Eq. 15)
                tau[3*j:3*j+3] = -J.T @ (R.T @ F_legs[j])
                if self.friction_comp and self.friction_in_stance:
                    tau[3*j:3*j+3] += self.friction_scale * \
                        joint_friction_torque(dq[3*j:3*j+3],
                                              self.joint_damping,
                                              self.joint_frictionloss)
                self._swing_start[leg] = None
                self._target[leg] = None
            else:
                # record the lift-off position at the start of the swing
                if self._swing_start[leg] is None:
                    self._swing_start[leg] = foot_world[j].copy()

                # footstep target: updated in the first part of the swing,
                # then frozen
                if self._target[leg] is None or swing_s[j] < self.target_hold:
                    p_new = footstep_target(
                        leg, state[0:3], state[5], state[6:9], self.v_cmd,
                        self.gait.stance_time, R=R, terrain_height=None)
                    # evaluate z at the target's own xy (slope)
                    p_new[2] = self.ground_z(p_new[0], p_new[1]) + R_FOOT
                    if self._target[leg] is not None:
                        self.target_move[j] = float(
                            np.linalg.norm(p_new - self._target[leg]))
                    self._target[leg] = p_new
                p_f = self._target[leg]

                T_sw = max(self.gait.swing_time, 1e-6)
                p_d_w, v_ds, a_dss = bezier_swing(self._swing_start[leg], p_f,
                                                  swing_s[j], self.swing_height)
                v_d_w = v_ds / T_sw
                a_d_w = a_dss / (T_sw * T_sw)       # m/s^2, world frame

                # Swing control entirely in the world frame. Measuring the
                # foot relative to the trunk origin but targeting it relative
                # to the COM once introduced a constant 19.6 mm offset. The
                # foot velocity includes the omega x r term.
                p_w = foot_world[j]
                r_w = p_w - state[0:3]
                v_w = (state[6:9]
                       + np.cross(state[9:12], r_w)
                       + R @ (J @ dq[3*j:3*j+3]))

                self.swing_err[j] = np.linalg.norm(p_d_w - p_w)
                f_cmd_w = (self.kp_swing * (p_d_w - p_w) +
                           self.kd_swing * (v_d_w - v_w))

                # tau = J^T R^T f   (opposite sign to stance, where F is the
                # ground reaction on the robot)
                tau_pd = J.T @ (R.T @ f_cmd_w)

                # feed-forward 1: the leg's own weight
                if self.gravity_comp:
                    tau_g = leg_gravity_torque(leg, q[3*j:3*j+3], R,
                                               self.link_masses,
                                               self.link_coms)
                else:
                    tau_g = 0.0

                # feed-forward 2: trajectory acceleration
                #   tau_ff = M(q) J^-1 (a_body - Jdot qdot), a_body = R^T a_world
                # (body acceleration neglected: ~1 m/s^2 vs a 56 m/s^2 demand)
                if self.accel_ff:
                    tau_a = self.ff_scale * swing_feedforward_torque(
                        leg, q[3*j:3*j+3], dq[3*j:3*j+3], R.T @ a_d_w,
                        self.link_masses, self.link_inertias, self.link_coms,
                        armature=self.joint_armature)
                    self.ff_tau[j] = float(np.linalg.norm(tau_a))
                else:
                    tau_a = 0.0

                # feed-forward 3: joint friction (viscous + Coulomb)
                if self.friction_comp:
                    tau_f = self.friction_scale * joint_friction_torque(
                        dq[3*j:3*j+3], self.joint_damping,
                        self.joint_frictionloss)
                    self.fric_tau[j] = float(np.linalg.norm(tau_f))
                else:
                    tau_f = 0.0

                tau[3*j:3*j+3] = tau_pd + tau_g + tau_a + tau_f

        # for the predictor: forces actually applied
        u_app = F_legs.copy()
        u_app[~contact] = 0.0
        self._u_applied = u_app.ravel()

        self._prev_contact = contact.copy()
        self.last_contact = contact
        return np.clip(tau, -TAU_MAX, TAU_MAX)
