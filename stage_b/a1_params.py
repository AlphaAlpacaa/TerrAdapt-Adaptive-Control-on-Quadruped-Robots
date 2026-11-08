"""
Unitree A1 physical parameters.

All values were read from the mujoco_menagerie/unitree_a1 model; none are
guessed. Parameters not given in the papers are marked under TUNED.
"""

import numpy as np

# ---------------------------------------------------------------------------
# MASS AND INERTIA
# ---------------------------------------------------------------------------

MASS_TOTAL = 12.453          # kg, sum of all bodies
MASS_TRUNK = 4.713           # kg

# Trunk inertia in its own principal frame (diagonal)
I_TRUNK = np.diag([0.045654, 0.037800, 0.015853])

# Composite inertia of the whole robot in the nominal stance, about the
# system COM. The SRB model assumes negligible leg inertia (Di Carlo 2018,
# Assumption 1; legs ~10% of the mass on Cheetah 3). On the A1 the legs are
# 62% of the mass (7.74 / 12.453), so the assumption fails: the yaw inertia
# differs by 25x (0.3996 vs 0.0159). The MPC therefore uses I_COMPOSITE.
I_COMPOSITE = np.array([
    [ 1.41070e-01, -2.54000e-04, -9.66100e-03],
    [-2.54000e-04,  3.67186e-01, -3.96000e-04],
    [-9.66100e-03, -3.96000e-04,  3.99641e-01],
])

# Offset of the system COM from the trunk origin (body frame)
COM_OFFSET = np.array([-0.01127, 0.00155, -0.01960])

# ---------------------------------------------------------------------------
# GEOMETRY  (body frame, trunk origin)
# ---------------------------------------------------------------------------

HIP_X = 0.183        # x offset of the hip abduction axis (front +, rear -)
HIP_Y = 0.047        # y offset of the hip abduction axis (left +, right -)
D_ABD = 0.085        # lateral distance from the abduction axis to the leg plane
L_THIGH = 0.200      # thigh length (hip pitch joint to knee)
L_CALF = 0.200       # calf length (knee to foot centre)
R_FOOT = 0.020       # foot sphere radius

# Leg order and signs. s = +1 left, -1 right.
LEGS = ["FR", "FL", "RR", "RL"]
HIP_POS = {                                   # hip position, body frame
    "FR": np.array([ HIP_X, -HIP_Y, 0.0]),
    "FL": np.array([ HIP_X,  HIP_Y, 0.0]),
    "RR": np.array([-HIP_X, -HIP_Y, 0.0]),
    "RL": np.array([-HIP_X,  HIP_Y, 0.0]),
}
LEG_SIGN = {"FR": -1.0, "FL": +1.0, "RR": -1.0, "RL": +1.0}

# ---------------------------------------------------------------------------
# JOINT LIMITS  (rad)
# ---------------------------------------------------------------------------

Q_MIN = np.array([-0.8029, -1.0472, -2.6965])   # [abduction, thigh, knee]
Q_MAX = np.array([ 0.8029,  4.1888, -0.9163])

# ---------------------------------------------------------------------------
# ACTUATORS
# ---------------------------------------------------------------------------

TAU_MAX = 33.5       # Nm per joint (forcerange in the model)

# The menagerie actuators are position servos (gaintype=0, biastype=1,
# kp=100, kv=0). Force control needs torque motors: make_torque_model.py.

# ---------------------------------------------------------------------------
# NOMINAL STANCE
# ---------------------------------------------------------------------------

Z_NOMINAL = 0.27                       # trunk height, keyframe
# The MPC state uses the COM (subtree_com), which is 0.0196 m below the
# trunk origin; using Z_NOMINAL as reference would lift the robot by 2 cm.
Z_COM_NOMINAL = Z_NOMINAL + COM_OFFSET[2]   # = 0.2504 m
Q_NOMINAL = np.array([0.0, 0.9, -1.8])  # per leg [abd, thigh, knee]

# ---------------------------------------------------------------------------
# CONTACT AND SIMULATION
# ---------------------------------------------------------------------------

MU_FOOT = 0.8        # foot sphere tangential friction
MU_FLOOR = 1.0       # floor plane
# MuJoCo rule: with equal priority the element-wise MAXIMUM is used, so the
# effective mu is 1.0. A low-friction scenario must lower both.
MU_EFFECTIVE = max(MU_FOOT, MU_FLOOR)

TIMESTEP = 0.002     # s  -> soft-ground solref timeconst must be >= 0.004
CONE = "elliptic"    # m.opt.cone = 1
SOLVER = "Newton"    # m.opt.solver = 2

# ---------------------------------------------------------------------------
# TUNED PARAMETERS  (not given in the papers)
# ---------------------------------------------------------------------------

F_MIN = 5.0          # N, minimum normal force (Cheetah 3: 10 N at 43 kg;
                     #    scaled by mass ~2.9 N; 5 N chosen)
F_MAX = 250.0        # N, maximum normal force
                     #    (static m*g = 122 N; ~2x margin on one leg)

GRAVITY = 9.81

# MPC weights from Sombolestan & Nguyen, Table I (given for the A1).
# Source value, kept unchanged for citation.
Q_MPC_PAPER = np.diag([2.5, 2.5, 20.0,      # position x, y, z
                       0.25, 0.25, 1.5,     # orientation roll, pitch, yaw
                       0.2, 0.2, 0.2,       # linear velocity
                       0.1, 0.1, 0.3])      # angular velocity

# The paper's roll/pitch weight (0.25) is too small for the A1.
# Measured (trot, vx = 0.3 m/s, 8 s, run_walk.py):
#     w_rp = 0.25  ->  roll mean +0.2101, std 0.4297 -> fell at t = 7.78 s
#     w_rp = 5.00  ->  roll mean +0.0026, std 0.0265 -> standing, 2.32 m
# Likely cause: the SRB orientation dynamics are less accurate on the A1
# (legs 62% of the mass), so a larger orientation weight is needed.
W_ROLL_PITCH = 5.0
Q_MPC = np.diag([2.5, 2.5, 20.0,
                 W_ROLL_PITCH, W_ROLL_PITCH, 1.5,
                 0.2, 0.2, 0.2,
                 0.1, 0.1, 0.3])
R_MPC = 1e-6                            # force weight (Cheetah 3: alpha = 1e-6)
MPC_HORIZON = 10                        # Sombolestan: horizon 10


def summary():
    print("A1 parameter summary")
    print(f"  total mass          : {MASS_TOTAL} kg  (trunk {MASS_TRUNK}, "
          f"legs {MASS_TOTAL-MASS_TRUNK:.3f} = {100*(MASS_TOTAL-MASS_TRUNK)/MASS_TOTAL:.0f}%)")
    print(f"  composite inertia   : {np.round(np.diag(I_COMPOSITE),4).tolist()}")
    print(f"  trunk inertia       : {np.round(np.diag(I_TRUNK),4).tolist()}")
    print(f"  ratio (comp./trunk) : "
          f"{np.round(np.diag(I_COMPOSITE)/np.diag(I_TRUNK),1).tolist()}")
    print(f"  leg: d={D_ABD}, l1={L_THIGH}, l2={L_CALF}, r_foot={R_FOOT}")
    print(f"  nominal height      : {Z_NOMINAL} m")
    print(f"  effective friction  : {MU_EFFECTIVE}")


if __name__ == "__main__":
    summary()
