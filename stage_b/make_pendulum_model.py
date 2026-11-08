"""
Swinging-payload (pendulum) model -- a disturbance M0 cannot represent.

M0 is theta_hat = alpha_hat*||e|| + beta_hat; beta_hat is a pure integrator
and removes a constant disturbance (rigid payload). A pendulum adds a
zero-mean oscillating part on top of the constant one. If it locks to the
gait (trot roll oscillation at 2.5 Hz), M1/M2 can see it through roll,
pitch, omega and sin/cos(phase); M0 has no phase in its regressor.

Model: added as the LAST child of the trunk,

    trunk
      ...(four legs)...
      pendulum_pivot   pos = payload position (set per run)
        hinge pendulum_hinge   axis x (roll plane) or y (pitch)
        PAYLOAD_BODY  pos = (0, 0, -L)   <- mass becomes 3 kg at t = 6 s

It must be the last child: read_state reads the leg joints from
d.qpos[7:] in order, and a hinge before the legs would shift every leg
index without any error. The check below asserts the qpos addresses.

childclass="a1" gives joints damping=2, armature=0.01, frictionloss=0.2,
which would lock a 0.0048 kg m^2 pendulum; all three are overridden.

Usage:
    python make_pendulum_model.py
"""

import re
import numpy as np
import mujoco

from make_course_model import (find_model_dir, PAYLOAD_BODY, PAYLOAD_GEOM,
                               PAYLOAD_SIZE)

PIVOT_BODY = "pendulum_pivot"
PIVOT_JOINT = "pendulum_hinge"
ROBOT_DST = "a1_pendulum.xml"
SCENE_DST = "scene_pendulum.xml"

# Defaults -- can be changed per run (body_pos / jnt_axis / dof_*)
L_DEFAULT = 0.04          # m, resonance: g/(2*pi*2.5 Hz)^2 = 0.0398 m
DAMPING_DEFAULT = 0.005   # N m s/rad -- light, so the swing is not damped out
ARMATURE = 1e-5           # keeps the joint non-singular before the payload


def pendulum_xml(L=L_DEFAULT, damping=DAMPING_DEFAULT, size=PAYLOAD_SIZE):
    sx, sy, sz = size
    return (
        f'\n      <body name="{PIVOT_BODY}" pos="0.12 0.05 0.06">\n'
        f'        <joint name="{PIVOT_JOINT}" type="hinge" axis="1 0 0" '
        f'damping="{damping}" armature="{ARMATURE}" frictionloss="0" '
        f'range="-3.1416 3.1416" limited="false"/>\n'
        f'        <body name="{PAYLOAD_BODY}" pos="0 0 {-L:.4f}">\n'
        f'          <geom name="{PAYLOAD_GEOM}" type="box" '
        f'size="{sx} {sy} {sz}" mass="1e-6" '
        f'contype="0" conaffinity="0" '
        f'rgba="0.85 0.15 0.10 0.0"/>\n'
        f'        </body>\n'
        f'      </body>'
    )


def trunk_close_index(text):
    """Index of the trunk's closing tag (counting nested <body>)."""
    m = re.search(r'<body\b[^>]*name\s*=\s*"trunk"[^>]*>', text)
    if m is None:
        raise RuntimeError('<body name="trunk"> not found')
    depth, i = 1, m.end()
    tok = re.compile(r'<body\b[^>]*?(/?)>|</body\s*>')
    while depth > 0:
        t = tok.search(text, i)
        if t is None:
            raise RuntimeError("closing tag of trunk not found")
        if t.group(0).startswith("</"):
            depth -= 1
            if depth == 0:
                return t.start()
        elif t.group(1) != "/":
            depth += 1
        i = t.end()


def extend_keyframes(text):
    """Append a 0 to the qpos (and qvel, if present) of every <key>."""
    n = 0

    def fix(mk):
        nonlocal n
        body = mk.group(0)
        for attr in ("qpos", "qvel"):
            m2 = re.search(rf'{attr}\s*=\s*"([^"]*)"', body)
            if m2 is not None:
                body = (body[:m2.start(1)] + m2.group(1).strip() + " 0"
                        + body[m2.end(1):])
        n += 1
        return body

    text = re.sub(r'<key\b[^>]*?/>', fix, text, flags=re.S)
    return text, n


def main():
    mdir = find_model_dir()
    src = (mdir / "a1_torque.xml").read_text(encoding="utf-8")
    k = trunk_close_index(src)
    out = src[:k] + pendulum_xml() + "\n    " + src[k:]
    out, nk = extend_keyframes(out)
    if nk == 0:
        raise RuntimeError("no keyframe found -- nq would not match")
    (mdir / ROBOT_DST).write_text(out, encoding="utf-8")

    stext = (mdir / "scene_torque.xml").read_text(encoding="utf-8")
    stext = stext.replace('"a1_torque.xml"', f'"{ROBOT_DST}"')
    (mdir / SCENE_DST).write_text(stext, encoding="utf-8")
    print(f"written: {ROBOT_DST}, {SCENE_DST}  ({mdir})")

    # --- checks --------------------------------------------------------
    print("\n=== CHECKS ===")
    m0 = mujoco.MjModel.from_xml_path(str(mdir / "scene_torque.xml"))
    m = mujoco.MjModel.from_xml_path(str(mdir / SCENE_DST))
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, PIVOT_JOINT)
    print(f"nq {m0.nq} -> {m.nq},  nv {m0.nv} -> {m.nv},  nu {m0.nu} -> {m.nu}")
    assert m.nq == m0.nq + 1 and m.nv == m0.nv + 1 and m.nu == m0.nu

    # leg joints at the same qpos addresses?
    for j in range(m0.njnt):
        name = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name is None:
            continue
        j1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert m.jnt_qposadr[j1] == m0.jnt_qposadr[j], name
        assert m.jnt_dofadr[j1] == m0.jnt_dofadr[j], name
    print("leg joint addresses unchanged -- PASS")
    print(f"pendulum joint qposadr={m.jnt_qposadr[jid]} (last, = nq-1 "
          f"{'PASS' if m.jnt_qposadr[jid] == m.nq - 1 else 'FAIL'})")
    assert m.jnt_qposadr[jid] == m.nq - 1
    da = m.jnt_dofadr[jid]
    print(f"pendulum joint damping={m.dof_damping[da]}, "
          f"armature={m.dof_armature[da]}, "
          f"frictionloss={m.dof_frictionloss[da]}  (childclass overridden?)")
    assert m.dof_damping[da] == DAMPING_DEFAULT and m.dof_frictionloss[da] == 0.0

    tot0, tot1 = float(np.sum(m0.body_mass)), float(np.sum(m.body_mass))
    print(f"total mass {tot0:.6f} -> {tot1:.6f} kg  (no effect before the payload)")
    assert abs(tot1 - tot0) < 1e-4

    # resonance
    w_g = 2 * np.pi / 0.40
    print(f"\ntrot period 0.40 s -> {w_g/2/np.pi:.2f} Hz; "
          f"resonant length g/w^2 = {9.81/w_g**2*100:.2f} cm")


if __name__ == "__main__":
    main()
