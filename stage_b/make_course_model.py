"""
Course model: adds a VISIBLE payload body to the trunk.

Adding the payload at run time only through m.body_mass works physically
but is invisible, and an off-centre mass would need the parallel-axis
theorem by hand. As a real <body>, MuJoCo computes the composite COM and
inertia itself and the box is visible in the viewer.

The payload starts almost massless (1e-6 kg) and invisible (alpha = 0).
At the payload event scenario.py sets body_mass, body_inertia and the
alpha channel, then calls mj_setConst.

The controller keeps using MASS_TOTAL = 12.453 kg: it is NOT told about the
payload. The original model files are not modified.

Box inertia (half sizes a, b, c): I = dm/3 * diag(b^2+c^2, a^2+c^2, a^2+b^2).
"""

import re
from pathlib import Path
import numpy as np
import mujoco

# Half sizes of the payload box (m): a 10 x 10 x 8 cm box.
PAYLOAD_SIZE = (0.05, 0.05, 0.04)
PAYLOAD_BODY = "payload"
PAYLOAD_GEOM = "payload_geom"


def find_model_dir():
    here = Path(__file__).resolve().parent
    for base in [here] + list(here.parents)[:4]:
        hit = list(base.glob("**/unitree_a1/a1_torque.xml"))
        if hit:
            return hit[0].parent
    raise FileNotFoundError("unitree_a1/a1_torque.xml missing -- "
                            "run make_torque_model.py first")


def payload_block(pos, size=PAYLOAD_SIZE):
    """
    Payload body to insert into the trunk.

    contype/conaffinity = 0 : collides with nothing (mass only)
    mass = 1e-6             : negligible at the start
    rgba alpha = 0          : invisible at the start
    """
    sx, sy, sz = size
    px, py, pz = pos
    return (
        f'\n      <body name="{PAYLOAD_BODY}" '
        f'pos="{px:.4f} {py:.4f} {pz:.4f}">\n'
        f'        <geom name="{PAYLOAD_GEOM}" type="box" '
        f'size="{sx} {sy} {sz}" mass="1e-6" '
        f'contype="0" conaffinity="0" '
        f'rgba="0.85 0.15 0.10 0.0"/>\n'
        f'      </body>'
    )


def patch_trunk(xml_text, pos):
    """Insert the payload body right after the trunk's opening tag."""
    m = re.search(r'<body\b[^>]*name\s*=\s*"trunk"[^>]*>', xml_text)
    if m is None:
        raise RuntimeError('<body name="trunk" ...> not found')
    i = m.end()
    return xml_text[:i] + payload_block(pos) + xml_text[i:]


def box_inertia(dm, size=PAYLOAD_SIZE):
    """Diagonal of the box inertia tensor (about its own centre)."""
    a, b, c = size
    return (dm / 3.0) * np.array([b*b + c*c, a*a + c*c, a*a + b*b])


def main(pos=(0.08, -0.05, 0.06)):
    mdir = find_model_dir()
    src = mdir / "a1_torque.xml"
    dst = mdir / "a1_course.xml"
    scene_src = mdir / "scene_torque.xml"
    scene_dst = mdir / "scene_course.xml"

    print("model folder:", mdir)
    text = src.read_text(encoding="utf-8")
    patched = patch_trunk(text, pos)
    dst.write_text(patched, encoding="utf-8")
    print(f"written: {dst.name}  (payload body pos={pos})")

    stext = scene_src.read_text(encoding="utf-8")
    stext = stext.replace('"a1_torque.xml"', '"a1_course.xml"')
    scene_dst.write_text(stext, encoding="utf-8")
    print(f"written: {scene_dst.name}")

    # --- checks --------------------------------------------------------
    print("\n=== CHECKS ===")
    m0 = mujoco.MjModel.from_xml_path(str(scene_src))
    m = mujoco.MjModel.from_xml_path(str(scene_dst))

    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PAYLOAD_BODY)
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, PAYLOAD_GEOM)
    print(f"payload body id : {pid}  (found: {pid >= 0})")
    print(f"payload geom id : {gid}  (found: {gid >= 0})")
    assert pid >= 0 and gid >= 0

    tot0 = float(np.sum(m0.body_mass))
    tot1 = float(np.sum(m.body_mass))
    print(f"total mass      : {tot0:.6f} -> {tot1:.6f} kg "
          f"(diff {tot1-tot0:.2e}, should be negligible)")
    ok_mass = abs(tot1 - tot0) < 1e-4
    print(f"  {'PASS' if ok_mass else 'FAIL'}: payload has no effect at the start")

    print(f"collision     : contype={int(m.geom_contype[gid])} "
          f"conaffinity={int(m.geom_conaffinity[gid])} "
          f"({'PASS -- no collision' if m.geom_contype[gid] == 0 else 'FAIL'})")
    print(f"visibility    : rgba={np.round(m.geom_rgba[gid],2).tolist()} "
          f"({'PASS -- invisible at the start' if m.geom_rgba[gid,3] == 0 else 'FAIL'})")

    print(f"geom count    : {m0.ngeom} -> {m.ngeom}")

    # attach the payload and measure the COM shift
    d = mujoco.MjData(m)
    if m.nkey > 0:
        d.qpos[:] = m.key_qpos[0]
    mujoco.mj_forward(m, d)
    com0 = d.subtree_com[1].copy()

    dm = 3.0
    m.body_mass[pid] = dm
    m.body_inertia[pid] = box_inertia(dm)
    # mj_setConst uses d as scratch and resets qpos to qpos0; save/restore.
    qpos, qvel = d.qpos.copy(), d.qvel.copy()
    mujoco.mj_setConst(m, d)
    d.qpos[:], d.qvel[:] = qpos, qvel
    mujoco.mj_forward(m, d)
    com1 = d.subtree_com[1].copy()

    print(f"\npayload {dm} kg attached:")
    print(f"  total mass : {float(np.sum(m.body_mass)):.4f} kg")
    print(f"  COM shift  : {np.round((com1-com0)*1000, 2).tolist()} mm")
    # expected shift dm/(M+dm) * (r_payload - COM_old), payload position
    # rotated from the body frame to the world frame
    trunk_pos = d.xpos[1].copy()
    R_trunk = d.xmat[1].reshape(3, 3)
    r_world = trunk_pos + R_trunk @ np.array(pos)
    M_old = tot0
    expected = dm / (M_old + dm) * (r_world - com0)
    print(f"  expected   : {np.round(expected*1000, 2).tolist()} mm")
    err = np.max(np.abs((com1 - com0) - expected))
    print(f"  difference : {err*1000:.3f} mm "
          f"({'PASS' if err < 1e-4 else 'FAIL'})")

    print("\nscene_course.xml ready; scenario.py uses it.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=float, nargs=3, default=[0.08, -0.05, 0.06],
                    help="payload position, body frame (m)")
    a = ap.parse_args()
    main(tuple(a.pos))
