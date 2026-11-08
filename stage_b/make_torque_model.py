"""
Make the A1 model torque-controlled.

The mujoco_menagerie/unitree_a1 actuators are position servos
(gaintype=0, biastype=1, gainprm=[100,0,0], biasprm=[0,-100,0]), i.e.
tau = 100*(ctrl - q). Force-based control commands torques directly,
tau_stance = -J(q)^T R^T F (Sombolestan & Nguyen, Eq. 15), so the
actuators are converted to <motor>.

This script:
  1. reads a1.xml
  2. replaces the <position .../> actuators with <motor .../>
  3. writes a1_torque.xml and scene_torque.xml
  4. checks that gaintype/biastype are 0 in the new model
  5. checks that the robot collapses with zero torque

The original files are not modified.
"""

import re
import shutil
from pathlib import Path
import numpy as np
import mujoco


def find_model_dir():
    here = Path(__file__).resolve().parent
    for base in [here] + list(here.parents)[:4]:
        hit = list(base.glob("**/unitree_a1/a1.xml"))
        if hit:
            return hit[0].parent
    raise FileNotFoundError("unitree_a1/a1.xml not found")


def patch_actuators(xml_text, tau_max=33.5):
    """
    Replace the position actuators in the <actuator> block with motors,
    keeping the joint names.
    """
    m = re.search(r"<actuator>(.*?)</actuator>", xml_text, re.S)
    if m is None:
        raise RuntimeError("<actuator> block not found")

    block = m.group(1)

    # name and joint of each actuator line
    entries = re.findall(
        r"<(?:position|general|motor)\b[^>]*?name\s*=\s*\"([^\"]+)\"[^>]*?"
        r"joint\s*=\s*\"([^\"]+)\"[^>]*?/>", block)
    if not entries:
        # name/joint may come in the other order
        entries = re.findall(
            r"<(?:position|general|motor)\b[^>]*?joint\s*=\s*\"([^\"]+)\"[^>]*?"
            r"name\s*=\s*\"([^\"]+)\"[^>]*?/>", block)
        entries = [(n, j) for j, n in entries]
    if not entries:
        raise RuntimeError("could not parse the actuator lines")

    lines = ["\n"]
    for name, joint in entries:
        lines.append(
            f'    <motor name="{name}" joint="{joint}" gear="1" '
            f'ctrlrange="{-tau_max} {tau_max}"/>\n')
    lines.append("  ")
    new_block = "".join(lines)

    return xml_text[:m.start(1)] + new_block + xml_text[m.end(1):], entries


def main():
    mdir = find_model_dir()
    src = mdir / "a1.xml"
    dst = mdir / "a1_torque.xml"
    scene_src = mdir / "scene.xml"
    scene_dst = mdir / "scene_torque.xml"

    print("model folder:", mdir)

    text = src.read_text(encoding="utf-8")
    patched, entries = patch_actuators(text)
    dst.write_text(patched, encoding="utf-8")
    print(f"written: {dst.name}  ({len(entries)} actuators converted to motors)")

    # scene.xml includes a1.xml -> copy it and change the include
    stext = scene_src.read_text(encoding="utf-8")
    stext = stext.replace('"a1.xml"', '"a1_torque.xml"')
    scene_dst.write_text(stext, encoding="utf-8")
    print(f"written: {scene_dst.name}")

    # --- checks --------------------------------------------------------
    print("\n=== CHECKS ===")
    m = mujoco.MjModel.from_xml_path(str(scene_dst))
    d = mujoco.MjData(m)

    ok = True
    for i in range(m.nu):
        gt, bt = int(m.actuator_gaintype[i]), int(m.actuator_biastype[i])
        if gt != 0 or bt != 0:
            ok = False
    print(f"actuators are torque motors : {'YES' if ok else 'NO'}")
    print(f"  example: gaintype={int(m.actuator_gaintype[0])} "
          f"biastype={int(m.actuator_biastype[0])} "
          f"gainprm={np.round(m.actuator_gainprm[0][:3],2).tolist()} "
          f"ctrlrange={np.round(m.actuator_ctrlrange[0],2).tolist()}")

    # mass and geometry must be unchanged
    m0 = mujoco.MjModel.from_xml_path(str(scene_src))
    same_mass = abs(float(np.sum(m.body_mass)) - float(np.sum(m0.body_mass))) < 1e-9
    same_n = (m.nq, m.nv, m.nu) == (m0.nq, m0.nv, m0.nu)
    print(f"mass preserved              : {'YES' if same_mass else 'NO'} "
          f"({float(np.sum(m.body_mass)):.4f} kg)")
    print(f"nq/nv/nu preserved          : {'YES' if same_n else 'NO'} "
          f"{(m.nq, m.nv, m.nu)}")

    # zero torque -> the robot must collapse
    if m.nkey > 0:
        d.qpos[:] = m.key_qpos[0]
    mujoco.mj_forward(m, d)
    z0 = float(d.qpos[2])
    d.ctrl[:] = 0.0
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    z1 = float(d.qpos[2])
    print(f"z after 0.5 s at zero torque: {z0:.4f} -> {z1:.4f} m "
          f"({'collapsed, as expected' if z1 < z0 - 0.02 else 'DID NOT COLLAPSE - problem'})")

    # static equilibrium torque
    if m.nkey > 0:
        d.qpos[:] = m.key_qpos[0]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    d.qacc[:] = 0.0
    mujoco.mj_inverse(m, d)
    tau = d.qfrc_inverse[6:]
    print(f"torque needed in the nominal stance: max |tau| = {np.max(np.abs(tau)):.2f} Nm "
          f"(limit {m.actuator_ctrlrange[0][1]:.1f} Nm)")
    print(f"  per leg [abd, thigh, knee]: {np.round(tau[:3], 3).tolist()}")


if __name__ == "__main__":
    main()
