"""A1: ayak geomlari, aktuator tipi, ileri kinematik, bilesik atalet."""
import mujoco, numpy as np
from pathlib import Path

def find_scene():
    here = Path(__file__).resolve().parent
    for base in [here] + list(here.parents)[:4]:
        hit = list(base.glob("**/unitree_a1/scene.xml"))
        if hit: return str(hit[0])
    raise FileNotFoundError("scene.xml yok")

m = mujoco.MjModel.from_xml_path(find_scene())
d = mujoco.MjData(m)
if m.nkey > 0:
    d.qpos[:] = m.key_qpos[0]
mujoco.mj_forward(m, d)

GTYPE = {0:"plane",1:"hfield",2:"sphere",3:"capsule",4:"ellipsoid",
         5:"cylinder",6:"box",7:"mesh"}

print("=== TUM GEOMLAR (govdesiyle) ===")
for i in range(m.ngeom):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
    b  = m.geom_bodyid[i]
    bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    print(f"{i:3d} {str(nm):16s} govde={str(bn):12s} "
          f"tip={GTYPE.get(int(m.geom_type[i]),'?'):9s} "
          f"boyut={np.round(m.geom_size[i],4).tolist()} "
          f"pos={np.round(m.geom_pos[i],4).tolist()} "
          f"contype={int(m.geom_contype[i])} conaff={int(m.geom_conaffinity[i])} "
          f"fric={np.round(m.geom_friction[i],3).tolist()}")

print("\n=== AKTUATOR TIPI ===")
for i in range(min(m.nu,3)):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"{nm:12s} gaintype={int(m.actuator_gaintype[i])} "
          f"biastype={int(m.actuator_biastype[i])} "
          f"gainprm={np.round(m.actuator_gainprm[i][:3],2).tolist()} "
          f"biasprm={np.round(m.actuator_biasprm[i][:3],2).tolist()} "
          f"forcerange={np.round(m.actuator_forcerange[i],2).tolist()}")
print("(gaintype=0 & biastype=0 -> tork motoru;  biastype=1 -> konum servosu)")

print("\n=== NOMINAL DURUSTA AYAK KONUMLARI (dunya) ===")
for leg in ["FR","FL","RR","RL"]:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
    gids = [g for g in range(m.ngeom) if m.geom_bodyid[g]==bid]
    for g in gids:
        gp = d.geom_xpos[g]
        print(f"{leg}_calf geom{g}: dunya={np.round(gp,4).tolist()} "
              f"yerel={np.round(m.geom_pos[g],4).tolist()} "
              f"r={np.round(m.geom_size[g],4).tolist()}")
    print(f"   {leg}_calf govde orijini: {np.round(d.xpos[bid],4).tolist()}")

print("\n=== GOVDE KONUMLARI (nominal durus) ===")
for nm in ["trunk","FR_hip","FR_thigh","FR_calf"]:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
    print(f"{nm:10s} xpos={np.round(d.xpos[bid],4).tolist()}")

print("\n=== BILESIK ATALET (tum robot, sistem COM'u etrafinda) ===")
M = float(np.sum(m.body_mass))
com = (m.body_mass[:,None] * d.xipos).sum(axis=0) / M
I = np.zeros((3,3))
for i in range(m.nbody):
    mi = float(m.body_mass[i])
    if mi <= 0: continue
    R = d.ximat[i].reshape(3,3)
    Ii = R @ np.diag(m.body_inertia[i]) @ R.T
    r = d.xipos[i] - com
    I += Ii + mi*(float(r@r)*np.eye(3) - np.outer(r,r))
print("toplam kutle:", round(M,4))
print("sistem COM (dunya):", np.round(com,5).tolist())
print("COM'un govde orijinine gore ofseti:",
      np.round(com - d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"trunk")],5).tolist())
print("bilesik atalet I_G:\n", np.round(I,6))
print("kosegen:", np.round(np.diag(I),6).tolist())
print("\nKIYAS - sadece govde ataleti:",
      np.round(m.body_inertia[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"trunk")],6).tolist())