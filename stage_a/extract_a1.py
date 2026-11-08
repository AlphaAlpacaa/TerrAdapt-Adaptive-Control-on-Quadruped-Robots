"""A1 modelinden MPC icin gereken fiziksel parametreleri cikarir."""
import mujoco, numpy as np
from pathlib import Path

# scene.xml'i otomatik bul: bu dosyadan yukari dogru tara
def find_scene():
    here = Path(__file__).resolve().parent
    for base in [here] + list(here.parents)[:4]:
        hit = list(base.glob("**/unitree_a1/scene.xml"))
        if hit:
            return str(hit[0])
    raise FileNotFoundError("unitree_a1/scene.xml bulunamadi")

PATH = find_scene()
print("model:", PATH, "\n")

m = mujoco.MjModel.from_xml_path(PATH)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

print("=== GENEL ===")
print("toplam kutle (kg):", round(float(np.sum(m.body_mass)), 4))
print("nq, nv, nu:", m.nq, m.nv, m.nu)
print("timestep:", m.opt.timestep, " integrator:", int(m.opt.integrator))
print("cone:", int(m.opt.cone), " solver:", int(m.opt.solver))

print("\n=== GOVDE ===")
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
if bid < 0: bid = 1
print("isim :", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid))
print("kutle:", float(m.body_mass[bid]))
print("atalet kosegeni:", m.body_inertia[bid].tolist())
print("ipos (COM ofseti):", m.body_ipos[bid].tolist())

print("\n=== TUM GOVDELER ===")
for i in range(m.nbody):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"{i:3d} {str(nm):18s} m={float(m.body_mass[i]):7.4f} "
          f"pos={np.round(m.body_pos[i],4).tolist()} "
          f"I={np.round(m.body_inertia[i],6).tolist()}")

print("\n=== EKLEMLER ===")
for i in range(m.njnt):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"{i:3d} {str(nm):22s} tip={int(m.jnt_type[i])} "
          f"aralik={np.round(m.jnt_range[i],4).tolist()} "
          f"eksen={np.round(m.jnt_axis[i],3).tolist()} "
          f"pos={np.round(m.jnt_pos[i],4).tolist()}")

print("\n=== AKTUATOR ===")
for i in range(m.nu):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"{i:3d} {str(nm):22s} ctrlrange={np.round(m.actuator_ctrlrange[i],2).tolist()} "
          f"gear={float(m.actuator_gear[i][0]):.3f}")

print("\n=== BASLANGIC DURUSU ===")
if m.nkey > 0:
    print("keyframe sayisi:", m.nkey)
    print("qpos0:", np.round(m.key_qpos[0], 4).tolist())
else:
    print("keyframe yok, qpos0:", np.round(m.qpos0, 4).tolist())

print("\n=== AYAK GEOMLARI ===")
for i in range(m.ngeom):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
    if nm and "foot" in nm.lower():
        print(f"{nm:18s} friction={m.geom_friction[i].tolist()} "
              f"solref={m.geom_solref[i].tolist()} "
              f"solimp={np.round(m.geom_solimp[i],4).tolist()} "
              f"boyut={np.round(m.geom_size[i],4).tolist()}")

print("\n=== AYAK SITE/TEMAS NOKTALARI (varsa) ===")
for i in range(m.nsite):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
    print(f"{i:3d} {str(nm):18s} pos={np.round(m.site_pos[i],4).tolist()}")