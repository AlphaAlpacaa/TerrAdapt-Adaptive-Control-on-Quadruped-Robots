"""
Time-series data for the report figures. Seed 0 only -- for ILLUSTRATION;
every number in the report comes from the three-seed sweeps.

    python report/figure_data.py combined   # Figs. 3 and 5 (combined course must be built)
    python report/figure_data.py pendulum   # Fig. 6 (scene_pendulum)

Output: report/data/<group>_<method>_g<gamma>[_<arm>].npz
"""
import sys
import io
import json
import contextlib
from pathlib import Path
import numpy as np

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK))
from scenario import run_course            # noqa: E402
from course_def import terrain_course       # noqa: E402
from make_course_model import find_model_dir  # noqa: E402

ODIR = Path(__file__).resolve().parent / "data"
ODIR.mkdir(exist_ok=True)
POS = np.array([0.12, 0.05, 0.06])


def save(name, L):
    np.savez_compressed(
        ODIR / f"{name}.npz", t=L["t"], x=L["x"], pitch=L["pitch"],
        roll=L["roll"], pred=L["pred_err"], pendulum=L["pendulum"],
        theta=L["theta_raw"],
        fell=-1.0 if L["fell"] is None else L["fell"],
        stuck=-1.0 if L["stuck"] is None else L["stuck"],
        done=-1.0 if L["done"] is None else L["done"],
        x_final=L["x_final"])
    st = ("FELL" if L["fell"] is not None else "STALLED"
          if L["stuck"] is not None else "COMPLETED"
          if L["done"] is not None else "TIME_UP")
    print(f"{name:28s} {st:10s} x={L['x_final']:.4f}", flush=True)


def run(kind, g, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        L, _ = run_course(kind, g, 0.3, kw.pop("T"), terrain_course(), 0,
                          False, **kw)
    return L


group = sys.argv[1]
if group == "combined":
    z = json.loads((find_model_dir() / "obstacle_zone.json").read_text())
    assert z["base"] == "terrain" and not z["loose"], z
    for k, g in [("none", None), ("M0I", 300.0), ("M0", 300.0),
                 ("M0F", 300.0), ("M1", 300.0), ("M2", 300.0),
                 ("M0", 1000.0), ("M1", 1000.0), ("M2", 1000.0)]:
        L = run(k, g, T=float(z["T"]), payload_pos=POS, terrain=True,
                obstacles=True)
        save(f"combined_{k}_g{0 if g is None else int(g)}", L)
elif group == "pendulum":
    rigid = POS - np.array([0, 0, 0.04])
    for k in ["none", "M0", "M0F", "M2"]:
        g = None if k == "none" else 300.0
        for arm, p, pend in [("pendulum", POS, {"L": 0.04, "axis": "x"}),
                             ("rigid", rigid, None)]:
            L = run(k, g, T=20.0, payload_pos=p, pendulum=pend)
            save(f"pendulum_{k}_g{0 if g is None else int(g)}_{arm}", L)
else:
    sys.exit("usage: figure_data.py combined|pendulum")
