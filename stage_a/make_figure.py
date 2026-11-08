"""Asama A ozet sekli: izleme hatasi, fonksiyon kestirim hatasi, agirlik normu."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generalization import run, traj_train
from approximators import M0_LinearRegressor, M1_RBF, M2_LbDNN

models = [
    ("M0  alpha||e||+beta  (4 par)",
     M0_LinearRegressor(gamma=800.0, seed=1), "#555555"),
    ("M1  RBF 7x7  (98 par)",
     M1_RBF(n_centers_per_dim=7, width=0.8, gamma=600.0, sigma=0.001, seed=1), "#1f77b4"),
    ("M2  Lb-DNN 2x6  (66 par)",
     M2_LbDNN(n_hidden=2, width=6, activation="lrelu", gamma=200.0,
              switch_gain=True, seed=1), "#d62728"),
]

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))

for label, m, c in models:
    ts, ee, fe = run(m, traj_train, 20.0)
    ax[0].semilogy(ts, np.maximum(ee, 1e-8), color=c, lw=1.2, label=label)
    ax[1].plot(ts, fe, color=c, lw=1.2, label=label)

# ||f|| referans cizgisi
from plant import f_true
tt = np.linspace(0, 20, 2000)
fn = np.array([np.linalg.norm(f_true(traj_train(t)[0])) for t in tt])
ax[1].plot(tt, fn, color="k", ls=":", lw=1.0, label="||f||  (yaklasiklama yok)")

ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("||e||")
ax[0].set_title("Izleme hatasi"); ax[0].grid(alpha=.3)
ax[0].legend(fontsize=7, loc="upper right")

ax[1].set_xlabel("t [s]"); ax[1].set_ylabel(r"$||f - \hat{\Phi}||$")
ax[1].set_title("Fonksiyon kestirim hatasi"); ax[1].grid(alpha=.3)
ax[1].set_ylim(0, 1.3); ax[1].legend(fontsize=7)

# Genelleme cubuk grafigi
from generalization import traj_test, rms_tail
names, trains, tests = [], [], []
for label, m, c in models:
    ts, ee, fe = run(m, traj_train, 20.0)
    trains.append(rms_tail(fe, ts, 10.0))
    ts2, _, fe2 = run(m, traj_test, 10.0, freeze=True)
    tests.append(rms_tail(fe2, ts2, 3.0))
    names.append(label.split()[0])

xpos = np.arange(3)
ax[2].bar(xpos - 0.18, trains, 0.36, label="egitim yorungesi", color="#2a9d8f")
ax[2].bar(xpos + 0.18, tests, 0.36, label="gorulmemis yorunge\n(agirliklar donuk)",
          color="#e76f51")
ax[2].axhline(0.81, color="k", ls=":", lw=1.0)
ax[2].text(2.35, 0.83, "||f||", fontsize=8, ha="right")
ax[2].set_xticks(xpos); ax[2].set_xticklabels(names)
ax[2].set_ylabel(r"$||f-\hat{\Phi}||$  RMS")
ax[2].set_title("Genelleme: hicbiri aktarmiyor")
ax[2].legend(fontsize=7); ax[2].grid(alpha=.3, axis="y")

plt.tight_layout()
plt.savefig("stage_a_summary.png", dpi=150)
print("stage_a_summary.png yazildi")
