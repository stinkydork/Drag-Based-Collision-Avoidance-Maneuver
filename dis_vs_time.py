# Separation at TCA vs maneuver duration t_m (nominal attitude reference: alpha = 0, beta = 0)
import io
import numpy as np
import torch
import time
import ussa1976
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from OrbitPropagatorGPU import OrbitPropagatorGPU
from OrbitPropagator import OrbitPropagator
import Constants as ct

start_time = time.time()

# ---------------- settings ----------------
h = 500.0                          # [km]
tspan = 4 * (24*60*60)             # time to TCA [s]  (set to 3 days to match the slide)
targets = [5, 10, 15, 20]          # desired separations to read off [km]
N = 5000                           # number of t_m values in the sweep
cpu_dt = 10.0
gpu_dt = 10.0

ob = ct.CubeSat_3U
maneuver_pitch = 0                 # [deg]
maneuver_yaw = -45                 # [deg]

# ---------------- CPU nominal (reference) ----------------
r0, v0 = OrbitPropagator.Kep2State(ct.Earth['radius']+h, 0, 0, 0, 0, 0, cenb=ct.Earth)
with redirect_stdout(io.StringIO()):            # silence per-step debug prints
    cpu = OrbitPropagator(r0, v0, tspan, cpu_dt, cenb=ct.Earth, ob=ob, pointing_mode='nadir')
reference_state_cpu = cpu.rs[-1]

# ---------------- analytic delta_a (same as your script) ----------------
table = ussa1976.compute(z=np.array([h*1000]), variables=['rho', 'p', 't'])
rho, p, T = table['rho'].values.item(), table['p'].values.item(), table['t'].values.item()
vrel = v0 - np.cross(ct.Earth['at_rot_vec'], r0)

F0 = OrbitPropagator.get_AeroForces_RectangularPrism(0, 0, np.linalg.norm(r0), np.linalg.norm(vrel), rho, p, T, ob)
F1 = OrbitPropagator.get_AeroForces_RectangularPrism(-maneuver_pitch, -maneuver_yaw, np.linalg.norm(r0), np.linalg.norm(vrel), rho, p, T, ob)
ca, sa = np.cos(np.radians(-maneuver_pitch)), np.sin(np.radians(-maneuver_pitch))
cb, sb = np.cos(np.radians(-maneuver_yaw)),   np.sin(np.radians(-maneuver_yaw))
proj = lambda F: (ca*cb)*F[0] + sb*F[1] + (cb*sa)*F[2]
delta_a = (abs(proj(F0) - proj(F1)) / ob["mass"]) / 1000      # [km/s^2]
print(f"delta_a = {delta_a:.3e} km/s^2")

# ---------------- GPU sweep ----------------
tm_candidates = np.linspace(0.0, tspan, N + 1)  # index 0 (t_m = 0) is the GPU nominal reference

with torch.no_grad():
    gpu = OrbitPropagatorGPU(
        np.repeat(np.asarray(r0).reshape(1, 3), N+1, axis=0),
        np.repeat(np.asarray(v0).reshape(1, 3), N+1, axis=0),
        tspan, gpu_dt, cenb=ct.Earth, ob=ob,
        pointing_mode="other",
        pointing_config={"mode": "custom_then_nadir", "time": tm_candidates,
                         "yaw": maneuver_yaw, "pitch": maneuver_pitch},
        save_history=False
    )

final_r_gpu = gpu.rs_final.cpu().numpy()
print(f"GPU vs CPU nominal: {np.linalg.norm(final_r_gpu[0] - reference_state_cpu)*1000:.3f} m")

tm = tm_candidates[1:]
sep = np.linalg.norm(final_r_gpu[1:] - np.asarray(reference_state_cpu).reshape(1, 3), axis=1)   # [km]

# ---------------- read t_m for each desired separation ----------------
if not np.all(np.diff(sep) >= 0):
    print("Warning: separation is not monotonic in t_m, so the interpolated t_m may be wrong.")
tm_for_target = np.interp(targets, sep, tm)      # only valid for targets inside [sep.min(), sep.max()]
for d, t in zip(targets, tm_for_target):
    ok = sep.min() <= d <= sep.max()
    print(f"desired {d:5.1f} km -> t_m = {t/3600:8.3f} h = {t/86400:6.3f} days" + ("" if ok else "  (outside sweep range)"))

# ---------------- analytic overlay: (3/2) * da * tm * (2*t_tca - tm) ----------------
sep_formula = 1.5 * delta_a * tm * (2*tspan - tm)

# ---------------- plot ----------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(tm/86400, sep, lw=2, label="Numerical propagation")
ax.plot(tm/86400, sep_formula, "--", lw=1.5, label="Analytic (Hill's equations)")
for d, t in zip(targets, tm_for_target):
    ax.plot([0, t/86400], [d, d], ":", color="gray")
    ax.plot([t/86400, t/86400], [0, d], ":", color="gray")
    ax.plot(t/86400, d, "o", color="k")
    ax.annotate(f"{t/86400:.2f} d", (t/86400, d), textcoords="offset points", xytext=(6, -14))
ax.set_xlabel(r"Maneuver duration $t_m$ [days]")
ax.set_ylabel("Separation at TCA [km]")
ax.set_title(f"3U CubeSat: yaw 0° → {abs(maneuver_yaw)}°, $t_{{TCA}}$ = {tspan/86400:.1f} days, h = {h:.0f} km")
ax.set_xlim(0, tspan/86400); ax.set_ylim(bottom=0)
ax.legend()
fig.tight_layout()
plt.show()

print(f"Code executed in {time.time() - start_time:.2f} seconds.")