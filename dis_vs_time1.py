# Separation at TCA vs maneuver duration t_m (nominal attitude reference: alpha = 0, beta = 0)
import io
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

# ---------------- settings ----------------
RECOMPUTE      = False            # True = redo the whole propagation sweep
CACHE_FILE     = "sweep_cache.npz"
h              = 500.0            # [km]
tspan          = 4 * 24 * 3600    # time to TCA [s]  (3 days to match the slide)
targets        = [5, 10, 15, 20]  # desired separations [km]
no_numeric_lbl = [20]             # targets whose NUMERICAL label/marker is hidden
N              = 1000             # number of t_m values
cpu_dt = gpu_dt = 10.0
maneuver_pitch = 0                # [deg]
maneuver_yaw   = -45              # [deg]

# ---------------- heavy part: propagation sweep ----------------
def compute_sweep():
    import torch
    import ussa1976
    from OrbitPropagatorGPU import OrbitPropagatorGPU
    from OrbitPropagator import OrbitPropagator
    import Constants as ct

    ob = ct.CubeSat_3U
    r0, v0 = OrbitPropagator.Kep2State(ct.Earth['radius']+h, 0, 0, 0, 0, 0, cenb=ct.Earth)
    with redirect_stdout(io.StringIO()):              # silence per-step prints
        cpu = OrbitPropagator(r0, v0, tspan, cpu_dt, cenb=ct.Earth, ob=ob, pointing_mode='nadir')
    ref = cpu.rs[-1]

    # analytic delta_a (same as your script)
    tbl = ussa1976.compute(z=np.array([h*1000]), variables=['rho', 'p', 't'])
    rho, p, T = tbl['rho'].values.item(), tbl['p'].values.item(), tbl['t'].values.item()
    vrel = v0 - np.cross(ct.Earth['at_rot_vec'], r0)
    F0 = OrbitPropagator.get_AeroForces_RectangularPrism(0, 0, np.linalg.norm(r0), np.linalg.norm(vrel), rho, p, T, ob)
    F1 = OrbitPropagator.get_AeroForces_RectangularPrism(-maneuver_pitch, -maneuver_yaw, np.linalg.norm(r0), np.linalg.norm(vrel), rho, p, T, ob)
    ca, sa = np.cos(np.radians(-maneuver_pitch)), np.sin(np.radians(-maneuver_pitch))
    cb, sb = np.cos(np.radians(-maneuver_yaw)),   np.sin(np.radians(-maneuver_yaw))
    proj = lambda F: (ca*cb)*F[0] + sb*F[1] + (cb*sa)*F[2]
    delta_a = (abs(proj(F0) - proj(F1)) / ob["mass"]) / 1000          # [km/s^2]

    tm_all = np.linspace(0.0, tspan, N + 1)                           # index 0 = GPU nominal
    with torch.no_grad():
        gpu = OrbitPropagatorGPU(
            np.repeat(np.asarray(r0).reshape(1, 3), N+1, axis=0),
            np.repeat(np.asarray(v0).reshape(1, 3), N+1, axis=0),
            tspan, gpu_dt, cenb=ct.Earth, ob=ob, pointing_mode="other",
            pointing_config={"mode": "custom_then_nadir", "time": tm_all,
                             "yaw": maneuver_yaw, "pitch": maneuver_pitch},
            save_history=False)
    final = gpu.rs_final.cpu().numpy()
    print(f"GPU vs CPU nominal: {np.linalg.norm(final[0] - ref)*1000:.3f} m")
    sep = np.linalg.norm(final[1:] - np.asarray(ref).reshape(1, 3), axis=1)
    return tm_all[1:], sep, delta_a

t0 = time.time()
use_cache = False
if os.path.exists(CACHE_FILE) and not RECOMPUTE:
    c = np.load(CACHE_FILE)
    if float(c["tspan"]) == tspan and float(c["h"]) == h and len(c["tm"]) == N \
       and float(c["yaw"]) == maneuver_yaw and float(c["pitch"]) == maneuver_pitch:
        tm, sep, delta_a = c["tm"], c["sep"], float(c["delta_a"])
        use_cache = True
        print("Loaded cached sweep.")
if not use_cache:
    tm, sep, delta_a = compute_sweep()
    np.savez(CACHE_FILE, tm=tm, sep=sep, delta_a=delta_a, tspan=tspan, h=h, N=N,
             yaw=maneuver_yaw, pitch=maneuver_pitch)
    print("Computed sweep and saved cache.")
print(f"delta_a = {delta_a:.3e} km/s^2")

# ---------------- t_m for each target: numerical and analytic ----------------
sep_formula = 1.5 * delta_a * tm * (2*tspan - tm)
tm_num, tm_ana = {}, {}
for d in targets:
    if sep.min() <= d <= sep.max():
        tm_num[d] = np.interp(d, sep, tm)
    q = tspan**2 - d / (1.5*delta_a)                 # inverse of the analytic formula
    if q >= 0:
        tm_ana[d] = tspan - np.sqrt(q)
for d in targets:
    print(f"desired {d:5.1f} km -> numerical t_m = {tm_num.get(d, np.nan)/3600:8.3f} h"
          f" | analytic t_m = {tm_ana.get(d, np.nan)/3600:8.3f} h")

# ---------------- plot ----------------
plt.rcParams.update({"font.size": 26, "axes.labelsize": 30, "axes.titlesize": 30,
                     "xtick.labelsize": 26, "ytick.labelsize": 26, "legend.fontsize": 26})
fig, ax = plt.subplots(figsize=(18, 10))
ln_n, = ax.plot(tm/86400, sep, lw=4, label="Numerical propagation")
ln_a, = ax.plot(tm/86400, sep_formula, "--", lw=4, label="Analytic (Hill's equations)")
cn, ca_ = ln_n.get_color(), ln_a.get_color()

for d in targets:
    if d in tm_num and d not in no_numeric_lbl:                    # numerical: lines, marker, label
        x = tm_num[d]/86400
        ax.plot([0, x], [d, d], ":", color=cn, lw=2.5)
        ax.plot([x, x], [0, d], ":", color=cn, lw=2.5)
        ax.plot(x, d, "o", color="k", ms=14)
        ax.annotate(f"{x:.2f} d", (x, d), textcoords="offset points", xytext=(14, -40),
                    fontsize=26, color=cn, fontweight="bold")
    if d in tm_ana:                                                # analytic: lines, marker, label
        x = tm_ana[d]/86400
        ax.plot([0, x], [d, d], ":", color=ca_, lw=2.5)
        ax.plot([x, x], [0, d], ":", color=ca_, lw=2.5)
        ax.plot(x, d, "s", color=ca_, ms=14, mec="k")
        ax.annotate(f"{x:.2f} d", (x, d), textcoords="offset points", xytext=(-110, 16),
                    fontsize=26, color=ca_, fontweight="bold")

ax.set_xlabel(r"Maneuver duration $t_m$ [days]")
ax.set_ylabel("Separation at TCA [km]")
ax.set_title(f"3U CubeSat: yaw 0° → {abs(maneuver_yaw)}°, $t_{{TCA}}$ = {tspan/86400:.1f} days, h = {h:.0f} km")
ax.set_xlim(0, tspan/86400); ax.set_ylim(bottom=0)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("separation_vs_tm.png", dpi=200)
plt.show()
print(f"Done in {time.time() - t0:.1f} s")