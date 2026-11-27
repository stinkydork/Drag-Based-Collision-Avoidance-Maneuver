import io
import numpy as np
from contextlib import redirect_stdout
from OrbitPropagator import OrbitPropagator, all_perturbations
import Constants as ct
import time

# ---------------- settings ----------------
TSPAN      = 4 * 24 * 60 * 60     # total time [s]
WINDOW     = 3600                 # exported window at the end [s]
DT         = 10                   # propagation / export step [s]
EPOCH_NOTE = "2000-01-01 12:00:00 UTC"   # t = 0 in your propagator

YAW_DEG    = -45.0                # held until T_MANEUVER, then nadir
PITCH_DEG  = 0.0
T_MANEUVER = 95 * 3600     #[s since epoch] = 55612.56 s

def run(r0, v0, tspan, dt, ob, mode, cfg=None):
    # silences the per-step debug prints inside OrbitPropagator
    with redirect_stdout(io.StringIO()):
        return OrbitPropagator(r0, v0, tspan, dt, cenb=ct.Earth, ob=ob,
                               pointing_mode=mode, pointing_config=cfg,
                               perturbations=perturbations)

def export_csv(name, t, rs_km):
    # columns: t [s since epoch], x, y, z [m], inertial (ICRF)
    np.savetxt(name, np.column_stack((t, rs_km * 1000.0)), delimiter=",",
               header="t_s,x_m,y_m,z_m", comments="", fmt="%.9f")

if __name__ == '__main__':
    start_time = time.time()
    print("Orbit propagator is running", flush=True)

    # Initial conditions of satellite
    r0_S, v0_S = OrbitPropagator.Kep2State(ct.Earth['radius']+500, 0, 0, 0, 0, 0, cenb=ct.Earth)

    # Initial conditions of debri
    r0_D = np.array([-9006.70098374, 3133.37789689, 3473.87915871])
    v0_D = np.array([-2.35811516, -4.91017985, -1.34957215])

    perturbations = all_perturbations()
    sat_cfg = {"mode": "custom_then_nadir", "time": T_MANEUVER,
               "yaw": YAW_DEG, "pitch": PITCH_DEG}

    assert TSPAN % DT == 0 and WINDOW % DT == 0, "DT must divide TSPAN and WINDOW"
    assert 0 < T_MANEUVER < TSPAN, "maneuver time must lie inside the run"

    # Propagate the full 4 days once
    ob_S = run(r0_S, v0_S, TSPAN, DT, ct.CubeSat_3U, 'other', sat_cfg)
    ob_D = run(r0_D, v0_D, TSPAN, DT, ct.Debri,      'N/A')

    # Keep only the last hour
    ts = ob_S.ts.ravel()
    m  = ts >= TSPAN - WINDOW
    export_csv('sat1.csv', ts[m], ob_S.rs[m])
    export_csv('deb1.csv', ts[m], ob_D.rs[m])

    # Quick check
    sep = np.linalg.norm(ob_S.rs[m] - ob_D.rs[m], axis=1)
    k = np.argmin(sep)
    print(f"Epoch (t = 0): {EPOCH_NOTE}")
    print(f"Maneuver: yaw {YAW_DEG} deg, pitch {PITCH_DEG} deg until t = {T_MANEUVER:.2f} s, then nadir")
    print(f"Exported: t = {ts[m][0]:.0f} to {ts[m][-1]:.0f} s, {m.sum()} samples each")
    print(f"Final separation vector [km]: {ob_S.rs[-1] - ob_D.rs[-1]}")
    print(f"Min separation in window: {sep[k]*1000:.1f} m at t = {ts[m][k]:.0f} s (step {DT} s)")
    print(f"Time taken: {time.time() - start_time:.1f} s")