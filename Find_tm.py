# Finds the exact time of maneuver required for a satellite to hold a specific attitude
# to achieve a desired amount of separation distance.
# Nominal (Reference) attitude is alpha = 0°,beta = 0°.

import numpy as np
import torch
import time
import ussa1976
from OrbitPropagatorGPU import OrbitPropagatorGPU
from OrbitPropagator import OrbitPropagator
import Constants as ct

start_time = time.time()

h = 500.0 # [km]
tspan = 4 * (24*60*60) # days
target_dist = 15 # [km]
N = 1000
cpu_dt = 10.0
gpu_dt = 10.0

ob = ct.CubeSat_3U
maneuver_pitch = 0 # [deg]
maneuver_yaw = -45 # [deg]

# CPU Propagation
r0, v0 = OrbitPropagator.Kep2State(ct.Earth['radius']+h, 0, 0, 0, 0, 0, cenb=ct.Earth)
cpu = OrbitPropagator(r0, v0, tspan, cpu_dt, cenb=ct.Earth, ob=ob, pointing_mode='nadir')
reference_state_cpu = cpu.rs[-1]

# Prediction
table = ussa1976.compute(z=np.array([h*1000]), variables=['rho', 'p', 't'])
rho, p, T = table['rho'].values.item(), table['p'].values.item(), table['t'].values.item()
vrel = v0 - np.cross(ct.Earth['at_rot_vec'], r0)

F_body_axial0, F_body_side0, F_body_normal0 = OrbitPropagator.get_AeroForces_RectangularPrism(0, 0, np.linalg.norm(r0), np.linalg.norm(vrel), rho, p, T, ob) # [N]
F_body_axial, F_body_side, F_body_normal = OrbitPropagator.get_AeroForces_RectangularPrism(-maneuver_pitch, -maneuver_yaw, np.linalg.norm(r0), np.linalg.norm(vrel), rho, p, T, ob) # [N]

ca, sa = np.cos(np.radians(-maneuver_pitch)), np.sin(np.radians(-maneuver_pitch))
cb, sb = np.cos(np.radians(-maneuver_yaw)), np.sin(np.radians(-maneuver_yaw))
F_v0 = (ca*cb)*F_body_axial0 + (sb)*F_body_side0 + (cb*sa)*F_body_normal0
F_v = (ca*cb)*F_body_axial + (sb)*F_body_side + (cb*sa)*F_body_normal
delta_a = (abs(F_v0-F_v)/ob["mass"])/1000 # [km/s^2]

inside_sqrt = tspan**2 - target_dist/(1.5*delta_a)
if inside_sqrt < 0:
    print("Error: No analytical tm solution found.")
    raise SystemExit
tm_formula = (target_dist/(1.5*delta_a)) / (tspan + np.sqrt(inside_sqrt))

# Using GPU to get exact required maneuver time
# Candidate maneuver durations
lowest_node = max(0.0, 0.8*tm_formula)
highest_node = min(tspan, 2*tm_formula)

tm_candidates = np.linspace(lowest_node, highest_node, N)
# Adding a reference GPU state and this will be used to compare CPU reference result.
tm_candidates = np.insert(tm_candidates, 0, 0.0)

# All candidates propagate together
with torch.no_grad():
    gpu = OrbitPropagatorGPU(np.repeat(np.asarray(r0).reshape(1, 3), N+1, axis=0), np.repeat(np.asarray(v0).reshape(1, 3), N+1, axis=0),
        tspan, gpu_dt, cenb=ct.Earth, ob=ob,
        pointing_mode="other",
        pointing_config={
            "mode": "custom_then_nadir",
            "time": tm_candidates,
            "yaw": maneuver_yaw,
            "pitch": maneuver_pitch
        },
        save_history = False
    )

# Transfer all final positions together
final_r_gpu = gpu.rs_final.cpu().numpy()
# Reference GPU state
reference_state_gpu = final_r_gpu[0]
tm_candidates = tm_candidates[1:]
final_r_gpu = final_r_gpu[1:]

separations = np.linalg.norm(final_r_gpu - np.asarray(reference_state_cpu).reshape(1, 3), axis=1)
errors = np.abs(separations - target_dist)
best_index = int(np.argmin(errors))
absolute_difference_in_distance = abs(np.linalg.norm(reference_state_gpu)-np.linalg.norm(reference_state_cpu))

print(f"Absolute Difference in Distance (GPU vs. CPU) = {absolute_difference_in_distance*1000:.6f} m")
print(f"tm_formula = {tm_formula / 3600:.6f} hours")
print(f"tm (Numerically Propagated) = {tm_candidates[best_index] / 3600:.6f} hours")
print(f"Separation at TCA (Numerically Propagated) = {separations[best_index]:.6f} km")
print(f"Error in Separation distance (Numerically Propagated vs. Desired) = {errors[best_index] * 1000:.6f} m")
print(f"Code executed in {time.time() - start_time:.2f} seconds.")
