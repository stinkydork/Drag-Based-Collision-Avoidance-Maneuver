import numpy as np
import torch
import time
from OrbitPropagatorGPU import OrbitPropagatorGPU
from OrbitPropagator import OrbitPropagator
import Constants as ct

a = ct.Earth['radius'] + 700.0
e = 0.04
inc = 45.0
RAAN = 30
omega = 20
nu = 90

r0, v0 = OrbitPropagator.Kep2State(a, e, inc, RAAN, omega, nu, cenb=ct.Earth)

tspan = 7 * (24*60*60)
dt = 1

print(f"Running.....")

pointing_config={"mode": "custom_then_nadir", "time": 0.4*tspan, "yaw": -35, "pitch": -50}
#cpu = OrbitPropagator(r0, v0, tspan, dt, cenb=ct.Earth, ob=ct.CubeSat_3U, pointing_mode='nadir')
#gpu = OrbitPropagatorGPU(r0, v0, tspan, dt, cenb=ct.Earth, ob=ct.CubeSat_3U, pointing_mode='nadir')
cpu = OrbitPropagator(r0, v0, tspan, dt, cenb=ct.Earth, ob=ct.CubeSat_3U, pointing_mode="other", pointing_config=pointing_config)
gpu = OrbitPropagatorGPU(r0, v0, tspan, dt, cenb=ct.Earth, ob=ct.CubeSat_3U, pointing_mode="other", pointing_config=pointing_config)

print("Final position difference [m]:", np.linalg.norm(cpu.rs[-1] - gpu.rs[-1, 0].cpu().numpy()) * 1000)