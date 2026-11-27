import numpy as np
import torch
import time
from OrbitPropagatorGPU import OrbitPropagatorGPU
from OrbitPropagator import OrbitPropagator
import Constants as ct

N_orbits = 10000
r0_batch = np.zeros((N_orbits, 3))
v0_batch = np.zeros((N_orbits, 3))

a = ct.Earth['radius'] + 500.0
e = 0.0
inc = 45.0                   
RAAN = 0.0                    
omega = 0.0                   

nu_list = np.linspace(0.0, 360.0, N_orbits, endpoint=False)

for i, nu in enumerate(nu_list):
    r0_batch[i], v0_batch[i] = OrbitPropagator.Kep2State(a, e, inc, RAAN, omega, nu, cenb=ct.Earth)

tspan = (24*60*60)
dt = 10.0

print(f"Running.....")
start_time = time.time()
propagator = OrbitPropagatorGPU(r0_batch, v0_batch, tspan, dt, cenb=ct.Earth, ob=ct.CubeSat_3U, pointing_mode='nadir')
end_time = time.time()

print(f"Code executed in {end_time - start_time:.2f} seconds.")