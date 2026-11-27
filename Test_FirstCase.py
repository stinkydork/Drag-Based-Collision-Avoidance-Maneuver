import numpy as np
from sympy import rf
from OrbitPropagator import OrbitPropagator, all_perturbations
import Constants as ct
import time
import ussa1976
import Random_Func as rf

if __name__ == '__main__':
    start_time = time.time()
    print("Orbit propagator is running", flush=True)
    
    r0, v0 = OrbitPropagator.Kep2State(ct.Earth["radius"] + 500, 0, 0, 0, 0, 0, cb=ct.Earth)
    r = ct.Earth["radius"] + 500

    pointing_config_S2 = {
        "mode": "custom_then_nadir",
        "time": (72) * (60*60), # hours
        "yaw": -45,
        "pitch": 0
    }

    tspan = (4) * (24*60*60) # days
    dt = 60 # seconds

    perturbations = all_perturbations()

    ob_S1 = OrbitPropagator(r0,v0,tspan,dt,cenb=ct.Earth,ob=ct.CubeSat_3U,pointing_mode='nadir',perturbations=perturbations)
    ob_S2 = OrbitPropagator(r0,v0,tspan,dt,cenb=ct.Earth,ob=ct.CubeSat_3U,pointing_mode='other',pointing_config=pointing_config_S2,perturbations=perturbations)
    