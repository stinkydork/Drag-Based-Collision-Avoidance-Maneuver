import numpy as np
from OrbitPropagator import OrbitPropagator, all_perturbations, no_perturbations
import Constants as ct
import time

# Use this file for copying the syntax.
if __name__ == '__main__':
    start_time = time.time()
    print("Orbit propagator is running", flush=True)

    # Initial conditions of satellite
    r0_S, v0_S = OrbitPropagator.Kep2State(ct.Earth['radius']+500, 0, 0, 0, 0, 0, cenb=ct.Earth)

    # Initial conditions of debri
    r0_D = np.array([-9006.70098374, 3133.37789689, 3473.87915871])
    v0_D = np.array([-2.35811516, -4.91017985, -1.34957215])

    # Time span and time step
    tspan = (4) * (24*60*60) # days
    dt = 10 # seconds
    perturbations = all_perturbations()

    # Propagate orbit
    ob_S = OrbitPropagator(r0_S,v0_S,tspan,dt,cenb=ct.Earth,ob=ct.CubeSat_3U,pointing_mode='nadir',perturbations=perturbations)
    ob_D = OrbitPropagator(r0_D,v0_D,tspan,dt,cenb=ct.Earth,ob=ct.Debri,pointing_mode='N/A',perturbations=perturbations)
    end_time = time.time()

    print(ob_S.rs[-1]-ob_D.rs[-1])
    print('Time Taken to execute propagator code: '+ str(end_time-start_time) +' seconds')

    # OrbitPropagator.PlotOrbits([ob_S.rs], labels=['Satellite'], cb=ct.Earth, title='3D plot of a propagated LEO trajectory around Earth')
    # OrbitPropagator.PlotOrbits([ob_S.rs, ob_D.rs], labels=['Satellite', 'Debri'], cb=ct.Earth, title='Synthetic Collision Situation')
    # OrbitPropagator.PlotKepOrbits(ob_S.rs,ob_S.vs,ob_S.ts,cb=ct.Earth,title="Evolution of Orbiting Body's Keplerian Elements")
    # OrbitPropagator.PlotEnergy(ob_S.rs,ob_S.vs,ob_S.ts,cb=ct.Earth,title="Specific Energy vs. Time")