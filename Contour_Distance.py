import numpy as np
import matplotlib.pyplot as plt
import ussa1976
import Constants as ct
from OrbitPropagator import OrbitPropagator

if __name__ == '__main__':

    plt.rcParams.update({"font.size": 20})

    tca = 3 * (24*60*60) # days
    tca_days = tca / (24*60*60)
    h = 500.0 # altitude [km]

    # Circular equatorial orbit
    r, v = OrbitPropagator.Kep2State(ct.Earth["radius"] + h, 0, 0, 0, 0, 0, cenb=ct.Earth)
    v_rel = (v - np.cross(ct.Earth["at_rot_vec"], r))

    # Atmosphere at the selected altitude
    atmosphere = ussa1976.compute(z=np.array([h * 1000.0]),variables=["rho", "p", "t"])

    rho = atmosphere["rho"].values.item()
    pressure = atmosphere["p"].values.item()
    temperature = atmosphere["t"].values.item()

    # Acceleration differences
    delta_accelerations = {}

    for name, ob in [("3U", ct.CubeSat_3U), ("12U", ct.CubeSat_12U), ("16U", ct.CubeSat_16U)]:
        F_A0, F_S0, F_N0 = OrbitPropagator.get_AeroForces_RectangularPrism(0, 0,
            np.linalg.norm(r), np.linalg.norm(v_rel),
            rho, pressure, temperature, ob, ct.Earth)
        
        F_AM, F_SM, F_NM = OrbitPropagator.get_AeroForces_RectangularPrism(0, -45,
            np.linalg.norm(r), np.linalg.norm(v_rel),
            rho, pressure, temperature, ob, ct.Earth)

        ca, sa = np.cos(0), np.sin(0)
        cb, sb = np.cos(np.deg2rad(-45)),  np.sin(np.deg2rad(-45))
        F_D0 = (ca*cb)*F_A0 + (sb)*F_S0 + (cb*sa)*F_N0
        F_DM = (ca*cb)*F_AM + (sb)*F_SM + (cb*sa)*F_NM
        delta_accelerations[name] = abs(F_DM - F_D0)/(ob["mass"]*1000)

    # Original acceleration range; expand only if needed for the lines
    delta_a_min = 0.0
    delta_a_max = max(
        1.0e-11,
        1.2 * max(delta_accelerations.values())
    )

    n_tm = 1000
    n_accel = 1000

    tm_ratio = np.linspace(0, 1, n_tm)
    delta_a = np.linspace(delta_a_min, delta_a_max, n_accel)

    X, Y = np.meshgrid(tm_ratio, delta_a)
    tm = X * tca
    delta_a_km = Y
    distance = 1.5*delta_a_km*tm*(2*tca-tm)

    fig, ax = plt.subplots(figsize=(9, 4.2))

    filled = ax.contourf(X,Y,distance,levels=20,cmap="cool",alpha=0.8)

    distance_levels = [
        level for level in [ 5, 10, 20, 50, 70]
        if level < np.max(distance)
    ]
    lines = ax.contour(X,Y,distance,levels=distance_levels,colors="black",linewidths=1.5)
    ax.clabel(lines,inline=True,fontsize=16,fmt="%d")

    for name, color in [("3U", "black"), ("12U", "red"), ("16U", "orange")]:
        ax.axhline(delta_accelerations[name],color=color,linestyle="--",linewidth=3,label=f"{name}: yaw 0° → 45°, pitch 0° → 0°")

    cbar = fig.colorbar(filled, ax=ax)
    cbar.set_label("Separation Distance [km]",fontsize=24)
    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel(r"$t_m/t_{TCA}$",fontsize=24)
    ax.set_ylabel(r"$|\Delta a_{Drag}|$ [km/s$^2$]",fontsize=24)
    ax.set_title(
        rf"$t_{{TCA}} = {tca_days:.1f}\text{{ days}}$, $h = {h:.0f}\text{{ km}}$",
        fontsize=26,
        pad=15
    )
    ax.tick_params(axis="both",labelsize=18)
    ax.set_xlim(0, 1)
    ax.set_ylim(delta_a_min, delta_a_max)

    ax.legend(loc="upper right",fontsize=18,framealpha=0.95)

    plt.tight_layout()
    plt.show()