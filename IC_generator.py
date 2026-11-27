import io
import numpy as np
from contextlib import redirect_stdout
import Constants as ct
import OrbitPropagator as OP

# ---------------- settings ----------------
T         = 4*86400          # time of collision [s]
dt        = 60               # output step [s]
heading   = 25.0             # debris heading rotated from sat heading about radial [deg] (~inclination here)
gamma     = 4.0              # debris flight-path angle [deg]
v_deb     = 8.3              # debris speed at collision [km/s]
offset_km = np.zeros(3)      # set nonzero for a near-miss case (e.g. for Pc2D tests)

R    = ct.Earth['radius']
pert = OP.all_perturbations()

def run(r0, v0, ob, mode, tspan=T, step=dt):
    with redirect_stdout(io.StringIO()):
        return OP.OrbitPropagator(r0, v0, tspan, step, ct.Earth, ob,
                                  pointing_mode=mode, perturbations=pert)

# 1) satellite forward to T
r0_S, v0_S = OP.OrbitPropagator.Kep2State(R + 500, 0, 0, 0, 0, 0, cenb=ct.Earth)
S = run(r0_S, v0_S, ct.CubeSat_3U, 'nadir')
rS, vS = S.rs[-1], S.vs[-1]

# 2) debris state at T: same position (+offset), rotated velocity
rhat = rS / np.linalg.norm(rS)
tS = vS - (vS @ rhat) * rhat
tS /= np.linalg.norm(tS)
psi, gam = np.radians(heading), np.radians(gamma)
tD = np.cos(psi) * tS + np.sin(psi) * np.cross(rhat, tS)
vD = v_deb * (np.cos(gam) * tD + np.sin(gam) * rhat)
rD = rS + offset_km

# 3) debris backward to t = 0 (negative tspan and dt)
Db = run(rD, vD, ct.Debri, 'N/A', -T, -dt)
r0_D, v0_D = Db.rs[-1], Db.vs[-1]
alt_min = np.linalg.norm(Db.rs, axis=1).min() - R
print(f"Debris min altitude over window: {alt_min:.1f} km")
assert alt_min > 120, "Debris reenters in the window, so change the geometry"

# 4) verify by propagating forward from the recovered initial state
D = run(r0_D, v0_D, ct.Debri, 'N/A')
dr = D.rs[-1] - S.rs[-1]
dv = D.vs[-1] - S.vs[-1]

kep = OP.OrbitPropagator.State2Kep(r0_D[None, :], v0_D[None, :], ct.Earth)[0]
print("Debris initial Kepler [a, e, i, RAAN, w, nu]:", kep)
print("r0_D =", r0_D)
print("v0_D =", v0_D)
print(f"Miss distance at T:  {np.linalg.norm(dr)*1000:.4f} m")
print(f"Relative speed at T: {np.linalg.norm(dv):.4f} km/s")