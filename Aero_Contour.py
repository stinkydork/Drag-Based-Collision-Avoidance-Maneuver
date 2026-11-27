# Plots the aerodynamic force components of the VNB frame as a function of 
# angle of attack (alpha) and sideslip angle (beta). 
# The contour plots show the change in each force component relative to 
# the reference condition (alpha = 0°,beta = 0°).

import numpy as np
from scipy.special import erf
import ussa1976
import Constants as ct
import matplotlib.pyplot as plt

# Inputs
ob = ct.CubeSat_3U
h = 500 # Altitude in km
v_cir = np.sqrt(ct.Earth["mu"]/(ct.Earth["radius"]+h)) # orbital velocity in km/s
T_wall = ob["T_wall"]
sigma_N = ob["sigma_N"]
sigma_T = ob["sigma_T"]
lx, ly, lz = ob["lx"], ob["ly"], ob["lz"] # [m]
A_ref = ly * lz # [m^2]

# Atmosphere (All are in SI units)
ds = ussa1976.compute(z=np.array([h*1000]), variables=["p", "rho", "t"])
rho = ds["rho"].values[0]
T_inf = ds["t"].values[0]
p = ds["p"].values[0]
s = (v_cir*1000) / np.sqrt(2 * (p/rho))
Tr = T_wall / T_inf

# Meshgrid for orientation angles
alpha_vals = np.linspace((0)*(np.pi/180), (90)*(np.pi/180), 1001)
beta_vals  = np.linspace((0)*(np.pi/180), (90)*(np.pi/180), 1001)
ALPHA, BETA = np.meshgrid(alpha_vals, beta_vals)

# Aerodynamic Coefficient Formulas
#=====================================================================================================================================================
#=====================================================================================================================================================
ca, sa = np.cos(ALPHA), np.sin(ALPHA)
cb, sb = np.cos(BETA),  np.sin(BETA)
# C_A
u = ca * cb
term1 = ( ((2 - sigma_N)/(s * np.sqrt(np.pi)) * u) + (np.sign(u) * (sigma_N/(2*s**2)) * np.sqrt(Tr)) ) * (np.exp(-s**2 * u**2))
term2 = (2 - sigma_N) * (u**2 + (1/(2*s**2))) * (np.sign(u) + erf(s*u))
term3 = (sigma_N/(2*s) * u * np.sqrt(np.pi*Tr)) * (1 + np.sign(u)*erf(s*u))
term4 = (sigma_T*u*(lx/ly)) * ( (1/(s*np.sqrt(np.pi)) * np.exp(-s**2 * sb**2)) + (sb*(np.sign(sb)+erf(s*sb))) )
term5 = (sigma_T*u*(lx/lz)) * ( (1/(s*np.sqrt(np.pi)) * np.exp(-s**2 * sa**2 * cb**2)) + (sa*cb*(erf(s*sa*cb)+np.sign(sa*cb))) )
C_A = term1 + term2 + term3 + term4 + term5
# C_S
term6 = (lx/ly) * ( ((2 - sigma_N)/(s * np.sqrt(np.pi)) * sb) + (np.sign(sb) * (sigma_N/(2*s**2)) * np.sqrt(Tr)) ) * (np.exp(-s**2 * sb**2))
term7 = (lx/ly) * (2 - sigma_N) * (sb**2 + (1/(2*s**2))) * (np.sign(sb) + erf(s*sb))
term8 = (lx/ly) * (sigma_N/(2*s) * sb * np.sqrt(np.pi*Tr)) * (1 + np.sign(sb)*erf(s*sb))
term9 = (sigma_T*sb) * ( (1/(s*np.sqrt(np.pi)) * np.exp(-s**2 * u**2)) + (u*(erf(s*u)+np.sign(u))) )
term10 = (sigma_T*sb*(lx/lz)) * ( (1/(s*np.sqrt(np.pi)) * np.exp(-s**2 * sa**2 * cb**2)) + (sa*cb*(erf(s*sa*cb)+np.sign(sa*cb))) )
C_S = term6 + term7 + term8 + term9 + term10
# C_N
y = sa * cb
term11 = (lx/lz) * ( ((2 - sigma_N)/(s * np.sqrt(np.pi)) * y) + (np.sign(y) * (sigma_N/(2*s**2)) * np.sqrt(Tr)) ) * (np.exp(-s**2 * y**2))
term12 = (lx/lz) * (2 - sigma_N) * (y**2 + (1/(2*s**2))) * (np.sign(y) + erf(s*y))
term13 = (lx/lz) * (sigma_N/(2*s) * y * np.sqrt(np.pi*Tr)) * (1 + np.sign(y)*erf(s*y))
term14 = (sigma_T*y) * ( (1/(s*np.sqrt(np.pi)) * np.exp(-s**2 * u**2)) + (u*(erf(s*u)+np.sign(u))) )
term15 = (sigma_T*y*(lx/ly)) * ( (1/(s*np.sqrt(np.pi)) * np.exp(-s**2 * sb**2)) + (sb*(erf(s*sb)+np.sign(sb))) )
C_N = term11 + term12 + term13 + term14 + term15
#=====================================================================================================================================================
#=====================================================================================================================================================

# Aerodynamic Forces in body frame
q = 0.5*rho*(v_cir*1000)**2
F_body_axial  = -q * A_ref * C_A
F_body_side   = -q * A_ref * C_S
F_body_normal = -q * A_ref * C_N

# Forces in VNB frame
# Source : https://flow.byu.edu/SixDOF.jl/v0.1/theory/
F_v = (ca*cb)*F_body_axial + (sb)*F_body_side + (cb*sa)*F_body_normal
F_n = (-ca*sb)*F_body_axial + (cb)*F_body_side + (-sa*sb)*F_body_normal
F_b = (-sa)*F_body_axial + (0)*F_body_side + (ca)*F_body_normal

# Plotting
plt.rcParams.update({
    "font.size": 25,
    "axes.titlesize": 25,
    "axes.labelsize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,})

fig, ax = plt.subplots(figsize=(10, 6))
Z = F_v - F_v[0, 0]
title = "Difference in Drag Force"
label = r"$\Delta F_{Drag}$ (N)"

cp = ax.contourf(np.degrees(ALPHA), np.degrees(BETA), Z, levels=20, cmap="YlOrRd")
cbar = plt.colorbar(cp, ax=ax)
cbar.set_label(label)
cbar.ax.tick_params()

ax.set_xlabel("Angle of Attack, α (deg)")
ax.set_ylabel("Sideslip Angle, β (deg)")
ax.set_title(title)
ax.tick_params(axis='both')

plt.tight_layout()
plt.show()

Z = F_v - F_v[0, 0]

# Find the minimum (most negative) value and its indices
min_idx = np.unravel_index(np.argmin(Z), Z.shape)
max_neg_val = Z[min_idx]
alpha_min = np.degrees(ALPHA[min_idx])
beta_min = np.degrees(BETA[min_idx])

print(f"Max negative ΔF_Drag: {max_neg_val:.6e} N at α = {alpha_min:.2f}°, β = {beta_min:.2f}°")