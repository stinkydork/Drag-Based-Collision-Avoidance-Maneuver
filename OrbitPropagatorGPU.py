import pathlib
import numpy as np
import torch
import ussa1976
import spiceypy as spice
import os
import sys
import Constants as ct

# For selecting which type of perturbation is needed.
def all_perturbations():
    return {
        'J2': True,
        'J3': True,
        'Drag': True,
        'SRP': True,
        'External Thrust': True
    }

# Kernels for NASA Spice files.
# Do not touch this portion of code
KERNEL_DIR = pathlib.Path("/home/bhavypat10/Engineering_Files/spice_kernels")
LSK  = KERNEL_DIR / "lsk" / "naif0012.tls"
SPK1 = KERNEL_DIR / "spk" / "de441_part-1.bsp"
SPK2 = KERNEL_DIR / "spk" / "de441_part-2.bsp"

def load_kernels():
    try:
        spice.kclear()
    except Exception:
        pass
    for k in (LSK, SPK1, SPK2):
        if not os.path.exists(k):
            raise FileNotFoundError(f"Kernel not found: {k}")
        spice.furnsh(str(k))

class OrbitPropagatorGPU:
    def __init__(self, r0, v0, tspan, dt, cenb, ob, pointing_mode, pointing_config=None, perturbations=all_perturbations(), device='cuda', save_history=True):
        load_kernels()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Convert initial states to batched tensors: (N, 3)
        self.r0 = torch.atleast_2d(torch.tensor(r0, dtype=torch.float64, device=self.device))
        self.v0 = torch.atleast_2d(torch.tensor(v0, dtype=torch.float64, device=self.device))
        self.N_bodies = self.r0.shape[0]
        
        self.tspan = tspan
        self.dt = dt
        self.cenb = cenb
        self.ob = ob
        self.perturbations = perturbations

        # Pointing Mode Validation
        self.pointing_mode = pointing_mode
        if self.pointing_mode == "nadir" and self.ob['shape'] == 'sphere_debri':
            sys.exit("Invalid Combination of pointing mode and shape of orbiting body")
        if self.pointing_mode == "N/A" and self.ob['shape'] == 'cube':
            sys.exit("Invalid Combination of pointing mode and shape of orbiting body")

        if self.pointing_mode == "other":
            if pointing_config is None:
                raise ValueError("Pointing_config must be provided")
            self.other_pointing_mode = pointing_config.get("mode")
            if self.other_pointing_mode == "custom_then_nadir":
                self.pointing_time = torch.as_tensor(pointing_config["time"], dtype=self.r0.dtype, device=self.device).reshape(-1)
                if self.pointing_time.numel() == 1:
                    self.pointing_time = self.pointing_time.expand(self.N_bodies)
                elif self.pointing_time.numel() != self.N_bodies:
                    raise ValueError("Provide one maneuver time or one time per satellite.")
                self.yaw = pointing_config.get("yaw")
                self.pitch = pointing_config.get("pitch")

        # Atmospheric tables - Evaluated once on CPU, then sent to GPU for fast lookup
        altitude = np.linspace(0, 1000000, 1000001)
        altitude_ds = ussa1976.compute(z=altitude, variables=["p", "rho", "t"])
        self.rho_table = torch.tensor(altitude_ds["rho"].values.squeeze(), device=self.device)
        self.pressure_table = torch.tensor(altitude_ds["p"].values.squeeze(), device=self.device)
        self.temp_table = torch.tensor(altitude_ds["t"].values.squeeze(), device=self.device)

        # Pre-allocate batched output tensors
        self.n_steps = int(np.ceil(self.tspan / self.dt)) + 1
        self.save_history = save_history

        if self.save_history:
            self.ys = torch.zeros((self.n_steps, self.N_bodies, 6), device=self.device, dtype=torch.float64)
            self.ts = torch.zeros((self.n_steps, 1), device=self.device, dtype=torch.float64)
            self.os = torch.zeros((self.n_steps, self.N_bodies, 3, 3), device=self.device, dtype=torch.float64)

        # Initial States
        self.y = torch.cat((self.r0, self.v0), dim=1) # (N, 6)
        self.t = 0.0
        
        # Initial orientation
        if self.pointing_mode == "nadir":
            self.C_cur = self.compute_lvlh_frame(self.r0, self.v0)
        elif self.pointing_mode == "N/A":
            self.C_cur = torch.zeros((self.N_bodies, 3, 3), device=self.device)
        elif self.pointing_mode == "other" and self.other_pointing_mode == "custom_then_nadir":
            self.C_cur = self.compute_switched_frame(0.0, self.r0, self.v0)

        if self.save_history:
            self.ts[0] = self.t
            self.ys[0] = self.y
            self.os[0] = self.C_cur.clone()

        # General RK integration
        self.step = 1

        while self.step < self.n_steps:
            t_next = min(self.step * self.dt, self.tspan)
            h = t_next - self.t

            self.y = self.rk4_step(self.t, self.y, h)
            self.t = t_next

            # Update orientation using the accepted final state
            r = self.y[:, :3]
            v = self.y[:, 3:]

            if self.pointing_mode == "nadir":
                self.C_cur = self.compute_lvlh_frame(r, v)

            elif (self.pointing_mode == "other"
                and self.other_pointing_mode == "custom_then_nadir"):
                self.C_cur = self.compute_switched_frame(self.t, r, v)

            if self.save_history:
                self.ts[self.step] = self.t
                self.ys[self.step] = self.y
                self.os[self.step] = self.C_cur.clone()
            #print(self.t)
            self.step += 1

        # Final output formatting
        if self.save_history:
            self.rs = self.ys[:, :, 0:3]
            self.vs = self.ys[:, :, 3:6]
        else:
            self.rs_final = self.y[:, 0:3]
            self.vs_final = self.y[:, 3:6]

    def rk4_step(self, t, y, dt):
        k1 = self.diffy_q(t, y)
        k2 = self.diffy_q(t + 0.5*dt, y + 0.5*dt*k1)
        k3 = self.diffy_q(t + 0.5*dt, y + 0.5*dt*k2)
        k4 = self.diffy_q(t + dt, y + dt*k3)
        return y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def rk3_step(self, t, y, dt):
        k1 = self.diffy_q(t, y)
        k2 = self.diffy_q(t + 0.5*dt, y + 0.5*dt*k1)
        k3 = self.diffy_q(t + dt, y - dt*k1 + 2*dt*k2)
        return y + (dt/6.0) * (k1 + 4*k2 + k3)

    def rk2_step(self, t, y, dt):
        k1 = self.diffy_q(t, y)
        k2 = self.diffy_q(t + 0.5*dt, y + 0.5*dt*k1)
        return y + dt * k2

    def euler_step(self, t, y, dt):
        k1 = self.diffy_q(t, y)
        return y + dt * k1

    def compute_switched_frame(self, t, r, v):
                C_custom = self.compute_custom_frame(r, v, self.yaw, self.pitch)
                C_nadir = self.compute_lvlh_frame(r, v)
                return torch.where((t < self.pointing_time)[:, None, None], C_custom, C_nadir)

    @staticmethod
    def compute_lvlh_frame(r, v):
        # r, v: (N, 3)
        r_hat = r / torch.linalg.norm(r, dim=1, keepdim=True)
        h = torch.cross(r, v, dim=1)
        h_hat = h / torch.linalg.norm(h, dim=1, keepdim=True)
        z = -r_hat
        y = -h_hat
        x = torch.cross(y, z, dim=1)
        x = x / torch.linalg.norm(x, dim=1, keepdim=True)
        return torch.stack((x, y, z), dim=-1) # (N, 3, 3)

    @staticmethod
    def compute_vnb_frame(r, v):
        v_hat = v / torch.linalg.norm(v, dim=1, keepdim=True)
        n = torch.cross(r, v, dim=1)
        n_hat = n / torch.linalg.norm(n, dim=1, keepdim=True)
        b = torch.cross(v_hat, n_hat, dim=1)
        b_hat = b / torch.linalg.norm(b, dim=1, keepdim=True)
        return torch.stack((v_hat, n_hat, b_hat), dim=-1) # (N, 3, 3)

    @staticmethod
    def compute_custom_frame(r, v, yaw, pitch):
        C_vnb = OrbitPropagatorGPU.compute_vnb_frame(r, v)
        
        alpha = torch.deg2rad(torch.tensor(-pitch, device=r.device, dtype=r.dtype))
        beta  = torch.deg2rad(torch.tensor(-yaw, device=r.device, dtype=r.dtype))
        ca, sa = torch.cos(alpha), torch.sin(alpha)
        cb, sb = torch.cos(beta), torch.sin(beta)

        C_body_vnb = torch.tensor([
            [ ca*cb,   sb,    sa*cb],
            [-ca*sb,   cb,   -sa*sb],
            [   -sa,  0.0,       ca]
        ], device=r.device, dtype=r.dtype)
        
        # Broadcast standard rotation matrix across N bodies
        C_body_vnb = C_body_vnb.unsqueeze(0).expand(r.shape[0], 3, 3)
        return torch.bmm(C_vnb, C_body_vnb)

    def diffy_q(self, t, y):
        r = y[:, 0:3]
        v = y[:, 3:6]
        norm_r = torch.linalg.norm(r, dim=1, keepdim=True) # (N, 1)

        # Two-body
        a = -r * self.cenb['mu'] / (norm_r**3)

        # Orientation updates
        if self.pointing_mode == 'nadir':
            self.C_cur = self.compute_lvlh_frame(r, v)
        elif self.pointing_mode == 'other' and self.other_pointing_mode == "custom_then_nadir":
            self.C_cur = self.compute_switched_frame(t, r, v)

        # J2
        if self.perturbations['J2']:
            z2 = r[:, 2:3]**2
            term = 5 * z2 / norm_r**2
            j2x = r[:, 0:1] * (term - 1)
            j2y = r[:, 1:2] * (term - 1)
            j2z = r[:, 2:3] * (term - 3)
            a_j2 = (1.5 * self.cenb['J2'] * self.cenb['mu'] * self.cenb['radius']**2 / norm_r**5) * torch.cat([j2x, j2y, j2z], dim=1)
            a += a_j2

        # J3
        if self.perturbations['J3']:
            z = r[:, 2:3]
            z2 = z**2
            z3 = z**3
            z_norm = z / norm_r
            term1 = 7 * (z3 / norm_r**3) - 3 * z_norm
            j3x = 5 * r[:, 0:1] * term1
            j3y = 5 * r[:, 1:2] * term1
            j3z = 35 * (z**4 / norm_r**3) - 30 * (z2 / norm_r) + 3*norm_r
            a_j3 = (0.5 * self.cenb['J3'] * self.cenb['mu'] * self.cenb['radius']**3 / norm_r**6) * torch.cat([j3x, j3y, j3z], dim=1)
            a += a_j3

        # Drag
        if self.perturbations['Drag']:
            alt = (norm_r - self.cenb['radius']) * 1000.0 # (N, 1) meters
            mask = (alt < 1000000).squeeze(-1) # Only apply to bodies below 1000km

            if mask.any():
                r_in = r[mask]
                v_in = v[mask]
                alt_in = alt[mask]

                at_rot_vec = torch.tensor(self.cenb['at_rot_vec'], device=self.device, dtype=r.dtype)
                vrel = (v_in - torch.cross(at_rot_vec.expand(r_in.shape[0], 3), r_in, dim=1)) * 1000.0
                vrel_norm = torch.linalg.norm(vrel, dim=1, keepdim=True)

                # Fetch from GPU atmosphere table
                z_idx = torch.clamp(torch.round(alt_in).long(), 0, len(self.rho_table)-1).squeeze(-1)
                rho_inf = self.rho_table[z_idx].unsqueeze(-1)
                p_inf = self.pressure_table[z_idx].unsqueeze(-1)
                T_inf = self.temp_table[z_idx].unsqueeze(-1)

                if self.ob['shape'] == 'sphere_debri':
                    C_d = OrbitPropagatorGPU.get_DragCoefficient_Sphere(vrel_norm, rho_inf, p_inf, T_inf, self.ob)
                    a_drag = (-((0.5 * rho_inf * C_d * (np.pi * self.ob['radius']**2)) / self.ob['mass']) * vrel_norm * vrel) / 1000.0
                    a[mask] += a_drag

                elif self.ob['shape'] == 'cube':
                    C_cur_in = self.C_cur[mask] # (M, 3, 3)
                    # Vectorized rotation: body to inertial
                    v_body = torch.bmm(C_cur_in.transpose(1, 2), vrel.unsqueeze(-1)).squeeze(-1) # (M, 3)

                    alpha = torch.atan2(v_body[:, 2], v_body[:, 0])
                    beta = torch.asin(v_body[:, 1] / torch.linalg.norm(v_body, dim=1))

                    F_ax, F_si, F_no = OrbitPropagatorGPU.get_AeroForces_RectangularPrism(
                        torch.rad2deg(alpha).unsqueeze(-1), 
                        torch.rad2deg(beta).unsqueeze(-1), 
                        vrel_norm, rho_inf, p_inf, T_inf, self.ob
                    )

                    F_body_vec = torch.stack([F_ax, F_si, F_no], dim=1).squeeze(-1) # (M, 3)
                    F_inertial = torch.bmm(C_cur_in, F_body_vec.unsqueeze(-1)).squeeze(-1)
                    a[mask] += (F_inertial / self.ob['mass']) / 1000.0

        return torch.cat([v, a], dim=1)

    @staticmethod
    def get_AeroForces_RectangularPrism(alpha, beta, v_mag, rho_inf, p_inf, T_inf, ob):
        T_wall = ob["T_wall"]
        sigma_N = ob["sigma_N"]
        sigma_T = ob["sigma_T"]
        lx, ly, lz = ob["lx"], ob["ly"], ob["lz"]

        alpha = torch.round(alpha)
        beta = torch.round(beta)

        s = v_mag / torch.sqrt(2 * (p_inf / rho_inf))
        Tr = T_wall / T_inf

        ca, sa = torch.cos(torch.deg2rad(alpha)), torch.sin(torch.deg2rad(alpha))
        cb, sb = torch.cos(torch.deg2rad(beta)),  torch.sin(torch.deg2rad(beta))

        # C_A
        u = ca * cb
        term1 = ( ((2 - sigma_N)/(s * np.sqrt(np.pi)) * u) + (torch.sign(u) * (sigma_N/(2*s**2)) * torch.sqrt(Tr)) ) * torch.exp(-s**2 * u**2)
        term2 = (2 - sigma_N) * (u**2 + (1/(2*s**2))) * (torch.sign(u) + torch.erf(s*u))
        term3 = (sigma_N/(2*s) * u * np.sqrt(np.pi) * torch.sqrt(Tr)) * (1 + torch.sign(u)*torch.erf(s*u))
        term4 = (sigma_T*u*(lx/ly)) * ( (1/(s*np.sqrt(np.pi)) * torch.exp(-s**2 * sb**2)) + (sb*(torch.sign(sb)+torch.erf(s*sb))) )
        term5 = (sigma_T*u*(lx/lz)) * ( (1/(s*np.sqrt(np.pi)) * torch.exp(-s**2 * sa**2 * cb**2)) + (sa*cb*(torch.erf(s*sa*cb)+torch.sign(sa*cb))) )
        C_A = term1 + term2 + term3 + term4 + term5

        # C_S
        term6 = (lx/ly) * ( ((2 - sigma_N)/(s * np.sqrt(np.pi)) * sb) + (torch.sign(sb) * (sigma_N/(2*s**2)) * torch.sqrt(Tr)) ) * torch.exp(-s**2 * sb**2)
        term7 = (lx/ly) * (2 - sigma_N) * (sb**2 + (1/(2*s**2))) * (torch.sign(sb) + torch.erf(s*sb))
        term8 = (lx/ly) * (sigma_N/(2*s) * sb * np.sqrt(np.pi) * torch.sqrt(Tr)) * (1 + torch.sign(sb)*torch.erf(s*sb))
        term9 = (sigma_T*sb) * ( (1/(s*np.sqrt(np.pi)) * torch.exp(-s**2 * u**2)) + (u*(torch.erf(s*u)+torch.sign(u))) )
        term10 = (sigma_T*sb*(lx/lz)) * ( (1/(s*np.sqrt(np.pi)) * torch.exp(-s**2 * sa**2 * cb**2)) + (sa*cb*(torch.erf(s*sa*cb)+torch.sign(sa*cb))) )
        C_S = term6 + term7 + term8 + term9 + term10

        # C_N
        y = sa * cb
        term11 = (lx/lz) * ( ((2 - sigma_N)/(s * np.sqrt(np.pi)) * y) + (torch.sign(y) * (sigma_N/(2*s**2)) * torch.sqrt(Tr)) ) * torch.exp(-s**2 * y**2)
        term12 = (lx/lz) * (2 - sigma_N) * (y**2 + (1/(2*s**2))) * (torch.sign(y) + torch.erf(s*y))
        term13 = (lx/lz) * (sigma_N/(2*s) * y * np.sqrt(np.pi) * torch.sqrt(Tr)) * (1 + torch.sign(y)*torch.erf(s*y))
        term14 = (sigma_T*y) * ( (1/(s*np.sqrt(np.pi)) * torch.exp(-s**2 * u**2)) + (u*(torch.erf(s*u)+torch.sign(u))) )
        term15 = (sigma_T*y*(lx/ly)) * ( (1/(s*np.sqrt(np.pi)) * torch.exp(-s**2 * sb**2)) + (sb*(torch.erf(s*sb)+torch.sign(sb))) )
        C_N = term11 + term12 + term13 + term14 + term15

        q = 0.5 * rho_inf * (v_mag)**2
        F_body_axial  = -q * ly * lz * C_A
        F_body_side   = -q * ly * lz * C_S
        F_body_normal = -q * ly * lz * C_N

        return F_body_axial, F_body_side, F_body_normal

    @staticmethod
    def get_DragCoefficient_Sphere(v_mag, rho_inf, p_inf, T_inf, ob):
        T_wall = ob["T_wall"]
        sigma_N = ob["sigma_N"]
        sigma_T = ob["sigma_T"]

        s = v_mag / torch.sqrt(2 * (p_inf / rho_inf))

        term1 = (2 - sigma_N + sigma_T) * (0.5 +
            0.5 * torch.erf(s) * (1 + 1/s**2 - 1/(4*s**4)) +
            (1 + 2*s**2) * torch.exp(-s**2) / (4*s**3*np.sqrt(np.pi)))
        term2 = (sigma_N / (3*s)) * np.sqrt(np.pi) * torch.sqrt(T_wall/T_inf) * (1 + torch.erf(s))
        term3 = (2 - sigma_N)/(2*s**2) 
        term4 = (sigma_N/(6*s**4)) * (1 + (2*s**2 - 1)*torch.exp(-s**2)) * torch.sqrt(T_wall/T_inf)
        
        return term1 + term2 + term3 + term4

 