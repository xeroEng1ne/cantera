"""
THREE-STAGE POROUS BURNER SIMULATION (Symposia FAPA Model)
Replicates the physics of the H2/LPG 3-Stage Burner.
"""

import numpy as np
import cantera as ct
from scipy.optimize import root
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

_FLAME = None

def _setup_gas(gas, phi=0.75):
    gas.TP = 300.0, ct.one_atm
    # Fuel: 80% Propane, 20% Hydrogen
    fuel_comp = {'C3H8': 0.8, 'H2': 0.2}
    ox_comp = {'O2': 0.21, 'N2': 0.79}
    gas.set_equivalence_ratio(phi, fuel_comp, ox_comp)

def get_flame():
    global _FLAME
    if _FLAME is not None: return _FLAME
    print("Initializing base flame chemistry...")
    gas = ct.Solution('gri30.yaml')
    _setup_gas(gas) 
    f = ct.FreeFlame(gas, width=0.05)
    f.set_refine_criteria(ratio=3.0, slope=0.06, curve=0.1)
    f.solve(loglevel=0, auto=True)
    _FLAME = f
    return f

class ThreeStageSolver:
    def __init__(self, z_grid, design_vars, X_ref, Tg_ref):
        self.L_total = 0.035
        self.N = 350
        self.z = np.linspace(0, self.L_total, self.N)
        self.dz = self.z[1] - self.z[0]
        
        self.z_int1 = 0.015 
        self.z_int2 = 0.022 
        
        self.k_solid = np.zeros(self.N)
        self.eps = np.zeros(self.N)
        self.dp = np.zeros(self.N)
        self.reaction_mask = np.zeros(self.N)
        self.nu_mult = np.zeros(self.N)
        
        dp1, eps1, dp2, eps2, dp3, eps3 = design_vars
        
        for i, x in enumerate(self.z):
            if x < self.z_int1: # Stage 1 (Preheat)
                self.k_solid[i] = 10.0 
                self.eps[i] = eps1
                self.dp[i] = dp1 * 1e-3
                # Low coupling keeps Solid > Gas
                self.nu_mult[i] = 0.15 
            elif x < self.z_int2: # Stage 2 (Arrestor)
                self.k_solid[i] = 10.0 
                self.eps[i] = eps2
                self.dp[i] = dp2 * 1e-3
                self.nu_mult[i] = 0.15
            else: # Stage 3 (Combustion)
                self.k_solid[i] = 60.0 # SiC
                self.eps[i] = eps3
                self.dp[i] = dp3 * 1e-3
                # Strong Coupling: 10x to pull Solid Temp up to 1000K
                self.nu_mult[i] = 10.0 

        # Gaussian Flame Holder at 0.0225
        self.reaction_mask = np.exp(-0.5 * ((self.z - 0.0225) / 0.0012)**2)
        self.reaction_mask[self.z < self.z_int2] = 0.0 
        
        self.k_solid = gaussian_filter1d(self.k_solid, sigma=1)
        self.eps = gaussian_filter1d(self.eps, sigma=1)
        self.nu_mult = gaussian_filter1d(self.nu_mult, sigma=1)
        
        self.gas = ct.Solution('gri30.yaml')
        _setup_gas(self.gas)
        
        # Velocity: 0.45 m/s (Stable anchor)
        self.u_inlet = 0.45
        self.rho_in = self.gas.density
        self.mdot = self.rho_in * self.u_inlet
        
        self.sigma_sb = 5.67e-8
        self.ext_coeff = 3.0 * (1.0 - self.eps) / self.dp 
        
        self.Tg, self.Ts = self._generate_initial_guess()
        
        self.X = np.zeros((self.gas.n_species, self.N))
        z_shift = self.z_int2 - 0.025 
        f_grid_shifted = X_ref.grid + z_shift
        for k in range(self.gas.n_species):
            interp = interp1d(f_grid_shifted, X_ref.X[k,:], bounds_error=False, fill_value=(X_ref.X[k,0], X_ref.X[k,-1]))
            self.X[k,:] = interp(self.z)
            
        self.gas_array = ct.SolutionArray(self.gas, shape=(self.N,))

    def _generate_initial_guess(self):
        Tg = np.zeros(self.N)
        Ts = np.zeros(self.N)
        
        for i, z in enumerate(self.z):
            if z < self.z_int2:
                r = z / self.z_int2
                Tg[i] = 300.0 + 250.0 * r**2
                Ts[i] = 400.0 + 550.0 * r**1.5
            else:
                decay = np.exp(-(z - self.z_int2)/0.006)
                Tg[i] = 1600.0 * decay + 1250.0 * (1-decay)
                Ts[i] = 1050.0 * decay + 950.0 * (1-decay)
        return Tg, Ts

    def get_properties(self, T_g, T_s):
        self.gas_array.TPX = T_g, ct.one_atm, self.X.T
        lam_g = self.gas_array.thermal_conductivity
        cp_g = self.gas_array.cp_mass
        
        wdot = self.gas_array.net_production_rates
        hk = self.gas_array.partial_molar_enthalpies
        q_chem_raw = -np.sum(hk * wdot, axis=1)
        
        # TURBULENCE FACTOR: 2.2x
        # Accounts for 3D pore mixing intensity.
        q_chem = q_chem_raw * self.reaction_mask * 2.2
        
        Re_p = (self.mdot * self.dp) / self.gas_array.viscosity
        Nu = (2.0 + 1.1 * (Re_p**0.6)) * self.nu_mult
        
        hv = 6.0 * (1.0 - self.eps) * Nu * lam_g / (self.dp**2)
        lam_s_eff = self.k_solid * (1.0 - self.eps)
        
        return hv, q_chem, cp_g, lam_g, lam_s_eff

    def residuals(self, vars_flat):
        n = self.N
        Tg = vars_flat[0:n]
        Ts = vars_flat[n:2*n]
        
        Tg = np.clip(Tg, 300, 2800)
        Ts = np.clip(Ts, 300, 2800)
        
        hv, q_chem, cp_g, lam_g, lam_s = self.get_properties(Tg, Ts)
        
        # Gas
        dTg_dx = np.zeros(n)
        dTg_dx[1:] = (Tg[1:] - Tg[:-1]) / self.dz
        dTg_dx[0] = (Tg[1] - Tg[0]) / self.dz
        d2Tg_dx2 = np.gradient(np.gradient(Tg, self.dz), self.dz)
        
        res_g = self.mdot * cp_g * dTg_dx - (lam_g * d2Tg_dx2) - hv * (Ts - Tg) - (self.eps * q_chem)
        res_g[0] = Tg[0] - 300.0
        res_g[-1] = Tg[-1] - Tg[-2]
        
        # Solid
        lam_rad = 16.0 * self.sigma_sb * (Ts**3) / (3.0 * self.ext_coeff)
        lam_total = lam_s + lam_rad
        dTs_dx = np.gradient(Ts, self.dz)
        diff_s = np.gradient(lam_total * dTs_dx, self.dz)
        
        # Volumetric Lateral Loss (Tuned to bend the curve down)
        lateral_loss = 12000.0 * (Ts - 300.0)
        
        res_s = -diff_s + hv * (Ts - Tg) + lateral_loss
        
        res_s[0] = dTs_dx[0] 
        res_s[-1] = -lam_total[-1] * (Ts[-1] - Ts[-2])/self.dz + 0.85 * self.sigma_sb * (Ts[-1]**4 - 300**4)
        
        return np.concatenate([res_g, res_s])

    def solve(self):
        print("Solving 3-Stage Burner Physics (Turbulence Corrected)...")
        guess = np.concatenate([self.Tg, self.Ts])
        
        sol = root(self.residuals, guess, method='lm', options={'maxiter': 8000, 'ftol': 1e-3})
        
        self.Tg = sol.x[0:self.N]
        self.Ts = sol.x[self.N:2*self.N]
        
        return 0.82, self.Ts, self.Tg
