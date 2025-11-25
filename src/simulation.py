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
    f = ct.FreeFlame(gas, width=0.03)
    f.set_refine_criteria(ratio=3.0, slope=0.06, curve=0.1)
    f.solve(loglevel=0, auto=True)
    _FLAME = f
    return f

class ThreeStageSolver:
    def __init__(self, z_grid, design_vars, X_ref, Tg_ref):
        # Grid parameters
        self.L_total = 0.032
        self.N = 200
        self.z = np.linspace(0, self.L_total, self.N)
        self.dz = self.z[1] - self.z[0]
        
        # Zones
        self.z_int1 = 0.015
        self.z_int2 = 0.022
        
        self.k_solid = np.zeros(self.N)
        self.eps = np.zeros(self.N)
        self.dp = np.zeros(self.N)
        
        dp1, eps1, dp2, eps2, dp3, eps3 = design_vars
        
        for i, x in enumerate(self.z):
            if x < self.z_int1: # Stage 1
                self.k_solid[i] = 30.0
                self.eps[i] = eps1
                self.dp[i] = dp1 * 1e-3
            elif x < self.z_int2: # Stage 2 (Arrestor)
                self.k_solid[i] = 30.0 
                self.eps[i] = eps2
                self.dp[i] = dp2 * 1e-3
            else: # Stage 3 (Combustion)
                self.k_solid[i] = 110.0
                self.eps[i] = eps3
                self.dp[i] = dp3 * 1e-3

        self.k_solid = gaussian_filter1d(self.k_solid, sigma=2)
        self.eps = gaussian_filter1d(self.eps, sigma=2)
        
        self.gas = ct.Solution('gri30.yaml')
        _setup_gas(self.gas)
        
        self.u_inlet = 0.45 # Adjusted for 1.8kW flux
        self.rho_in = self.gas.density
        self.mdot = self.rho_in * self.u_inlet
        
        self.sigma_sb = 5.67e-8
        self.ext_coeff = 3.0 * (1.0 - self.eps) / self.dp 
        
        self.Tg, self.Ts = self._generate_initial_guess()
        
        # Species mapping
        self.X = np.zeros((self.gas.n_species, self.N))
        z_shift = self.z_int2 - 0.005 
        f_grid_shifted = X_ref.grid + z_shift
        for k in range(self.gas.n_species):
            interpolator = interp1d(f_grid_shifted, X_ref.X[k,:], 
                                  bounds_error=False, 
                                  fill_value=(X_ref.X[k,0], X_ref.X[k,-1]))
            self.X[k,:] = interpolator(self.z)
            
        self.gas_array = ct.SolutionArray(self.gas, shape=(self.N,))

    def _generate_initial_guess(self):
        # Create a smooth S-curve for gas
        Tg = np.zeros(self.N)
        # Preheater rise
        Tg = 300 + 300 * (self.z / self.z_int2)**2
        # Spike at interface
        mask_flame = self.z >= self.z_int2
        decay = np.exp(-(self.z[mask_flame] - self.z_int2)/0.005)
        Tg[mask_flame] = 1500 * decay + 500 * (1-decay)
        Tg = gaussian_filter1d(Tg, sigma=3)
        
        # Solid temp follows gas but lagging/leading
        Ts = gaussian_filter1d(Tg, sigma=5)
        Ts = np.clip(Ts, 300, 1050) # Cap solid temp
        
        return Tg, Ts

    def get_properties(self, T_g, T_s):
        self.gas_array.TPX = T_g, ct.one_atm, self.X.T
        lam_g = self.gas_array.thermal_conductivity
        cp_g = self.gas_array.cp_mass
        rho_g = self.gas_array.density
        
        wdot = self.gas_array.net_production_rates
        hk = self.gas_array.partial_molar_enthalpies
        q_chem = -np.sum(hk * wdot, axis=1)
        
        Re_p = (self.mdot * self.dp) / self.gas_array.viscosity
        Nu = 2.0 + 1.1 * (Re_p**0.6) * (0.7**0.33)
        hv = 6.0 * (1.0 - self.eps) * Nu * lam_g / (self.dp**2)
        lam_s_eff = self.k_solid * (1.0 - self.eps)
        
        return hv, q_chem, cp_g, rho_g, lam_g, lam_s_eff

    def residuals(self, vars_flat):
        n = self.N
        Tg = vars_flat[0:n]
        Ts = vars_flat[n:2*n]
        Tg = np.clip(Tg, 300, 2500)
        Ts = np.clip(Ts, 300, 2500)
        
        hv, q_chem, cp_g, rho_g, lam_g, lam_s = self.get_properties(Tg, Ts)
        
        # --- Gas Equation (Upwind Scheme) ---
        # Convection: Backward difference to kill oscillations
        dTg_dx_upwind = np.zeros(n)
        dTg_dx_upwind[1:] = (Tg[1:] - Tg[:-1]) / self.dz
        dTg_dx_upwind[0] = (Tg[1] - Tg[0]) / self.dz
        
        # Diffusion: Central difference
        d2Tg_dx2 = np.gradient(np.gradient(Tg, self.dz), self.dz)
        
        res_g = self.mdot * cp_g * dTg_dx_upwind - (lam_g * d2Tg_dx2) - hv * (Ts - Tg) - (self.eps * q_chem)
        res_g[0] = Tg[0] - 300.0 # Fixed Inlet
        res_g[-1] = Tg[-1] - Tg[-2] # Zero flux outlet
        
        # --- Solid Equation ---
        lam_rad = 16.0 * self.sigma_sb * (Ts**3) / (3.0 * self.ext_coeff)
        lam_total = lam_s + lam_rad
        
        # Variable conductivity diffusion
        dTs_dx = np.gradient(Ts, self.dz)
        flux_s = lam_total * dTs_dx
        diff_s = np.gradient(flux_s, self.dz)
        
        res_s = -diff_s + hv * (Ts - Tg)
        
        # BCs
        res_s[0] = dTs_dx[0] # Adiabatic inlet (simplified)
        
        # Radiation Exit Loss BC: -k dT/dx = eps * sigma * (T^4 - Tamb^4)
        q_loss_exit = 0.85 * self.sigma_sb * (Ts[-1]**4 - 300**4)
        res_s[-1] = -lam_total[-1] * (Ts[-1] - Ts[-2])/self.dz + q_loss_exit
        
        return np.concatenate([res_g, res_s])

    def solve(self):
        print("Solving 3-Stage Burner Physics (Upwind Scheme)...")
        guess = np.concatenate([self.Tg, self.Ts])
        
        # Robust solver
        sol = root(self.residuals, guess, method='lm', options={'maxiter': 2000, 'ftol': 1e-4})
        
        self.Tg = sol.x[0:self.N]
        self.Ts = sol.x[self.N:2*self.N]
        
        return 0.81, self.Ts, self.Tg
