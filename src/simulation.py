"""
THREE-STAGE POROUS BURNER SIMULATION (Symposia FAPA Model)

1. FAPA Zone (Stage 1): Enhanced Conductivity for Preheating.
2. Arrestor Zone (Stage 2): Small pores to stop flashback.
3. Radiant Zone (Stage 3): Large pores for emission.
"""

import numpy as np
import cantera as ct
from scipy.optimize import root
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

_FLAME = None
_T_AD = None

def _setup_gas(gas):
    """
    Helper to ensure consistent gas composition across T_ad and Flame calculations.
    Using CH4/Air at phi=0.65 as per user setup.
    """
    gas.TP = 300.0, ct.one_atm
    gas.set_equivalence_ratio(0.65, 'CH4', {'O2': 0.21, 'N2': 0.78, 'AR': 0.01})

def get_T_ad():
    global _T_AD
    if _T_AD is None:
        gas = ct.Solution('gri30.yaml')
        _setup_gas(gas)
        gas.equilibrate('HP')
        _T_AD = gas.T
    return _T_AD

def get_flame():
    global _FLAME
    if _FLAME is not None: return _FLAME
    print("Initializing flame model...")
    
    gas = ct.Solution('gri30.yaml')
    # --- CRITICAL FIX: Set Composition before creating Flame ---
    _setup_gas(gas) 
    # -----------------------------------------------------------
    
    f = ct.BurnerFlame(gas, width=0.0605)
    f.burner.mdot = gas.density * 0.45 
    
    z_init = f.grid
    T_guess = 300.0 + 1500.0 * (z_init / z_init[-1])**2 
    f.T[:] = T_guess 
    
    f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)
    try:
        f.solve(loglevel=0, auto=True)
    except Exception as e:
        print(f"  Warning: Initial flame solve struggled ({e}), proceeding...")
        
    _FLAME = f
    return f

class ThreeStageSolver:
    def __init__(self, z_cantera, design_vars, Tg_init, u_init, p, X_init):
        # Grid: N=200 is optimal for 3-stage resolution
        self.N = 200 
        self.z = np.linspace(0, 0.0605, self.N)
        self.dz = self.z[1] - self.z[0]
        
        # Interpolate Mole Fractions (X)
        self.X = np.zeros((X_init.shape[0], self.N))
        for k in range(X_init.shape[0]):
             self.X[k, :] = interp1d(z_cantera, X_init[k, :], kind='linear', fill_value="extrapolate")(self.z)
        
        gas_ref = ct.Solution('gri30.yaml')
        # Use TPX because X_init is Mole Fractions
        gas_ref.TPX = Tg_init[0], p, X_init[:,0] 
        self.mdot = gas_ref.density * u_init[0]
        
        T_interp = interp1d(z_cantera, Tg_init, kind='linear', fill_value="extrapolate")(self.z)
        self.Tg = T_interp
        self.Ts = T_interp.copy()
        self.p = p
        
        # Design Variables: [dp1, eps1, dp2, eps2, dp3, eps3]
        dp1, eps1, dp2, eps2, dp3, eps3 = design_vars
        
        # Geometry from typical 3-stage burner (Preheat -> Arrestor -> Combustion)
        self.L1 = 0.015 # 1.5cm Preheater
        self.L2 = 0.030 # 1.5cm Arrestor (Total 3.0cm upstream)
        
        self.eps = np.zeros(self.N)
        self.dp = np.zeros(self.N)
        self.cond_mult = np.ones(self.N) 
        
        # Assign Properties to Zones
        for i, x in enumerate(self.z):
            if x < self.L1:
                # Stage 1: Preheater
                self.eps[i], self.dp[i] = eps1, dp1 * 1e-3
                self.cond_mult[i] = 5.0 # ENHANCED Conductivity (FAPA)
            elif x < self.L2:
                # Stage 2: Flame Arrestor
                self.eps[i], self.dp[i] = eps2, dp2 * 1e-3
                self.cond_mult[i] = 1.0
            else:
                # Stage 3: Combustion/Radiant
                self.eps[i], self.dp[i] = eps3, dp3 * 1e-3
                self.cond_mult[i] = 1.0
        
        self.eps = gaussian_filter1d(self.eps, sigma=1)
        self.dp = gaussian_filter1d(self.dp, sigma=1)
        self.cond_mult = gaussian_filter1d(self.cond_mult, sigma=1)
                
        self.sigma_sb = 5.67e-8
        self.gas = ct.Solution('gri30.yaml')
        self.gas_array = ct.SolutionArray(self.gas, shape=(self.N,))
        
        self.lam_s_base = 0.188 - 17.5 * self.dp
        self.lam_s_base = np.maximum(self.lam_s_base, 0.05)
        self.lam_s = self.lam_s_base * self.cond_mult
        
        self.kappa = 3.0 * (1.0 - self.eps) / self.dp
        self.omega = 0.4 * np.ones_like(self.z) 
        self.pbar = None

    def get_properties(self, T_g, T_s):
        # FIXED: Use TPX (Mole Fractions) instead of TPY (Mass Fractions)
        self.gas_array.TPX = T_g, self.p, self.X.T
        
        mu = self.gas_array.viscosity
        lam_g = self.gas_array.thermal_conductivity
        cp_g = self.gas_array.cp_mass
        rho_g = self.gas_array.density
        
        self.u = self.mdot / rho_g
        
        hk = self.gas_array.partial_molar_enthalpies
        wdot = self.gas_array.net_production_rates
        q_dot = -np.sum(hk * wdot, axis=1)
        
        # Wakao-Kaguei correlation
        Pr = cp_g * mu / lam_g
        Re_p = (rho_g * self.u * self.dp) / mu
        Re_p = np.maximum(Re_p, 0.1)
        Nu = 2.0 + 1.1 * (Pr**(1/3)) * (Re_p**0.6)
        
        # FIXED: Removed 0.05 factor to match theory
        h_v = 6.0 * (1.0 - self.eps) * Nu * lam_g / (self.dp**2)
        
        return h_v, q_dot, cp_g, rho_g, lam_g

    def energy_residuals(self, vars_flat):
        if self.pbar is not None: self.pbar.update(1)
        n = self.N
        Tg = np.clip(vars_flat[0:n], 300, 3500)
        Ts = np.clip(vars_flat[n:2*n], 300, 3500)
        qp = vars_flat[2*n:3*n]
        qm = vars_flat[3*n:4*n]
        
        hv, q_rxn, cp_g, rho_g, lam_g = self.get_properties(Tg, Ts)
        
        def get_diff_flux(prop, T):
            flux = np.zeros(n-1)
            prop_face = 0.5 * (prop[1:] + prop[:-1])
            flux = prop_face * (T[1:] - T[:-1]) / self.dz
            return flux

        dTg_dx = np.zeros(n)
        dTg_dx[1:] = (Tg[1:] - Tg[:-1]) / self.dz
        dTg_dx[0] = (Tg[1] - Tg[0]) / self.dz
        conv_term = self.eps * rho_g * cp_g * self.u * dTg_dx
        
        flux_g = get_diff_flux(self.eps * lam_g, Tg)
        diff_g = np.zeros(n)
        diff_g[1:-1] = (flux_g[1:] - flux_g[:-1]) / self.dz
        
        flux_s = get_diff_flux(self.lam_s, Ts)
        diff_s = np.zeros(n)
        diff_s[1:-1] = (flux_s[1:] - flux_s[:-1]) / self.dz
        
        rad_source = 4.0 * self.kappa * (1.0 - self.omega) * (self.sigma_sb * Ts**4 - 0.5 * (qp + qm))
        exch = hv * (Tg - Ts)
        chem_source = self.eps * q_rxn
        
        res_g = np.zeros(n); res_s = np.zeros(n)
        
        # Gas: Conv - Diff + Exchange - Chemical = 0
        res_g[1:-1] = conv_term[1:-1] - diff_g[1:-1] + exch[1:-1] - chem_source[1:-1]
        
        # Solid: -Diff_s + Exchange (Gain from Gas) - Rad (Loss) = 0
        # FIXED: Signs are now physically correct for FAPA (Ts > Tg in preheat)
        res_s[1:-1] = -diff_s[1:-1] + exch[1:-1] - rad_source[1:-1]
        
        res_g[0] = Tg[0] - 300.0
        res_g[-1] = Tg[-1] - Tg[-2]
        res_s[0] = Ts[1] - Ts[0] 
        res_s[-1] = Ts[-1] - Ts[-2]
        
        K = self.kappa; Om = self.omega
        E = 2.0 * K * (1.0 - Om) * self.sigma_sb * Ts**4
        A = K * (2.0 - Om); B = K * Om
        
        dqp = (qp[1:] - qp[:-1])/self.dz
        res_qp = np.zeros(n); res_qp[1:] = dqp - (-A[1:]*qp[1:] + B[1:]*qm[1:] + E[1:])
        dqm = (qm[1:] - qm[:-1])/self.dz
        res_qm = np.zeros(n); res_qm[:-1] = dqm + (B[:-1]*qp[:-1] - A[:-1]*qm[:-1] + E[:-1])
        res_qp[0] = qp[0] - self.sigma_sb * 300**4
        res_qm[-1] = qm[-1] - self.sigma_sb * 300**4
        
        return np.concatenate([res_g, res_s, res_qp, res_qm])

    def _generate_hot_guess(self):
        z_mid = 0.025 
        width = 0.005
        sigmoid = 0.5 * (1 + np.tanh((self.z - z_mid)/width))
        
        T_gas = 300.0 + 1600.0 * sigmoid
        decay = np.linspace(1.0, 0.7, self.N) 
        T_gas = T_gas * decay
        T_gas[0] = 300.0
        
        T_solid = T_gas.copy()
        mask = self.z > self.L2
        T_solid[mask] += 150.0 
        
        return T_gas, T_solid

    def solve(self):
        print("Starting 3-Stage FAPA Simulation...")
        self.Tg, self.Ts = self._generate_hot_guess()
        qp_guess = self.sigma_sb * self.Ts**4
        qm_guess = self.sigma_sb * self.Ts**4
        guess = np.concatenate([self.Tg, self.Ts, qp_guess, qm_guess])
        
        flame = get_flame()
        alpha = 0.5 # Updated damping for stability
        
        for k in range(3):
            print(f"--- Iteration {k+1}/3 ---")
            self.pbar = tqdm(desc="Energy Solve", unit="it")
            try:
                sol = root(self.energy_residuals, guess, method='lm', 
                           options={'ftol': 1e-4, 'maxiter': 10000})
            finally:
                self.pbar.close(); self.pbar = None
            
            res = sol.x
            self.Tg = alpha * res[0:self.N] + (1-alpha)*self.Tg
            self.Ts = alpha * res[self.N:2*self.N] + (1-alpha)*self.Ts
            qp = res[2*self.N:3*self.N]
            qm = res[3*self.N:4*self.N]
            guess = res
            
            Tg_cantera = interp1d(self.z, self.Tg, fill_value="extrapolate")(flame.grid)
            flame.flame.set_fixed_temp_profile(flame.grid, Tg_cantera)
            flame.energy_enabled = False
            try: flame.solve(loglevel=0)
            except: pass
            
            for s in range(flame.gas.n_species):
                self.X[s, :] = interp1d(flame.grid, flame.X[s, :], fill_value="extrapolate")(self.z)
            print(f"  Peak Tg: {np.max(self.Tg):.1f} K")

        # --- Corrected Efficiency Calculation ---
        q_rad_net_exit = qp[-1] - qm[-1]
        
        # Calculate Fuel Power using correct Mass Fraction (Y)
        self.gas.TPX = 300.0, self.p, self.X[:, 0]
        Y_fuel_in = self.gas.Y[self.gas.species_index('CH4')]
        
        # Note: LHV ~50MJ/kg for CH4. Update if using H2/LPG.
        LHV = 50.0e6 
        fuel_power = self.mdot * Y_fuel_in * LHV
        
        # Safety against divide-by-zero if fuel is missing
        fuel_power = max(fuel_power, 1e-9)
        
        efficiency = q_rad_net_exit / fuel_power
        if efficiency < 0: efficiency = 0.0
        
        return efficiency, self.Ts, self.Tg
