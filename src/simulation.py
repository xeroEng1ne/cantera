"""
THREE-STAGE POROUS BURNER SIMULATION
Revised Solver:
1. Uses Uniform Grid for Energy Equations (Fixes Jaggedness).
2. Uses Tanh Initial Guess (Fixes Convergence).
3. Conservative Finite Volume Scheme (Fixes Physics).
4. Fixes UnboundLocalError and increases solver iterations.
"""

import numpy as np
import cantera as ct
from scipy.optimize import root
from scipy.interpolate import interp1d
from tqdm import tqdm

_FLAME = None
_T_AD = None

def get_T_ad():
    global _T_AD  
    if _T_AD is None:
        gas = ct.Solution('gri30.yaml')
        gas.TP = 300.0, ct.one_atm
        gas.set_equivalence_ratio(0.65, 'CH4', {'O2': 0.21, 'N2': 0.78, 'AR': 0.01})
        gas.equilibrate('HP')
        _T_AD = gas.T
    return _T_AD

def get_flame():
    global _FLAME
    if _FLAME is not None: return _FLAME
    print("Initializing flame model...")
    gas = ct.Solution('gri30.yaml')
    # Use wider grid to capture full domain
    f = ct.BurnerFlame(gas, width=0.0605)
    f.burner.mdot = gas.density * 0.45 
    
    # Force a hot initial guess to ensure ignition and grid refinement
    z_init = f.grid
    # Ramped profile to prevent division by zero in Cantera
    T_guess = 300.0 + 1500.0 * (z_init / z_init[-1])**2 
    f.T[:] = T_guess 
    
    f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)
    
    try:
        f.solve(loglevel=0, auto=True)
    except Exception as e:
        print(f"  Warning: Initial flame solve struggled ({e}), proceeding with best effort...")
        
    _FLAME = f
    return f

class ThreeStageSolver:
    def __init__(self, z_cantera, design_vars, Tg_init, u_init, p, X_init):
        # --- 1. RE-GRID TO UNIFORM MESH ---
        self.N = 200 
        self.z = np.linspace(0, 0.0605, self.N)
        self.dz = self.z[1] - self.z[0]
        
        # Interpolate Cantera inputs to this new grid
        self.u = interp1d(z_cantera, u_init, kind='linear', fill_value="extrapolate")(self.z)
        self.X = np.zeros((X_init.shape[0], self.N))
        for k in range(X_init.shape[0]):
             self.X[k, :] = interp1d(z_cantera, X_init[k, :], kind='linear', fill_value="extrapolate")(self.z)
        
        # Initial T Guess
        T_interp = interp1d(z_cantera, Tg_init, kind='linear', fill_value="extrapolate")(self.z)
        self.Tg = T_interp
        self.Ts = T_interp.copy()
        self.p = p
        
        # Design Variables
        dp2_mm, eps2, dp3_mm, eps3 = design_vars
        self.L1 = 0.020 
        self.L2 = 0.035 
        
        self.eps = np.zeros(self.N)
        self.dp = np.zeros(self.N)
        
        dp1 = 0.5e-3; eps1 = 0.40
        dp2 = dp2_mm * 1e-3
        dp3 = dp3_mm * 1e-3
        
        for i, x in enumerate(self.z):
            if x < self.L1:
                self.eps[i], self.dp[i] = eps1, dp1
            elif x < self.L2:
                self.eps[i], self.dp[i] = eps2, dp2
            else:
                self.eps[i], self.dp[i] = eps3, dp3
                
        # Constants
        self.sigma_sb = 5.67e-8
        self.gas = ct.Solution('gri30.yaml')
        self.gas_array = ct.SolutionArray(self.gas, shape=(self.N,))
        
        # Solid Properties
        self.lam_s = 0.188 - 17.5 * self.dp
        self.lam_s = np.maximum(self.lam_s, 0.05)
        self.kappa = 3.0 * (1.0 - self.eps) / self.dp
        self.omega = 0.8 * np.ones_like(self.z)
        
        self.C_Nu = -400 * self.dp + 0.687
        self.m_Nu = 443.7 * self.dp + 0.361
        self.pbar = None

    def get_properties(self, T_g, T_s):
        self.gas_array.TPY = T_g, self.p, self.X.T
        
        mu = self.gas_array.viscosity
        lam_g = self.gas_array.thermal_conductivity
        cp_g = self.gas_array.cp_mass
        rho_g = self.gas_array.density
        
        hk = self.gas_array.partial_molar_enthalpies
        wdot = self.gas_array.net_production_rates
        q_dot = -np.sum(hk * wdot, axis=1)
        
        Re_p = (rho_g * self.u * self.eps * self.dp) / mu
        Re_p = np.maximum(Re_p, 0.1)
        Nu = self.C_Nu * (Re_p ** self.m_Nu)
        
        h_v = Nu * lam_g / (self.dp**2)
        return h_v, q_dot, cp_g, rho_g, lam_g

    def energy_residuals(self, vars_flat):
        if self.pbar is not None: 
            self.pbar.update(1)
        
        n = self.N
        # Clamp inputs for safety
        Tg = np.clip(vars_flat[0:n], 300, 3500)
        Ts = np.clip(vars_flat[n:2*n], 300, 3500)
        qp = vars_flat[2*n:3*n]
        qm = vars_flat[3*n:4*n]
        
        hv, q_rxn, cp_g, rho_g, lam_g = self.get_properties(Tg, Ts)
        
        res_g = np.zeros(n)
        res_s = np.zeros(n)
        
        # Precompute fluxes (Central Difference)
        def get_diff_flux(prop, T):
            flux = np.zeros(n-1)
            prop_face = 0.5 * (prop[1:] + prop[:-1])
            flux = prop_face * (T[1:] - T[:-1]) / self.dz
            return flux

        dTg_dx_up = np.zeros(n)
        dTg_dx_up[1:] = (Tg[1:] - Tg[:-1]) / self.dz
        dTg_dx_up[0] = (Tg[1] - Tg[0]) / self.dz
        conv_term = self.eps * rho_g * cp_g * self.u * dTg_dx_up
        
        flux_g = get_diff_flux(self.eps * lam_g, Tg)
        diff_g = np.zeros(n)
        diff_g[1:-1] = (flux_g[1:] - flux_g[:-1]) / self.dz
        
        flux_s = get_diff_flux(self.lam_s, Ts)
        diff_s = np.zeros(n)
        diff_s[1:-1] = (flux_s[1:] - flux_s[:-1]) / self.dz
        
        rad_source = 4.0 * self.kappa * (1.0 - self.omega) * (self.sigma_sb * Ts**4 - 0.5 * (qp + qm))
        exch = hv * (Tg - Ts)
        chem_source = self.eps * q_rxn
        
        res_g[1:-1] = conv_term[1:-1] - diff_g[1:-1] + exch[1:-1] - chem_source[1:-1]
        res_s[1:-1] = -diff_s[1:-1] - exch[1:-1] + rad_source[1:-1]
        
        # BCs
        res_g[0] = Tg[0] - 300.0
        res_g[-1] = Tg[-1] - Tg[-2]
        res_s[0] = Ts[1] - Ts[0]
        res_s[-1] = Ts[-1] - Ts[-2]
        
        K = self.kappa
        Om = self.omega
        E = 2.0 * K * (1.0 - Om) * self.sigma_sb * Ts**4
        A = K * (2.0 - Om)
        B = K * Om
        
        res_qp = np.zeros(n)
        res_qm = np.zeros(n)
        
        dqp = (qp[1:] - qp[:-1])/self.dz
        res_qp[1:] = dqp - (-A[1:]*qp[1:] + B[1:]*qm[1:] + E[1:])
        
        dqm = (qm[1:] - qm[:-1])/self.dz
        res_qm[:-1] = dqm + (B[:-1]*qp[:-1] - A[:-1]*qm[:-1] + E[:-1])
        
        res_qp[0] = qp[0] - self.sigma_sb * 300**4
        res_qm[-1] = qm[-1] - self.sigma_sb * 300**4
        
        return np.concatenate([res_g, res_s, res_qp, res_qm])

    def _generate_hot_guess(self):
        """Creates a smooth tanh profile with Ts > Tg downstream to bias solution"""
        z_mid = 0.022
        width = 0.005
        # Basic sigmoid
        sigmoid = 0.5 * (1 + np.tanh((self.z - z_mid)/width))
        
        T_gas = 300.0 + 1600.0 * sigmoid
        # Add decay to gas to simulate heat loss to solid
        decay = np.linspace(1.0, 0.8, self.N)
        T_gas = T_gas * decay
        T_gas[0] = 300.0
        
        # Solid: Biased hotter in radiant zone (stage 3)
        T_solid = T_gas.copy()
        mask_rad = self.z > self.L2
        T_solid[mask_rad] += 150.0  # Bias Ts > Tg to help solver find radiant mode
        
        return T_gas, T_solid

    def solve(self):
        print("Starting coupled solution on Uniform Grid...")
        
        # 1. Force Hot Guess
        self.Tg, self.Ts = self._generate_hot_guess()
        
        qp_guess = self.sigma_sb * self.Ts**4
        qm_guess = self.sigma_sb * self.Ts**4
        guess = np.concatenate([self.Tg, self.Ts, qp_guess, qm_guess])
        
        flame = get_flame()
        
        for k in range(3):
            print(f"--- Outer Iteration {k+1}/3 ---")
            
            self.pbar = tqdm(desc="Energy Minimize", unit="it")
            try:
                # Increased maxiter significantly for LM solver
                sol = root(self.energy_residuals, guess, method='lm', 
                           options={'ftol': 1e-4, 'maxiter': 5000})
            finally:
                self.pbar.close()
                self.pbar = None
            
            if sol.success:
                res = sol.x
                self.Tg = res[0:self.N]
                self.Ts = res[self.N:2*self.N]
                guess = res
            else:
                print(f"  Warning: Energy solver stalled: {sol.message}")
                res = sol.x
                self.Tg = res[0:self.N]
                self.Ts = res[self.N:2*self.N]
                guess = res
            
            print("  Updating Chemistry (Cantera)...")
            Tg_cantera = interp1d(self.z, self.Tg, fill_value="extrapolate")(flame.grid)
            flame.flame.set_fixed_temp_profile(flame.grid, Tg_cantera)
            flame.energy_enabled = False
            try:
                flame.solve(loglevel=0)
            except:
                print("  Cantera solve struggled...")
            
            for s in range(flame.gas.n_species):
                self.X[s, :] = interp1d(flame.grid, flame.X[s, :], fill_value="extrapolate")(self.z)
            
            print(f"  Peak Tg: {np.max(self.Tg):.1f} K, Peak Ts: {np.max(self.Ts):.1f} K")

        efficiency = (self.Ts[-1]/get_T_ad())**4
        return efficiency, self.Ts, self.Tg
