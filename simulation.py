# simulation_three_phase_fixed.py
"""
Simplified three-phase porous burner with better convergence
"""

import numpy as np
import cantera as ct
from scipy.integrate import solve_bvp

# ========================================
# Global cached objects
# ========================================
_FLAME = None
_T_AD = None

def get_T_ad():
    """Cached adiabatic flame temperature"""
    global _T_AD
    if _T_AD is None:
        gas = ct.Solution('gri30.yaml')
        gas.TP = 300.0, ct.one_atm
        gas.set_equivalence_ratio(0.65, 'CH4', {'O2': 0.21, 'N2': 0.78, 'AR': 0.01})
        gas.equilibrate('HP')
        _T_AD = gas.T
    return _T_AD

def _build_adiabatic_flame():
    """Initial adiabatic flame for starting point"""
    gas = ct.Solution('gri30.yaml')
    Tin, p, phi, uin, width = 300.0, ct.one_atm, 0.65, 0.45, 0.0605
    gas.TP = Tin, p
    gas.set_equivalence_ratio(phi, 'CH4', {'O2': 0.21, 'N2': 0.78, 'AR': 0.01})
    
    f = ct.BurnerFlame(gas, width=width)
    f.burner.mdot = gas.density * uin
    f.burner.T = Tin
    f.burner.X = gas.X
    f.transport_model = 'mixture-averaged'
    f.set_refine_criteria(ratio=3, slope=0.15, curve=0.30)
    f.solve(loglevel=0, auto=True)
    
    return f

def get_flame():
    """Get cached flame object"""
    global _FLAME
    if _FLAME is None:
        _FLAME = _build_adiabatic_flame()
    return _FLAME


# ========================================
# Three-phase porous media profiles
# ========================================

def _porosity_dp_profile_threephase(z, eps2, dp2_m, eps3, dp3_m):
    """Three-layer porosity and pore diameter profile"""
    eps1, dp1 = 0.835, 0.00029
    z0, z1, z2 = 0.033, 0.037, 0.050
    
    eps = np.empty_like(z)
    dp = np.empty_like(z)
    
    for i, x in enumerate(z):
        if x < z0:
            eps[i], dp[i] = eps1, dp1
        elif x < z1:
            f = (x - z0) / (z1 - z0)
            eps[i] = eps1 + f * (eps2 - eps1)
            dp[i] = dp1 + f * (dp2_m - dp1)
        elif x < z2:
            eps[i], dp[i] = eps2, dp2_m
        else:
            f = min((x - z2) / (0.060 - z2), 1.0)
            eps[i] = eps2 + f * (eps3 - eps2)
            dp[i] = dp2_m + f * (dp3_m - dp2_m)
    
    return eps, np.maximum(dp, 1e-6)


# ========================================
# Simplified sequential three-phase solver
# ========================================

class SimplifiedThreePhaseSolver:
    """
    Sequential solver: Gas -> Solid -> Liquid -> iterate
    Much more stable than fully coupled approach
    """
    
    def __init__(self, z, design_vars, gas_solution_dict):
        self.z = z
        self.N = len(z)
        
        # Extract design variables
        dp2_mm, eps2, dp3_mm, eps3 = design_vars
        self.dp2 = max(dp2_mm / 1000.0, 1e-6)
        self.eps2 = float(eps2)
        self.dp3 = max(dp3_mm / 1000.0, 1e-6)
        self.eps3 = float(eps3)
        
        # Get profiles
        self.eps, self.dp = _porosity_dp_profile_threephase(
            z, self.eps2, self.dp2, self.eps3, self.dp3
        )
        
        # Material properties
        self.lam_s = np.maximum(0.188 - 17.5 * self.dp, 0.01)
        self.kappa = np.maximum((3.0 / self.dp) * (1.0 - self.eps), 1e-8)
        
        # Gas properties
        self.Tg_input = gas_solution_dict['T']
        self.u = gas_solution_dict['u']
        self.X = gas_solution_dict['X']
        self.p = gas_solution_dict['p']
        
        # Constants
        self.sigma = ct.stefan_boltzmann
        self.omega = 0.7
        self.T_surround = 300.0
        
        # Liquid properties
        self.k_l = 0.6
        self.h_fg = 2.26e6
        
        self.prop = ct.Solution('gri30.yaml')
    
    def _compute_h_v(self, Tg):
        """Volumetric heat transfer coefficient"""
        h_v = np.zeros(self.N)
        
        for i in range(self.N):
            self.prop.TPX = Tg[i], self.p, self.X[:, i]
            rho = self.prop.density
            mu = max(self.prop.viscosity, 1e-8)
            k_g = self.prop.thermal_conductivity
            
            C = -400.0 * self.dp[i] + 0.687
            m = 443.7 * self.dp[i] + 0.361
            Re_p = rho * self.eps[i] * self.u[i] * self.dp[i] / mu
            Nu_v = C * (max(Re_p, 0.1) ** m)
            h_v[i] = Nu_v * k_g / (self.dp[i]**2)
        
        return np.clip(h_v, 1e3, 1e7)
    
    def _solve_solid_tridiagonal(self, Tg, Ts_old):
        """Solve solid phase using tridiagonal system"""
        Ts = Ts_old.copy()
        h_v = self._compute_h_v(Tg)
        
        # Tridiagonal coefficients
        a = np.zeros(self.N)
        b = np.zeros(self.N)
        c = np.zeros(self.N)
        d = np.zeros(self.N)
        
        # Boundary conditions
        b[0] = 1.0
        d[0] = Tg[0]
        
        for i in range(1, self.N - 1):
            dz = self.z[i] - self.z[i-1]
            dz_next = self.z[i+1] - self.z[i]
            dz_avg = 0.5 * (dz + dz_next)
            
            # Coefficients for d²T/dz²
            a[i] = self.lam_s[i] / dz**2
            c[i] = self.lam_s[i] / dz_next**2
            
            # Radiation term (linearized)
            q_rad = 8.0 * self.kappa[i] * (1.0 - self.omega) * self.sigma * Ts[i]**3
            
            b[i] = -(a[i] + c[i] + h_v[i] + q_rad)
            d[i] = -h_v[i] * Tg[i] - 2.0 * self.kappa[i] * (1.0 - self.omega) * self.sigma * self.T_surround**4
        
        # Outlet BC (zero gradient)
        a[-1] = -1.0
        b[-1] = 1.0
        d[-1] = 0.0
        
        # Solve tridiagonal system
        Ts_new = self._thomas_algorithm(a, b, c, d)
        
        return np.clip(Ts_new, 300, 2400)
    
    def _thomas_algorithm(self, a, b, c, d):
        """Thomas algorithm for tridiagonal system"""
        n = len(d)
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)
        x = np.zeros(n)
        
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        
        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i-1]
            if abs(denom) < 1e-10:
                denom = 1e-10
            c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom
        
        x[-1] = d_prime[-1]
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
        return x
    
    def _solve_liquid_simple(self, Tg, Ts):
        """Simple liquid phase solution"""
        Tl = np.zeros(self.N)
        Tl[0] = 300.0  # Inlet
        
        for i in range(1, self.N):
            # Simple forward integration
            dz = self.z[i] - self.z[i-1]
            
            # Heat exchange with gas
            h_gl = 0.05 * self._compute_h_v(Tg)
            Q_gl = h_gl[i] * (Tg[i] - Tl[i-1])
            
            # Conduction
            if i > 1:
                Q_cond = self.k_l * (Tl[i-1] - Tl[i-2]) / dz
            else:
                Q_cond = 0.0
            
            # Temperature update (explicit)
            Tl[i] = Tl[i-1] + dz * (Q_gl + Q_cond) / (self.k_l + 1e-6)
        
        return np.clip(Tl, 280, 400)
    
    def solve(self, max_iter=30, tol=5.0):
        """Sequential iteration"""
        Tg = self.Tg_input.copy()
        Ts = self.Tg_input.copy()
        Tl = np.ones(self.N) * 300.0
        
        for iteration in range(max_iter):
            Ts_old = Ts.copy()
            
            # Solve solid with current gas temperature
            Ts = self._solve_solid_tridiagonal(Tg, Ts_old)
            
            # Solve liquid with current gas/solid
            Tl = self._solve_liquid_simple(Tg, Ts)
            
            # Update gas temperature (weighted average toward solid)
            Tg = 0.7 * Tg + 0.3 * Ts
            
            # Check convergence
            err = np.linalg.norm(Ts - Ts_old)
            if err < tol:
                print(f"    3-phase converged in {iteration+1} iters (err={err:.2f}K)")
                break
        
        return Tg, Ts, Tl


# ========================================
# Gas phase solver
# ========================================

def solve_gas_phase_threephase(flame_obj, T_target=None, loglevel=0, refine=False):
    """Update BurnerFlame with fixed temperature profile"""
    f = flame_obj
    
    if T_target is not None:
        zloc = f.grid / f.grid.max()
        f.flame.set_fixed_temp_profile(zloc, np.asarray(T_target))
        f.energy_enabled = False
    else:
        f.energy_enabled = True
    
    f.transport_model = 'mixture-averaged'
    
    if refine:
        f.set_refine_criteria(ratio=3.0, slope=0.15, curve=0.30)
    else:
        f.set_refine_criteria(ratio=10.0, slope=0.8, curve=0.8, prune=0.0)
    
    try:
        f.solve(loglevel=loglevel, auto=True)
    except:
        f.solve(loglevel=0, auto=False)
    
    return f
