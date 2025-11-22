"""
THREE-STAGE POROUS BURNER SIMULATION - CUSTOM CONFIGURATION (FIXED)
Stage 1: 1.2 cm, 40% porosity
Stage 2: 0.5 cm, 10% porosity
Stage 3: 1.0 cm, 90% porosity
"""

import numpy as np
import cantera as ct

_FLAME = None
_T_AD = None

def get_T_ad():
    """Adiabatic flame temperature (cached)"""
    global _T_AD
    if _T_AD is None:
        gas = ct.Solution('gri30.yaml')
        gas.TP = 300.0, ct.one_atm
        gas.set_equivalence_ratio(0.65, 'CH4', {'O2': 0.21, 'N2': 0.78, 'AR': 0.01})
        gas.equilibrate('HP')
        _T_AD = gas.T
    return _T_AD


def _build_adiabatic_flame():
    """Build initial adiabatic flame"""
    gas = ct.Solution('gri30.yaml')
    Tin, p, phi, uin = 300.0, ct.one_atm, 0.65, 0.45
    width = 0.027  # Total length: 2.7 cm (1.2 + 0.5 + 1.0)

    gas.TP = Tin, p
    gas.set_equivalence_ratio(phi, 'CH4', {'O2': 0.21, 'N2': 0.78, 'AR': 0.01})

    f = ct.BurnerFlame(gas, width=width)
    f.burner.mdot = gas.density * uin
    f.burner.T = Tin
    f.burner.X = gas.X
    f.transport_model = 'mixture-averaged'
    f.set_refine_criteria(ratio=3, slope=0.15, curve=0.30)
    f.solve(loglevel=0, refine_grid=False)

    return f


def get_flame():
    """Get cached flame or build new one"""
    global _FLAME
    if _FLAME is None:
        _FLAME = _build_adiabatic_flame()
    return _FLAME


class ThreeStageSolver:
    """
    Three-stage porous burner solver - CUSTOM CONFIGURATION

    Configuration:
    - Stage 1: 0 to 1.2 cm (preheating, porosity=40%, FIXED)
    - Stage 2: 1.2 to 1.7 cm (combustion, porosity=10%, DESIGN VARIABLES)
    - Stage 3: 1.7 to 2.7 cm (radiation, porosity=90%, DESIGN VARIABLES)
    """

    def __init__(self, z, design_vars, Tg, u, p, X):
        """
        Args:
            z: Grid points [m]
            design_vars: [dp2_mm, eps2, dp3_mm, eps3]
            Tg: Gas temperature profile [K]
            u: Velocity profile [m/s]
            p: Pressure [Pa]
            X: Species mole fractions (can be 1D or 2D from flame object)
        """
        self.z = z
        self.Tg = Tg
        self.u = u
        self.p = p

        # Handle X dimensionality - flame.X is 2D array (nSpecies x nPoints)
        if X.ndim == 2:
            # X is 2D: use middle point composition as representative
            self.X = X[:, len(z)//2]
        else:
            # X is already 1D
            self.X = X

        # CUSTOM Stage boundaries [m]
        self.L1 = 0.012  # 1.2 cm
        self.L2 = 0.017  # 1.7 cm (1.2 + 0.5)
        self.L3 = 0.027  # 2.7 cm (1.2 + 0.5 + 1.0)

        # Stage 1: FIXED with custom porosity
        self.dp1 = 0.5e-3   # 0.5 mm pore diameter
        self.eps1 = 0.40    # 40% porosity

        # Stage 2 & 3: Design variables with custom porosities
        dp2_mm, eps2, dp3_mm, eps3 = design_vars
        self.dp2 = dp2_mm * 1e-3
        self.eps2 = eps2  # Will be close to 10%
        self.dp3 = dp3_mm * 1e-3
        self.eps3 = eps3  # Will be close to 90%

        # Material properties
        self.k_s = 5.0  # Solid thermal conductivity [W/m/K]
        self.sigma = 5.67e-8  # Stefan-Boltzmann constant
        self.emissivity = 0.85

        # Gas properties
        self.gas = ct.Solution('gri30.yaml')

    def _get_stage_properties(self, z_val):
        """Get properties at axial position z"""
        if z_val <= self.L1:
            return self.dp1, self.eps1, 1
        elif z_val <= self.L2:
            return self.dp2, self.eps2, 2
        else:
            return self.dp3, self.eps3, 3

    def _heat_transfer_coeff(self, dp, eps, u_local, T_avg):
        """Gas-solid heat transfer coefficient [W/m³/K]"""
        # Update gas properties
        self.gas.TPX = T_avg, self.p, self.X

        rho = self.gas.density
        cp = self.gas.cp_mass
        mu = self.gas.viscosity
        k_g = self.gas.thermal_conductivity

        # Reynolds and Prandtl numbers
        Re = max(rho * u_local * dp / mu, 1e-6)
        Pr = mu * cp / k_g

        # Nusselt number (Wakao correlation)
        Nu = 2.0 + 1.1 * (Re**0.6) * (Pr**(1/3))

        # Volumetric heat transfer coefficient
        h = Nu * k_g / dp
        av = 6.0 * (1 - eps) / dp

        return h * av

    def _effective_conductivity(self, eps):
        """Effective solid conductivity [W/m/K]"""
        return self.k_s * (1 - eps)

    def solve(self, max_iter=50, tol=1e-3):
        """Solve solid energy balance with tridiagonal matrix"""
        n = len(self.z)
        Ts = self.Tg.copy()

        for iteration in range(max_iter):
            # Build tridiagonal system
            a = np.zeros(n)
            b = np.zeros(n)
            c = np.zeros(n)
            d = np.zeros(n)

            for i in range(n):
                z_val = self.z[i]
                dp, eps, stage = self._get_stage_properties(z_val)
                k_eff = self._effective_conductivity(eps)

                T_avg = 0.5 * (self.Tg[i] + Ts[i])
                hv = self._heat_transfer_coeff(dp, eps, self.u[i], T_avg)

                if i == 0:
                    # Inlet BC: Ts = Tg (CRITICAL FIX!)
                    b[i] = 1.0
                    d[i] = self.Tg[i]

                elif i == n-1:
                    # Outlet BC: Radiation loss
                    dz = self.z[i] - self.z[i-1]
                    q_rad = self.emissivity * self.sigma * (Ts[i]**4 - 300**4)

                    a[i] = -k_eff / dz
                    b[i] = k_eff / dz + hv + 4 * self.emissivity * self.sigma * Ts[i]**3
                    d[i] = hv * self.Tg[i] + 3 * self.emissivity * self.sigma * Ts[i]**4

                else:
                    # Interior points
                    dz_prev = self.z[i] - self.z[i-1]
                    dz_next = self.z[i+1] - self.z[i]

                    a[i] = -k_eff / dz_prev
                    c[i] = -k_eff / dz_next
                    b[i] = k_eff / dz_prev + k_eff / dz_next + hv
                    d[i] = hv * self.Tg[i]

            # Solve tridiagonal system
            Ts_new = self._solve_tridiagonal(a, b, c, d)

            # Check convergence
            error = np.max(np.abs(Ts_new - Ts))
            Ts = Ts_new

            if error < tol:
                break

        # Calculate efficiency
        T_ad = get_T_ad()
        efficiency = (Ts[-1] / T_ad) ** 4

        return efficiency, Ts

    def _solve_tridiagonal(self, a, b, c, d):
        """Thomas algorithm for tridiagonal system"""
        n = len(d)
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)
        x = np.zeros(n)

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i-1]
            c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom

        x[-1] = d_prime[-1]
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]

        return x

    def get_stage_info(self):
        """Return stage configuration info for visualization"""
        return {
            'L1': self.L1 * 100,  # Convert to cm
            'L2': self.L2 * 100,
            'L3': self.L3 * 100,
            'stage1': {'dp': self.dp1*1e3, 'eps': self.eps1},
            'stage2': {'dp': self.dp2*1e3, 'eps': self.eps2},
            'stage3': {'dp': self.dp3*1e3, 'eps': self.eps3}
        }


def solve_gas_phase(f, T_target, loglevel=0):
    """Update gas phase without expensive refinement"""
    f.set_refine_criteria(ratio=20.0, slope=1.0, curve=1.0)
    f.solve(loglevel=loglevel, refine_grid=False)
    return f


def get_detailed_results(design_vars):
    """
    Run simulation and return detailed results for analysis/plotting

    Args:
        design_vars: [dp2_mm, eps2, dp3_mm, eps3]

    Returns:
        dict with keys: z, Tg, Ts, efficiency, u, stage_info
    """
    f = get_flame()

    # Initial solve
    z = f.grid
    Tg = f.T
    u = f.velocity
    p = f.gas.P
    X = f.X

    # Solve three-stage system
    solver = ThreeStageSolver(z, design_vars, Tg, u, p, X)
    eta, Ts = solver.solve()

    # Update gas phase
    Ttarget = np.clip(0.6*Tg + 0.4*Ts, 300, 2200)
    f = solve_gas_phase(f, T_target=Ttarget, loglevel=0)

    # Final solve
    z_final = f.grid
    Tg_final = f.T
    u_final = f.velocity
    X_final = f.X
    p_final = f.gas.P

    solver_final = ThreeStageSolver(z_final, design_vars, Tg_final, u_final, p_final, X_final)
    efficiency, Ts_final = solver_final.solve()

    return {
        'z': z_final,
        'Tg': Tg_final,
        'Ts': Ts_final,
        'efficiency': efficiency,
        'u': u_final,
        'stage_info': solver_final.get_stage_info(),
        'T_ad': get_T_ad()
    }
