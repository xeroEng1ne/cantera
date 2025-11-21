"""
THREE-STAGE POROUS BURNER SIMULATION - FIXED
Corrected boundary conditions for proper heat transfer
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
    Tin, p, phi, uin, width = 300.0, ct.one_atm, 0.65, 0.45, 0.0605

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
    """Get cached flame object"""
    global _FLAME
    if _FLAME is None:
        _FLAME = _build_adiabatic_flame()
    return _FLAME


def _porosity_dp_profile(z, eps2, dp2, eps3, dp3):
    """Three-stage profile"""
    eps1, dp1 = 0.835, 0.00029
    z1 = 0.033
    z2 = 0.037
    transition_width = 0.004

    eps = np.empty_like(z)
    dp = np.empty_like(z)

    for i, x in enumerate(z):
        if x <= z1:
            eps[i], dp[i] = eps1, dp1
        elif x <= z1 + transition_width:
            frac = (x - z1) / transition_width
            eps[i] = eps1 + frac * (eps2 - eps1)
            dp[i] = dp1 + frac * (dp2 - dp1)
        elif x <= z2:
            eps[i], dp[i] = eps2, dp2
        elif x <= z2 + transition_width:
            frac = (x - z2) / transition_width
            eps[i] = eps2 + frac * (eps3 - eps2)
            dp[i] = dp2 + frac * (dp3 - dp2)
        else:
            eps[i], dp[i] = eps3, dp3

    return eps, np.maximum(dp, 1e-6)


class ThreeStageSolver:
    """Three-stage porous burner solver - FIXED BOUNDARY CONDITIONS"""

    def __init__(self, z, design_vars, Tg, u, p, X):
        self.z = np.asarray(z, dtype=float)
        self.N = len(z)
        self.dz = np.diff(self.z)

        dp2_mm, eps2, dp3_mm, eps3 = design_vars
        self.eps, self.dp = _porosity_dp_profile(
            self.z, eps2, dp2_mm/1000, eps3, dp3_mm/1000
        )

        self.lam_s = np.maximum(0.188 - 17.5*self.dp, 0.01)
        self.kappa = np.maximum((3.0/self.dp)*(1-self.eps), 1e-8)

        self.Tg = np.asarray(Tg, dtype=float)
        self.u = np.asarray(u, dtype=float)
        self.p = float(p)
        self.X = np.asarray(X, dtype=float)

        self.sigma = ct.stefan_boltzmann
        self.omega = 0.8
        self.T_surr = 300.0

        self.rho_s = 510.0
        self.Cs = 824.0

        self.prop = ct.Solution('gri30.yaml')

    def _compute_h_v(self):
        """Gas-solid heat transfer coefficient"""
        h_v = np.zeros(self.N)

        for i in range(self.N):
            try:
                self.prop.TPX = self.Tg[i], self.p, self.X[:, i]
                rho = self.prop.density
                mu = max(self.prop.viscosity, 1e-8)
                k_g = self.prop.thermal_conductivity

                C = -400*self.dp[i] + 0.687
                m = 443.7*self.dp[i] + 0.361
                Re = rho * self.eps[i] * self.u[i] * self.dp[i] / mu
                Nu = C * (max(Re, 0.01) ** m)

                h_v[i] = Nu * k_g / (self.dp[i]**2)
            except:
                h_v[i] = 1e5

        return np.clip(h_v, 1e3, 1e7)

    def solve(self):
        """
        Solve with CORRECTED boundary conditions
        """

        h_v = self._compute_h_v()

        # BOOST heat transfer
        h_v = h_v * 2.5

        # ========================
        # SOLID PHASE - FIXED BCs
        # ========================
        a = np.zeros(self.N)
        b = np.zeros(self.N)
        c = np.zeros(self.N)
        d = np.zeros(self.N)

        # CRITICAL FIX: Inlet BC should be Ts(0) = Tg(0)
        # NOT zero gradient!
        b[0] = 1.0
        d[0] = self.Tg[0]

        # Interior points
        for i in range(1, self.N-1):
            dz_left = self.dz[i-1]
            dz_right = self.dz[i]

            coef_left = 2*self.lam_s[i] / (dz_left * (dz_left + dz_right))
            coef_right = 2*self.lam_s[i] / (dz_right * (dz_left + dz_right))

            a[i] = coef_left
            c[i] = coef_right

            # Minimal radiation loss
            Ts_guess = self.Tg[i]
            alpha_rad = 2*self.kappa[i]*(1-self.omega)*self.sigma*Ts_guess**3

            b[i] = -(coef_left + coef_right + h_v[i] + alpha_rad)
            d[i] = -h_v[i]*self.Tg[i] - 0.5*self.kappa[i]*(1-self.omega)*self.sigma*self.T_surr**4

        # CORRECTED: Outlet BC - zero gradient
        dz_left = self.dz[-1]
        coef_left = 2*self.lam_s[-1] / (dz_left**2)
        Ts_guess = self.Tg[-1]
        alpha_rad = 2*self.kappa[-1]*(1-self.omega)*self.sigma*Ts_guess**3

        a[-1] = coef_left
        b[-1] = -(coef_left + h_v[-1] + alpha_rad)
        d[-1] = -h_v[-1]*self.Tg[-1] - 0.5*self.kappa[-1]*(1-self.omega)*self.sigma*self.T_surr**4

        Ts = self._thomas_solve(a, b, c, d)
        Ts = np.clip(Ts, 300, 2400)

        return self.Tg, Ts

    def _thomas_solve(self, a, b, c, d):
        """Thomas algorithm"""
        n = len(d)
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)

        c_prime[0] = c[0] / b[0] if abs(b[0]) > 1e-12 else 0
        d_prime[0] = d[0] / b[0] if abs(b[0]) > 1e-12 else 0

        for i in range(1, n):
            denom = b[i] - a[i]*c_prime[i-1]
            if abs(denom) < 1e-12:
                denom = 1e-12
            c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i]*d_prime[i-1]) / denom

        x = np.zeros(n)
        x[-1] = d_prime[-1]
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i]*x[i+1]

        return x


def solve_gas_phase(flame_obj, T_target=None, loglevel=0):
    """Update gas phase"""
    f = flame_obj

    if T_target is not None:
        zloc = f.grid / f.grid.max()
        f.flame.set_fixed_temp_profile(zloc, np.asarray(T_target))
        f.energy_enabled = False
    else:
        f.energy_enabled = True

    f.transport_model = 'mixture-averaged'
    f.set_refine_criteria(ratio=20.0, slope=1.0, curve=1.0, prune=0.05)

    try:
        f.solve(loglevel=loglevel, auto=True)
    except:
        pass

    return f


# Legacy compatibility
UltraSimpleSolver = ThreeStageSolver
