import numpy as np
import cantera as ct

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

def _build_adiabatic_flame():
    gas = ct.Solution('gri30.yaml')
    Tin, p, phi, uin = 300.0, ct.one_atm, 1.0, 0.45   # Try phi=1.0 (easier ignition)
    width = 0.035                                    # Try wider burner
    gas.TP = Tin, p
    gas.set_equivalence_ratio(phi, 'CH4', {'O2': 0.21, 'N2': 0.78, 'AR': 0.01})
    f = ct.BurnerFlame(gas, width=width)
    f.burner.mdot = gas.density * uin
    f.burner.T = Tin
    f.burner.X = gas.X
    f.transport_model = 'mixture-averaged'

    f.set_refine_criteria(ratio=2.0, slope=0.004, curve=0.005, prune=0.001)
    f.solve(loglevel=0, refine_grid=True)
    f.set_initial_guess()
    f.T[:] = np.linspace(400, 1800, len(f.grid))

    print(f"  Grid points: {len(f.grid)}")
    return f




def get_flame():
    global _FLAME
    if _FLAME is None:
        _FLAME = _build_adiabatic_flame()
    return _FLAME

class ThreeStageSolver:
    def __init__(self, z, design_vars, Tg, u, p, X):
        self.z = z
        self.Tg_init = Tg.copy()
        self.u = u
        self.p = p
        if X.ndim == 2:
            self.X = X[:, len(z)//2]
        else:
            self.X = X
        self.L1 = 0.012
        self.L2 = 0.017
        self.L3 = 0.027
        self.dp1 = 0.5e-3
        self.eps1 = 0.40
        dp2_mm, eps2, dp3_mm, eps3 = design_vars
        self.dp2 = dp2_mm * 1e-3
        self.eps2 = eps2
        self.dp3 = dp3_mm * 1e-3
        self.eps3 = eps3
        self.k_s = 5.0
        self.sigma = 5.67e-8
        self.emissivity = 0.85
        self.gas = ct.Solution('gri30.yaml')
        self.relax = 0.6

    def _get_stage_properties(self, z_val):
        if z_val <= self.L1:
            return self.dp1, self.eps1
        elif z_val <= self.L2:
            return self.dp2, self.eps2
        else:
            return self.dp3, self.eps3

    def _heat_transfer_coeff(self, dp, eps, u_local, T_avg):
        self.gas.TPX = T_avg, self.p, self.X
        rho = self.gas.density
        cp = self.gas.cp_mass
        mu = self.gas.viscosity
        k_g = self.gas.thermal_conductivity
        Re = max(rho * (u_local/eps) * dp / mu, 1e-6)  # Effective velocity
        Pr = mu * cp / k_g
        if eps < 0.3:
            Nu = 2.0 + 1.6 * (Re**0.6) * (Pr**(1/3))
        elif eps > 0.85:
            Nu = 2.0 + 0.7 * (Re**0.6) * (Pr**(1/3))
        else:
            Nu = 2.0 + 1.1 * (Re**0.6) * (Pr**(1/3))
        h = Nu * k_g / dp
        av = 6.0 * (1 - eps) / dp
        return h * av

    def _effective_conductivity(self, eps):
        return self.k_s * (1 - eps)**1.2 if eps > 0.85 else self.k_s * (1 - eps)

    def _solve_solid(self, Tg, Ts_old):
        n = len(self.z)
        a = np.zeros(n); b = np.zeros(n); c = np.zeros(n); d = np.zeros(n)
        for i in range(n):
            dp, eps = self._get_stage_properties(self.z[i])
            k_eff = self._effective_conductivity(eps)
            hv = self._heat_transfer_coeff(dp, eps, self.u[i], 0.5*(Tg[i] + Ts_old[i]))
            if i == 0:
                b[i] = 1.0; d[i] = Tg[i]
            elif i == n-1:
                dz = self.z[i] - self.z[i-1]
                a[i] = -k_eff/dz
                b[i] = k_eff/dz + hv + 4*self.emissivity*self.sigma*Ts_old[i]**3
                d[i] = hv*Tg[i] + 3*self.emissivity*self.sigma*Ts_old[i]**4
            else:
                dz_prev = self.z[i]-self.z[i-1]
                dz_next = self.z[i+1]-self.z[i]
                a[i]=-k_eff/dz_prev
                c[i]=-k_eff/dz_next
                b[i]=k_eff/dz_prev + k_eff/dz_next + hv
                d[i]=hv*Tg[i]
        return self._solve_tridiagonal(a, b, c, d)

    def _update_gas(self, Tg_old, Ts):
        n = len(self.z)
        Tg_new = Tg_old.copy()
        for i in range(1, n):
            dp, eps = self._get_stage_properties(self.z[i])
            hv = self._heat_transfer_coeff(dp, eps, self.u[i], 0.5*(Tg_old[i]+Ts[i]))
            self.gas.TPX = 0.5*(Tg_old[i]+Ts[i]), self.p, self.X
            rho = self.gas.density; cp = self.gas.cp_mass
            dz = self.z[i] - self.z[i-1]
            denom = rho * cp * self.u[i]
            Q_exchange = hv * (Ts[i] - Tg_old[i])
            if denom > 1e-6:
                Tg_new[i] = Tg_old[i-1] + (Q_exchange * dz) / denom
            else:
                Tg_new[i] = Tg_old[i]
            Tg_new[i] = np.clip(Tg_new[i], 300, 2200)
        Tg_new[0] = 300.0
        return Tg_new

    def solve(self, max_iter=12, tol=10.0):
        n = len(self.z)
        Tg = self.Tg_init.copy()
        Ts = self.Tg_init.copy()
        for iteration in range(max_iter):
            Ts_new = self._solve_solid(Tg, Ts)
            Tg_new = self._update_gas(Tg, Ts_new)
            Ts = self.relax * Ts_new + (1-self.relax) * Ts
            Tg = self.relax * Tg_new + (1-self.relax) * Tg
            if max(np.abs(Ts-Ts_new)) < tol and max(np.abs(Tg-Tg_new)) < tol:
                break
        T_ad = get_T_ad()
        efficiency = (Ts[-1]/T_ad)**4
        return efficiency, Ts, Tg

    def _solve_tridiagonal(self, a, b, c, d):
        n = len(d)
        c_prime = np.zeros(n); d_prime = np.zeros(n); x = np.zeros(n)
        c_prime[0] = c[0] / b[0]; d_prime[0] = d[0] / b[0]
        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i-1]
            c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom
        x[-1] = d_prime[-1]
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
        return x
