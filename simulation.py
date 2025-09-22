# simulation.py
import numpy as np
import cantera as ct

# -----------------------------
# Cached gas-phase flame context
# -----------------------------
_FLAME_CTX = None

def _compute_flame_ctx():
    """
    Solve one FreeFlame (mixture-averaged, moderate refinement) and cache grid/state.
    """
    gas = ct.Solution('gri30.yaml')
    Tin = 300.0
    p = ct.one_atm
    phi = 0.65
    uin = 0.45
    width = 0.0605

    gas.TP = Tin, p
    gas.set_equivalence_ratio(phi=phi, fuel='CH4',
                              oxidizer={'O2': 0.21, 'N2': 0.78, 'Ar': 0.01})

    f = ct.FreeFlame(gas, width=width)
    f.inlet.T = Tin
    f.inlet.X = gas.X
    f.inlet.mdot = gas.density * uin  # kg/m^2/s

    # Faster transport and moderate refinement for repeated optimization calls
    f.transport_model = 'mixture-averaged'
    f.set_refine_criteria(ratio=3, slope=0.15, curve=0.30)
    f.solve(loglevel=0, auto=True)

    # Cache flame state and inlet mass flux for power normalization
    return dict(
        z=f.grid.copy(),
        Tg=f.T.copy(),
        u=f.velocity.copy(),
        X=f.X.copy(),
        p=p,
        Tin=Tin,
        phi=phi,
        mdot=float(f.inlet.mdot)
    )

def _get_flame_ctx():
    global _FLAME_CTX
    if _FLAME_CTX is None:
        _FLAME_CTX = _compute_flame_ctx()
    return _FLAME_CTX  # Reusing a cached flame/grid is the intended fast pattern for repeated 1D runs [web:220][web:206].

# -----------------------------
# Helpers: solid/radiation models
# -----------------------------
def _porosity_dp_profile(z, porosity_stage2, pore_diameter):
    """
    Two-stage porous medium with linear transition from 0.033 m to 0.037 m.
    """
    porosity1, dp1 = 0.835, 0.00029
    eps = np.empty_like(z)
    dp = np.empty_like(z)
    for i, x in enumerate(z):
        if x < 0.033:
            eps[i], dp[i] = porosity1, dp1
        elif x > 0.037:
            eps[i], dp[i] = porosity_stage2, pore_diameter
        else:
            f = (x - 0.033) / 0.004
            eps[i] = porosity1 + f * (porosity_stage2 - porosity1)
            dp[i]  = dp1 + f * (pore_diameter - dp1)
    dp = np.maximum(dp, 1e-6)
    return eps, dp  # Keeps dp positive while matching the two-stage/transition description [web:255].

def _assemble_conduction(z, lambda_s, h_v):
    """
    Assemble insulated-end operator: -d/dx(λ dTs/dx) + h_v * Ts = RHS with Neumann BCs.
    """
    N = len(z)
    A = np.zeros((N, N))
    # Neumann at x=0: T0 - T1 = 0
    A[0, 0] = 1.0
    A[0, 1] = -1.0

    def hmean(a, b):
        return 2.0 * a * b / (a + b) if (a + b) != 0.0 else 0.0

    for i in range(1, N - 1):
        dz_w = z[i] - z[i - 1]
        dz_e = z[i + 1] - z[i]
        lam_w = hmean(lambda_s[i - 1], lambda_s[i])
        lam_e = hmean(lambda_s[i], lambda_s[i + 1])
        a_w = lam_w / dz_w
        a_e = lam_e / dz_e
        A[i, i - 1] = -a_w
        A[i, i]     = a_w + a_e + h_v[i]
        A[i, i + 1] = -a_e

    # Neumann at x=L: TN-1 - TN-2 = 0
    A[N - 1, N - 1] = 1.0
    A[N - 1, N - 2] = -1.0
    return A  # Discretization mirrors the 1D reference approach for second-order operators [web:255].

def _build_s2_matrix(z, kappa, omega=0.7, T_surround=300.0):
    """
    Build constant S2 (two-flux) system for unknowns [q+_0, q-_0, q+_1, q-_1, ...].
    BCs: q+(0)=σTsur^4, q-(N-1)=σTsur^4; ODEs enforced interior and one-sided at ends.
    """
    N = len(z)
    sigma = ct.stefan_boltzmann
    kappa = np.maximum(kappa, 1e-12)
    om = float(np.clip(omega, 0.0, 0.999999))

    A = np.zeros((2 * N, 2 * N))
    b = np.zeros(2 * N)
    ip = lambda i: 2 * i
    im = lambda i: 2 * i + 1

    # Dirichlet incoming intensities
    A[ip(0), ip(0)] = 1.0
    b[ip(0)] = sigma * T_surround**4
    A[im(N - 1), im(N - 1)] = 1.0
    b[im(N - 1)] = sigma * T_surround**4

    # Interior ODEs (central differences over span z[i+1]-z[i-1])
    for i in range(1, N - 1):
        dz = (z[i + 1] - z[i - 1])
        ki = kappa[i]
        # d(q+)/dx = -k(2-om) q+ + k*om q- + S
        A[ip(i), ip(i - 1)] = -1.0 / dz
        A[ip(i), ip(i + 1)] =  1.0 / dz
        A[ip(i), ip(i)]     =  ki * (2.0 - om)
        A[ip(i), im(i)]     = -ki * om
        # -d(q-)/dx = k*om q+ - k(2-om) q- + S
        A[im(i), im(i - 1)] = -1.0 / dz
        A[im(i), im(i + 1)] =  1.0 / dz
        A[im(i), ip(i)]     = -ki * om
        A[im(i), im(i)]     =  ki * (2.0 - om)

    # One-sided ODEs at boundaries
    dz0 = z[1] - z[0]
    k0  = kappa[0]
    A[im(0), im(0)] += (1.0 / dz0) + k0 * (2.0 - om)
    A[im(0), im(1)] += -(1.0 / dz0)
    A[im(0), ip(0)] += -k0 * om

    dzN = z[N - 1] - z[N - 2]
    kN  = kappa[N - 1]
    A[ip(N - 1), ip(N - 1)] += (1.0 / dzN) + kN * (2.0 - om)
    A[ip(N - 1), ip(N - 2)] += -(1.0 / dzN)
    A[ip(N - 1), im(N - 1)] += -kN * om

    return {'A': A, 'b_base': b, 'ip': ip, 'im': im, 'omega': om}  # Closed S2 system avoids singularity and is constant per design [web:255].

def _build_s2_rhs(z, Ts, kappa, rad_ctx):
    """
    Build RHS with emission S(T)=2*kappa*(1-omega)*σ*Ts^4 at ODE rows (with Ts clipped).
    """
    sigma = ct.stefan_boltzmann
    N = len(z)
    b = rad_ctx['b_base'].copy()
    ip, im = rad_ctx['ip'], rad_ctx['im']
    om = rad_ctx['omega']
    Ts_eff = np.clip(Ts, 250.0, 2600.0)
    S = 2.0 * kappa * (1.0 - om) * sigma * (Ts_eff**4)

    # Left boundary ODE for q-
    b[im(0)] += S[0]
    # Interior ODEs
    for i in range(1, N - 1):
        b[ip(i)] += S[i]
        b[im(i)] += S[i]
    # Right boundary ODE for q+
    b[ip(N - 1)] += S[N - 1]
    return b

# -----------------------------
# Objective wrapper
# -----------------------------
def run_burner_simulation(design_vars, gas_obj=None):
    """
    Computes radiant efficiency based on the thesis definition: η = (Ts,out / T_ad)^4.
    Returns -η for minimization.
    """
    try:
        ctx = _get_flame_ctx()
        z, Tg, u, X, p = ctx['z'], ctx['Tg'], ctx['u'], ctx['X'], ctx['p']
        Tin, phi = ctx['Tin'], ctx['phi']
        N = len(z)

        pore_diameter_mm, porosity_stage2 = design_vars
        pore_diameter = max(pore_diameter_mm / 1000.0, 1e-6)
        omega_s2 = 0.8

        eps, dp = _porosity_dp_profile(z, porosity_stage2, pore_diameter)
        lambda_s = np.maximum(0.188 - 17.5 * dp, 1e-4)
        kappa = np.maximum((3.0 / dp) * (1.0 - eps), 1e-12)

        prop_gas = ct.Solution('gri30.yaml')
        h_v = np.zeros(N)
        for i in range(N):
            prop_gas.TPX = Tg[i], p, X[:, i]
            rho, mu, k_g = prop_gas.density, prop_gas.viscosity, prop_gas.thermal_conductivity
            C = -400.0 * dp[i] + 0.687
            m = 443.7 * dp[i] + 0.361
            Re_p = rho * eps[i] * u[i] * dp[i] / mu if mu > 0 else 0.0
            Nu_v = C * (Re_p ** m) if Re_p > 0 else 0.0
            h_v[i] = Nu_v * k_g / (dp[i] ** 2) if dp[i] > 0 else 0.0
        h_v = np.clip(h_v, 0.0, 5e6)

        A = _assemble_conduction(z, lambda_s, h_v)
        solve_A = lambda rhs: np.linalg.solve(A, rhs)

        Rad = _build_s2_matrix(z, kappa, omega=omega_s2, T_surround=300.0)
        rad_solve = lambda b: np.linalg.solve(Rad['A'], b)

        Ts = np.clip(Tg.copy(), 300.0, 2000.0)
        max_iter, rtol, alpha = 80, 1e-6, 0.1

        for iter_count in range(max_iter):
            Ts_old = Ts.copy()
            b_rad = _build_s2_rhs(z, Ts, kappa, Rad)
            sol = rad_solve(b_rad)
            J = 0.5 * (sol[0::2] + sol[1::2])
            Ts_eff = np.clip(Ts, 250.0, 2600.0)
            dq_dx = 4.0 * kappa * (1.0 - omega_s2) * (ct.stefan_boltzmann * Ts_eff**4 - J)
            rhs = np.zeros(N); rhs[1:-1] = h_v[1:-1] * Tg[1:-1] - dq_dx[1:-1] # Corrected RHS
            Ts_new = solve_A(rhs)
            if not np.all(np.isfinite(Ts_new)): return 1.0
            Ts = alpha * Ts_new + (1.0 - alpha) * Ts_old
            Ts = np.clip(Ts, 250.0, 2600.0)
            if np.linalg.norm(Ts - Ts_old) / np.linalg.norm(Ts_old) < rtol: break
        if iter_count == max_iter - 1: return 1.0

        gas_ad = ct.Solution('gri30.yaml')
        gas_ad.TP = Tin, p
        gas_ad.set_equivalence_ratio(phi=phi, fuel='CH4', oxidizer={'O2': 0.21, 'N2': 0.78, 'Ar': 0.01})
        gas_ad.equilibrate('HP')
        T_ad = gas_ad.T
        
        efficiency = (Ts[-1] / T_ad)**4
        efficiency = float(np.clip(efficiency, 0.0, 1.0))
        
        return -efficiency if np.isfinite(efficiency) else 1.0

    except (ct.CanteraError, np.linalg.LinAlgError, ValueError, FloatingPointError):
        return 1.0
