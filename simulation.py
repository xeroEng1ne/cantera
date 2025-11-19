# simulation.py
import numpy as np
import cantera as ct

# -----------------------------
# Cached BurnerFlame and gases
# -----------------------------
_FLAME = None
_PROP_GAS = None
_GAS_AD = None
_T_AD = None

def _get_prop_gas():
    global _PROP_GAS
    if _PROP_GAS is None:
        _PROP_GAS = ct.Solution('gri30.yaml')
    return _PROP_GAS

def get_T_ad():
    global _GAS_AD, _T_AD
    if _T_AD is None:
        _GAS_AD = ct.Solution('gri30.yaml')
        _GAS_AD.TP = 300.0, ct.one_atm
        _GAS_AD.set_equivalence_ratio(0.65, 'CH4', {'O2': 0.21, 'N2': 0.78, 'AR': 0.01})
        _GAS_AD.equilibrate('HP')
        _T_AD = _GAS_AD.T
    return _T_AD

def _build_adiabatic_flame():
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
    global _FLAME
    if _FLAME is None:
        _FLAME = _build_adiabatic_flame()
    return _FLAME

# -----------------------------
# Solid phase (S2 + conduction)
# -----------------------------
def _porosity_dp_profile(z, eps2, dp2_m):
    eps1, dp1 = 0.835, 0.00029
    z0, z1 = 0.033, 0.037
    eps = np.empty_like(z)
    dp = np.empty_like(z)
    for i, x in enumerate(z):
        if x < z0:
            eps[i], dp[i] = eps1, dp1
        elif x <= z1:
            f = (x - z0) / (z1 - z0)
            eps[i] = eps1 + f * (eps2 - eps1)
            dp[i]  = dp1  + f * (dp2_m - dp1)
        else:
            eps[i], dp[i] = eps2, dp2_m
    return eps, np.maximum(dp, 1e-6)

def _assemble_conduction(z, lam_s, h_v):
    N = len(z)
    A = np.zeros((N, N))
    A[0, 0] = 1.0; A[0, 1] = -1.0
    def hmean(a, b): return 2*a*b/(a+b) if (a+b) != 0.0 else 0.0
    for i in range(1, N-1):
        dz_w = z[i] - z[i-1]; dz_e = z[i+1] - z[i]
        lam_w = hmean(lam_s[i-1], lam_s[i]); lam_e = hmean(lam_s[i], lam_s[i+1])
        a_w, a_e = lam_w/dz_w, lam_e/dz_e
        A[i, i-1] = -a_w
        A[i, i]   = a_w + a_e + h_v[i]
        A[i, i+1] = -a_e
    A[N-1, N-1] = 1.0; A[N-1, N-2] = -1.0
    return A

def _build_s2_matrix(z, kappa, omega=0.7, T_surround=300.0):
    N = len(z); sigma = ct.stefan_boltzmann; om = float(np.clip(omega, 0.0, 0.999999))
    kappa = np.maximum(kappa, 1e-12)
    A = np.zeros((2*N, 2*N)); b = np.zeros(2*N)
    ip = lambda i: 2*i; im = lambda i: 2*i+1
    A[ip(0), ip(0)] = 1.0; b[ip(0)] = sigma * T_surround**4
    A[im(N-1), im(N-1)] = 1.0; b[im(N-1)] = sigma * T_surround**4
    for i in range(1, N-1):
        dz = (z[i+1] - z[i-1]); ki = kappa[i]
        A[ip(i), ip(i-1)] = -1.0/dz; A[ip(i), ip(i+1)] =  1.0/dz
        A[ip(i), ip(i)]   =  ki*(2.0-om); A[ip(i), im(i)] = -ki*om
        A[im(i), im(i-1)] = -1.0/dz; A[im(i), im(i+1)] =  1.0/dz
        A[im(i), ip(i)]   = -ki*om;    A[im(i), im(i)]  =  ki*(2.0-om)
    dz0, k0 = z[1]-z[0], kappa[0]
    A[im(0), im(0)] += 1.0/dz0 + k0*(2.0-om); A[im(0), im(1)] += -1.0/dz0; A[im(0), ip(0)] += -k0*om
    dzN, kN = z[-1]-z[-2], kappa[-1]
    A[ip(N-1), ip(N-1)] += 1.0/dzN + kN*(2.0-om); A[ip(N-1), ip(N-2)] += -1.0/dzN; A[ip(N-1), im(N-1)] += -kN*om
    return {'A': A, 'b_base': b, 'ip': ip, 'im': im, 'omega': om}

def _build_s2_rhs(z, Ts, kappa, rad_ctx):
    sigma = ct.stefan_boltzmann; N = len(z)
    b = rad_ctx['b_base'].copy(); ip, im = rad_ctx['ip'], rad_ctx['im']; om = rad_ctx['omega']
    Ts_eff = np.clip(Ts, 250.0, 2600.0)
    S = 2.0 * kappa * (1.0 - om) * sigma * (Ts_eff**4)
    b[im(0)] += S[0]
    for i in range(1, N-1):
        b[ip(i)] += S[i]; b[im(i)] += S[i]
    b[ip(N-1)] += S[N-1]
    return b

def solve_solid_phase(gas_solution_dict, design_vars):
    z = gas_solution_dict['z']; Tg = gas_solution_dict['T']; u = gas_solution_dict['u']
    X = gas_solution_dict['X']; p = gas_solution_dict['p']
    N = len(z)
    dp2_mm, eps2 = float(design_vars[0]), float(design_vars[1])
    dp2 = max(dp2_mm/1000.0, 1e-6)
    eps, dp = _porosity_dp_profile(z, eps2, dp2)
    lam_s = np.maximum(0.188 - 17.5 * dp, 1e-4)
    kappa = np.maximum((3.0/dp) * (1.0 - eps), 1e-12)

    prop = _get_prop_gas()
    h_v = np.zeros(N)
    for i in range(N):
        prop.TPX = Tg[i], p, X[:, i]
        rho = prop.density; mu = prop.viscosity; k_g = prop.thermal_conductivity
        C = -400.0 * dp[i] + 0.687; m = 443.7 * dp[i] + 0.361
        Re_p = rho * eps[i] * u[i] * dp[i] / mu if mu > 0 else 0.0
        Nu_v = C * (Re_p ** m) if Re_p > 0 else 0.0
        h_v[i] = Nu_v * k_g / (dp[i]**2) if dp[i] > 0 else 0.0
    h_v = np.clip(h_v, 0.0, 5e6)

    A = _assemble_conduction(z, lam_s, h_v)
    Rad = _build_s2_matrix(z, kappa, omega=0.7, T_surround=300.0)
    try:
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import splu
        A_lu = splu(csc_matrix(A)); solve_A = lambda rhs: A_lu.solve(rhs)
        R_lu = splu(csc_matrix(Rad['A'])); rad_solve = lambda b: R_lu.solve(b)
    except Exception:
        solve_A = lambda rhs: np.linalg.solve(A, rhs)
        rad_solve = lambda b: np.linalg.solve(Rad['A'], b)

    Ts = np.clip(Tg.copy(), 300.0, 2000.0)
    max_iter = 20
    for it in range(max_iter):
        Ts_old = Ts.copy()
        b_rad = _build_s2_rhs(z, Ts, kappa, Rad)
        sol = rad_solve(b_rad)
        J = 0.5 * (sol[0::2] + sol[1::2])
        Ts_eff = np.clip(Ts, 250.0, 2600.0)
        dq_dx = 4.0 * kappa * (1.0 - Rad['omega']) * (ct.stefan_boltzmann * Ts_eff**4 - J)
        rhs = np.zeros(N); rhs[1:-1] = h_v[1:-1] * Tg[1:-1] - dq_dx[1:-1]
        Ts_new = solve_A(rhs)
        if not np.all(np.isfinite(Ts_new)):
            return None, None
        Ts = 0.5 * Ts_new + 0.5 * Ts_old
        Ts = np.clip(Ts, 250.0, 2600.0)
        err = np.linalg.norm(Ts - Ts_old, ord=np.inf)/max(1.0, np.linalg.norm(Ts_old, ord=np.inf))
        if (it >= 8 and err < 2e-6) or err < 5e-7:
            break
    return Ts, h_v

# -----------------------------
# Gas phase with fixed T profile
# -----------------------------
def solve_gas_phase(flame_obj, T_target=None, loglevel=0, refine=False):
    """
    Update the same BurnerFlame using a fixed temperature profile when T_target is given
    (energy disabled), else solve normally. Refinement can be disabled for speed.
    """
    f = flame_obj
    if T_target is not None:
        zloc = f.grid / f.grid.max()
        f.flame.set_fixed_temp_profile(zloc, np.asarray(T_target))
        f.energy_enabled = False
    else:
        f.energy_enabled = True

    f.transport_model = 'mixture-averaged'
    if refine:
        f.set_refine_criteria(ratio=3, slope=0.15, curve=0.30)
    else:
        # Effectively disable refinement during optimization sweeps
        f.set_refine_criteria(ratio=10, slope=1e9, curve=1e9)

    f.solve(loglevel=loglevel, auto=True)
    return f
