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
    f.inlet.mdot = gas.density * uin

    # Faster and sufficient for optimization sweeps
    f.transport_model = 'mixture-averaged'
    f.set_refine_criteria(ratio=3, slope=0.15, curve=0.30)
    f.solve(loglevel=0, auto=True)

    return dict(
        z=f.grid.copy(),
        Tg=f.T.copy(),
        u=f.velocity.copy(),
        X=f.X.copy(),
        p=p,
        Tin=Tin,
        phi=phi
    )  # Reusing a cached flame/grid is the recommended fast pattern for repeated runs [web:220][web:295].

def _get_flame_ctx():
    global _FLAME_CTX
    if _FLAME_CTX is None:
        _FLAME_CTX = _compute_flame_ctx()
    return _FLAME_CTX  # The cached context avoids re-solving the flame each objective call [web:220][web:295].

# -----------------------------
# Helpers: solid/radiation models
# -----------------------------
def _porosity_dp_profile(z, porosity_stage2, pore_diameter):
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
    return eps, dp  # This profile matches the two-stage/transition description while keeping dp positive [web:220].

def _assemble_conduction(z, lambda_s, h_v):
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
    return A  # The insulated-end operator is fixed per design and can be factorized once for speed [web:255][web:295].

def _build_s2_matrix(z, kappa, omega=0.8, T_surround=300.0):
    """
    Build constant S2 system matrix for q_plus/q_minus with closed BCs.
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

    # Interior ODEs (central difference on span z[i+1]-z[i-1])
    for i in range(1, N - 1):
        dz = (z[i + 1] - z[i - 1])
        ki = kappa[i]
        # d(q+)/dx
        A[ip(i), ip(i - 1)] = -1.0 / dz
        A[ip(i), ip(i + 1)] =  1.0 / dz
        A[ip(i), ip(i)]     =  ki * (2.0 - om)
        A[ip(i), im(i)]     = -ki * om
        # -d(q-)/dx
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

    return {'A': A, 'b_base': b, 'ip': ip, 'im': im, 'omega': om}  # The radiation matrix is constant per design and re-used within the iteration [web:255][web:295].

def _build_s2_rhs(z, Ts, kappa, rad_ctx):
    sigma = ct.stefan_boltzmann
    N = len(z)
    b = rad_ctx['b_base'].copy()
    ip, im = rad_ctx['ip'], rad_ctx['im']
    om = rad_ctx['omega']
    # Use Ts clipped to physical bounds during RHS build to prevent emission blow-up
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
    return b  # Building only the RHS each iteration avoids refactorization cost and improves stability [web:255][web:295].

# -----------------------------
# Objective wrapper
# -----------------------------
def run_burner_simulation(design_vars, gas_obj):
    """
    Compute -efficiency for (pore_diameter_mm, porosity_stage2) by:
      1) Reusing a cached FreeFlame grid/state,
      2) Solving solid Ts with conduction + hv*(Tg - Ts) + S2 radiation,
      3) Returning negative radiant efficiency for minimization.
    """
    try:
        ctx = _get_flame_ctx()
        z, Tg, u, X, p = ctx['z'], ctx['Tg'], ctx['u'], ctx['X'], ctx['p']
        Tin, phi = ctx['Tin'], ctx['phi']
        N = len(z)

        pore_diameter_mm, porosity_stage2 = design_vars
        pore_diameter = max(pore_diameter_mm / 1000.0, 1e-6)

        eps, dp = _porosity_dp_profile(z, porosity_stage2, pore_diameter)
        lambda_s = 0.188 - 17.5 * dp
        kappa = np.maximum((3.0 / dp) * (1.0 - eps), 1e-12)

        # Gas properties at local (T, X)
        prop_gas = ct.Solution('gri30.yaml')
        h_v = np.zeros(N)
        for i in range(N):
            prop_gas.TPX = Tg[i], p, X[:, i]
            rho = prop_gas.density
            mu  = prop_gas.viscosity
            k_g = prop_gas.thermal_conductivity
            C = -400.0 * dp[i] + 0.687
            m = 443.7 * dp[i] + 0.361
            Re_p = rho * eps[i] * u[i] * dp[i] / mu
            Nu_v = C * (Re_p ** m) if Re_p > 0 else 0.0
            h_v[i] = Nu_v * k_g / (dp[i] ** 2)

        # Cap h_v to guard extreme correlations during optimization sweeps
        h_v = np.clip(h_v, 0.0, 5e6)  # W/m^3/K (tunable guardrail for stability) [web:255].

        # Conduction operator factorization
        A = _assemble_conduction(z, lambda_s, h_v)
        try:
            from scipy.sparse import csc_matrix
            from scipy.sparse.linalg import splu
            A_lu = splu(csc_matrix(A))
            solve_A = lambda rhs: A_lu.solve(rhs)
        except Exception:
            solve_A = lambda rhs: np.linalg.solve(A, rhs)

        # Radiation operator factorization
        Rad = _build_s2_matrix(z, kappa, omega=0.8, T_surround=300.0)
        try:
            from scipy.sparse import csc_matrix
            from scipy.sparse.linalg import splu
            R_lu = splu(csc_matrix(Rad['A']))
            rad_solve = lambda b: R_lu.solve(b)
        except Exception:
            rad_solve = lambda b: np.linalg.solve(Rad['A'], b)

        # Fixed-point iteration for Ts with adaptive relaxation and clamping
        Ts = np.clip(Tg.copy(), 300.0, 2000.0)  # start from gas temperature for stability [web:220].
        max_iter = 40
        rtol = 5e-7
        alpha = 0.5

        for _ in range(max_iter):
            # Build radiation RHS using clipped Ts to avoid overflow in emission
            b_rad = _build_s2_rhs(z, Ts, kappa, Rad)
            sol = rad_solve(b_rad)
            q_plus = sol[0::2]
            q_minus = sol[1::2]
            J = 0.5 * (q_plus + q_minus)

            # Compute dq/dx with safeguarded Ts
            Ts_eff = np.clip(Ts, 250.0, 2600.0)
            dq_dx = 4.0 * kappa * (1.0 - 0.8) * (ct.stefan_boltzmann * Ts_eff**4 - J)

            rhs = np.zeros(N)
            rhs[1:-1] = h_v[1:-1] * (Tg[1:-1] - Ts_eff[1:-1]) - dq_dx[1:-1]

            Ts_new = solve_A(rhs)

            # Backtracking if update is unstable/non-finite
            local_alpha = alpha
            ok = False
            for _bt in range(6):
                Ts_upd = local_alpha * Ts_new + (1.0 - local_alpha) * Ts
                if np.all(np.isfinite(Ts_upd)) and np.max(np.abs(Ts_upd)) < 1e5:
                    ok = True
                    break
                local_alpha *= 0.5
            if not ok:
                return 1.0  # penalize if cannot stabilize update [web:295].

            # Clamp to physical bounds each iteration to prevent runaway emission
            Ts_upd = np.clip(Ts_upd, 250.0, 2600.0)

            err = np.linalg.norm(Ts_upd - Ts, ord=np.inf) / max(1.0, np.linalg.norm(Ts, ord=np.inf))
            Ts = Ts_upd
            if err < rtol:
                break  # tight per-iteration tolerance improves robustness without over-iterating [web:295].

        # Guard against non-finite values
        if not np.all(np.isfinite(Ts)):
            return 1.0  # penalty path to keep optimizer stable [web:295].

        # Adiabatic reference
        gas_ad = ct.Solution('gri30.yaml')
        gas_ad.TP = Tin, p
        gas_ad.set_equivalence_ratio(phi=phi, fuel='CH4',
                                     oxidizer={'O2': 0.21, 'N2': 0.78, 'Ar': 0.01})
        gas_ad.equilibrate('HP')
        T_ad = gas_ad.T

        efficiency = (Ts[-1] / T_ad)**4
        if not np.isfinite(efficiency):
            return 1.0  # guard [web:295].
        return -efficiency  # minimization target [web:220].

    except ct.CanteraError:
        return 1.0  # penalty on solver issues keeps optimization moving [web:220].
