import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

# -----------------------------
# 1) Gas-phase free flame solve
# -----------------------------
gas = ct.Solution('gri30.yaml')

Tin = 300.0
p = ct.one_atm
phi = 0.65
uin = 0.45  # m/s, only for inlet mdot set below

gas.TP = Tin, p
gas.set_equivalence_ratio(phi=phi, fuel='CH4',
                          oxidizer={'O2': 0.21, 'N2': 0.78, 'Ar': 0.01})

width = 0.0605  # m
flame = ct.FreeFlame(gas, width=width)
flame.inlet.T = Tin
flame.inlet.X = gas.X
flame.inlet.mdot = gas.density * uin

# Solve flame
flame.set_refine_criteria(ratio=3, slope=0.1, curve=0.2)
flame.solve(loglevel=1, auto=True)

# Extract 1D solution on the converged grid
z = flame.grid.copy()           # m
Tg = flame.T.copy()             # K
u = flame.velocity.copy()       # m/s
X = flame.X.copy()              # shape: (n_species, n_points)

# -----------------------------
# 2) Solid with S2 radiation
# -----------------------------

# Two-flux (S2) solver: returns dq/dx given Ts, kappa, omega
def solve_radiation(z, Ts, kappa, omega, T_surround=300.0):
    """
    Two-flux (Schuster–Schwarzschild) solver on the given nonuniform grid.
    Unknowns: q_plus[i], q_minus[i], i=0..N-1
    BCs: q_plus(0) = sigma*T_sur^4, q_minus(L) = sigma*T_sur^4
    ODEs are enforced at all interior nodes with central differences,
    and at both boundaries with one-sided differences to close the system.
    Returns dq/dx = 4*kappa*(1-omega)*(sigma*Ts^4 - 0.5*(q_plus+q_minus)).
    """
    import numpy as np
    import cantera as ct

    z = np.asarray(z, dtype=float)
    Ts = np.asarray(Ts, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    n = len(z)

    # Robustness: avoid zero/degenerate coefficients
    kappa = np.maximum(kappa, 1e-12)
    if np.isscalar(omega):
        omega_vec = np.full(n, float(omega))
    else:
        omega_vec = np.asarray(omega, dtype=float)
    omega_vec = np.clip(omega_vec, 0.0, 0.999999)

    sigma = ct.stefan_boltzmann  # W/m^2/K^4

    A = np.zeros((2 * n, 2 * n), dtype=float)
    b = np.zeros(2 * n, dtype=float)

    # Index helpers for packed unknowns [q+_0, q-_0, q+_1, q-_1, ...]
    def ip(i): return 2 * i       # q_plus index
    def im(i): return 2 * i + 1   # q_minus index

    # Left boundary: Dirichlet for incoming from outside
    A[0, ip(0)] = 1.0
    b[0] = sigma * T_surround**4

    # Enforce ODE for q_minus at i=0 using forward difference
    dz0 = z[1] - z[0]
    k0, om0 = kappa[0], omega_vec[0]
    S0 = 2.0 * k0 * (1.0 - om0) * sigma * Ts[0]**4
    r = 1
    # -d(q-)/dx = k*om*q+ - k*(2-om)*q- + S
    # -> (q-_0 - q-_1)/dz0 - k*om*q+_0 + k*(2-om)*q-_0 = S
    A[r, im(0)] += (1.0 / dz0) + k0 * (2.0 - om0)
    A[r, im(1)] += -(1.0 / dz0)
    A[r, ip(0)] += -k0 * om0
    b[r] = S0

    # Interior nodes: central differences on span Δz = z[i+1]-z[i-1]
    for i in range(1, n - 1):
        dz = (z[i + 1] - z[i - 1])
        ki, omi = kappa[i], omega_vec[i]
        S = 2.0 * ki * (1.0 - omi) * sigma * Ts[i]**4

        # d(q+)/dx = -k(2-om) q+ + k*om q- + S
        r = 2 * i
        A[r, ip(i - 1)] += -1.0 / dz
        A[r, ip(i + 1)] +=  1.0 / dz
        A[r, ip(i)]     +=  ki * (2.0 - omi)
        A[r, im(i)]     += -ki * omi
        b[r] = S

        # -d(q-)/dx = k*om q+ - k(2-om) q- + S
        r = 2 * i + 1
        A[r, im(i - 1)] += -1.0 / dz
        A[r, im(i + 1)] +=  1.0 / dz
        A[r, ip(i)]     += -ki * omi
        A[r, im(i)]     +=  ki * (2.0 - omi)
        b[r] = S

    # Enforce ODE for q_plus at i=n-1 using backward difference
    dzN = z[n - 1] - z[n - 2]
    kN, omN = kappa[n - 1], omega_vec[n - 1]
    SN = 2.0 * kN * (1.0 - omN) * sigma * Ts[n - 1]**4
    r = 2 * n - 2
    # d(q+)/dx ≈ (q+_N - q+_{N-1})/dzN
    # -> (q+_N - q+_{N-1})/dzN + k(2-om) q+_N - k*om q-_N = S
    A[r, ip(n - 1)] += (1.0 / dzN) + kN * (2.0 - omN)
    A[r, ip(n - 2)] += -(1.0 / dzN)
    A[r, im(n - 1)] += -kN * omN
    b[r] = SN

    # Right boundary: Dirichlet for incoming from outside
    A[2 * n - 1, im(n - 1)] = 1.0
    b[2 * n - 1] = sigma * T_surround**4

    # Solve packed system
    x = np.linalg.solve(A, b)
    q_plus = x[0::2]
    q_minus = x[1::2]
    J = 0.5 * (q_plus + q_minus)

    dq_dx = 4.0 * kappa * (1.0 - omega_vec) * (sigma * Ts**4 - J)
    return dq_dx


# Porosity and particle size with linear transition
def porosity_and_dp(x):
    porosity1, dp1 = 0.835, 0.00029
    porosity2, dp2 = 0.87,  0.00152
    if x < 0.033:
        return porosity1, dp1
    elif x > 0.037:
        return porosity2, dp2
    else:
        f = (x - 0.033) / 0.004
        return porosity1 + f*(porosity2 - porosity1), dp1 + f*(dp2 - dp1)

N = len(z)
lambda_s = np.zeros(N)   # solid conductivity
h_v = np.zeros(N)        # volumetric gas-solid heat transfer
kappa = np.zeros(N)      # extinction coefficient
omega = 0.8              # scattering albedo (assumed constant)

# Use local composition and temperature for gas properties where available
prop_gas = ct.Solution('gri30.yaml')

for i in range(N):
    eps, dp = porosity_and_dp(z[i])
    # solid conductivity correlation
    lambda_s[i] = 0.188 - 17.5 * dp
    # extinction coefficient
    kappa[i] = (3.0 / dp) * (1.0 - eps)

    # gas properties at local (T, X)
    prop_gas.TPX = Tg[i], p, X[:, i]
    rho = prop_gas.density
    mu  = prop_gas.viscosity
    k_g = prop_gas.thermal_conductivity

    # Re_p and h_v correlation
    C = -400.0 * dp + 0.687
    m = 443.7 * dp + 0.361
    Re_p = rho * eps * u[i] * dp / mu
    Nu_v = C * (Re_p ** m) if Re_p > 0 else 0.0
    h_v[i] = Nu_v * k_g / (dp ** 2)

# Assemble operator for conduction + convection: -d/dx(λ dTs/dx) + h_v*Ts = RHS
def assemble_matrix(z, lambda_s, h_v):
    N = len(z)
    A = np.zeros((N, N))

    # Neumann at x=0: T0 - T1 = 0
    A[0, 0] = 1.0
    A[0, 1] = -1.0

    # Interior
    def hmean(a, b):
        return 2.0*a*b/(a+b) if (a+b) != 0.0 else 0.0

    for i in range(1, N-1):
        dz_w = z[i] - z[i-1]
        dz_e = z[i+1] - z[i]
        lam_w = hmean(lambda_s[i-1], lambda_s[i])
        lam_e = hmean(lambda_s[i], lambda_s[i+1])
        a_w = lam_w / dz_w
        a_e = lam_e / dz_e

        A[i, i-1] = -a_w
        A[i, i]   = a_w + a_e + h_v[i]
        A[i, i+1] = -a_e

    # Neumann at x=L: TN-1 - TN-2 = 0
    A[N-1, N-1] = 1.0
    A[N-1, N-2] = -1.0
    return A

A = assemble_matrix(z, lambda_s, h_v)

# Fixed-point iteration because dq/dx depends on Ts
Ts = np.linspace(Tin, 1800.0, N)  # initial guess
max_iter = 50
rtol = 1e-6
alpha_relax = 0.5

for it in range(max_iter):
    dq_dx = solve_radiation(z, Ts, kappa, omega)  # W/m^3

    # RHS: h_v*Tg - dq_dx
    rhs = np.zeros(N)
    # Neumann BCs -> 0 at the boundaries
    for i in range(1, N-1):
        rhs[i] = h_v[i] * Tg[i] - dq_dx[i]

    Ts_new = np.linalg.solve(A, rhs)
    # Under-relaxation for robustness
    Ts_upd = alpha_relax * Ts_new + (1 - alpha_relax) * Ts
    err = np.linalg.norm(Ts_upd - Ts, ord=np.inf) / max(1.0, np.linalg.norm(Ts, ord=np.inf))
    Ts = Ts_upd
    if err < rtol:
        print(f"Radiative solid solve converged in {it+1} iterations; max rel change={err:.2e}")
        break
else:
    print(f"Radiative solid solve reached max_iter={max_iter}; max rel change={err:.2e}")

# ----------------
# 3) Visualization
# ----------------
plt.figure()
plt.plot(z * 100.0, Tg, '-o', label='Gas Temperature (Tg)')
plt.plot(z * 100.0, Ts, '-s', label='Solid Temperature (Ts)')
plt.xlabel('Position (cm)')
plt.ylabel('Temperature (K)')
plt.title('Porous Burner: Gas–Solid with S2 Radiation (post-processing)')
plt.axvline(x=3.5, color='grey', linestyle='--', label='Stage Interface')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('burner_temperature_profile_with_radiation.png', dpi=150)
print("Plot saved to 'burner_temperature_profile_with_radiation.png'")
plt.show()
