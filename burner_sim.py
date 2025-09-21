# burner_sim_corrected.py
# Requires: cantera >= 3.0, numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

# -----------------------------
# 1) Gas-phase free flame solve
# -----------------------------
gas = ct.Solution('gri30.yaml')

inlet_temp = 300.0
pressure = ct.one_atm
equivalence_ratio = 0.65
inlet_velocity = 0.45  # used only for inlet mdot

# Set inlet gas state
gas.TP = inlet_temp, pressure
gas.set_equivalence_ratio(phi=equivalence_ratio, fuel='CH4',
                          oxidizer={'O2': 0.21, 'N2': 0.78, 'Ar': 0.01})

width = 0.0605  # m
flame = ct.FreeFlame(gas, width=width)

# Set inlet boundary conditions
flame.inlet.T = inlet_temp
flame.inlet.X = gas.X
flame.inlet.mdot = gas.density * inlet_velocity

# Solve free flame
flame.set_refine_criteria(ratio=3, slope=0.1, curve=0.2)
flame.solve(loglevel=1, auto=True)

# Extract grid and gas solution needed for post-processing
z = flame.grid.copy()                     # m
Tg = flame.T.copy()                       # K
u_gas = flame.velocity.copy()             # m/s
p = pressure                              # Pa (FreeFlame is constant-p)

# -----------------------------------------------------------
# 2) Compute Ts by solving: d/dx(λs dTs/dx) + h_v*(Tg - Ts) = 0
#    with insulated boundaries: dTs/dx|x=0 = dTs/dx|x=L = 0
# -----------------------------------------------------------

# Helper: piecewise/transition properties for porosity and particle size dp
def porosity_and_dp(x):
    # Transition from 0.033 m to 0.037 m (4 mm) with linear blending
    porosity1, dp1 = 0.835, 0.00029
    porosity2, dp2 = 0.87, 0.00152
    if x < 0.033:
        return porosity1, dp1
    elif x > 0.037:
        return porosity2, dp2
    else:
        frac = (x - 0.033) / 0.004
        return (porosity1 + frac * (porosity2 - porosity1),
                dp1 + frac * (dp2 - dp1))

N = len(z)
lambda_s = np.zeros(N)   # solid conductivity profile
h_v = np.zeros(N)        # volumetric heat transfer coefficient profile

# Reuse a Cantera gas object for transport properties; composition is held at the inlet mix
prop_gas = ct.Solution('gri30.yaml')
prop_gas.TPX = inlet_temp, p, gas.X

for i in range(N):
    porosity, dp = porosity_and_dp(z[i])

    # Update gas state at local temperature (composition held at inlet mix; replace
    # with local composition if available in the workflow for higher fidelity)
    prop_gas.TP = Tg[i], p

    rho = prop_gas.density
    mu = prop_gas.viscosity
    k_g = prop_gas.thermal_conductivity

    # Correlations from the user’s formulation
    # Solid conductivity (Eq. 2.2): lambda_s = 0.188 - 17.5 * dp
    lambda_s[i] = 0.188 - 17.5 * dp

    # Volumetric heat transfer coefficient:
    # Re_p = rho * porosity * u * dp / mu
    # Nu_v = C * Re_p**m (zero if Re_p <= 0)
    # h_v = Nu_v * k_g / dp**2
    C = -400.0 * dp + 0.687
    m = 443.7 * dp + 0.361
    Re_p = rho * porosity * u_gas[i] * dp / mu
    Nu_v = C * (Re_p ** m) if Re_p > 0 else 0.0
    h_v[i] = Nu_v * k_g / (dp ** 2)

# Assemble tridiagonal linear system for Ts on nonuniform grid:
# -d/dx(λ dTs/dx) + h_v * Ts = h_v * Tg   (interior)
# Neumann BCs (insulated): T0 - T1 = 0,  TN-1 - TN-2 = 0

A = np.zeros((N, N))
rhs = np.zeros(N)

def harmonic_mean(a, b):
    return 2.0 * a * b / (a + b) if (a + b) != 0.0 else 0.0

# Left boundary (Neumann: dT/dx = 0) -> T0 - T1 = 0
A[0, 0] = 1.0
A[0, 1] = -1.0
rhs[0] = 0.0

# Interior points
for i in range(1, N - 1):
    dx_w = z[i]   - z[i - 1]
    dx_e = z[i + 1] - z[i]

    lam_w = harmonic_mean(lambda_s[i - 1], lambda_s[i])
    lam_e = harmonic_mean(lambda_s[i], lambda_s[i + 1])

    a_w = lam_w / dx_w
    a_e = lam_e / dx_e

    A[i, i - 1] = -a_w
    A[i, i]     = a_w + a_e + h_v[i]
    A[i, i + 1] = -a_e

    rhs[i] = h_v[i] * Tg[i]

# Right boundary (Neumann: dT/dx = 0) -> TN-1 - TN-2 = 0
A[N - 1, N - 1] = 1.0
A[N - 1, N - 2] = -1.0
rhs[N - 1] = 0.0

# Solve linear system
Ts = np.linalg.solve(A, rhs)

# ----------------
# 3) Visualization
# ----------------
plt.figure()
plt.plot(z * 100.0, Tg, '-o', label='Gas Temperature (Tg)')
plt.plot(z * 100.0, Ts, '-s', label='Solid Temperature (Ts)')
plt.xlabel('Position (cm)')
plt.ylabel('Temperature (K)')
plt.title('Porous Burner: Gas–Solid Temperature Profiles')
plt.axvline(x=3.5, color='grey', linestyle='--', label='Stage Interface')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('burner_temperature_profile.png', dpi=150)
print("Plot saved to 'burner_temperature_profile.png'")
plt.show()
