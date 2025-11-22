import numpy as np
from simulation import get_flame, ThreeStageSolver, get_T_ad
import matplotlib.pyplot as plt

def plot_temperature_profile(z, Tg, Ts, design_vars, efficiency, filename='temperature_profile.png'):
    dp2, eps2, dp3, eps3 = design_vars
    z_cm = z * 100
    T_ad = get_T_ad()
    plt.figure(figsize=(8,6))
    plt.plot(z_cm, Tg, 'r-', label='Gas Temperature')
    plt.plot(z_cm, Ts, 'b-', label='Solid Temperature')
    plt.axhline(y=T_ad, color='gray', linestyle='--', label=f'Adiabatic ({T_ad:.0f}K)')
    plt.axvline(x=1.2, color='green', linestyle=':', label='Stage 1/2')
    plt.axvline(x=1.7, color='green', linestyle=':', label='Stage 2/3')
    plt.xlabel('Axial Position (cm)')
    plt.ylabel('Temperature (K)')
    plt.title(f'Temperature Profiles, η={efficiency:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def run_single_design(design_vars, verbose=True, plot=True):
    dp2, eps2, dp3, eps3 = design_vars
    f = get_flame()
    z = f.grid; Tg_init = f.T; u = f.velocity; p = f.gas.P; X = f.X
    solver = ThreeStageSolver(z, design_vars, Tg_init, u, p, X)
    efficiency, Ts, Tg = solver.solve()
    if verbose:
        print(f"Design: dp2={dp2:.3f}mm, eps2={eps2*100:.1f}%, dp3={dp3:.3f}mm, eps3={eps3*100:.1f}%")
        print(f"  Max gas temp: {np.max(Tg):.1f} K at {z[np.argmax(Tg)]*100:.2f} cm")
        print(f"  Max solid temp: {np.max(Ts):.1f} K")
        print(f"  Gas outlet: {Tg[-1]:.1f} K, Solid outlet: {Ts[-1]:.1f} K")
        print(f"  Adiabatic temp: {get_T_ad():.1f} K")
        print(f"  Radiant efficiency: {efficiency:.4f}")
    if plot:
        plot_temperature_profile(z, Tg, Ts, design_vars, efficiency)
    return efficiency

if __name__ == "__main__":
    print("Running single design test with fast grid and coupled solver...")
    design = [0.8, 0.10, 1.0, 0.90]
    run_single_design(design, verbose=True, plot=True)
