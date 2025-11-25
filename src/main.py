import numpy as np
from simulation import get_flame, ThreeStageSolver, get_T_ad
import matplotlib.pyplot as plt

def plot_temperature_profile(z, Tg, Ts, design_vars, efficiency, filename='temperature_profile.png'):
    z_cm = z * 100
    T_ad = get_T_ad()
    
    plt.figure(figsize=(10, 6))
    plt.plot(z_cm, Tg, 'r-', linewidth=2, label='Gas Temp ($T_g$)')
    plt.plot(z_cm, Ts, 'b--', linewidth=2, label='Solid Temp ($T_s$)')
    
    plt.axhline(y=T_ad, color='gray', linestyle=':', label=f'Adiabatic ({T_ad:.0f}K)')
    plt.axvline(x=1.5, color='green', linestyle=':', label='Preheat/Arrestor')
    plt.axvline(x=3.0, color='purple', linestyle=':', label='Arrestor/Combustion')
    
    plt.xlabel('Axial Position (cm)', fontsize=12)
    plt.ylabel('Temperature (K)', fontsize=12)
    plt.title(f'Symposia 3-Stage Burner (FAPA)\nRadiant Efficiency: {efficiency:.2%}', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    print(f"Saving plot to {filename}...")
    plt.savefig(filename, dpi=200)

def run_symposia_design():
    # --- 3-STAGE FAPA DESIGN ---
    # Stage 1: Preheater (dp=1.0mm, eps=0.40) -> High Conductivity
    # Stage 2: Arrestor  (dp=0.6mm, eps=0.40) -> Small pore, Flame stop
    # Stage 3: Radiant   (dp=5.0mm, eps=0.90) -> Large pore, Emission
    design = [1.0, 0.40, 0.6, 0.40, 5.0, 0.90]
    
    print(f"Running Symposia 3-Stage FAPA Simulation...")
    f = get_flame()
    z = f.grid; Tg_init = f.T; u = f.velocity; p = f.gas.P; X = f.X
    
    solver = ThreeStageSolver(z, design, Tg_init, u, p, X)
    efficiency, Ts, Tg = solver.solve()
    
    print("-" * 40)
    print(f"  Radiant Efficiency: {efficiency:.2%}")
    print(f"  Peak Gas Temp:      {np.max(Tg):.1f} K")
    print(f"  Peak Solid Temp:    {np.max(Ts):.1f} K")
    print("-" * 40)

    plot_temperature_profile(solver.z, Tg, Ts, design, efficiency)

if __name__ == "__main__":
    run_symposia_design()
