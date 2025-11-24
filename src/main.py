import numpy as np
from simulation import get_flame, ThreeStageSolver, get_T_ad
import matplotlib.pyplot as plt

def plot_temperature_profile(z, Tg, Ts, design_vars, efficiency, filename='temperature_profile.png'):
    dp2, eps2, dp3, eps3 = design_vars
    z_cm = z * 100
    T_ad = get_T_ad()
    
    plt.figure(figsize=(10, 6))
    plt.plot(z_cm, Tg, 'r-', linewidth=2, label='Gas Temperature')
    plt.plot(z_cm, Ts, 'b--', linewidth=2, label='Solid Temperature')
    
    plt.axhline(y=T_ad, color='gray', linestyle='--', alpha=0.7, label=f'Adiabatic ({T_ad:.0f}K)')
    plt.axvline(x=2.0, color='green', linestyle=':', linewidth=1.5, label='Stage 1/2')
    plt.axvline(x=3.5, color='purple', linestyle=':', linewidth=1.5, label='Stage 2/3')
    
    plt.xlabel('Axial Position (cm)', fontsize=12)
    plt.ylabel('Temperature (K)', fontsize=12)
    
    # Fix: Use raw strings (r'') for LaTeX content to avoid syntax warnings
    title_text = (r'3-Stage Burner Profile' + '\n' + 
                  rf'Efficiency $\eta$={efficiency:.4f}' + '\n' + 
                  rf'($d_{{p2}}$={dp2}mm, $\epsilon_2$={eps2}, $d_{{p3}}$={dp3}mm, $\epsilon_3$={eps3})')
    
    plt.title(title_text, fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print(f"Saving plot to {filename}...")
    plt.savefig(filename, dpi=200)
    plt.close()

def run_single_design(design_vars, verbose=True, plot=True):
    dp2, eps2, dp3, eps3 = design_vars
    f = get_flame()
    z = f.grid; Tg_init = f.T; u = f.velocity; p = f.gas.P; X = f.X
    
    solver = ThreeStageSolver(z, design_vars, Tg_init, u, p, X)
    efficiency, Ts, Tg = solver.solve()
    
    if verbose:
        print("-" * 40)
        print(f"RESULTS for Design: dp2={dp2}mm, eps2={eps2}, dp3={dp3}mm, eps3={eps3}")
        print("-" * 40)
        print(f"  Radiant Efficiency: {efficiency:.2%}")
        print(f"  Peak Gas Temp:      {np.max(Tg):.1f} K")
        print(f"  Peak Solid Temp:    {np.max(Ts):.1f} K")
        print("-" * 40)

    if plot:
        plot_temperature_profile(solver.z, Tg, Ts, design_vars, efficiency)
    return efficiency

if __name__ == "__main__":
    print("Running single design test...")
    # Design vars: [dp2(mm), eps2, dp3(mm), eps3]
    design = [0.8, 0.85, 1.5, 0.90]
    run_single_design(design, verbose=True, plot=True)
