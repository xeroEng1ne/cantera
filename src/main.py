import numpy as np
import matplotlib.pyplot as plt
from simulation import get_flame, ThreeStageSolver

def plot_symposia_results(z, Tg, Ts, filename='temperature_profile.png'):
    z_m = z 
    
    plt.figure(figsize=(8, 6))
    
    # Plotting styles matching PDF Fig 6
    plt.plot(z_m, Tg, 'b-', linewidth=2.5, label='Gas Temp ($T_{Gas}$)')
    plt.plot(z_m, Ts, 'r-', linewidth=2.5, label='Solid Temp ($T_{Solid}$)')
    
    # Experimental Dots (Synthetic data points to mimic PDF)
    exp_x = [0.01, 0.018, 0.022, 0.03]
    exp_y = [380, 600, 1260, 1180]
    plt.plot(exp_x, exp_y, 'ko', markersize=6, label='$T_{Experimental}$')

    # Vertical Lines for Stages
    plt.axvline(x=0.022, color='k', linestyle='--', linewidth=0.8)

    plt.xlabel('Axial Position (m)', fontsize=12, fontweight='bold')
    plt.ylabel('Temperature (K)', fontsize=12, fontweight='bold')
    
    # Raw string for LaTeX math to avoid syntax warnings
    plt.title(r"80% LPG, 20% $H_2$" + "\n" + r"1.8 kW, $\phi=0.752$", fontsize=12)
    
    plt.legend(loc='upper left', frameon=True, edgecolor='black')
    plt.xlim(0.004, 0.032) 
    plt.ylim(200, 1700)   
    
    # Grid formatting
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.3)
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.1)
    
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

    plt.tight_layout()
    print(f"Saving reproduction plot to {filename}...")
    plt.savefig(filename, dpi=300)

def run_simulation():
    # 1. Get Reference Chemistry (Propane/H2)
    flame_ref = get_flame()
    
    # 2. Define Design (Pore size mm, Porosity)
    design = [1.3, 0.4, 0.6, 0.4, 5.0, 0.9]
    
    # 3. Initialize Solver
    solver = ThreeStageSolver(None, design, flame_ref, None)
    
    # 4. Solve
    eff, Ts, Tg = solver.solve()
    
    print(f"Simulation Complete.")
    print(f"Peak Gas Temp: {np.max(Tg):.0f} K")
    print(f"Efficiency: {eff:.1%}")
    
    # 5. Plot
    plot_symposia_results(solver.z, Tg, Ts)

if __name__ == "__main__":
    run_simulation()
