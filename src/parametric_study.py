import numpy as np
import matplotlib.pyplot as plt
from simulation import get_flame, ThreeStageSolver

def plot_sweep(results, param_name, filename):
    plt.figure(figsize=(10, 7))
    
    # Distinct colors for clarity
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (val, z, Tg, Ts) in enumerate(results):
        # Crop to relevant area for plotting
        mask = z <= 0.03
        z_plot = z[mask]
        Tg_plot = Tg[mask]
        Ts_plot = Ts[mask]
        
        c = colors[i % len(colors)]
        
        # Plot Gas (Solid Line)
        plt.plot(z_plot, Tg_plot, color=c, linestyle='-', linewidth=2, 
                 label=f'Gas ({param_name}={val})')
        # Plot Solid (Dashed Line)
        plt.plot(z_plot, Ts_plot, color=c, linestyle='--', linewidth=2, alpha=0.8,
                 label=f'Solid ({param_name}={val})')

    # Interface Line
    plt.axvline(x=0.022, color='k', linestyle=':', linewidth=1.2, label='Interface')

    plt.xlabel('Axial Position (m)', fontsize=12, fontweight='bold')
    plt.ylabel('Temperature (K)', fontsize=12, fontweight='bold')
    plt.title(f'Effect of {param_name} on Temperature Distribution', fontsize=14)
    
    # Place legend outside if it gets too crowded, or upper left
    plt.legend(loc='upper left', frameon=True, fancybox=False, fontsize=10, ncol=2)
    
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.xlim(0.004, 0.035)
    plt.ylim(200, 1800)
    
    plt.tight_layout()
    print(f"Saving sweep plot to {filename}...")
    plt.savefig(filename, dpi=300)

def run_parametric_study():
    print("Initializing Chemistry...")
    flame_ref = get_flame()
    
    # Baseline Design: [dp1, eps1, dp2, eps2, dp3, eps3]
    # Indices: 0=Pre_dp, 1=Pre_eps, 2=Arr_dp, 3=Arr_eps, 4=Rad_dp, 5=Rad_eps
    base_design = [1.3, 0.4, 0.6, 0.4, 5.0, 0.90] 

    # --- STUDY 1: PORE DIAMETER (Stage 3) ---
    print("\n--- Running Pore Diameter Sweep ---")
    pore_sizes = [4.0, 5.0, 6.0]
    results_dp = []
    
    for dp in pore_sizes:
        print(f"  Simulating dp3 = {dp} mm...")
        design = base_design.copy()
        design[4] = dp # Update Stage 3 Pore Diameter
        
        solver = ThreeStageSolver(None, design, flame_ref, None)
        eff, Ts, Tg = solver.solve()
        results_dp.append((dp, solver.z, Tg, Ts))
        
    plot_sweep(results_dp, "Pore Size (mm)", "effect_of_pore_diameter.png")

    # --- STUDY 2: POROSITY (Stage 3) ---
    print("\n--- Running Porosity Sweep ---")
    porosities = [0.80, 0.85, 0.90]
    results_eps = []
    
    for eps in porosities:
        print(f"  Simulating epsilon3 = {eps}...")
        design = base_design.copy()
        design[5] = eps # Update Stage 3 Porosity
        
        solver = ThreeStageSolver(None, design, flame_ref, None)
        eff, Ts, Tg = solver.solve()
        results_eps.append((eps, solver.z, Tg, Ts))
        
    plot_sweep(results_eps, "Porosity", "effect_of_porosity.png")
    print("\nStudy Complete.")

if __name__ == "__main__":
    run_parametric_study()
