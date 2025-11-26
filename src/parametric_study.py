import numpy as np
import matplotlib.pyplot as plt
from simulation import get_flame, ThreeStageSolver
import os

def plot_sweep(results, param_name, stage_name, filename):
    plt.figure(figsize=(10, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (val, z, Tg, Ts) in enumerate(results):
        # Crop to relevant area
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
    plt.axvline(x=0.022, color='k', linestyle=':', linewidth=1.2, label='Reaction Interface')

    plt.xlabel('Axial Position (m)', fontsize=12, fontweight='bold')
    plt.ylabel('Temperature (K)', fontsize=12, fontweight='bold')
    plt.title(f'Effect of {stage_name} {param_name} on Temperature', fontsize=14)
    
    plt.legend(loc='upper left', frameon=True, fancybox=False, fontsize=10, ncol=2)
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.xlim(0.004, 0.03)
    plt.ylim(200, 1800)
    
    plt.tight_layout()
    print(f"Saving sweep plot to {filename}...")
    plt.savefig(filename, dpi=300)
    plt.close()

def run_sweep(flame_ref, base_design, param_index, param_values, param_label, stage_label, filename):
    """
    Generic function to run a parameter sweep.
    """
    print(f"\n--- Running {stage_label} {param_label} Sweep ---")
    results = []
    
    for val in param_values:
        print(f"  Simulating {param_label} = {val}...")
        
        # Copy baseline and update specific parameter
        design = base_design.copy()
        design[param_index] = val 
        
        try:
            solver = ThreeStageSolver(None, design, flame_ref, None)
            eff, Ts, Tg = solver.solve()
            results.append((val, solver.z, Tg, Ts))
        except Exception as e:
            print(f"  Failed to solve for {val}: {e}")
            
    plot_sweep(results, param_label, stage_label, filename)

def run_parametric_study():
    print("Initializing Chemistry...")
    flame_ref = get_flame()
    
    # Baseline Design: [dp1, eps1, dp2, eps2, dp3, eps3]
    # Indices:
    # 0: Stage 1 (Preheat) Pore Diameter (mm)
    # 1: Stage 1 (Preheat) Porosity
    # 2: Stage 2 (Arrestor) Pore Diameter (mm)
    # 3: Stage 2 (Arrestor) Porosity
    # 4: Stage 3 (Radiant) Pore Diameter (mm)
    # 5: Stage 3 (Radiant) Porosity
    base_design = [1.3, 0.4, 0.6, 0.4, 5.0, 0.90] 

    # --- STAGE 1 SWEEPS ---
    # Pore Diameter
    run_sweep(flame_ref, base_design, 
              param_index=0, 
              param_values=[1.0, 1.3, 1.6], 
              param_label="d_p1 (mm)", 
              stage_label="Stage 1", 
              filename="sweep_stage1_dp.png")
    
    # Porosity
    run_sweep(flame_ref, base_design, 
              param_index=1, 
              param_values=[0.3, 0.4, 0.5], 
              param_label="epsilon1", 
              stage_label="Stage 1", 
              filename="sweep_stage1_eps.png")

    # --- STAGE 2 SWEEPS ---
    # Pore Diameter
    run_sweep(flame_ref, base_design, 
              param_index=2, 
              param_values=[0.4, 0.6, 0.8], 
              param_label="d_p2 (mm)", 
              stage_label="Stage 2", 
              filename="sweep_stage2_dp.png")
    
    # Porosity
    run_sweep(flame_ref, base_design, 
              param_index=3, 
              param_values=[0.3, 0.4, 0.5], 
              param_label="epsilon2", 
              stage_label="Stage 2", 
              filename="sweep_stage2_eps.png")

    # --- STAGE 3 SWEEPS (Original) ---
    # Pore Diameter
    run_sweep(flame_ref, base_design, 
              param_index=4, 
              param_values=[4.0, 5.0, 6.0], 
              param_label="d_p3 (mm)", 
              stage_label="Stage 3", 
              filename="sweep_stage3_dp.png")
    
    # Porosity
    run_sweep(flame_ref, base_design, 
              param_index=5, 
              param_values=[0.80, 0.85, 0.90], 
              param_label="epsilon3", 
              stage_label="Stage 3", 
              filename="sweep_stage3_eps.png")

    print("\nAll Parametric Studies Complete.")

if __name__ == "__main__":
    run_parametric_study()
