"""
Main driver - Three-STAGE Porous Burner Optimization (CUSTOM CONFIGURATION)
Stage 1: 1.2 cm, 40% porosity (FIXED)
Stage 2: 0.5 cm, 10% porosity (optimizable)
Stage 3: 1.0 cm, 90% porosity (optimizable)
"""

import os
import numpy as np
from tqdm import tqdm
from simulation import (
    get_flame,
    ThreeStageSolver,
    solve_gas_phase,
    get_T_ad
)
import matplotlib.pyplot as plt


def plot_temperature_profile(z, Tg, Ts, design_vars, efficiency, filename='temperature_profile.png'):
    """
    Plot temperature profiles for the three-stage burner

    Args:
        z: Grid points [m]
        Tg: Gas temperature [K]
        Ts: Solid temperature [K]
        design_vars: [dp2_mm, eps2, dp3_mm, eps3]
        efficiency: Radiant efficiency
        filename: Output filename
    """
    dp2, eps2, dp3, eps3 = design_vars
    z_cm = z * 100  # Convert to cm
    T_ad = get_T_ad()

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Temperature profiles
    ax1 = axes[0]
    ax1.plot(z_cm, Tg, 'r-', linewidth=2.5, label='Gas Temperature', marker='o', markersize=4)
    ax1.plot(z_cm, Ts, 'b-', linewidth=2.5, label='Solid Temperature', marker='s', markersize=4)
    ax1.axhline(y=T_ad, color='gray', linestyle='--', linewidth=2, label=f'Adiabatic Temp ({T_ad:.0f} K)')

    # CUSTOM Stage boundaries (1.2 cm, 1.7 cm)
    ax1.axvline(x=1.2, color='green', linestyle=':', linewidth=2.5, alpha=0.7, label='Stage Boundaries')
    ax1.axvline(x=1.7, color='green', linestyle=':', linewidth=2.5, alpha=0.7)

    # Add stage labels with custom porosities
    ax1.text(0.6, max(Tg)*0.95, 'Stage 1\nPreheating\nε=40%\n(Fixed)', 
             ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax1.text(1.45, max(Tg)*0.95, f'Stage 2\nCombustion\nε={eps2*100:.0f}%\ndp={dp2:.2f}mm', 
             ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    ax1.text(2.2, max(Tg)*0.95, f'Stage 3\nRadiation\nε={eps3*100:.0f}%\ndp={dp3:.2f}mm', 
             ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax1.set_xlabel('Axial Position (cm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Temperature (K)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Three-Stage Porous Burner (Custom Config: 1.2cm/0.5cm/1.0cm) | η = {efficiency:.4f}', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(z_cm))

    # Plot 2: Temperature difference (heat transfer driving force)
    ax2 = axes[1]
    delta_T = Tg - Ts
    ax2.plot(z_cm, delta_T, 'purple', linewidth=2.5, marker='o', markersize=4)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(x=1.2, color='green', linestyle=':', linewidth=2.5, alpha=0.7)
    ax2.axvline(x=1.7, color='green', linestyle=':', linewidth=2.5, alpha=0.7)

    ax2.set_xlabel('Axial Position (cm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ΔT = T_gas - T_solid (K)', fontsize=12, fontweight='bold')
    ax2.set_title('Gas-Solid Temperature Difference (Heat Transfer Driving Force)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(z_cm))

    # Plot 3: Both temperatures on same axis with fill
    ax3 = axes[2]
    ax3.plot(z_cm, Tg, 'r-', linewidth=2.5, label='Gas', marker='o', markersize=4)
    ax3.plot(z_cm, Ts, 'b-', linewidth=2.5, label='Solid', marker='s', markersize=4)
    ax3.fill_between(z_cm, Tg, Ts, alpha=0.3, color='orange', label='Heat Transfer Zone')
    ax3.axvline(x=1.2, color='green', linestyle=':', linewidth=2.5, alpha=0.7)
    ax3.axvline(x=1.7, color='green', linestyle=':', linewidth=2.5, alpha=0.7)

    # Add text annotations for key temperatures
    ax3.text(max(z_cm)*0.98, Ts[-1], f'T_s,out={Ts[-1]:.0f}K', 
             ha='right', va='bottom', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax3.text(max(z_cm)*0.98, Tg[-1], f'T_g,out={Tg[-1]:.0f}K', 
             ha='right', va='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    ax3.set_xlabel('Axial Position (cm)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Temperature (K)', fontsize=12, fontweight='bold')
    ax3.set_title('Temperature Profiles with Heat Transfer Visualization', 
                  fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max(z_cm))

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Temperature profile saved: {filename}")
    plt.close()


def run_single_design(design_vars, verbose=False, plot=False):
    """
    Run single three-stage design
    Args:
        design_vars: [dp2_mm, eps2, dp3_mm, eps3]
        verbose: Print detailed output
        plot: Generate temperature profile plot
    """
    try:
        dp2, eps2, dp3, eps3 = design_vars

        # Validation ranges (FIXED: inclusive boundaries)
        # Stage 2: ~10% porosity (0.07-0.13)
        # Stage 3: ~90% porosity (0.87-0.93)
        if not (0.4 <= dp2 <= 1.6 and 0.07 <= eps2 <= 0.13 and
                0.4 <= dp3 <= 1.6 and 0.87 <= eps3 <= 0.93):
            return -1.0

        f = get_flame()

        # Initial solve
        z = f.grid
        Tg = f.T
        u = f.velocity
        p = f.gas.P
        X = f.X

        # Solve three-stage system
        solver = ThreeStageSolver(z, design_vars, Tg, u, p, X)
        eta, Ts = solver.solve()

        # Update gas phase
        Ttarget = np.clip(0.6*Tg + 0.4*Ts, 300, 2200)
        f = solve_gas_phase(f, T_target=Ttarget, loglevel=0)

        # Final solve
        z = f.grid
        Tg_final = f.T
        u_final = f.velocity
        X_final = f.X
        p_final = f.gas.P

        solver_final = ThreeStageSolver(z, design_vars, Tg_final, u_final, p_final, X_final)
        efficiency, Ts_final = solver_final.solve()

        if verbose:
            T_ad = get_T_ad()
            print(f"\n{'='*60}")
            print(f"Design: dp2={dp2:.3f}mm, eps2={eps2*100:.1f}%, dp3={dp3:.3f}mm, eps3={eps3*100:.1f}%")
            print(f"{'='*60}")
            print(f"  Max gas temp:     {np.max(Tg_final):.1f} K")
            print(f"  Solid outlet:     {Ts_final[-1]:.1f} K")
            print(f"  Adiabatic temp:   {T_ad:.1f} K")
            print(f"  Radiant efficiency: {efficiency:.4f}")
            print(f"  Formula: η = (T_s,out / T_ad)^4 = ({Ts_final[-1]:.1f}/{T_ad:.1f})^4")

        # Generate plot if requested
        if plot:
            plot_temperature_profile(z, Tg_final, Ts_final, design_vars, efficiency)

        return efficiency

    except Exception as e:
        if verbose:
            print(f"ERROR: {e}")
        return -1.0


def run_optimization():
    """Run grid search optimization"""
    print("\n" + "="*60)
    print("THREE-STAGE POROUS BURNER OPTIMIZATION")
    print("CUSTOM CONFIGURATION")
    print("="*60)
    print("\nStage Configuration:")
    print("  Stage 1: 0.0 - 1.2 cm  | Porosity: 40% (FIXED)")
    print("  Stage 2: 1.2 - 1.7 cm  | Porosity: ~10% (optimizable)")
    print("  Stage 3: 1.7 - 2.7 cm  | Porosity: ~90% (optimizable)")

    # Initialize flame once
    print("\nInitializing gas-phase flame (one-time setup)...")
    _ = get_flame()
    print("  ✓ Flame initialized")

    # Test single design first
    print("\n" + "-"*60)
    print("TESTING SINGLE DESIGN")
    print("-"*60)
    test_design = [0.8, 0.10, 1.0, 0.90]  # Custom porosities: 10% and 90%
    print(f"Test design: dp2={test_design[0]}mm, eps2={test_design[1]*100:.0f}%, "
          f"dp3={test_design[2]}mm, eps3={test_design[3]*100:.0f}%")

    eta_test = run_single_design(test_design, verbose=True, plot=True)

    if eta_test < 0:
        print("\n❌ Test failed! Check configuration.")
        return

    print(f"\n✓ Test passed: η = {eta_test:.4f}")

    # Grid search with custom porosity ranges
    print("\n" + "-"*60)
    print("GRID SEARCH OPTIMIZATION")
    print("-"*60)

    # Search around 10% for Stage 2, 90% for Stage 3
    dp2_range = np.linspace(0.6, 1.2, 4)
    eps2_range = np.linspace(0.08, 0.12, 3)  # 8% to 12%
    dp3_range = np.linspace(0.7, 1.2, 4)
    eps3_range = np.linspace(0.88, 0.92, 3)  # 88% to 92%

    designs = [(dp2, eps2, dp3, eps3) 
               for dp2 in dp2_range 
               for eps2 in eps2_range 
               for dp3 in dp3_range 
               for eps3 in eps3_range]

    print(f"\nSearching {len(designs)} designs...")
    print(f"  Stage 2 porosity range: {eps2_range[0]*100:.0f}% - {eps2_range[-1]*100:.0f}%")
    print(f"  Stage 3 porosity range: {eps3_range[0]*100:.0f}% - {eps3_range[-1]*100:.0f}%")

    results = []
    for design in tqdm(designs, desc="Evaluating"):
        eta = run_single_design(design, verbose=False, plot=False)
        results.append((eta, design))

    # Filter out failed designs and find best
    valid_results = [(eta, design) for eta, design in results if eta > 0]

    if not valid_results:
        print("\n❌ All designs failed! Check parameter ranges.")
        return

    valid_results.sort(reverse=True, key=lambda x: x[0])
    best_eta, best_design = valid_results[0]

    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"\nBest efficiency: {best_eta:.4f}")
    print(f"Optimal design:")
    print(f"  Stage 2: dp2 = {best_design[0]:.3f} mm, eps2 = {best_design[1]*100:.1f}%")
    print(f"  Stage 3: dp3 = {best_design[2]:.3f} mm, eps3 = {best_design[3]*100:.1f}%")
    print(f"\nValid designs: {len(valid_results)}/{len(designs)}")

    # Plot best design
    print(f"\nGenerating plot for optimal design...")
    run_single_design(best_design, verbose=True, plot=True)

    # Show top 5
    print(f"\nTop 5 designs:")
    for i, (eta, design) in enumerate(valid_results[:5], 1):
        print(f"  {i}. η={eta:.4f} | dp2={design[0]:.2f}mm, eps2={design[1]*100:.0f}%, "
              f"dp3={design[2]:.2f}mm, eps3={design[3]*100:.0f}%")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Quick test mode
        print("Running single design test with visualization...")
        print("\nCustom Configuration:")
        print("  Stage 1: 1.2 cm, 40% porosity")
        print("  Stage 2: 0.5 cm, 10% porosity")
        print("  Stage 3: 1.0 cm, 90% porosity")
        _ = get_flame()
        design = [0.8, 0.10, 1.0, 0.90]  # dp2, eps2, dp3, eps3
        run_single_design(design, verbose=True, plot=True)
    else:
        # Full optimization
        run_optimization()
