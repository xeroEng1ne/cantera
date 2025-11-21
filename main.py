"""
Main driver - Three-STAGE Porous Burner Optimization
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

def run_single_design(design_vars, verbose=False):
    """
    Run single three-stage design
    
    Args:
        design_vars: [dp2_mm, eps2, dp3_mm, eps3]
    """
    try:
        dp2, eps2, dp3, eps3 = design_vars
        
        # Validation (based on Horsman thesis ranges)
        if not (0.6 < dp2 < 1.6 and 0.86 < eps2 < 0.96 and
                0.5 < dp3 < 1.5 and 0.80 < eps3 < 0.93):
            return 1.0
        
        f = get_flame()
        
        # Initial solve
        z = f.grid
        Tg = f.T
        u = f.velocity
        p = f.gas.P
        X = f.X
        
        # Solve three-stage system
        solver = ThreeStageSolver(z, design_vars, Tg, u, p, X)
        _, Ts = solver.solve()
        
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
        _, Ts_final = solver_final.solve()
        
        # Check outlet temperature
        Ts_out = Ts_final[-1]
        
        # Lower threshold for three-stage (more stages = more complexity)
        if Ts_out < 500:
            if verbose:
                print(f"  Low solid outlet: {Ts_out:.0f}K")
            return 1.0
        
        # Radiant efficiency (Horsman Eq. in Chapter 5)
        T_ad = get_T_ad()
        eta = (Ts_out / T_ad) ** 4
        
        if not np.isfinite(eta) or eta <= 0 or eta > 1.5:
            if verbose:
                print(f"  Invalid eta: {eta:.4f}")
            return 1.0
        
        if verbose:
            print(f"  Ts_out={Ts_out:.0f}K, η={eta:.4f}")
        
        return -eta
    
    except Exception as e:
        if verbose:
            print(f"  Error: {str(e)[:60]}")
        return 1.0


def quick_optimization():
    """Grid search for three-stage design"""
    print("\n" + "="*70)
    print("THREE-STAGE POROUS RADIANT BURNER OPTIMIZATION")
    print("="*70 + "\n")
    
    # Coarse grid for initial testing
    dp2_range = np.linspace(0.8, 1.4, 3)
    eps2_range = np.linspace(0.88, 0.94, 2)
    dp3_range = np.linspace(0.7, 1.2, 3)
    eps3_range = np.linspace(0.83, 0.90, 2)
    
    best_eta = -1.0
    best_design = None
    results = []
    
    total = len(dp2_range) * len(eps2_range) * len(dp3_range) * len(eps3_range)
    
    with tqdm(total=total, desc="Evaluating") as pbar:
        for dp2 in dp2_range:
            for eps2 in eps2_range:
                for dp3 in dp3_range:
                    for eps3 in eps3_range:
                        design = [dp2, eps2, dp3, eps3]
                        eta_neg = run_single_design(design)
                        eta = -eta_neg
                        
                        if eta < 2.0 and eta > 0:
                            results.append((design, eta))
                            if eta > best_eta:
                                best_eta = eta
                                best_design = design
                        
                        pbar.update(1)
                        pbar.set_postfix(best_η=f"{best_eta:.4f}")
    
    return best_design, best_eta, results


def main():
    os.makedirs('output_threestage', exist_ok=True)

    print("\n" + "="*70)
    print("Initializing flame solver...")
    print("="*70)
    flame = get_flame()
    T_ad = get_T_ad()
    print(f"✓ Flame initialized successfully")
    print(f"  Grid points: {len(flame.grid)}")
    print(f"  Max temperature: {flame.T.max():.1f} K")
    print(f"  Adiabatic flame temp: {T_ad:.1f} K")
    print()
    
    # Test single case first
    print("Testing single three-stage design...")
    test_design = [1.0, 0.90, 1.2, 0.87]
    test_result = run_single_design(test_design, verbose=True)
    print(f"Test result: η = {-test_result:.4f}\n")
    
    # Run optimization
    best_design, best_eta, results = quick_optimization()
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    if best_design is not None:
        print(f"\nBest Three-Stage Design:")
        print(f"  Stage 2: dp2={best_design[0]:.3f}mm, eps2={best_design[1]:.4f}")
        print(f"  Stage 3: dp3={best_design[2]:.3f}mm, eps3={best_design[3]:.4f}")
        print(f"\nRadiant Efficiency: η = {best_eta:.4f}")
        print(f"Valid designs found: {len(results)}")
    else:
        print("No valid solution found!")
    
    print("="*70 + "\n")
    
    # Save results
    if results:
        results_sorted = sorted(results, key=lambda x: x[1], reverse=True)[:10]
        with open('output_threestage/results.txt', 'w') as f:
            f.write("Top 10 Three-Stage Designs:\n")
            for i, (design, eta) in enumerate(results_sorted):
                f.write(f"{i+1}. η={eta:.4f}, dp2={design[0]:.3f}, eps2={design[1]:.4f}, "
                       f"dp3={design[2]:.3f}, eps3={design[3]:.4f}\n")

if __name__ == "__main__":
    main()
