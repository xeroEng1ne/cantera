# main_three_phase_fixed.py
"""
Main driver - simplified and faster
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from simulation import (
    get_flame,
    SimplifiedThreePhaseSolver,
    solve_gas_phase_threephase,
    get_T_ad
)


def run_threephase_simulation(design_vars):
    """
    Simplified three-phase simulation
    Sequential solver is much more stable
    """
    try:
        dp2_mm, eps2, dp3_mm, eps3 = design_vars
        
        # Validate bounds
        if not (0.6 < dp2_mm < 1.6 and 0.86 < eps2 < 0.96 and 
                0.5 < dp3_mm < 1.5 and 0.80 < eps3 < 0.93):
            return 1.0
        
        f = get_flame()
        
        # Outer coupling loop (reduced iterations)
        max_outer = 5
        for outer in range(max_outer):
            Tg_old = f.T.copy()
            
            # Subsample for speed
            stride = 3 if outer < 2 else 2
            idx = np.arange(0, len(f.grid), stride)
            
            gas_dict = {
                'z': f.grid[idx],
                'T': f.T[idx],
                'u': f.velocity[idx],
                'X': f.X[:, idx],
                'p': f.gas.P
            }
            
            # Solve three-phase system
            solver = SimplifiedThreePhaseSolver(gas_dict['z'], design_vars, gas_dict)
            Tg_sub, Ts_sub, Tl_sub = solver.solve()
            
            # Interpolate back
            Ts_full = np.interp(f.grid, gas_dict['z'], Ts_sub)
            
            # Continuation
            for fac in [0.5, 0.8, 1.0]:
                Ttarget = np.clip((1-fac)*f.T + fac*Ts_full, 300, 2200)
                f = solve_gas_phase_threephase(f, T_target=Ttarget, loglevel=0, refine=False)
            
            # Convergence check
            if len(f.T) == len(Tg_old):
                err = np.linalg.norm(f.T - Tg_old)
                if err < 3.0:
                    break
        
        # Final solution
        gas_dict_final = {
            'z': f.grid,
            'T': f.T,
            'u': f.velocity,
            'X': f.X,
            'p': f.gas.P
        }
        
        solver_final = SimplifiedThreePhaseSolver(f.grid, design_vars, gas_dict_final)
        _, Ts_final, _ = solver_final.solve()
        
        # Efficiency
        T_ad = get_T_ad()
        Ts_out = Ts_final[-1]
        
        if not (600 < Ts_out < 2300):
            return 1.0
        
        eta = (Ts_out / T_ad) ** 4
        
        if not np.isfinite(eta) or eta <= 0:
            return 1.0
        
        print(f"  ✓ Ts_out={Ts_out:.0f}K, η={eta:.4f} | dp2={dp2_mm:.2f}, ε2={eps2:.3f}, dp3={dp3_mm:.2f}, ε3={eps3:.3f}")
        return -eta
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 1.0


def simple_grid_search():
    """
    Simple grid search instead of complex RSM
    Much faster and more reliable
    """
    print("Starting 3-Phase Grid Search Optimization\n")
    
    # Coarse grid
    dp2_range = np.linspace(0.8, 1.5, 5)
    eps2_range = np.linspace(0.87, 0.94, 4)
    dp3_range = np.linspace(0.7, 1.3, 5)
    eps3_range = np.linspace(0.82, 0.90, 4)
    
    best_eta = -1.0
    best_design = None
    results = []
    
    total = len(dp2_range) * len(eps2_range) * len(dp3_range) * len(eps3_range)
    
    with tqdm(total=total, desc="Grid Search") as pbar:
        for dp2 in dp2_range:
            for eps2 in eps2_range:
                for dp3 in dp3_range:
                    for eps3 in eps3_range:
                        design = [dp2, eps2, dp3, eps3]
                        eta_neg = run_threephase_simulation(design)
                        eta = -eta_neg
                        
                        results.append((design, eta))
                        
                        if eta > best_eta and eta < 1.5:
                            best_eta = eta
                            best_design = design
                        
                        pbar.update(1)
                        pbar.set_postfix(best_η=f"{best_eta:.4f}")
    
    return best_design, best_eta, results


def main():
    """Main optimization"""
    os.makedirs('output_three_phase_simple', exist_ok=True)
    
    best_design, best_eta, results = simple_grid_search()
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    
    if best_design is not None:
        print(f"Optimal Design:")
        print(f"  Layer 2: dp={best_design[0]:.3f} mm, ε={best_design[1]:.4f}")
        print(f"  Layer 3: dp={best_design[2]:.3f} mm, ε={best_design[3]:.4f}")
        print(f"Maximum Efficiency: η = {best_eta:.4f}")
    else:
        print("No valid solution found!")
    
    print("="*60)
    
    # Save results
    np.save('output_three_phase_simple/results.npy', results)
    
    # Plot top 20 designs
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)[:20]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    etas = [r[1] for r in results_sorted]
    ax.bar(range(len(etas)), etas)
    ax.set_xlabel('Design Rank')
    ax.set_ylabel('Efficiency η')
    ax.set_title('Top 20 Designs')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output_three_phase_simple/top_designs.png', dpi=150)
    plt.close()
    
    print(f"\nResults saved to: output_three_phase_simple/")


if __name__ == "__main__":
    main()
