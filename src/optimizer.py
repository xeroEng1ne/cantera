"""
THREE-STAGE BURNER OPTIMIZER
Uses the corrected 'ThreeStageSolver' to find the design parameters
that MAXIMIZE Radiant Efficiency.
"""

import numpy as np
from scipy.optimize import minimize
from simulation import get_flame, ThreeStageSolver
import sys
import os

# --- Helper to suppress the chatty solver output ---
class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# Global Cache
_FLAME_CACHE = None
_Z_CACHE = None
_T_INIT_CACHE = None
_U_INIT_CACHE = None
_P_CACHE = None
_X_CACHE = None

def objective_function(x):
    """
    Objective: Minimize (-Efficiency)
    Variables x: [dp3_mm] (Optimizing Stage 3 Pore Size)
    """
    global _FLAME_CACHE, _Z_CACHE, _T_INIT_CACHE, _U_INIT_CACHE, _P_CACHE, _X_CACHE
    
    # Initialize base flame once
    if _FLAME_CACHE is None:
        print("Initializing Base Flame for Optimization...")
        with SuppressPrints():
            f = get_flame()
        _FLAME_CACHE = f
        _Z_CACHE = f.grid
        _T_INIT_CACHE = f.T
        _U_INIT_CACHE = f.velocity
        _P_CACHE = f.gas.P
        _X_CACHE = f.X

    # We fix Stage 2 properties and Optimize Stage 3
    # Design: [dp2=0.8, eps2=0.85, dp3=VARIABLE, eps3=0.90]
    dp3 = x[0]
    design_vars = [0.8, 0.85, dp3, 0.90]
    
    print(f"  Testing dp3 = {dp3:.2f} mm ... ", end="", flush=True)
    
    try:
        # Run solver silently
        with SuppressPrints():
            solver = ThreeStageSolver(_Z_CACHE, design_vars, _T_INIT_CACHE, _U_INIT_CACHE, _P_CACHE, _X_CACHE)
            efficiency, _, _ = solver.solve()
        
        print(f"Efficiency = {efficiency:.2%}")
        return -efficiency # Negative because we want to MAXIMIZE
        
    except Exception as e:
        print(f"Failed ({e})")
        return 0.0

def run_optimization():
    print("========================================")
    print("   STARTING PORE SIZE OPTIMIZATION")
    print("========================================")
    
    # Initial Guess: 4.0 mm
    x0 = [4.0]
    
    # Bounds: Search between 1.0mm and 6.0mm
    bounds = [(1.0, 6.0)]
    
    # Nelder-Mead is robust for this type of noise
    res = minimize(
        objective_function, 
        x0, 
        bounds=bounds,
        method='Nelder-Mead',
        options={'maxiter': 20, 'xatol': 0.1}
    )
    
    print("\n========================================")
    print("   OPTIMIZATION COMPLETE")
    print("========================================")
    print(f"Optimal Stage 3 Pore Diameter: {res.x[0]:.4f} mm")
    print(f"Maximum Efficiency: {-res.fun:.2%}")

if __name__ == "__main__":
    run_optimization()
