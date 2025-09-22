import numpy as np
import cantera as ct
from simulation import run_burner_simulation
from optimizer import RSMOptimizer
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    # --- Create Output Directory ---
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # --- Optimization Setup ---
    initial_guess = (1.52, 0.87)          # (pore_diameter_mm, porosity_stage2)
    bounds = [(0.69, 1.52), (0.865, 0.95)]
    initial_step = (0.075, 0.01)

    # Preload (unused by simulation, kept for API compatibility)
    gas = ct.Solution('gri30.yaml')

    print("--- Porous Burner Optimization ---")
    optimizer = RSMOptimizer(
        objective_function=run_burner_simulation,
        initial_guess=initial_guess,
        bounds=bounds,
        initial_step=initial_step,
        gas_obj=gas,
        n_jobs=-1  # parallelize the 9 stencil evaluations; set to 1 to disable
    )
    print(f"Initial Efficiency: {optimizer.history[0]['f']:.4f}")

    # --- Main Optimization Loop ---
    max_iterations = 15
    min_iterations = 6
    rel_tol = 1e-3

    for i in tqdm(range(max_iterations), desc="Main Optimization Loop"):
        prev_best = optimizer.best_f
        x_prev = optimizer.x_current.copy()
        x_new, f_new = optimizer.step()

        print(f"\n--- Iteration {i+1} ---")
        print(f"  Best point so far: d={optimizer.best_x[0]:.4f} mm, ε={optimizer.best_x[1]:.4f}")
        print(f"  Best efficiency so far: {-optimizer.best_f:.4f}")

        rel_impr = abs(prev_best - optimizer.best_f) / max(1.0, abs(prev_best))
        if (i + 1) >= min_iterations and rel_impr < rel_tol:
            print("\nConvergence reached: relative improvement below threshold.")
            break
    else:
        print("\nMax iterations reached.")

    print("\n--- Optimization Finished ---")
    print(f"Optimal Design: Pore Diameter = {optimizer.best_x[0]:.4f} mm, Porosity = {optimizer.best_x[1]:.4f}")
    print(f"Maximum Radiant Efficiency: {-optimizer.best_f:.4f}")

    # --- Plot Optimization History ---
    iterations = np.arange(len(optimizer.history))
    efficiencies = [h['f'] for h in optimizer.history]
    plt.figure()
    plt.plot(iterations, efficiencies, 'o-', ms=6)
    plt.xlabel("Iteration")
    plt.ylabel("Radiant Efficiency")
    plt.title("Optimization History")
    plt.grid(True)
    plt.ylim(0, max(1e-3, 1.1 * max(efficiencies)))
    plt.savefig(os.path.join(output_dir, 'optimization_history.png'), dpi=150)
    print(f"\nOptimization history plot saved to '{output_dir}/optimization_history.png'")

if __name__ == "__main__":
    main()
