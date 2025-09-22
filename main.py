# main.py
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simulation import run_burner_simulation
from optimizer import RSMOptimizer

def main():
    print("--- Porous Burner Optimization ---")
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use initial conditions and bounds from the thesis for 2-D optimization
    initial_guess = [1.52, 0.87]
    step_size = [0.075, 0.01] # Initial gamma values from Table 5.3
    bounds = [[0.69, 1.52], [0.865, 0.95]] # From section 5.3

    optimizer = RSMOptimizer(
        objective_fn=run_burner_simulation,
        initial_guess=initial_guess,
        step_size=step_size,
        bounds=bounds
    )

    print(f"Initial Efficiency: {-optimizer.best_f:.4f}")

    # Main optimization loop with progress bar
    max_iter = 15
    with tqdm(total=max_iter, desc="Main Optimization Loop") as pbar:
        for best_x, best_f in optimizer.optimize(max_iterations=max_iter):
            pbar.set_postfix(d=f"{best_x[0]:.3f}", eps=f"{best_x[1]:.3f}", eta=f"{-best_f:.4f}")
            pbar.update(1)

    # --- Results ---
    final_x, final_f = optimizer.best_x, optimizer.best_f
    print("\n--- Optimization Finished ---")
    print(f"Optimal Design: Pore Diameter = {final_x[0]:.4f} mm, Porosity = {final_x[1]:.4f}")
    print(f"Maximum Radiant Efficiency: {-final_f:.4f}")

    # Plotting
    # The history is a list of tuples (design_vector, objective_value).
# We unpack it into separate arrays for plotting.
    history_points = np.array([item[0] for item in optimizer.history])
    history_f_values = np.array([item[1] for item in optimizer.history])

    iterations = np.arange(len(history_f_values))
    efficiencies = -history_f_values

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, efficiencies, 'o-', label='Optimizer Path')
# Add a line for the best value found
    plt.axhline(y=-optimizer.best_f, color='r', linestyle='--', label=f'Best Found: {-optimizer.best_f:.4f}')
    plt.xlabel('Iteration Number')
    plt.ylabel('Radiant Efficiency (η)')
    plt.title('Optimization History')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'optimization_history.png')
    plt.savefig(plot_path)
    print(f"\nOptimization history plot saved to '{plot_path}'")

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, efficiencies, 'o-', label='Optimizer Path')
    plt.xlabel('Iteration Number')
    plt.ylabel('Radiant Efficiency (η)')
    plt.title('Optimization History')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'optimization_history.png')
    plt.savefig(plot_path)
    print(f"\nOptimization history plot saved to '{plot_path}'")

if __name__ == "__main__":
    main()
