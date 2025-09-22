# main.py
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simulation import run_burner_simulation
from optimizer import RSMOptimizer

def plot_response_surfaces(plot_data_history, output_dir):
    """
    Generates and saves contour plots of the quadratic response surface for each iteration.
    """
    print("\n--- Generating Response Surface Plots ---")
    for data in plot_data_history:
        iteration = data['iteration']
        b = data['coefficients']
        center_x, center_y = data['center_point']
        step_x, step_y = data['step_size']
        bounds = data['bounds']

        # Create a grid of points to evaluate the surface model
        x_range = np.linspace(bounds['x_min'], bounds['x_max'], 100)
        y_range = np.linspace(bounds['y_min'], bounds['y_max'], 100)
        X, Y = np.meshgrid(x_range, y_range)

        # Evaluate the quadratic model at each grid point
        # Z = b0 + b1*x + b2*y + b3*x^2 + b4*y^2 + b5*x*y
        Z = b[0] + b[1]*X + b[2]*Y + b[3]*X**2 + b[4]*Y**2 + b[5]*X*Y
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Filled contour plot (like the colored background)
        contourf = ax.contourf(X, Y, -Z, levels=50, cmap='jet')
        # Line contour plot (the black lines)
        ax.contour(X, Y, -Z, levels=50, colors='black', linewidths=0.5)
        
        # Draw the white search box (trust region)
        rect = plt.Rectangle(
            (center_x - step_x, center_y - step_y),
            2 * step_x, 2 * step_y,
            fill=False, edgecolor='white', linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Draw the white dot for the center point
        ax.plot(center_x, center_y, 'w.', markersize=8)

        ax.set_xlabel('Pore Diameter, $d_{p,2}$ (mm)')
        ax.set_ylabel('Porosity, $\epsilon_2$')
        ax.set_title(f'Response Surface for Iteration {iteration}')
        ax.set_xlim(bounds['x_min'], bounds['x_max'])
        ax.set_ylim(bounds['y_min'], bounds['y_max'])

        # Save the figure
        plot_path = os.path.join(output_dir, f'response_surface_iter_{iteration:02d}.png')
        plt.savefig(plot_path)
        plt.close(fig)
        
    print(f"Saved {len(plot_data_history)} surface plots to '{output_dir}'.")

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
    history_f_values = np.array([item[1] for item in optimizer.history])
    iterations = np.arange(len(history_f_values))
    
    # Calculate the cumulative best (maximum) efficiency at each step
    best_f_history = np.minimum.accumulate(history_f_values)
    best_efficiencies = -best_f_history

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, best_efficiencies, 'o-', label='Best Efficiency Found')
    plt.xlabel('Iteration Number')
    plt.ylabel('Radiant Efficiency (η)')
    plt.title('Optimization Progress (Best-So-Far)')
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=best_efficiencies.min() - 0.02, top=best_efficiencies.max() + 0.02) # Adjust y-axis
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'optimization_history_best.png')
    plt.savefig(plot_path)
    print(f"\nBest-so-far optimization plot saved to '{plot_path}'")
    plot_response_surfaces(optimizer.plot_data_history,output_dir)

if __name__ == "__main__":
    main()
