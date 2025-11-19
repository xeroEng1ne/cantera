# main.py
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cantera as ct

from simulation import get_flame, solve_solid_phase, solve_gas_phase, get_T_ad
from optimizer import RSMOptimizer

def run_coupled_simulation(design_vars):
    """
    Double-loop continuation coupling with speed optimizations:
      - Subsample solid grid early (stride 4 → 3 → 2),
      - Fixed-T continuation toward solid: fac in {0.6, 0.8, 1.0} with clamp,
      - Disabled grid refinement during sweeps,
      - Thesis metric η = (Ts_out/T_ad)^4 using cached T_ad.
    Returns negative efficiency for minimization.
    """
    try:
        f = get_flame()            # cached BurnerFlame reused across calls
        max_outer, tolK = 6, 1.5   # K (L2) convergence

        for outer in range(max_outer):
            Tg_old = f.T.copy()

            # Coarse grid early for speed, finer near convergence
            stride = 4 if outer < 2 else (3 if outer < 4 else 2)
            idx = np.arange(0, len(f.grid), stride)

            gas_dict_sub = {
                'z': f.grid[idx],
                'T': f.T[idx],
                'u': f.velocity[idx],
                'X': f.X[:, idx],
                'p': f.gas.P
            }
            Ts_sub, _ = solve_solid_phase(gas_dict_sub, design_vars)
            if Ts_sub is None:
                return 1.0

            # Interpolate Ts back to full grid
            Ts = np.interp(f.grid, gas_dict_sub['z'], Ts_sub)

            # Continuation toward solid with clamped target
            for fac in (0.6, 0.8, 1.0):
                Ttarget = np.clip((1.0 - fac) * f.T + fac * Ts, 300.0, 2300.0)
                f = solve_gas_phase(f, T_target=Ttarget, loglevel=0, refine=False)

            # Relax once with energy enabled (still no refine for speed)
            f = solve_gas_phase(f, T_target=None, loglevel=0, refine=False)

            if len(Tg_old) == len(f.T):
                err = np.linalg.norm(f.T - Tg_old)
                if err < tolK:
                    break

        # Final solid on full grid for η
        gas_dict_full = {
            'z': f.grid,
            'T': f.T,
            'u': f.velocity,
            'X': f.X,
            'p': f.gas.P
        }
        Ts_final, _ = solve_solid_phase(gas_dict_full, design_vars)
        if Ts_final is None:
            return 1.0

        T_ad = get_T_ad()  # cached adiabatic reference
        eta = (Ts_final[-1] / T_ad)**4
        return -eta if np.isfinite(eta) else 1.0

    except ct.CanteraError:
        return 1.0

def plot_response_surfaces(plot_data_history, output_dir):
    """Generates and saves contour plots of the quadratic response surface."""
    if not plot_data_history:
        print("No data available to plot response surfaces.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Generating {len(plot_data_history)} Response Surface Plots ---")

    for data in plot_data_history:
        iteration   = data['iteration']
        center      = data['center_point']
        step        = data['step_size']
        b           = data['coefficients']
        bounds      = data['bounds']

        if len(b) < 6:
            print(f"Skipping plot for iteration {iteration}: insufficient coefficients.")
            continue

        x_range = np.linspace(bounds['x_min'], bounds['x_max'], 150)
        y_range = np.linspace(bounds['y_min'], bounds['y_max'], 150)
        X, Y = np.meshgrid(x_range, y_range)
        Z = b[0] + b[1]*X + b[2]*Y + b[3]*X**2 + b[4]*Y**2 + b[5]*X*Y

        fig, ax = plt.subplots(figsize=(8, 7))
        csf = ax.contourf(X, Y, -Z, levels=40, cmap='viridis')
        ax.contour(X, Y, -Z, levels=40, colors='black', linewidths=0.4)
        rect = plt.Rectangle((center[0]-step[0], center[1]-step[1]),
                             2*step[0], 2*step[1],
                             fill=False, edgecolor='white', linewidth=1.5)
        ax.add_patch(rect)
        ax.plot(center[0], center[1], 'w.', markersize=8)
        ax.set_xlabel(r'Pore Diameter, $d_{p,2}$ (mm)')
        ax.set_ylabel(r'Porosity, $\epsilon_2$')
        ax.set_title(f'Response Surface (Iteration {iteration})')
        ax.set_xlim(bounds['x_min'], bounds['x_max'])
        ax.set_ylim(bounds['y_min'], bounds['y_max'])
        cbar = fig.colorbar(csf, ax=ax); cbar.set_label('Predicted Efficiency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'response_surface_iter_{iteration:02d}.png'), dpi=150)
        plt.close(fig)

def main():
    """Main driver for the optimization and plotting."""
    # Prevent thread oversubscription with process pool
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    output_dir = 'output_double_coupled'
    os.makedirs(output_dir, exist_ok=True)

    # Design variables: (pore_diameter_mm, porosity_stage2)
    initial_guess = [1.445, 0.88]
    step_size     = [0.075, 0.01]
    bounds        = [[0.69, 1.52], [0.865, 0.95]]

    optimizer = RSMOptimizer(
        objective_fn=run_coupled_simulation,
        initial_guess=initial_guess,
        step_size=step_size,
        bounds=bounds
    )
    print(f"Initial Radiant Efficiency: {-optimizer.best_f:.4f}")

    max_iter = 10
    with tqdm(total=max_iter, desc="Main Optimization Loop") as pbar:
        for best_x, best_f in optimizer.optimize(max_iterations=max_iter):
            pbar.set_postfix(d=f"{best_x[0]:.3f}",
                             eps=f"{best_x[1]:.3f}",
                             eta=f"{-best_f:.4f}")
            pbar.update(1)

    final_x, final_f = optimizer.best_x, optimizer.best_f
    print("\n--- Optimization Finished ---")
    print(f"Optimal Design: Pore Diameter = {final_x[0]:.4f} mm, Porosity = {final_x[1]:.4f}")
    print(f"Maximum Radiant Efficiency: {-final_f:.4f}")

    # Plot Best-So-Far Efficiency
    history_f = np.array([item[1] for item in optimizer.history])
    best_f = np.minimum.accumulate(history_f)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(best_f)), -best_f, 'o-')
    plt.xlabel('Iteration Number')
    plt.ylabel('Radiant Efficiency (η)')
    plt.title('Optimization Progress (Best-So-Far)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimization_best_history.png'), dpi=150)
    plt.close()

    # Plot response-surface contours
    plot_response_surfaces(optimizer.plot_data_history, output_dir)

if __name__ == "__main__":
    main()
