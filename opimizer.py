import numpy as np
import time
from custom_cantera_solver import run_combustion_simulation

# -------------------- Helper Functions --------------------

def newfun2d(x, b):
    """Quadratic 2D model: value, gradient, Hessian"""
    val = b[0] + b[1]*x[0] + b[2]*x[1] + b[3]*x[0]**2 + b[4]*x[1]**2 + b[5]*x[0]*x[1]
    grad = np.array([b[1] + 2*b[3]*x[0] + b[5]*x[1],
                     b[2] + 2*b[4]*x[1] + b[5]*x[0]])
    hess = np.array([[2*b[3], b[5]],
                     [b[5], 2*b[4]]])
    return val, grad, hess

def fun2D(pore_diameter_mm, porosity):
    """Wrapper for optimizer: maximize efficiency"""
    print(f"--> Testing: Pore Diameter = {pore_diameter_mm:.4f} mm, Porosity = {porosity:.4f}")
    return -run_combustion_simulation(pore_diameter_mm, porosity)

# -------------------- RSM Optimization --------------------

def rsm2d():
    x0 = np.array([1.445, 0.88])
    x_prev = np.copy(x0)

    alpha1, alpha2 = 0.075, 0.01
    bounds = {'lcon': 0.69, 'rcon': 1.52, 'bcon': 0.865, 'tcon': 0.95}

    points = np.array([
        [x0[0]-alpha1, x0[1]-alpha2], [x0[0]-alpha1, x0[1]], [x0[0]-alpha1, x0[1]+alpha2],
        [x0[0], x0[1]-alpha2],        [x0[0], x0[1]],        [x0[0], x0[1]+alpha2],
        [x0[0]+alpha1, x0[1]-alpha2], [x0[0]+alpha1, x0[1]], [x0[0]+alpha1, x0[1]+alpha2]
    ])

    f_values = np.zeros(9)
    diff2 = 1.0

    while diff2 > 1e-4:
        for i in range(len(points)):
            f_values[i] = fun2D(points[i,0], points[i,1])

        # Quadratic fit (no .columns, just NumPy)
        X = np.c_[
            np.ones(len(points)),
            points[:,0], points[:,1],
            points[:,0]**2, points[:,1]**2,
            points[:,0]*points[:,1]
        ]
        b = np.linalg.lstsq(X, f_values, rcond=None)[0]

        f_model, grad, hess = newfun2d(x0, b)

        try:
            if np.all(np.linalg.eigvals(hess) < 0):
                d = -np.linalg.solve(hess, grad)
            else:
                d = grad
        except np.linalg.LinAlgError:
            d = grad

        # Trust region
        d[0] = np.clip(d[0], -alpha1, alpha1)
        d[1] = np.clip(d[1], -alpha2, alpha2)

        x_prev = np.copy(x0)
        x0 += d
        x0[0] = np.clip(x0[0], bounds['lcon'], bounds['rcon'])
        x0[1] = np.clip(x0[1], bounds['bcon'], bounds['tcon'])

        diff2 = np.linalg.norm(x0 - x_prev)
        print(f"    Step diff: {diff2:.6f}, New position: [{x0[0]:.4f}, {x0[1]:.4f}]")

        # Update points around new center
        points = np.array([
            [x0[0]-alpha1, x0[1]-alpha2], [x0[0]-alpha1, x0[1]], [x0[0]-alpha1, x0[1]+alpha2],
            [x0[0], x0[1]-alpha2],        [x0[0], x0[1]],        [x0[0], x0[1]+alpha2],
            [x0[0]+alpha1, x0[1]-alpha2], [x0[0]+alpha1, x0[1]], [x0[0]+alpha1, x0[1]+alpha2]
        ])

    final_efficiency = -fun2D(x0[0], x0[1])
    return x0, final_efficiency

# -------------------- Main Script --------------------

if __name__ == "__main__":
    start_time = time.time()
    print("Starting Response Surface Method Optimization...")
    print(f"Initial Guess: {[1.445,0.88]}")
    print("-"*60)

    optimal_vars, max_efficiency = rsm2d()

    print("-"*60)
    print("Optimization Finished.")
    print("\nOptimal Design Found:")
    print(f"  - Pore Diameter: {optimal_vars[0]:.4f} mm")
    print(f"  - Porosity:      {optimal_vars[1]:.4f}")
    print(f"  - Maximum Radiant Efficiency: {max_efficiency:.4f}")

    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
