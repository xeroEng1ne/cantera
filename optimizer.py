# optimizer.py
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

class RSMOptimizer:
    """
    A Python implementation of the Response Surface Methodology (RSM)
    optimizer described in the Horsman thesis, Appendix D.
    """
    def __init__(self, objective_fn, initial_guess, step_size, bounds, tol=1e-4):
        self.fn = objective_fn
        self.x = np.array(initial_guess, dtype=float)
        self.step_size = np.array(step_size, dtype=float)
        self.bounds = {
            'x_min': bounds[0][0], 'x_max': bounds[0][1],
            'y_min': bounds[1][0], 'y_max': bounds[1][1]
        }
        self.tol = tol

        self.best_x = self.x.copy()
        self.history = []
        # Evaluate initial point
        self.best_f = self.fn(self.x)
        if not np.isfinite(self.best_f) or self.best_f >= 0:
             # If initial point fails, use a known good starting point from the thesis
            print("Warning: Initial design point failed. Starting from a known good point.")
            self.x = np.array([1.445, 0.88]) # A known good point from the thesis C++ code
            self.best_x = self.x.copy()
            self.best_f = self.fn(self.x)

        self.history.append((self.best_x.copy(), self.best_f))

    def _get_fcc_points(self):
        """Generates the 9 Face-Centered Central design points."""
        x_center, y_center = self.x
        dx, dy = self.step_size
        
        points = [
            (x_center - dx, y_center - dy), (x_center, y_center - dy), (x_center + dx, y_center - dy),
            (x_center - dx, y_center),     (x_center, y_center),     (x_center + dx, y_center),
            (x_center - dx, y_center + dy), (x_center, y_center + dy), (x_center + dx, y_center + dy)
        ]
        
        # Clamp points to be within the hard bounds
        return [self._clamp(p) for p in points]

    def _clamp(self, point):
        """Ensure a point is within the defined optimization bounds."""
        x, y = point
        x = np.clip(x, self.bounds['x_min'], self.bounds['x_max'])
        y = np.clip(y, self.bounds['y_min'], self.bounds['y_max'])
        return np.array([x,y])

    def _fit_quadratic_surface(self, points, f_values):
        """
        Fits z = b0 + b1*x + b2*y + b3*x^2 + b4*y^2 + b5*x*y
        This replaces the C++ MM, MT, and LUSolve functions with np.linalg.lstsq.
        """
        # Design matrix X
        X = np.array([
            [1, p[0], p[1], p[0]**2, p[1]**2, p[0]*p[1]] for p in points
        ])
        # Use least squares to find the coefficients b
        b, _, _, _ = np.linalg.lstsq(X, f_values, rcond=None)
        return b

    def _find_surface_optimum(self, b):
        """
        Finds the optimum of the quadratic surface using Newton's method.
        This replaces the Hessian calculation and linear solve in the C++ code.
        """
        # Hessian H = [[2*b3, b5], [b5, 2*b4]]
        # Gradient g = [b1 + 2*b3*x + b5*y, b2 + 2*b4*y + b5*x]
        H = np.array([[2 * b[3], b[5]], [b[5], 2 * b[4]]])
        
        # We are maximizing, so we want to move in the direction of the gradient
        # and the Hessian should be negative definite.
        try:
            # Newton's step: d = -H^-1 * g
            grad_at_center = np.array([b[1] + 2*b[3]*self.x[0] + b[5]*self.x[1],
                                       b[2] + 2*b[4]*self.x[1] + b[5]*self.x[0]])
            d = -np.linalg.solve(H, grad_at_center)
            
            # Check for maximization direction (H should be negative definite)
            # A simple check is to see if the step is an ascent direction.
            if np.dot(grad_at_center, d) < 0:
                # If not, fall back to steepest ascent
                d = grad_at_center
        except np.linalg.LinAlgError:
            # If Hessian is singular, use steepest ascent
            grad_at_center = np.array([b[1] + 2*b[3]*self.x[0] + b[5]*self.x[1],
                                       b[2] + 2*b[4]*self.x[1] + b[5]*self.x[0]])
            d = grad_at_center
        
        return d

    def optimize(self, max_iterations=15):
        with ProcessPoolExecutor() as executor:
            for i in range(max_iterations):
                print(f"\n--- Iteration {i+1} ---")
                
                # 1. Generate sample points and evaluate objective function in parallel
                fcc_points = self._get_fcc_points()
                future_to_point = {executor.submit(self.fn, p): p for p in fcc_points}
                
                points, f_values = [], []
                for future in as_completed(future_to_point):
                    points.append(future_to_point[future])
                    f_values.append(future.result())

                # 2. Fit the quadratic response surface
                b = self._fit_quadratic_surface(points, f_values)
                
                # 3. Find the optimal direction on the surface (Newton's step)
                d = self._find_surface_optimum(b)

                # 4. Constrain the step to the trust region (the box defined by step_size)
                # This logic mimics the gamma/alpha scaling in the C++ code.
                scale = np.min([1.0, 
                                self.step_size[0] / (abs(d[0]) + 1e-9), 
                                self.step_size[1] / (abs(d[1]) + 1e-9)])
                
                x_new = self.x + d * scale
                x_new = self._clamp(x_new) # Ensure it's within hard bounds
                
                # 5. Evaluate the new point
                f_new = self.fn(x_new)

                print(f"  Proposed step: d={x_new[0]:.4f}, ε={x_new[1]:.4f}, η={-f_new:.4f}")

                # 6. Update state and step size
                if f_new < self.best_f:
                    self.best_f = f_new
                    self.best_x = x_new
                
                self.history.append((x_new.copy(), f_new))

                # Check for convergence
                if np.linalg.norm(x_new - self.x) < self.tol:
                    print("\nConvergence reached: step size below tolerance.")
                    break
                
                self.x = x_new

                # Shrink the trust region if the step was small (interior optimum)
                # This corresponds to the logic in the C++ version.
                if scale > 0.99: 
                    self.step_size /= 2.0
                    print(f"  Shrinking step size to: [{self.step_size[0]:.4f} {self.step_size[1]:.4f}]")

                # Yield for tqdm
                yield self.best_x, self.best_f
