# optimizer.py
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

class RSMOptimizer:
    """
    9-point face-centered central design with quadratic fit and safeguarded move.
    Parallelizes the 9 simulation calls per iteration using processes.
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
        self.plot_data_history = []

        # Initial evaluation with fallback
        self.best_f = self.fn(self.x)
        if not np.isfinite(self.best_f) or self.best_f >= 0:
            print("Warning: Initial design point failed. Starting from a known good point.")
            self.x = np.array([1.445, 0.88])
            self.best_x = self.x.copy()
            self.best_f = self.fn(self.x)
        self.history.append((self.best_x.copy(), self.best_f))

    def _get_fcc_points(self):
        xc, yc = self.x; dx, dy = self.step_size
        pts = [
            (xc-dx,yc-dy),(xc,yc-dy),(xc+dx,yc-dy),
            (xc-dx,yc),(xc,yc),(xc+dx,yc),
            (xc-dx,yc+dy),(xc,yc+dy),(xc+dx,yc+dy)
        ]
        return [self._clamp(p) for p in pts]

    def _clamp(self, point):
        x, y = point
        x = np.clip(x, self.bounds['x_min'], self.bounds['x_max'])
        y = np.clip(y, self.bounds['y_min'], self.bounds['y_max'])
        return np.array([x, y])

    def _fit_quadratic_surface(self, points, f_values):
        X = np.array([[1, p[0], p[1], p[0]**2, p[1]**2, p[0]*p[1]] for p in points])
        b, _, _, _ = np.linalg.lstsq(X, f_values, rcond=None)
        return b

    def _find_surface_optimum(self, b):
        H = np.array([[2*b[3], b[5]], [b[5], 2*b[4]]], dtype=float)
        grad = np.array([
            b[1] + 2*b[3]*self.x[0] + b[5]*self.x[1],
            b[2] + 2*b[4]*self.x[1] + b[5]*self.x[0]
        ], dtype=float)
        try:
            d = np.linalg.solve(H, -grad)
            if np.dot(grad, d) > 0:
                d = -grad
        except np.linalg.LinAlgError:
            d = -grad
        return d

    def optimize(self, max_iterations=10):
        with ProcessPoolExecutor() as executor:
            for i in range(max_iterations):
                fcc_points = self._get_fcc_points()
                future_to_point = {executor.submit(self.fn, p): p for p in fcc_points}
                points, f_values = [], []
                for future in as_completed(future_to_point):
                    points.append(future_to_point[future])
                    f_values.append(future.result())
                f_values = np.array(f_values, dtype=float)

                b = self._fit_quadratic_surface(points, f_values)
                self.plot_data_history.append({
                    'iteration': i + 1,
                    'center_point': self.x.copy(),
                    'step_size': self.step_size.copy(),
                    'coefficients': b.copy(),
                    'bounds': self.bounds
                })

                # Forced progress on flat surfaces: compare model vs best sampled
                d = self._find_surface_optimum(b)
                scale = np.min([1.0,
                                self.step_size[0] / (abs(d[0]) + 1e-9),
                                self.step_size[1] / (abs(d[1]) + 1e-9)])
                x_model = self._clamp(self.x + d * scale)
                f_model = self.fn(x_model)

                best_idx = int(np.argmin(f_values))
                x_best_sample = points[best_idx]
                f_best_sample = f_values[best_idx]

                if f_model <= f_best_sample:
                    x_new, f_new = x_model, f_model
                else:
                    x_new, f_new = x_best_sample, f_best_sample

                is_improvement = f_new < self.best_f
                if is_improvement:
                    self.best_f, self.best_x = f_new, x_new

                self.history.append((x_new.copy(), f_new))

                min_iters = 4
                if (i + 1) >= min_iters and np.linalg.norm(x_new - self.x) < self.tol and not is_improvement:
                    print("\nConvergence reached: step size below tolerance.")
                    break

                self.x = self.best_x.copy()
                if scale > 0.99 or not is_improvement:
                    self.step_size /= 2.0

                yield self.best_x, self.best_f
