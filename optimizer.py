# optimizer.py
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

class RSMOptimizer:
    """Response Surface Methodology with fallback to best sampled design and optional parallel evaluations."""
    def __init__(self, objective_function, initial_guess, bounds, initial_step, gas_obj, n_jobs=1):
        self.func = objective_function
        self.gas = gas_obj
        self.x_current = np.array(initial_guess, dtype=float)
        self.bounds = np.array(bounds, dtype=float)
        self.step_size = np.array(initial_step, dtype=float)
        self.history = []
        self.n_jobs = int(n_jobs) if _HAS_JOBLIB else 1

        print("Evaluating initial design point...")
        initial_f = self.func(self.x_current, self.gas)
        self.best_x = self.x_current.copy()
        self.best_f = initial_f
        self.history.append({'x': self.x_current.copy(), 'f': -initial_f})

    def step(self):
        x_center = self.best_x.copy()
        x_points = self._select_points(x_center)

        # Evaluate 9-point stencil (parallel if available/allowed)
        if self.n_jobs != 1 and _HAS_JOBLIB:
            f_values = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self.func)(p, self.gas) for p in x_points
            )
        else:
            f_values = []
            for p in tqdm(x_points, desc="  Evaluating designs", unit="sim"):
                f_values.append(self.func(p, self.gas))
        f_values = np.array(f_values, dtype=float)

        model_and_poly = self._fit_surface(x_points, f_values)
        x_surface_opt = self._find_surface_optimum(model_and_poly)
        x_surface_opt = self._handle_constraints(x_surface_opt)

        # Best sampled design ensures forward progress
        idx_best = int(np.argmin(f_values))
        x_best_sample = x_points[idx_best]
        f_best_sample = f_values[idx_best]

        # Choose between model’s suggestion and sampled best
        if model_and_poly is None or np.allclose(x_surface_opt, x_center):
            x_new, f_new = x_best_sample, f_best_sample
        else:
            f_model = self.func(x_surface_opt, self.gas)
            if f_model <= f_best_sample:
                x_new, f_new = x_surface_opt, f_model
            else:
                x_new, f_new = x_best_sample, f_best_sample

        # Step-size adaptation
        if np.all(np.abs(x_new - x_center) < self.step_size):
            self.step_size *= 0.5
            print(f"  Shrinking step size to: {self.step_size}")

        self.x_current = x_new.copy()
        if f_new < self.best_f:
            self.best_f = f_new
            self.best_x = x_new.copy()

        self.history.append({'x': self.x_current.copy(), 'f': -f_new})
        return self.x_current.copy(), -f_new

    def _select_points(self, center):
        d, p = center
        dd, dp = self.step_size
        pts = np.array([
            [d-dd, p-dp], [d, p-dp], [d+dd, p-dp],
            [d-dd, p],   [d, p],    [d+dd, p],
            [d-dd, p+dp], [d, p+dp], [d+dd, p+dp]
        ], dtype=float)
        # Clip each dimension to bounds
        pts[:, 0] = np.clip(pts[:, 0], self.bounds[0, 0], self.bounds[0, 1])
        pts[:, 1] = np.clip(pts[:, 1], self.bounds[1, 0], self.bounds[1, 1])
        return pts

    def _fit_surface(self, X, y):
        valid = np.isfinite(y)
        if not np.any(valid):
            return None
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X[valid])
        model = LinearRegression().fit(X_poly, y[valid])
        return model, poly

    def _find_surface_optimum(self, model_and_poly):
        if model_and_poly is None:
            return self.best_x
        model, _ = model_and_poly
        # PolynomialFeatures order: [1, d, p, d^2, d*p, p^2]
        c = model.coef_
        if c.shape[0] < 6:
            return self.best_x
        b1, b2 = c[1], c[2]
        b4, b3, b5 = c[3], c[4], c[5]
        H = np.array([[2*b4, b3], [b3, 2*b5]], dtype=float)
        g = np.array([b1, b2], dtype=float)
        try:
            # Seek minimum (H positive definite)
            if np.all(np.linalg.eigvals(H) > 0):
                return np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            pass
        return self.best_x

    def _handle_constraints(self, x):
        x = np.asarray(x, dtype=float)
        x[0] = np.clip(x[0], self.bounds[0, 0], self.bounds[0, 1])
        x[1] = np.clip(x[1], self.bounds[1, 0], self.bounds[1, 1])
        return x
