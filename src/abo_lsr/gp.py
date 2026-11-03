from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

class ExactRBFGaussianProcess:
    """Small-data exact GP with ARD RBF kernel and Gaussian observation noise."""
    def __init__(self, noise=1e-6, optimize_every=1, restarts=2, seed=0):
        self.noise = float(noise)
        self.optimize_every = int(optimize_every)
        self.restarts = int(restarts)
        self.rng = np.random.default_rng(seed)
        self.fitted = False
        self.lengthscales = None
        self.signal = 1.0
        self.mean_y = 0.0
        self.scale_y = 1.0
        self.fit_count = 0

    @staticmethod
    def _kernel(a, b, lengthscales, signal):
        delta = (a[:, None, :] - b[None, :, :]) / lengthscales
        return signal**2 * np.exp(-0.5 * np.sum(delta**2, axis=2))

    def _nll(self, theta, x, y):
        d = x.shape[1]
        ls = np.exp(theta[:d])
        signal = np.exp(theta[d])
        noise = np.exp(theta[d+1])
        k = self._kernel(x, x, ls, signal) + (noise + 1e-8) * np.eye(len(x))
        try:
            cf = cho_factor(k, lower=True, check_finite=False)
            alpha = cho_solve(cf, y, check_finite=False)
            return 0.5*y@alpha + np.sum(np.log(np.diag(cf[0]))) + 0.5*len(x)*np.log(2*np.pi)
        except np.linalg.LinAlgError:
            return 1e25

    def fit(self, x, y, force=False):
        x = np.asarray(x, float); y = np.asarray(y, float).reshape(-1)
        if len(x) != len(y) or len(y) < 2:
            raise ValueError("GP requires at least two paired observations")
        self.x = x.copy()
        self.mean_y = float(np.mean(y))
        self.scale_y = float(np.std(y)) or 1.0
        self.y = (y-self.mean_y)/self.scale_y
        self.fit_count += 1
        if not self.fitted or force or self.fit_count % self.optimize_every == 0:
            d = x.shape[1]
            bounds = [(-3.5, 2.0)]*d + [(-3.0, 2.0), (-12.0, -2.0)]
            starts = [np.r_[np.zeros(d), 0.0, np.log(max(self.noise, 1e-8))]]
            for _ in range(self.restarts):
                starts.append(np.r_[self.rng.uniform(-1, 1, d),
                                    self.rng.uniform(-0.5, 0.5),
                                    np.log(max(self.noise, 1e-8))])
            best = None
            for start in starts:
                res = minimize(self._nll, start, args=(self.x, self.y),
                               method="L-BFGS-B", bounds=bounds)
                if best is None or res.fun < best.fun:
                    best = res
            theta = best.x
            self.lengthscales = np.exp(theta[:d])
            self.signal = float(np.exp(theta[d]))
            self.noise_fit = float(np.exp(theta[d+1]))
        elif self.lengthscales is None:
            self.lengthscales = np.ones(x.shape[1])
            self.noise_fit = self.noise
        k = self._kernel(self.x, self.x, self.lengthscales, self.signal)
        k += (self.noise_fit + 1e-8)*np.eye(len(self.x))
        self.cf = cho_factor(k, lower=True, check_finite=False)
        self.alpha = cho_solve(self.cf, self.y, check_finite=False)
        self.fitted = True
        return self

    def predict(self, x):
        if not self.fitted:
            raise RuntimeError("Call fit before predict")
        x = np.atleast_2d(np.asarray(x, float))
        kxs = self._kernel(x, self.x, self.lengthscales, self.signal)
        mean_n = kxs @ self.alpha
        v = cho_solve(self.cf, kxs.T, check_finite=False)
        var_n = np.maximum(self.signal**2 - np.sum(kxs*v.T, axis=1), 1e-12)
        return mean_n*self.scale_y+self.mean_y, np.sqrt(var_n)*self.scale_y
