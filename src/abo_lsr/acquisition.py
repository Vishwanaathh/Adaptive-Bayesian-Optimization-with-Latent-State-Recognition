from __future__ import annotations
import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc

def ucb(mu, sigma, beta):
    return np.asarray(mu) + np.sqrt(max(float(beta), 0.0))*np.asarray(sigma)

def expected_improvement(mu, sigma, best, xi=0.01):
    sigma = np.maximum(np.asarray(sigma), 1e-12)
    improvement = np.asarray(mu) - float(best) - float(xi)
    z = improvement/sigma
    pdf = np.exp(-0.5*z*z)/np.sqrt(2*np.pi)
    return np.maximum(improvement*ndtr(z) + sigma*pdf, 0.0)

def sobol_candidates(dim, n, seed):
    m = int(np.ceil(np.log2(max(n, 2))))
    return qmc.Sobol(dim, scramble=True, seed=seed).random_base2(m)[:n]

def local_candidates(incumbent, n, radius, rng):
    return np.clip(np.asarray(incumbent)[None, :] +
                   rng.normal(0.0, radius, (n, len(incumbent))), 0.0, 1.0)
