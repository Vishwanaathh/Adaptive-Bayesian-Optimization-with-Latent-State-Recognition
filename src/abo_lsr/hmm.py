from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp

@dataclass
class HMMResult:
    belief: np.ndarray
    log_likelihood: float
    entropy: float

class GaussianHMM:
    """K-state scalar Gaussian HMM with log-space filtering and regularized batch EM."""
    def __init__(self, means=(0.0, 0.0, 0.0), stds=(1.8, 0.8, 0.35),
                 sticky_prior=8.0, transition_prior=1.0, min_variance=0.10):
        self.names = ("explore", "exploit", "refine")
        self.k = len(means)
        if self.k != 3:
            raise ValueError("This research implementation currently expects exactly 3 states")
        self.means = np.asarray(means, float)
        self.vars = np.maximum(np.asarray(stds, float)**2, min_variance)
        self.min_variance = float(min_variance)
        self.sticky_prior = float(sticky_prior)
        self.transition_prior = float(transition_prior)
        self.A = np.full((self.k, self.k), 1.0/self.k)
        for i in range(self.k):
            self.A[i, i] += self.sticky_prior
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.pi = np.full(self.k, 1.0/self.k)
        self.log_likelihood = 0.0
        self.history = []
        self.last_belief = self.pi.copy()

    def _log_emission(self, obs):
        x = np.asarray(obs, float).reshape(-1, 1)
        return -0.5*(np.log(2*np.pi*self.vars)[None, :] +
                     (x-self.means[None, :])**2/self.vars[None, :])

    def filter(self, observation, prior=None):
        b0 = self.last_belief if prior is None else np.asarray(prior, float)
        pred = np.maximum(b0, 1e-300) @ self.A
        logw = np.log(np.maximum(pred, 1e-300)) + self._log_emission([observation])[0]
        log_norm = logsumexp(logw)
        belief = np.exp(logw-log_norm)
        self.last_belief = belief
        self.history.append(float(observation))
        entropy = float(-np.sum(belief*np.log(np.maximum(belief, 1e-300))))
        return HMMResult(belief, float(log_norm), entropy)

    def reset_filter(self):
        self.last_belief = self.pi.copy()

    def fit_em(self, observations, n_iter=20, min_occupancy=0.03):
        obs = np.asarray(observations, float).reshape(-1)
        if len(obs) < max(6, self.k*2):
            return False
        log_b = self._log_emission(obs)
        log_A = np.log(np.maximum(self.A, 1e-300))
        log_pi = np.log(np.maximum(self.pi, 1e-300))
        prior_A = np.full_like(self.A, self.transition_prior)
        prior_A[np.diag_indices(self.k)] += self.sticky_prior
        old_ll = -np.inf
        for _ in range(int(n_iter)):
            T = len(obs)
            la = np.empty((T, self.k))
            la[0] = log_pi + log_b[0]
            for t in range(1, T):
                la[t] = log_b[t] + logsumexp(la[t-1][:, None]+log_A, axis=0)
            ll = float(logsumexp(la[-1]))
            lb = np.zeros((T, self.k))
            for t in range(T-2, -1, -1):
                lb[t] = logsumexp(log_A + log_b[t+1][None, :] +
                                  lb[t+1][None, :], axis=1)
            lg = la+lb
            lg -= logsumexp(lg, axis=1, keepdims=True)
            gamma = np.exp(lg)
            xi_sum = np.zeros((self.k, self.k))
            for t in range(T-1):
                lx = la[t][:, None] + log_A + log_b[t+1][None, :] + lb[t+1][None, :]
                lx -= logsumexp(lx)
                xi_sum += np.exp(lx)
            occ = gamma.sum(axis=0)
            new_A = xi_sum + prior_A
            new_A /= new_A.sum(axis=1, keepdims=True)
            new_means = (gamma.T @ obs)/np.maximum(occ, 1e-12)
            new_vars = (gamma * (obs[:, None]-new_means[None, :])**2).sum(axis=0) / np.maximum(occ, 1e-12)
            new_vars = np.maximum(new_vars, self.min_variance)
            new_pi = gamma[0] + 1.0
            new_pi /= new_pi.sum()
            # Avoid a degenerate state collapse by retaining parameters for nearly empty states.
            weak = occ < min_occupancy*T
            new_means[weak] = self.means[weak]
            new_vars[weak] = self.vars[weak]
            # Keep semantic labels stable by ordering states from broadest to
            # narrowest emission variance: explore, exploit, refine. This is an
            # explicit identifying constraint, not something inferred from names.
            order = np.argsort(-new_vars, kind="stable")
            self.A = new_A[np.ix_(order, order)]
            self.means = new_means[order]
            self.vars = new_vars[order]
            self.pi = new_pi[order]
            log_A = np.log(np.maximum(self.A, 1e-300))
            log_pi = np.log(np.maximum(self.pi, 1e-300))
            log_b = self._log_emission(obs)
            if abs(ll-old_ll) < 1e-5:
                break
            old_ll = ll
        self.log_likelihood = ll
        self.reset_filter()
        self.history = []
        # Replay all observations to produce a posterior consistent with updated parameters.
        for value in obs:
            self.filter(float(value))
        return True

    def diagnostics(self):
        return {
            "transition_matrix": self.A.tolist(),
            "means": self.means.tolist(),
            "stds": np.sqrt(self.vars).tolist(),
            "initial_probabilities": self.pi.tolist(),
            "log_likelihood": float(self.log_likelihood),
        }
