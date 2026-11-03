from __future__ import annotations
import numpy as np
from .acquisition import ucb, expected_improvement, local_candidates, sobol_candidates

REGIMES = ("explore", "exploit", "refine")
METHODS = ("abo_lsr", "random", "gp_ucb", "gp_ei", "scheduled_ucb", "gp_hedge")
CONTROLLERS = ("soft", "hard")

class PolicyController:
    def __init__(self, method="abo_lsr", controller="soft", candidate_pool=2048,
                 beta=None, radius=None, xi=0.01, seed=0):
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
        if controller not in CONTROLLERS:
            raise ValueError(f"Unknown controller {controller!r}; choose from {CONTROLLERS}")
        self.method = method
        self.controller = controller
        self.candidate_pool = int(candidate_pool)
        self.beta = beta or {"explore": 6.0, "exploit": 1.5, "refine": 0.25}
        self.radius = radius or {"explore": 0.30, "exploit": 0.12, "refine": 0.035}
        self.xi = float(xi)
        self.rng = np.random.default_rng(seed)
        self.hedge_gains = np.zeros(3, dtype=float)
        self.hedge_eta = 0.5

    def update_hedge(self, gp, expert_points):
        """Update cumulative GP-Hedge gains from the posterior after the new observation."""
        means, _ = gp.predict(np.asarray(expert_points, dtype=float))
        self.hedge_gains += np.asarray(means, dtype=float)
        return self.hedge_gains.copy()

    def choose(self, gp, X, y, belief=None, incumbent=None, seed=0):
        d = X.shape[1]
        candidates = sobol_candidates(d, self.candidate_pool, seed)
        mu, sigma = gp.predict(candidates)
        best = float(np.max(y))
        if self.method == "random":
            return candidates[self.rng.integers(len(candidates))], "random", {}
        if self.method == "gp_hedge":
            # GP-Hedge: each expert proposes a candidate; cumulative gains are updated
            # using the GP posterior mean at each expert's proposal, then Hedge samples
            # an expert from a softmax distribution over cumulative gains.
            ucb_point = candidates[int(np.argmax(ucb(mu, sigma, 6.0)))]
            ei_point = candidates[int(np.argmax(expected_improvement(mu, sigma, best, self.xi)))]
            local_pool = local_candidates(incumbent, max(128, self.candidate_pool // 4),
                                          0.10, self.rng)
            lm, ls = gp.predict(local_pool)
            local_point = local_pool[int(np.argmax(ucb(lm, ls, 0.5)))]
            expert_points = np.vstack([ucb_point, ei_point, local_point])
            logits = self.hedge_eta * (self.hedge_gains - np.max(self.hedge_gains))
            probs = np.exp(logits); probs /= probs.sum()
            expert = int(self.rng.choice(3, p=probs))
            return expert_points[expert], ("ucb", "ei", "local")[expert], {
                "hedge_probabilities": probs.tolist(),
                "hedge_gains": self.hedge_gains.tolist(),
                "hedge_expert_points": expert_points.tolist(),
                "hedge_selected_expert": expert
            }
        if self.method in {"gp_ucb", "scheduled_ucb", "gp_ei"}:
            if self.method == "gp_ucb":
                beta = 2.0
            elif self.method == "scheduled_ucb":
                beta = max(0.15, 6.0*(0.97**len(y)))
            elif self.method == "gp_ei":
                beta = None
            else:
                beta = None
            if beta is None:
                acq = expected_improvement(mu,sigma,best,self.xi); name="ei"
            else:
                acq = ucb(mu,sigma,beta); name=self.method
            i = int(np.argmax(acq))
            return candidates[i], name, {"candidate_scores": [float(np.max(s)) for s in
                    ([ucb(mu,sigma,6.0), expected_improvement(mu,sigma,best,self.xi),
                      -np.linalg.norm(candidates-incumbent,axis=1)] if self.method=="gp_hedge" else [acq])]}
        b = np.asarray(belief if belief is not None else [1/3]*3, float)
        b = b/np.sum(b)
        if self.controller == "hard":
            k = int(np.argmax(b))
            name = REGIMES[k]
            if name == "refine":
                local = local_candidates(incumbent, self.candidate_pool, self.radius[name], self.rng)
                lm, ls = gp.predict(local)
                i = int(np.argmax(ucb(lm,ls,self.beta[name])))
                return local[i], name, {"belief":b.tolist()}
            acq = ucb(mu,sigma,self.beta[name]) if name=="explore" else expected_improvement(mu,sigma,best,self.xi)
            return candidates[int(np.argmax(acq))], name, {"belief":b.tolist()}
        # Soft policy: normalize each expert's scores to [0,1] before belief-weighted mixing.
        expert_scores = []
        for name in REGIMES:
            if name == "explore":
                s = ucb(mu,sigma,self.beta[name])
            elif name == "exploit":
                s = expected_improvement(mu,sigma,best,self.xi)
            else:
                dist = np.linalg.norm(candidates-incumbent,axis=1)
                s = -dist
            lo, hi = np.min(s), np.max(s)
            expert_scores.append((s-lo)/(hi-lo+1e-12))
        combined = np.sum(np.asarray(expert_scores)*b[:,None], axis=0)
        i = int(np.argmax(combined))
        return candidates[i], "soft_mixture", {"belief":b.tolist()}
