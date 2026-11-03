from __future__ import annotations
import time, uuid, json
from pathlib import Path
import numpy as np
from .gp import ExactRBFGaussianProcess
from .hmm import GaussianHMM
from .policies import PolicyController
from .storage import ExperimentStore

class BayesianOptimizer:
    def __init__(self, space, objective, config=None, output_dir=None):
        config = config or {}
        self.space, self.objective, self.config = space, objective, config
        self.seed = int(config.get("seed",17))
        self.rng = np.random.default_rng(self.seed)
        oc = config.get("optimizer", {})
        hc = config.get("hmm", {})
        ac = config.get("acquisition", {})
        self.gp = ExactRBFGaussianProcess(
            noise=oc.get("gp_noise",1e-6),
            optimize_every=oc.get("gp_optimize_every",3),
            restarts=oc.get("gp_restarts",2), seed=self.seed
        )
        self.hmm = GaussianHMM(
            means=hc.get("initial_means",[0,0,0]),
            stds=hc.get("initial_stds",[1.8,.8,.35]),
            sticky_prior=hc.get("sticky_prior",8),
            transition_prior=hc.get("transition_prior",1),
            min_variance=hc.get("min_variance",.1)
        )
        self.controller = PolicyController(
            method=oc.get("method","abo_lsr"),
            controller=oc.get("controller","soft"),
            candidate_pool=oc.get("candidate_pool",2048),
            beta=ac.get("ucb_beta"), radius=ac.get("local_radius"),
            xi=ac.get("ei_xi",.01), seed=self.seed
        )
        # With output_dir=None nothing is written to disk.
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.store = None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.store = ExperimentStore(self.output_dir/"experiments.sqlite")
        self.run_id = str(uuid.uuid4())
        self.X=[]; self.y=[]; self.zs=[]; self.records=[]
        self.belief=self.hmm.pi.copy()

    def run(self, budget=40, initial_points=8):
        started = time.time()
        budget = int(budget)
        initial_points = min(int(initial_points), budget)
        init = self.space.sample(self.rng, initial_points)
        for i in range(initial_points):
            self._evaluate(init[i], i, None, None, None, "initial_random")
        for step in range(initial_points, budget):
            # Failed objective calls consume budget but are not assigned fabricated GP targets.
            # Until two successful observations exist, continue with random proposals.
            if len(self.y) < 2:
                x = self.space.sample(self.rng, 1)[0]
                self._evaluate(x, step, None, None, "recovery_random", "recovery_random")
                continue
            self.gp.fit(np.asarray(self.X), np.asarray(self.y))
            incumbent = np.asarray(self.X)[int(np.argmax(self.y))]
            if len(self.y) < initial_points + 1:
                x = self.space.sample(self.rng, 1)[0]
                policy, meta = "warmup_random", {}
            else:
                x, policy, meta = self.controller.choose(
                    self.gp, np.asarray(self.X), np.asarray(self.y), self.belief,
                    incumbent, seed=self.seed + step
                )
            # Store the pre-evaluation GP prediction. Never recompute this after observing y_t.
            m, s = self.gp.predict(x[None, :])
            successful_before = len(self.y)
            self._evaluate(x, step, float(m[0]), float(s[0]), policy, policy, meta)
            if self.controller.method == "gp_hedge" and meta and "hedge_expert_points" in meta and len(self.y) > successful_before and len(self.y) >= 2:
                self.gp.fit(np.asarray(self.X), np.asarray(self.y), force=True)
                self.controller.update_hedge(self.gp, meta["hedge_expert_points"])
        if len(self.y) >= 2:
            self.gp.fit(np.asarray(self.X), np.asarray(self.y), force=True)
            best_i = int(np.argmax(self.y))
            best_score = float(self.y[best_i])
            best_x = self.X[best_i].tolist()
            best_params = self.space.decode(self.X[best_i])
        else:
            best_score, best_x, best_params = None, None, None
        summary = {"run_id": self.run_id, "best_score": best_score,
                   "best_x": best_x, "best_params": best_params,
                   "evaluations": len(self.records),
                   "successful_evaluations": len(self.y),
                   "failed_evaluations": len(self.records)-len(self.y),
                   "elapsed_seconds": time.time()-started,
                   "hmm": self.hmm.diagnostics()}
        if self.output_dir is not None:
            (self.output_dir/"summary.json").write_text(json.dumps(summary, indent=2))
            self.store.close()
        return summary

    def _evaluate(self,x,step,mu_pre,sd_pre,policy,policy_label,meta=None):
        t0=time.time(); params=self.space.decode(x)
        try:
            score=float(self.objective(params))
            if not np.isfinite(score): raise ValueError("Objective returned non-finite score")
            status="ok"
        except Exception as exc:
            score=None; status=f"failed:{type(exc).__name__}:{exc}"
        elapsed=time.time()-t0
        z=None
        if score is not None:
            if mu_pre is not None and sd_pre is not None:
                z=float((score-mu_pre)/max(sd_pre,1e-8))
                self.zs.append(z)
                if self.config.get("hmm",{}).get("enabled",True):
                    result=self.hmm.filter(z)
                    self.belief=result.belief
                    hc=self.config.get("hmm",{})
                    if len(self.zs)>=hc.get("warmup",8) and len(self.zs)%hc.get("em_every",3)==0:
                        self.hmm.fit_em(self.zs,n_iter=hc.get("em_iterations",20),
                                        min_occupancy=hc.get("min_state_occupancy",.03))
                        self.belief=self.hmm.last_belief.copy()
            self.X.append(np.asarray(x,float)); self.y.append(score)
        self.records.append({"step":step,"score":score,"status":status,"z":z,
                             "policy":policy_label,"belief":self.belief.tolist(),
                             "elapsed":elapsed,"params":params})
        if self.store is not None:
            self.store.add(self.run_id,step,np.asarray(x).tolist(),params,score,status,
                           mu_pre,sd_pre,z,self.belief.tolist(),policy_label,elapsed)
