from abo_lsr.space import SearchSpace, Dimension
from abo_lsr.optimizer import BayesianOptimizer

def test_optimizer_runs(tmp_path):
    space=SearchSpace([Dimension("x","continuous",0,1),Dimension("y","continuous",0,1)])
    def objective(p): return -(p["x"]-.3)**2-(p["y"]-.7)**2
    cfg={"seed":2,"optimizer":{"candidate_pool":128,"gp_restarts":0},
         "hmm":{"enabled":True,"warmup":5,"em_every":2,"em_iterations":5},
         "acquisition":{}}
    opt=BayesianOptimizer(space,objective,cfg,tmp_path)
    result=opt.run(budget=12,initial_points=5)
    assert result["evaluations"]==12
    assert result["best_score"]<=0

def test_gp_hedge_policy_defers_gain_update_until_explicit_observation():
    import numpy as np
    from abo_lsr.gp import ExactRBFGaussianProcess
    from abo_lsr.policies import PolicyController

    x = np.array([[0.0], [0.5], [1.0]])
    y = np.array([0.0, 0.4, 0.1])
    gp = ExactRBFGaussianProcess(restarts=0).fit(x, y, force=True)
    controller = PolicyController(method="gp_hedge", candidate_pool=64, seed=3)
    before = controller.hedge_gains.copy()
    point, expert, meta = controller.choose(gp, x, y, incumbent=x[1], seed=4)
    assert point.shape == (1,)
    assert len(meta["hedge_expert_points"]) == 3
    assert np.array_equal(controller.hedge_gains, before)
    gp.fit(np.vstack((x, point)), np.r_[y, 0.5], force=True)
    gains = controller.update_hedge(gp, meta["hedge_expert_points"])
    assert np.any(gains != before)


def test_optimizer_writes_nothing_without_output_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    space = SearchSpace([Dimension("x", "continuous", 0, 1)])
    opt = BayesianOptimizer(space, lambda p: -(p["x"] - .5) ** 2,
                            {"optimizer": {"candidate_pool": 64, "gp_restarts": 0}})
    result = opt.run(budget=8, initial_points=4)
    assert result["evaluations"] == 8
    assert list(tmp_path.iterdir()) == []


def test_unknown_method_and_controller_are_rejected():
    import pytest
    from abo_lsr.policies import PolicyController
    with pytest.raises(ValueError):
        PolicyController(method="gp_eI")
    with pytest.raises(ValueError):
        PolicyController(controller="medium")
