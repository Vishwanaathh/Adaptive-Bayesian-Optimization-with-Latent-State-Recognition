"""Maximize a toy objective with ABO-LSR. Runs in a few seconds, writes no files."""
from abo_lsr import BayesianOptimizer, Dimension, SearchSpace

space = SearchSpace([
    Dimension("x", "continuous", 0.0, 1.0),
    Dimension("lr", "log", 1e-4, 1e-1),
])


def objective(params):  # the optimizer MAXIMIZES this
    return -((params["x"] - 0.3) ** 2) - (abs(params["lr"] - 1e-2) * 10)


config = {"seed": 0, "optimizer": {"candidate_pool": 256, "gp_restarts": 1}}
result = BayesianOptimizer(space, objective, config).run(budget=25, initial_points=6)
print("best score :", result["best_score"])
print("best params:", result["best_params"])
