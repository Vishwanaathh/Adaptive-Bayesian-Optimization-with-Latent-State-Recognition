"""ABO-LSR: Bayesian optimization with latent Gaussian-HMM state recognition."""
from .gp import ExactRBFGaussianProcess
from .hmm import GaussianHMM
from .optimizer import BayesianOptimizer
from .space import Dimension, SearchSpace

__version__ = "0.3.0"
__all__ = ["BayesianOptimizer", "SearchSpace", "Dimension",
           "GaussianHMM", "ExactRBFGaussianProcess", "__version__"]
