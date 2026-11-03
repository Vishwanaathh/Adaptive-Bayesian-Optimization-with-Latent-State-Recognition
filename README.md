# ABO-LSR: Adaptive Bayesian Optimization with Latent State Recognition

This repository implements a research experiment for adaptive Bayesian optimization driven by a three-state Gaussian HMM. The maintained experiment is **real-data-only** and tunes Ridge, Lasso, and ElasticNet regression on the California Housing dataset distributed by scikit-learn.

## Method

For a candidate hyperparameter vector \(\lambda_t\), the GP prediction is recorded **before** evaluating the validation objective:

\[
z_t = \frac{y_t-\mu_{t-1}(\lambda_t)}{\max(\sigma_{t-1}(\lambda_t),\epsilon)}.
\]

The scalar innovation is passed to a Gaussian HMM with three variance-ordered states: Explore (broadest emission), Exploit, and Refine (narrowest emission). The HMM uses log-space online forward filtering. At configured intervals, forward-backward batch EM updates the transition and emission parameters using the observed innovation history, after which the filter is replayed to produce a belief consistent with the updated parameters. State beliefs control either a hard MAP-state policy or a soft belief-weighted acquisition policy.

The variance ordering is an explicit state-label identification constraint. It does not prove that the inferred states correspond to meaningful optimization phases. The residual-only emission model is a research hypothesis: calibrated GP innovations may be close to standard normal and may not contain useful regime information.

## Experiment design

- Dataset: the real 20,640-row California Housing dataset in scikit-learn format. The loader first uses `fetch_california_housing`; if Figshare returns an HTTP error, it tries a checksum-verified archive mirror and caches the parsed arrays in scikit-learn's data-home directory.
- Model families: Ridge, Lasso, ElasticNet.
- Hyperparameter objective: validation R².
- Splits: per-seed, fixed train/validation/test partitions shared by every method for that seed.
- Preprocessing: `StandardScaler` is fitted inside each model pipeline on training data only; validation and test observations do not influence scaler fitting.
- Test set: never used during optimization. The selected hyperparameters are refit on train + validation before one final test evaluation.
- Methods: Random Search, GP-UCB, GP-EI, Scheduled UCB, GP-Hedge, ABO-LSR hard, ABO-LSR soft, plus no-HMM and fixed-HMM ablations.
- OLS: a non-optimized reference, refit on train + validation and evaluated on the same held-out test set.
- Paired seeds, equal objective evaluation budgets, and common initial designs within each model-family/seed comparison.

`GP-Hedge` uses three experts (UCB, EI, local UCB). It samples an expert from the softmax of cumulative gains, evaluates the selected proposal, refits the GP with the new observation, then updates every expert's gain using the updated posterior mean at that expert's proposal. This is the actual posterior expert-gain update sequence rather than a pre-evaluation prediction shortcut.

## Install

Python 3.10 or newer is required.

```bash
pip install abo-lsr                  # optimizer only (numpy, scipy)
pip install "abo-lsr[experiments]"   # + California Housing runner, CLI, plotting
```

Until it is published to PyPI, install from a checkout: `pip install .` (or `pip install ".[experiments]"`).

For development:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Use as a library

```python
from abo_lsr import BayesianOptimizer, Dimension, SearchSpace

space = SearchSpace([
    Dimension("alpha", "log", 1e-4, 1e2),
    Dimension("l1_ratio", "continuous", 0.05, 1.0),
])

def objective(params):          # the optimizer MAXIMIZES this (e.g. validation R^2)
    return my_validation_score(**params)

opt = BayesianOptimizer(space, objective, {"seed": 0})
result = opt.run(budget=30, initial_points=8)
print(result["best_score"], result["best_params"])
```

Dimension kinds: `continuous`, `log`, `integer`, `categorical` (with `choices=`).
Pass `output_dir="runs/exp1"` to also write `summary.json` and an SQLite log of every evaluation;
by default nothing is written to disk. Failed objective calls (exceptions or non-finite values) use up
budget and are recorded but never fed to the GP. See `examples/quickstart.py` for a runnable script.

`config` keys (all optional): `seed`; `optimizer.{method, controller, candidate_pool, gp_restarts,
gp_noise, gp_optimize_every}`; `hmm.{enabled, warmup, em_every, em_iterations, initial_stds, ...}`;
`acquisition.{ucb_beta, local_radius, ei_xi}`. `optimizer.method` is one of `abo_lsr` (default),
`gp_ucb`, `gp_ei`, `scheduled_ucb`, `gp_hedge`, `random`; `optimizer.controller` is `soft` or `hard`
and applies to `abo_lsr`.

## Run the experiment

Requires the `experiments` extra. The default config is bundled with the package:

```bash
abo-lsr california
abo-lsr california --config my_config.yaml --output results/my_run
```

Or without the console script: `python -m abo_lsr.cli california`. A copy of the default config is
at `src/abo_lsr/configs/california_housing.yaml`.

The first run requires internet access to download California Housing unless it is already cached. If the default Figshare URL returns HTTP 403, the loader tries a checksum-verified S3 mirror. The fallback cache is `abo_lsr_california_housing.npz` inside scikit-learn's data-home directory. For a quick smoke test, make a copy of the YAML and reduce `seeds` to one seed, `budget` to 8, `initial_points` to 4, and `candidate_pool` to 128. Such a smoke test is not a publication-quality experiment.

## Output files

The default output directory is `results/california_housing/`:

- `per_seed_metrics.csv`: one row per family, method, and seed, including validation/test R², test RMSE/MAE, runtime, evaluation count, selected parameters, and HMM diagnostics.
- `aggregate_metrics.csv`: mean, sample standard deviation, and percentile-bootstrap 95% confidence intervals.
- `paired_statistical_tests.csv`: paired two-sided Wilcoxon signed-rank comparisons against ABO-LSR soft, with Holm correction within each model family.
- `optimization_traces.csv`: per-evaluation validation score, incumbent score, predictive surprise, state beliefs, policy and elapsed objective time.
- `experiment_metadata.json`: dataset, seeds, methods, budget, and test-leakage declaration.

The confidence intervals are percentile bootstrap intervals over independent seeds. With the default five seeds they should be interpreted as descriptive and low-powered. The Wilcoxon test is also low-powered at small seed counts. Increase the number of seeds before making strong inferential claims.

## Ablations

- `abo_no_hmm`: removes online HMM inference and fixes the belief to the exploit expert (equivalent to an EI-only soft policy).
- `abo_fixed_hmm`: retains online filtering but disables EM parameter updates.
- `abo_lsr_hard` versus `abo_lsr_soft`: isolates MAP-state switching versus belief-weighted acquisition.

These ablations are included as comparative variants; they do not establish causal mechanism without sufficient repeated experiments and diagnostics.

## Tests

```bash
pytest
```

Tests validate core GP/HMM behaviour, search-space encoding, expert gain update timing, and the regression experiment pipeline on a locally available real scikit-learn regression dataset. Tests do not download California Housing or claim that the full research experiment has run.

## Limitations and research cautions

1. A residual-only HMM may not identify optimization phases; inspect learned emissions, occupancy, transition matrix, entropy, and ablation results.
2. Variance ordering stabilizes labels but is an identifying convention, not semantic validation.
3. HMM updates use batch EM over the innovation history, not stochastic online EM. Filtering is online; parameter updates are periodic batch updates.
4. Candidate acquisition is optimized over a reproducible Sobol pool, not by continuous global acquisition optimization.
5. The exact GP has cubic fitting cost in the number of evaluations.
6. California Housing is a single regression dataset. Results do not establish generality across datasets.
7. Runtime depends on hardware, package versions, cache state, and dataset download. No results are bundled or represented as having been obtained until the experiment is actually run.

## Project layout

- `src/abo_lsr/gp.py`: exact ARD-RBF Gaussian process.
- `src/abo_lsr/hmm.py`: three-state Gaussian HMM, filtering, and batch EM.
- `src/abo_lsr/acquisition.py`: UCB, EI, Sobol candidate pool, and local candidates.
- `src/abo_lsr/policies.py`: legacy policy controller retained for component-level compatibility.
- `src/abo_lsr/experiment.py`: California Housing experiment, baselines, ablations, metrics, and statistics.
- `src/abo_lsr/cli.py`: command-line entry point.
- `src/abo_lsr/optimizer.py`: general-purpose `BayesianOptimizer` (the library entry point).
- `examples/`, `scripts/`: quickstart example; results-digest script.
- `tests/`: unit and integration tests.


### California Housing download fallback

The experiment first uses `sklearn.datasets.fetch_california_housing`. If its default
Figshare endpoint returns an HTTP error (including HTTP 403), ABO-LSR tries a
scikit-learn-compatible archive mirror and verifies the archive against the SHA-256
checksum published by scikit-learn before parsing it. The parsed data are cached as
`abo_lsr_california_housing.npz` in scikit-learn's data-home directory. If all sources
are unavailable, the run stops with an explicit download error; it does not substitute
a synthetic or different dataset.

## License

MIT. See [LICENSE](LICENSE).
