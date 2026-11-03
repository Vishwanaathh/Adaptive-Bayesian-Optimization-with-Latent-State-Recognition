# ABO-LSR: Adaptive Bayesian Optimization with Latent State Recognition

A small Bayesian-optimization library for expensive, low-dimensional problems such as hyperparameter
tuning. A Gaussian process proposes points; a three-state Gaussian HMM watches how surprising each result
was and shifts the search between exploring, exploiting and refining. It also ships the experiment code
for the accompanying research study (tuning Ridge, Lasso and ElasticNet on California Housing).

Needs only `numpy` and `scipy` for the optimizer. Python 3.10+.

## Install

```bash
pip install abo-lsr                  # optimizer only (numpy, scipy)
pip install "abo-lsr[experiments]"   # + California Housing runner, CLI, plotting
```

From a checkout instead: `pip install .` (or `pip install ".[experiments]"`).

## Quick start

```python
from abo_lsr import BayesianOptimizer, Dimension, SearchSpace

# 1. Describe what to search over
space = SearchSpace([
    Dimension("alpha", "log", 1e-4, 1e1),
    Dimension("l1_ratio", "continuous", 0.05, 1.0),
])

# 2. Write a function that returns a score to MAXIMIZE
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)

def objective(params):
    model = make_pipeline(StandardScaler(),
                          ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], max_iter=20000))
    return cross_val_score(model, X, y, cv=5, scoring="r2").mean()

# 3. Run
opt = BayesianOptimizer(space, objective, {"seed": 0})
result = opt.run(budget=20, initial_points=6)
print(result["best_score"], result["best_params"])
```

(`scikit-learn` is used here only to make the objective; the optimizer itself does not need it.)
A runnable version without scikit-learn is in `examples/quickstart.py`.

**Rules**
- The optimizer **maximizes**. To minimize a loss, return `-loss`.
- `budget` is the total number of objective calls, including the `initial_points` random ones.
- `params` arrives as a dict with the names and types you declared.
- Tune on training or validation data only; keep your final test set out of the objective.

## Usage guide

### Dimension types

| Kind | Example |
|---|---|
| `continuous` | `Dimension("x", "continuous", 0.0, 1.0)` |
| `log` | `Dimension("lr", "log", 1e-4, 1e-1)` (low must be > 0) |
| `integer` | `Dimension("max_depth", "integer", 1, 4)` |
| `categorical` | `Dimension("loss", "categorical", choices=("squared_error", "huber"))` |

Integer and categorical values reach your objective as `int` and `str`.

### What you get back

- `result`: dict with `best_score`, `best_params`, `best_x`, `evaluations`, `successful_evaluations`,
  `failed_evaluations`, `elapsed_seconds`, `hmm`, `run_id`.
- `opt.records`: one dict per evaluation with `step`, `score`, `status`, `z`, `policy`, `belief`,
  `elapsed`, `params`. `z` is how surprising the score was versus the GP's prediction; `belief` is the
  HMM's probability over [explore, exploit, refine].
- `opt.hmm.diagnostics()`: learned transition matrix, state means and stds, log-likelihood.

### Configuration (all keys optional)

```python
config = {
    "seed": 0,
    "optimizer": {
        "method": "abo_lsr",      # abo_lsr | gp_ucb | gp_ei | scheduled_ucb | gp_hedge | random
        "controller": "soft",     # soft | hard (abo_lsr only)
        "candidate_pool": 2048,
        "gp_restarts": 2,
    },
    "hmm": {"enabled": True, "warmup": 8, "em_every": 3, "em_iterations": 20},
    "acquisition": {"ei_xi": 0.01},
}
```

Lower `candidate_pool` and `gp_restarts` for faster runs. To compare against a baseline, rerun with the
same `seed` and `budget` and `"method": "gp_ei"` or `"random"`. An unknown `method` or `controller`
raises `ValueError`.

### Saving a run

```python
BayesianOptimizer(space, objective, config, output_dir="runs/exp1").run(budget=30)
```

Writes `runs/exp1/summary.json` and `runs/exp1/experiments.sqlite` (one row per evaluation).
Without `output_dir`, nothing is written to disk.

### Failing objectives

If the objective raises or returns NaN/inf, that call is recorded with a `failed:...` status and uses up
budget, but is never fed to the GP. The run continues.

### Limits

Best for small, expensive problems (roughly 10-200 evaluations): the exact GP costs O(n^3) in the number
of evaluations. The HMM states are a modelling hypothesis, not proof of distinct optimization phases; see
the limitations below and compare against baselines over several seeds.

## Development

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS/Linux:         source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

## Run the experiment

Requires the `experiments` extra. The default config is bundled with the package:

```bash
abo-lsr california
abo-lsr california --config my_config.yaml --output results/my_run
```

Or without the console script: `python -m abo_lsr.cli california`. A copy of the default config is
at `src/abo_lsr/configs/california_housing.yaml`.

The first run requires internet access to download California Housing unless it is already cached. If the default Figshare URL returns HTTP 403, the loader tries a checksum-verified S3 mirror. The fallback cache is `abo_lsr_california_housing.npz` inside scikit-learn's data-home directory. For a quick smoke test, make a copy of the YAML and reduce `seeds` to one seed, `budget` to 8, `initial_points` to 4, and `candidate_pool` to 128. Such a smoke test is not a publication-quality experiment.

## Research study

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
