"""Leakage-controlled California Housing regression BO experiments."""
from __future__ import annotations

import hashlib
import io
import json
import logging
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.datasets import fetch_california_housing, get_data_home
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .acquisition import expected_improvement, local_candidates, sobol_candidates, ucb
from .gp import ExactRBFGaussianProcess
from .hmm import GaussianHMM

LOGGER = logging.getLogger(__name__)

CALIFORNIA_ARCHIVE_SHA256 = "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681"
CALIFORNIA_ARCHIVE_URLS = (
    "https://s3.amazonaws.com/sagemaker-sample-files/datasets/tabular/california_housing/cal_housing.tgz",
    "https://ndownloader.figshare.com/files/5976036",
)

METHODS = ("random", "gp_ucb", "gp_ei", "scheduled_ucb", "gp_hedge",
           "abo_lsr_hard", "abo_lsr_soft", "abo_no_hmm", "abo_fixed_hmm")
FAMILIES = ("ridge", "lasso", "elasticnet")


def _space(family: str, unit: np.ndarray) -> dict:
    alpha = float(10.0 ** (-4.0 + 6.0 * unit[0]))
    if family == "ridge":
        return {"alpha": alpha}
    if family == "lasso":
        return {"alpha": alpha}
    return {"alpha": alpha, "l1_ratio": float(0.05 + 0.95 * unit[1])}


def _model(family: str, params: dict, seed: int):
    common = {"alpha": params["alpha"], "max_iter": 20000, "tol": 1e-5}
    if family == "ridge":
        estimator = Ridge(alpha=common["alpha"])
    elif family == "lasso":
        estimator = Lasso(**common, selection="cyclic")
    elif family == "elasticnet":
        estimator = ElasticNet(**common, l1_ratio=params["l1_ratio"], selection="cyclic")
    else:
        raise ValueError(f"Unknown model family: {family}")
    return make_pipeline(StandardScaler(), estimator)


def _fit_score(family, params, seed, x_train, y_train, x_val, y_val):
    model = _model(family, params, seed)
    model.fit(x_train, y_train)
    return float(r2_score(y_val, model.predict(x_val)))


def _proposal(method, controller, gp, X, y, belief, incumbent, rng, step, pool_size, hedge_gains, eta):
    d = X.shape[1]
    meta = {}
    if method == "random":
        return rng.random(d), "random", None, hedge_gains, meta
    candidates = sobol_candidates(d, pool_size, int(rng.integers(0, 2**31 - 1)))
    mu, sd = gp.predict(candidates)
    best = float(np.max(y))
    if method == "gp_ucb":
        scores = ucb(mu, sd, 2.0); return candidates[int(np.argmax(scores))], "gp_ucb", None, hedge_gains, meta
    if method == "gp_ei":
        scores = expected_improvement(mu, sd, best); return candidates[int(np.argmax(scores))], "gp_ei", None, hedge_gains, meta
    if method == "scheduled_ucb":
        beta = max(0.15, 6.0 * (0.97 ** step))
        return candidates[int(np.argmax(ucb(mu, sd, beta)))], "scheduled_ucb", None, hedge_gains, {"beta": beta}
    if method == "gp_hedge":
        local = local_candidates(incumbent, max(128, pool_size // 4), 0.10, rng)
        lm, ls = gp.predict(local)
        points = np.vstack((candidates[int(np.argmax(ucb(mu, sd, 6.0)))],
                            candidates[int(np.argmax(expected_improvement(mu, sd, best)))] ,
                            local[int(np.argmax(ucb(lm, ls, 0.5)))]))
        logits = eta * (hedge_gains - np.max(hedge_gains))
        probs = np.exp(logits); probs /= probs.sum()
        expert = int(rng.choice(3, p=probs))
        return points[expert], ("ucb", "ei", "local")[expert], points, hedge_gains, {"hedge_probabilities": probs.tolist(), "selected_expert": expert}

    if method == "abo_no_hmm":
        belief = np.array([0.0, 1.0, 0.0])
    elif method == "abo_fixed_hmm":
        pass
    b = np.asarray(belief, float); b = b / max(float(b.sum()), 1e-15)
    beta = {"explore": 6.0, "exploit": 1.5, "refine": 0.25}
    if controller == "hard":
        state = int(np.argmax(b)); name = ("explore", "exploit", "refine")[state]
        if name == "refine":
            local = local_candidates(incumbent, pool_size, 0.035, rng)
            lm, ls = gp.predict(local)
            return local[int(np.argmax(ucb(lm, ls, beta[name])))], name, None, hedge_gains, {"belief": b.tolist()}
        score = ucb(mu, sd, beta[name]) if name == "explore" else expected_improvement(mu, sd, best)
        return candidates[int(np.argmax(score))], name, None, hedge_gains, {"belief": b.tolist()}

    scores = [ucb(mu, sd, beta["explore"]), expected_improvement(mu, sd, best),
              -np.linalg.norm(candidates - incumbent, axis=1)]
    normalized = []
    for score in scores:
        lo, hi = float(np.min(score)), float(np.max(score))
        normalized.append((score - lo) / (hi - lo + 1e-12))
    combined = np.sum(np.asarray(normalized) * b[:, None], axis=0)
    return candidates[int(np.argmax(combined))], "soft_mixture", None, hedge_gains, {"belief": b.tolist()}



def _update_hedge_gains(gp, expert_points, gains):
    """Apply GP-Hedge's post-observation posterior-mean expert gain update."""
    posterior_means, _ = gp.predict(np.asarray(expert_points, float))
    return np.asarray(gains, float) + np.asarray(posterior_means, float)


def _run_optimizer(family, method, controller, seed, budget, initial_points, split, config):
    x_train, x_val, y_train, y_val = split[:4]
    dim = 2 if family == "elasticnet" else 1
    rng = np.random.default_rng(seed)
    initial = rng.random((initial_points, dim))
    X, y, records = [], [], []
    hmm = GaussianHMM(**config.get("hmm_init", {}))
    belief = np.array([0.0, 1.0, 0.0]) if method == "abo_no_hmm" else hmm.pi.copy()
    residuals = []
    gp = ExactRBFGaussianProcess(noise=1e-5, optimize_every=3,
                                 restarts=config.get("gp_restarts", 1), seed=seed)
    hedge_gains = np.zeros(3, dtype=float)
    started = time.perf_counter()
    for step in range(budget):
        if step < initial_points:
            point, policy, hedge_points, meta = initial[step], "initial_random", None, {}
            mu_pre = sd_pre = None
        else:
            if method == "random":
                point, policy, hedge_points, hedge_gains, meta = _proposal(
                    method, controller, gp, np.asarray(X), np.asarray(y), belief,
                    np.asarray(X)[int(np.argmax(y))], rng, step,
                    int(config.get("candidate_pool", 1024)), hedge_gains,
                    float(config.get("hedge_eta", 0.5)))
                mu_pre = sd_pre = None
            else:
                gp.fit(np.asarray(X), np.asarray(y))
                incumbent = np.asarray(X)[int(np.argmax(y))]
                point, policy, hedge_points, hedge_gains, meta = _proposal(
                    method, controller, gp, np.asarray(X), np.asarray(y), belief, incumbent,
                    rng, step, int(config.get("candidate_pool", 1024)), hedge_gains,
                    float(config.get("hedge_eta", 0.5)))
                m, s = gp.predict(point[None, :]); mu_pre, sd_pre = float(m[0]), float(s[0])
        params = _space(family, point)
        eval_start = time.perf_counter()
        score = _fit_score(family, params, seed, x_train, y_train, x_val, y_val)
        eval_seconds = time.perf_counter() - eval_start
        z = None
        if mu_pre is not None:
            z = float((score - mu_pre) / max(sd_pre, 1e-8)); residuals.append(z)
            if method not in {"abo_no_hmm"}:
                hmm.filter(z)
                if method != "abo_fixed_hmm" and len(residuals) >= int(config.get("hmm_warmup", 8)) and len(residuals) % int(config.get("hmm_em_every", 3)) == 0:
                    hmm.fit_em(residuals, n_iter=int(config.get("hmm_em_iterations", 15)))
                belief = hmm.last_belief.copy()
        X.append(point.copy()); y.append(score)
        # GP-Hedge gains are updated only after the selected candidate is observed and the GP is refit.
        if method == "gp_hedge" and hedge_points is not None:
            gp.fit(np.asarray(X), np.asarray(y), force=True)
            hedge_gains = _update_hedge_gains(gp, hedge_points, hedge_gains)
        records.append({"seed": seed, "family": family, "method": method,
                        "evaluation": step + 1, "validation_r2": score,
                        "best_validation_r2": float(np.max(y)), "policy": policy,
                        "z": z, "belief_explore": float(belief[0]),
                        "belief_exploit": float(belief[1]), "belief_refine": float(belief[2]),
                        "evaluation_seconds": eval_seconds, "params": json.dumps(params),
                        **{f"meta_{k}": json.dumps(v) if isinstance(v, (list, dict)) else v for k,v in meta.items()}})
    best_i = int(np.argmax(y)); best_params = _space(family, X[best_i])
    # Final test assessment: refit on train + validation, then evaluate test exactly once.
    x_test, y_test = split[4], split[5]
    x_trainval = np.concatenate((x_train, x_val)); y_trainval = np.concatenate((y_train, y_val))
    final_model = _model(family, best_params, seed)
    final_model.fit(x_trainval, y_trainval)
    prediction = final_model.predict(x_test)
    elapsed = time.perf_counter() - started
    metrics = {"seed": seed, "family": family, "method": method,
               "validation_r2": float(y[best_i]), "test_r2": float(r2_score(y_test, prediction)),
               "test_rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
               "test_mae": float(mean_absolute_error(y_test, prediction)),
               "runtime_seconds": float(elapsed), "evaluation_count": len(records),
               "initial_evaluation_count": initial_points, "best_params": json.dumps(best_params),
               "hmm_diagnostics": json.dumps(hmm.diagnostics())}
    return metrics, records


def _decode_california_housing_archive(archive_bytes: bytes):
    """Decode the official scikit-learn California Housing archive format."""
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        member = archive.extractfile("CaliforniaHousing/cal_housing.data")
        if member is None:
            raise ValueError("Archive does not contain CaliforniaHousing/cal_housing.data")
        raw = np.loadtxt(member, delimiter=",")

    if raw.shape != (20640, 9) or not np.isfinite(raw).all():
        raise ValueError(f"Unexpected California Housing data shape/content: {raw.shape}")

    # Match sklearn.datasets.fetch_california_housing's column order and transforms.
    raw = raw[:, [8, 7, 2, 3, 4, 5, 6, 1, 0]]
    target = raw[:, 0] / 100000.0
    features = raw[:, 1:].copy()
    features[:, 2] /= features[:, 5]  # average rooms per household
    features[:, 3] /= features[:, 5]  # average bedrooms per household
    features[:, 5] = features[:, 4] / features[:, 5]  # average occupancy
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("California Housing archive produced non-finite values")
    return features, target


def _load_california_housing_data():
    """Load sklearn's California Housing data, with a checksum-verified mirror fallback.

    The fallback is useful when Figshare returns HTTP 403 for sklearn's default URL.
    The archive checksum is the one published in scikit-learn's dataset loader.
    """
    cache_path = Path(get_data_home()) / "abo_lsr_california_housing.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["data"], cached["target"]

    try:
        bunch = fetch_california_housing(n_retries=1, delay=0.2)
        return bunch.data, bunch.target
    except Exception as sklearn_error:
        LOGGER.warning(
            "scikit-learn California Housing download failed (%s). Trying verified mirrors.",
            sklearn_error,
        )

    errors = []
    for url in CALIFORNIA_ARCHIVE_URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "abo-lsr/0.2.1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                archive_bytes = response.read()
            digest = hashlib.sha256(archive_bytes).hexdigest()
            if digest != CALIFORNIA_ARCHIVE_SHA256:
                raise ValueError(
                    f"SHA-256 mismatch for {url}: expected {CALIFORNIA_ARCHIVE_SHA256}, got {digest}"
                )
            features, target = _decode_california_housing_archive(archive_bytes)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, data=features, target=target)
            LOGGER.info("Downloaded and cached California Housing data at %s", cache_path)
            return features, target
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    detail = "\n".join(errors)
    raise RuntimeError(
        "Could not download the real California Housing dataset. The default scikit-learn "
        "Figshare URL and configured mirrors were unavailable. Check internet/proxy access. "
        "No experiment results were generated. Attempts:\n" + detail
    )


def _split_dataset(seed, test_size, validation_size, dataset=None):
    if dataset is None:
        dataset = _load_california_housing_data()
    features, target = dataset
    x_dev, x_test, y_dev, y_test = train_test_split(features, target,
        test_size=test_size, random_state=seed)
    relative_val = validation_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_dev, y_dev,
        test_size=relative_val, random_state=seed + 10000)
    return x_train, x_val, y_train, y_val, x_test, y_test


def _ols_reference(seed, split):
    x_train, x_val, y_train, y_val, x_test, y_test = split
    start = time.perf_counter()
    validation_model = make_pipeline(StandardScaler(), LinearRegression())
    validation_model.fit(x_train, y_train)
    validation_score = float(r2_score(y_val, validation_model.predict(x_val)))
    final_model = make_pipeline(StandardScaler(), LinearRegression())
    final_model.fit(np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val)))
    pred = final_model.predict(x_test); runtime = time.perf_counter() - start
    return {"seed": seed, "family": "ols", "method": "ols_reference", "validation_r2": validation_score,
            "test_r2": float(r2_score(y_test, pred)), "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "test_mae": float(mean_absolute_error(y_test, pred)), "runtime_seconds": runtime,
            "evaluation_count": 1, "initial_evaluation_count": 0, "best_params": "{}", "hmm_diagnostics": "{}"}


def _bootstrap_ci(values, confidence=0.95, n_boot=5000, seed=123):
    values = np.asarray(values, float)
    if len(values) == 0: return [None, None]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2
    return [float(np.quantile(samples, alpha)), float(np.quantile(samples, 1-alpha))]


def _aggregate(metrics):
    rows = []
    for (family, method), group in metrics.groupby(["family", "method"]):
        row = {"family": family, "method": method, "n_seeds": len(group)}
        for col in ("validation_r2", "test_r2", "test_rmse", "test_mae", "runtime_seconds", "evaluation_count"):
            vals = group[col].astype(float).to_numpy()
            row[f"{col}_mean"] = float(np.mean(vals)); row[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0
            stable_seed = sum(ord(ch) for ch in f"{family}|{method}|{col}")
            row[f"{col}_ci95_low"], row[f"{col}_ci95_high"] = _bootstrap_ci(vals, seed=stable_seed)
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_tests(metrics, reference="abo_lsr_soft"):
    rows=[]
    for family in FAMILIES:
        sub=metrics[metrics.family==family]
        ref=sub[sub.method==reference].set_index("seed").test_r2
        if ref.empty: continue
        for method in sorted(set(sub.method)-{reference}):
            other=sub[sub.method==method].set_index("seed").test_r2
            common=ref.index.intersection(other.index)
            a,b=ref.loc[common].to_numpy(),other.loc[common].to_numpy()
            if len(a)<2: stat,p=np.nan,np.nan
            elif np.allclose(a,b): stat,p=0.0,1.0
            else:
                try: result=wilcoxon(a,b,alternative="two-sided"); stat,p=float(result.statistic),float(result.pvalue)
                except ValueError: stat,p=np.nan,np.nan
            rows.append({"family":family,"reference":reference,"comparison":method,"n_pairs":len(a),"wilcoxon_statistic":stat,"p_value":p,"mean_paired_difference_reference_minus_comparison":float(np.mean(a-b)) if len(a) else np.nan})
    result = pd.DataFrame(rows)
    if len(result):
        # Holm step-down family-wise correction within each model family.
        result["p_value_holm"] = np.nan
        for family, indexes in result.groupby("family").groups.items():
            valid = result.loc[indexes, "p_value"].dropna().sort_values()
            m = len(valid); adjusted = []
            running = 0.0
            for rank, (idx, pval) in enumerate(valid.items()):
                running = max(running, min(1.0, (m-rank)*float(pval)))
                adjusted.append((idx, running))
            for idx, value in adjusted: result.loc[idx, "p_value_holm"] = value
    return result


def run_california_experiment(config: dict, output_dir="results/california_housing"):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    seeds=list(config.get("seeds",[17,23,41,59,71]))
    methods=list(config.get("methods",METHODS)); families=list(config.get("families",FAMILIES))
    budget=int(config.get("budget",24)); initial=int(config.get("initial_points",6))
    if budget <= initial or initial < 2: raise ValueError("Require budget > initial_points >= 2")
    metric_rows=[]; trace_rows=[]
    dataset = _load_california_housing_data()
    split_cache={seed:_split_dataset(seed,float(config.get("test_size",0.2)),float(config.get("validation_size",0.2)),dataset) for seed in seeds}
    for seed in seeds:
        split=split_cache[seed]
        metric_rows.append(_ols_reference(seed,split))
        for family in families:
            for method in methods:
                controller="hard" if method=="abo_lsr_hard" else "soft"
                metrics, traces = _run_optimizer(family,method,controller,seed,budget,initial,split,config)
                metric_rows.append(metrics); trace_rows.extend(traces)
                pd.DataFrame(metric_rows).to_csv(out/"per_seed_metrics.csv",index=False)
                pd.DataFrame(trace_rows).to_csv(out/"optimization_traces.csv",index=False)
    metrics=pd.DataFrame(metric_rows); traces=pd.DataFrame(trace_rows)
    aggregate=_aggregate(metrics); paired=_paired_tests(metrics)
    metrics.to_csv(out/"per_seed_metrics.csv",index=False)
    aggregate.to_csv(out/"aggregate_metrics.csv",index=False)
    paired.to_csv(out/"paired_statistical_tests.csv",index=False)
    traces.to_csv(out/"optimization_traces.csv",index=False)
    metadata={"dataset":"California Housing (scikit-learn fetch_california_housing)","seeds":seeds,
              "methods":methods,"families":families,"budget":budget,"initial_points":initial,
              "train_fraction":1.0-float(config.get("test_size",0.2))-float(config.get("validation_size",0.2)),
              "validation_fraction":float(config.get("validation_size",0.2)),
              "test_fraction":float(config.get("test_size",0.2)),
              "test_set_used_for_optimization":False,"status":"completed"}
    (out/"experiment_metadata.json").write_text(json.dumps(metadata,indent=2))
    return {"output_dir":str(out),"runs":len(metrics),"aggregate":aggregate,"paired_tests":paired}
