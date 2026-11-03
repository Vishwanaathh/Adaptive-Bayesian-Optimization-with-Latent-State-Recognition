import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from abo_lsr.experiment import _run_optimizer, _update_hedge_gains
from abo_lsr.gp import ExactRBFGaussianProcess


def test_hedge_gain_uses_current_gp_posterior():
    x = np.array([[0.0], [0.4], [0.8]])
    y = np.array([0.0, 0.2, 0.9])
    gp = ExactRBFGaussianProcess(restarts=0).fit(x, y, force=True)
    proposals = np.array([[0.1], [0.5], [0.95]])
    gains = np.zeros(3)
    updated = _update_hedge_gains(gp, proposals, gains)
    means, _ = gp.predict(proposals)
    assert np.allclose(updated, means)
    assert not np.shares_memory(updated, gains)


def test_regression_experiment_runner_reports_test_metrics():
    data = load_diabetes()
    x_dev, x_test, y_dev, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=7)
    x_train, x_val, y_train, y_val = train_test_split(x_dev, y_dev, test_size=0.25, random_state=8)
    metrics, traces = _run_optimizer(
        family="ridge", method="abo_lsr_soft", controller="soft", seed=7,
        budget=8, initial_points=4,
        split=(x_train, x_val, y_train, y_val, x_test, y_test),
        config={"candidate_pool": 48, "gp_restarts": 0, "hmm_warmup": 8},
    )
    assert metrics["evaluation_count"] == 8
    assert np.isfinite(metrics["validation_r2"])
    assert np.isfinite(metrics["test_r2"])
    assert np.isfinite(metrics["test_rmse"])
    assert np.isfinite(metrics["test_mae"])
    assert len(traces) == 8


def test_aggregate_and_paired_statistics_are_emitted():
    import pandas as pd
    from abo_lsr.experiment import _aggregate, _paired_tests

    rows = []
    for seed, shift in [(1, 0.0), (2, 0.1), (3, -0.1), (4, 0.05)]:
        rows.append({"seed": seed, "family": "ridge", "method": "abo_lsr_soft",
                     "validation_r2": .6 + shift, "test_r2": .55 + shift,
                     "test_rmse": 0.7 - shift, "test_mae": 0.5 - shift,
                     "runtime_seconds": 1.0, "evaluation_count": 12})
        rows.append({"seed": seed, "family": "ridge", "method": "gp_ei",
                     "validation_r2": .5 + shift, "test_r2": .45 + shift,
                     "test_rmse": 0.8 - shift, "test_mae": 0.6 - shift,
                     "runtime_seconds": 0.8, "evaluation_count": 12})
    frame = pd.DataFrame(rows)
    aggregate = _aggregate(frame)
    paired = _paired_tests(frame)
    assert len(aggregate) == 2
    assert "test_r2_ci95_low" in aggregate.columns
    assert len(paired) == 1
    assert paired.iloc[0]["n_pairs"] == 4
    assert "p_value_holm" in paired.columns


def test_california_archive_decoder_matches_expected_schema():
    import io
    import tarfile

    from abo_lsr.experiment import _decode_california_housing_archive

    # Parser fixture only: this is not used as an optimization objective or reported result.
    raw = np.ones((20640, 9), dtype=float)
    raw[:, 0] = 200000.0
    raw[:, 1] = 2.0
    raw[:, 2] = 3.0
    raw[:, 3] = 4.0
    raw[:, 4] = 5.0
    raw[:, 5] = 6.0
    raw[:, 6] = 2.0
    raw[:, 7] = 34.0
    raw[:, 8] = 200000.0
    csv_bytes = ("\n".join(",".join(map(str, row)) for row in raw)).encode()

    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("CaliforniaHousing/cal_housing.data")
        info.size = len(csv_bytes)
        archive.addfile(info, io.BytesIO(csv_bytes))

    x, y = _decode_california_housing_archive(archive_buffer.getvalue())
    assert x.shape == (20640, 8)
    assert y.shape == (20640,)
    assert np.allclose(y, 2.0)
    assert np.isfinite(x).all()
