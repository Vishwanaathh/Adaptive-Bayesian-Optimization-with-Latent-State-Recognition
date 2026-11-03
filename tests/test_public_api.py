import abo_lsr


def test_public_api_exports():
    for name in ("BayesianOptimizer", "SearchSpace", "Dimension",
                 "GaussianHMM", "ExactRBFGaussianProcess"):
        assert hasattr(abo_lsr, name)
    assert abo_lsr.__version__


def test_bundled_default_config_loads():
    from abo_lsr.cli import DEFAULT_CONFIG, load_config
    config = load_config(DEFAULT_CONFIG)
    assert config["budget"] > config["initial_points"]
