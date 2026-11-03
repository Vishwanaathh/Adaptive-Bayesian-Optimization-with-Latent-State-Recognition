from __future__ import annotations
import argparse
import sys
from importlib import resources
from pathlib import Path

DEFAULT_CONFIG = resources.files("abo_lsr") / "configs" / "california_housing.yaml"


def load_config(path):
    import yaml
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="abo-lsr")
    sub = parser.add_subparsers(dest="command", required=True)
    experiment = sub.add_parser("california", help="Run the real California Housing regression study")
    experiment.add_argument("--config", default=None,
                            help="YAML config (default: the config bundled with the package)")
    experiment.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    try:
        from .experiment import run_california_experiment
        config = load_config(args.config or DEFAULT_CONFIG)
    except ImportError as exc:
        sys.exit(f"The experiment runner needs extra dependencies ({exc}). "
                 'Install them with: pip install "abo-lsr[experiments]"')
    output = args.output or config.get("output_dir", "results/california_housing")
    result = run_california_experiment(config, Path(output))
    print(f"Completed {result['runs']} model/method/seed rows. Results: {result['output_dir']}")
    print(result["aggregate"].to_string(index=False))


if __name__ == "__main__":
    main()
