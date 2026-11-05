from __future__ import annotations

import argparse
from importlib import import_module


def main() -> None:
    commands = {
        "generate-data": "data.generate_dataset",
        "validate": "experiments.validation.run_credit_risk_validation",
        "baseline": "experiments.baseline.run_catboost",
        "tune": "experiments.tuning.run_tuning",
        "woe": "experiments.linear.run_woe_logistic_regression",
        "lightgbm": "experiments.trees.run_lightgbm",
        "tabpfn": "experiments.foundation.run_tabpfn_v2",
        "qwen": "experiments.foundation.run_qwen_llm",
        "alora": "experiments.foundation.run_qwen_alora",
    }
    parser = argparse.ArgumentParser(
        description="Run Bree's synthetic credit-risk modeling workflow."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=tuple(commands),
        default="validate",
        help="Choose the validation, data-generation, or standalone model workflow.",
    )
    args = parser.parse_args()

    module = import_module(commands[args.command])
    module.main()


if __name__ == "__main__":
    main()
