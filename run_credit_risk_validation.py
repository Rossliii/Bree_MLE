from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

from catboost_model import create_model as create_catboost
from credit_risk_validation import (
    SCENARIOS,
    evaluate_at_approval_rate,
    make_scenario_test_data,
)
from lightgbm_model import create_model as create_lightgbm
from logistic_regression_model import create_model as create_logistic_regression
from model_utils import (
    DATA_SEED,
    MODEL_SEED,
    _choose_threshold,
    _collect_metrics,
    _make_fairness_table,
    add_features,
    generate_dataset,
)
from tabpfn_v2_model import create_model as create_tabpfn
from woe_logistic_regression_model import create_model as create_woe_logistic


EXPERIMENT_SEEDS = (42, 2025, 2027)
TEST_SAMPLE_SIZE = 1200
APPROVAL_RATES = (0.45, 0.70)


def _probability_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def _fit_platt_scaler(
    raw_probabilities: np.ndarray,
    y_calibration: pd.Series,
    seed: int,
) -> LogisticRegression:
    calibrator = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=seed,
    )
    calibrator.fit(_probability_logits(raw_probabilities), y_calibration)
    return calibrator


def _calibrate(
    calibrator: LogisticRegression,
    raw_probabilities: np.ndarray,
) -> np.ndarray:
    return calibrator.predict_proba(_probability_logits(raw_probabilities))[:, 1]


def _fit_candidate_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    common = {
        "scoring": "average_precision",
        "cv": cv,
        "n_jobs": 1,
        "refit": True,
        "error_score": "raise",
        "return_train_score": False,
    }
    searches: dict[str, GridSearchCV | RandomizedSearchCV] = {
        "Logistic Regression (tuned)": GridSearchCV(
            create_logistic_regression(),
            {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__class_weight": ["balanced", None],
            },
            **common,
        ),
        "Logistic Regression (WOE)": GridSearchCV(
            create_woe_logistic(),
            {
                "woe__n_bins": [3, 5, 8],
                "classifier__C": [0.01, 0.1, 1.0],
                "classifier__class_weight": ["balanced", None],
            },
            **common,
        ),
        "CatBoost (tuned)": RandomizedSearchCV(
            create_catboost(),
            {
                "classifier__iterations": [250, 400, 600],
                "classifier__depth": [3, 4, 5],
                "classifier__learning_rate": [0.03, 0.06],
                "classifier__l2_leaf_reg": [3, 7, 10],
            },
            n_iter=5,
            random_state=seed,
            **common,
        ),
        "LightGBM (tuned)": RandomizedSearchCV(
            create_lightgbm(),
            {
                "classifier__n_estimators": [100, 200, 400],
                "classifier__learning_rate": [0.02, 0.05, 0.1],
                "classifier__num_leaves": [7, 15, 31],
                "classifier__max_depth": [-1, 3, 5],
                "classifier__min_child_samples": [10, 20, 30],
                "classifier__reg_lambda": [0.0, 1.0, 5.0],
                "classifier__class_weight": [None, "balanced"],
                "classifier__colsample_bytree": [0.8, 1.0],
            },
            n_iter=6,
            random_state=seed,
            **common,
        ),
    }

    fitted_models: dict[str, object] = {}
    search_rows: list[dict[str, object]] = []
    for model_name, search in searches.items():
        print(f"Seed {seed}: tuning {model_name}...", flush=True)
        search.fit(x_train, y_train)
        fitted_models[model_name] = search.best_estimator_
        search_rows.append(
            {
                "seed": seed,
                "model": model_name,
                "cv_folds": cv.get_n_splits(x_train, y_train),
                "candidate_count": len(search.cv_results_["params"]),
                "cv_mean_pr_auc": search.best_score_,
                "cv_std_pr_auc": search.cv_results_["std_test_score"][
                    search.best_index_
                ],
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            }
        )
    return fitted_models, search_rows


def _calibration_rows(
    y_true: pd.Series,
    probabilities: np.ndarray,
    model_name: str,
    scenario: str,
    seed: int,
    calibration_method: str,
) -> list[dict[str, object]]:
    bins = pd.qcut(
        pd.Series(probabilities),
        q=min(10, len(probabilities)),
        labels=False,
        duplicates="drop",
    )
    calibration_data = pd.DataFrame(
        {
            "bin": bins,
            "predicted": probabilities,
            "observed": np.asarray(y_true),
        }
    )
    grouped = calibration_data.groupby("bin", observed=True).agg(
        mean_predicted_pd=("predicted", "mean"),
        observed_default_rate=("observed", "mean"),
        applicant_count=("observed", "size"),
    )
    return [
        {
            "seed": seed,
            "scenario": scenario,
            "model": model_name,
            "calibration_method": calibration_method,
            "bin": bin_index + 1,
            "mean_predicted_pd": float(row["mean_predicted_pd"]),
            "observed_default_rate": float(row["observed_default_rate"]),
            "applicant_count": int(row["applicant_count"]),
        }
        for bin_index, (_, row) in enumerate(grouped.iterrows())
    ]


def _rule_baseline_metrics(
    data: pd.DataFrame,
    y_true: pd.Series,
    scenario: str,
    seed: int,
) -> dict[str, object]:
    predictions = (
        data["rule_based_decision"].to_numpy() != "approved"
    ).astype(int)
    risk_scores = (100 - data["rule_based_score"].to_numpy()) / 100
    row = _collect_metrics(
        "Rule-based baseline",
        "test",
        y_true,
        predictions,
        risk_scores,
        0.25,
    )
    row.update(
        {
            "seed": seed,
            "scenario": scenario,
            "default_prevalence": float(y_true.mean()),
            "brier_score_raw": np.nan,
            "brier_score_calibrated": np.nan,
            "log_loss_raw": np.nan,
            "log_loss_calibrated": np.nan,
        }
    )
    return row


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "reports_credit_risk_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []
    approval_rows: list[dict[str, object]] = []
    fairness_frames: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, object]] = []

    for seed in EXPERIMENT_SEEDS:
        source_data = generate_dataset(n=2000, random_state=seed)
        labeled = source_data.loc[
            source_data["actual_outcome"].isin(["repaid", "defaulted"])
        ].copy()
        y = (labeled["actual_outcome"] == "defaulted").astype(int)
        x = add_features(labeled)
        x_train, x_holdout, y_train, y_holdout = train_test_split(
            x,
            y,
            test_size=0.4,
            random_state=seed,
            stratify=y,
        )
        x_calibration, x_validation, y_calibration, y_validation = train_test_split(
            x_holdout,
            y_holdout,
            test_size=0.5,
            random_state=seed + 1,
            stratify=y_holdout,
        )

        fitted_models, seed_search_rows = _fit_candidate_models(
            x_train,
            y_train,
            seed,
        )
        search_rows.extend(seed_search_rows)
        print(f"Seed {seed}: fitting TabPFNv2 (Nature paper)...", flush=True)
        tabpfn_model = create_tabpfn(random_state=seed)
        tabpfn_model.fit(x_train, y_train)
        fitted_models["TabPFNv2 (Nature paper)"] = tabpfn_model

        calibrators: dict[str, LogisticRegression] = {}
        thresholds: dict[str, float] = {}
        for model_name, model in fitted_models.items():
            raw_calibration = model.predict_proba(x_calibration)[:, 1]
            calibrator = _fit_platt_scaler(raw_calibration, y_calibration, seed)
            calibrators[model_name] = calibrator
            raw_validation = model.predict_proba(x_validation)[:, 1]
            calibrated_validation = _calibrate(calibrator, raw_validation)
            thresholds[model_name] = _choose_threshold(
                y_validation,
                calibrated_validation,
            )

        for scenario_index, scenario in enumerate(SCENARIOS):
            scenario_data = make_scenario_test_data(
                scenario,
                n=TEST_SAMPLE_SIZE,
                random_state=seed + 10_000 + scenario_index * 100,
            )
            test_labeled = scenario_data.loc[
                scenario_data["actual_outcome"].isin(["repaid", "defaulted"])
            ].copy()
            y_test = (
                test_labeled["actual_outcome"] == "defaulted"
            ).astype(int)
            x_test = add_features(test_labeled)
            metric_rows.append(
                _rule_baseline_metrics(
                    test_labeled,
                    y_test,
                    scenario,
                    seed,
                )
            )

            model_predictions: dict[str, np.ndarray] = {}
            risk_scores_for_approval: dict[str, np.ndarray] = {
                "Rule-based baseline": (
                    100 - test_labeled["rule_based_score"].to_numpy()
                )
            }
            for model_name, model in fitted_models.items():
                raw_probabilities = model.predict_proba(x_test)[:, 1]
                calibrated_probabilities = _calibrate(
                    calibrators[model_name],
                    raw_probabilities,
                )
                predictions = (
                    calibrated_probabilities >= thresholds[model_name]
                ).astype(int)
                model_predictions[model_name] = predictions
                risk_scores_for_approval[model_name] = calibrated_probabilities

                metric_row = _collect_metrics(
                    model_name,
                    "test",
                    y_test,
                    predictions,
                    calibrated_probabilities,
                    thresholds[model_name],
                )
                metric_row.update(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "default_prevalence": float(y_test.mean()),
                        "brier_score_raw": brier_score_loss(
                            y_test,
                            raw_probabilities,
                        ),
                        "brier_score_calibrated": brier_score_loss(
                            y_test,
                            calibrated_probabilities,
                        ),
                        "log_loss_raw": log_loss(
                            y_test,
                            raw_probabilities,
                            labels=[0, 1],
                        ),
                        "log_loss_calibrated": log_loss(
                            y_test,
                            calibrated_probabilities,
                            labels=[0, 1],
                        ),
                    }
                )
                metric_rows.append(metric_row)

                for method, probabilities in (
                    ("raw", raw_probabilities),
                    ("platt", calibrated_probabilities),
                ):
                    calibration_rows.extend(
                        _calibration_rows(
                            y_test,
                            probabilities,
                            model_name,
                            scenario,
                            seed,
                            method,
                        )
                    )

            fairness = _make_fairness_table(
                test_labeled,
                y_test,
                model_predictions,
            )
            fairness["seed"] = seed
            fairness["scenario"] = scenario
            fairness_frames.append(fairness)

            for model_name, risk_scores in risk_scores_for_approval.items():
                for target_approval_rate in APPROVAL_RATES:
                    approval_rows.append(
                        {
                            "seed": seed,
                            "scenario": scenario,
                            "model": model_name,
                            "target_approval_rate": target_approval_rate,
                            **evaluate_at_approval_rate(
                                y_test,
                                np.asarray(risk_scores),
                                target_approval_rate,
                            ),
                        }
                    )
            print(
                f"Seed {seed}, scenario {scenario}: "
                f"default rate={y_test.mean():.3f}, "
                f"test applicants={len(y_test)}",
                flush=True,
            )

    metrics = pd.DataFrame(metric_rows)
    searches = pd.DataFrame(search_rows)
    approval_comparison = pd.DataFrame(approval_rows)
    fairness = pd.concat(fairness_frames, ignore_index=True)
    calibration_curves = pd.DataFrame(calibration_rows)

    metrics.to_csv(output_dir / "robustness_metrics.csv", index=False)
    searches.to_csv(output_dir / "model_search_summary.csv", index=False)
    approval_comparison.to_csv(
        output_dir / "equal_approval_comparison.csv",
        index=False,
    )
    fairness.to_csv(output_dir / "group_fairness_metrics.csv", index=False)
    calibration_curves.to_csv(
        output_dir / "calibration_curves.csv",
        index=False,
    )

    summary = (
        metrics.groupby(["scenario", "model"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            default_prevalence_mean=("default_prevalence", "mean"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            approval_rate_mean=("approval_rate", "mean"),
            brier_score_raw_mean=("brier_score_raw", "mean"),
            brier_score_calibrated_mean=("brier_score_calibrated", "mean"),
            log_loss_raw_mean=("log_loss_raw", "mean"),
            log_loss_calibrated_mean=("log_loss_calibrated", "mean"),
        )
        .sort_values(["scenario", "model"])
    )
    summary.to_csv(output_dir / "robustness_summary.csv", index=False)

    selected_columns = [
        "scenario",
        "model",
        "seed_count",
        "default_prevalence_mean",
        "f1_mean",
        "f1_std",
        "recall_mean",
        "pr_auc_mean",
        "brier_score_raw_mean",
        "brier_score_calibrated_mean",
    ]
    print("\nMulti-seed robustness summary:")
    print(
        summary[selected_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print("\nEqual-approval-rate comparison (mean across seeds):")
    approval_summary = (
        approval_comparison.groupby(
            ["scenario", "target_approval_rate", "model"],
            as_index=False,
        )
        .agg(
            actual_approval_rate=("approval_rate", "mean"),
            approved_default_rate=("approved_default_rate", "mean"),
            defaults_in_approved=("defaults_in_approved", "mean"),
            defaults_captured=("defaults_captured", "mean"),
            good_applicants_rejected=("good_applicants_rejected", "mean"),
        )
        .sort_values(["scenario", "target_approval_rate", "model"])
    )
    approval_summary.to_csv(
        output_dir / "equal_approval_summary.csv",
        index=False,
    )
    print(
        approval_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print(
        f"\nReports saved to {output_dir}. "
        "Expected Loss was not calculated because LGD/EAD assumptions were not provided.\n"
        f"Data seed base={DATA_SEED}; model seed base={MODEL_SEED}; "
        f"experiment seeds={EXPERIMENT_SEEDS}"
    )


if __name__ == "__main__":
    main()
