from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold

from common.model_utils import (
    DATA_SEED,
    MODEL_SEED,
    TRAIN_TEST_SEED,
    TRAIN_VALIDATION_SEED,
    _choose_threshold,
    _collect_metrics,
    _make_fairness_table,
)
from common.paths import PROJECT_ROOT
from experiments.tuning.run_tuning import load_splits
from models.trees.lightgbm_model import create_model


def main() -> None:
    project_dir = PROJECT_ROOT
    x_train, y_train, x_validation, y_validation, x_test, y_test = load_splits(
        project_dir
    )
    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=2,
        random_state=MODEL_SEED,
    )
    search = RandomizedSearchCV(
        estimator=create_model(),
        param_distributions={
            "classifier__n_estimators": [100, 200, 400],
            "classifier__learning_rate": [0.02, 0.05, 0.1],
            "classifier__num_leaves": [7, 15, 31],
            "classifier__max_depth": [-1, 3, 5],
            "classifier__min_child_samples": [10, 20, 30],
            "classifier__reg_lambda": [0.0, 1.0, 5.0],
            "classifier__class_weight": [None, "balanced"],
            "classifier__colsample_bytree": [0.8, 1.0],
        },
        n_iter=12,
        scoring="average_precision",
        cv=cv,
        n_jobs=1,
        random_state=MODEL_SEED,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_

    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    threshold = _choose_threshold(y_validation, validation_probabilities)
    test_probabilities = model.predict_proba(x_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)
    model_name = "LightGBM (tuned)"

    metrics_frame = pd.DataFrame(
        [
            _collect_metrics(
                model_name,
                "validation",
                y_validation,
                (validation_probabilities >= threshold).astype(int),
                validation_probabilities,
                threshold,
            ),
            _collect_metrics(
                model_name,
                "test",
                y_test,
                test_predictions,
                test_probabilities,
                threshold,
            ),
        ]
    )

    output_dir = project_dir / "reports_lightgbm"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(output_dir / "lightgbm_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": model_name,
                "cv_splits": cv.get_n_splits(x_train, y_train),
                "candidate_count": len(search.cv_results_["params"]),
                "cv_mean_average_precision": search.best_score_,
                "cv_std_average_precision": search.cv_results_["std_test_score"][
                    search.best_index_
                ],
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                "validation_threshold": threshold,
            }
        ]
    ).to_csv(output_dir / "lightgbm_search_summary.csv", index=False)

    data = pd.read_csv(project_dir / "loan_applications.csv")
    labeled = data.loc[data["actual_outcome"].isin(["repaid", "defaulted"])].copy()
    test_data = labeled.loc[x_test.index]
    fairness = _make_fairness_table(
        test_data,
        y_test,
        {model_name: test_predictions},
    )
    fairness.to_csv(output_dir / "lightgbm_fairness_metrics.csv", index=False)
    pd.DataFrame(
        {
            "applicant_id": test_data["applicant_id"].to_numpy(),
            "actual_outcome": test_data["actual_outcome"].to_numpy(),
            "employment_status": test_data["employment_status"].to_numpy(),
            "model": model_name,
            "default_probability": test_probabilities,
            "threshold": threshold,
            "predicted_default": test_predictions,
            "predicted_decision": np.where(
                test_predictions == 1,
                "deny_or_review",
                "approve",
            ),
        }
    ).to_csv(output_dir / "lightgbm_test_predictions.csv", index=False)

    comparison_paths = [
        project_dir / "reports_catboost" / "model_metrics.csv",
        project_dir / "reports_tuned" / "tuned_model_metrics.csv",
        project_dir / "reports_woe" / "woe_logistic_metrics.csv",
        project_dir / "reports_tabpfn_v2" / "tabpfn_v2_metrics.csv",
    ]
    comparison_rows: list[pd.DataFrame] = []
    for path in comparison_paths:
        if not path.exists():
            continue
        existing = pd.read_csv(path)
        existing_test = existing.loc[existing["split"] == "test"]
        if path.name == "tuned_model_metrics.csv":
            existing_test = existing_test.loc[
                existing_test["model"] == "Logistic Regression (tuned)"
            ]
        elif path.name == "woe_logistic_metrics.csv":
            existing_test = existing_test.loc[
                existing_test["model"] == "Logistic Regression (WOE)"
            ]
        elif path.name == "tabpfn_v2_metrics.csv":
            existing_test = existing_test.loc[
                existing_test["model"] == "TabPFNv2 (Nature paper)"
            ]
        elif path.name == "model_metrics.csv":
            existing_test = existing_test.loc[
                existing_test["model"] == "Rule-based baseline"
            ]
        if not existing_test.empty:
            comparison_rows.append(existing_test)

    comparison_rows.append(metrics_frame.loc[metrics_frame["split"] == "test"])
    comparison = pd.concat(comparison_rows, ignore_index=True)
    order = {
        "Rule-based baseline": 0,
        "Logistic Regression (tuned)": 1,
        "Logistic Regression (WOE)": 2,
        model_name: 3,
        "TabPFNv2 (Nature paper)": 4,
    }
    comparison["_order"] = comparison["model"].map(order)
    comparison = comparison.sort_values("_order").drop(columns="_order")
    comparison.to_csv(output_dir / "risk_model_comparison.csv", index=False)

    print(
        metrics_frame[
            [
                "model",
                "split",
                "threshold",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "defaults_caught",
                "good_applicants_flagged_or_denied",
                "approval_rate",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print("\nTest-set comparison:")
    print(
        comparison[
            [
                "model",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "defaults_caught",
                "good_applicants_flagged_or_denied",
                "approval_rate",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print(
        f"\nLightGBM reports saved to {output_dir}\n"
        f"Seeds: data={DATA_SEED}, train/test={TRAIN_TEST_SEED}, "
        f"train/validation={TRAIN_VALIDATION_SEED}, CV/model={MODEL_SEED}"
    )


if __name__ == "__main__":
    main()
