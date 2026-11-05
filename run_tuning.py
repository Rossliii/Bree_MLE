from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)

from catboost_model import create_model as create_catboost
from logistic_regression_model import create_model as create_logistic_regression
from model_utils import (
    MODEL_SEED,
    TRAIN_TEST_SEED,
    TRAIN_VALIDATION_SEED,
    _choose_threshold,
    _collect_metrics,
    _make_fairness_table,
    add_features,
)
from random_forest_model import create_model as create_random_forest


MODEL_FACTORIES: dict[str, Callable[[], object]] = {
    "Logistic Regression": create_logistic_regression,
    "Random Forest": create_random_forest,
    "CatBoost": create_catboost,
}

PARAMETER_SEARCHES: dict[str, dict[str, list[object]]] = {
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "Random Forest": {
        "classifier__n_estimators": [250, 400],
        "classifier__max_depth": [None, 6, 10],
        "classifier__min_samples_leaf": [2, 5, 10, 20],
        "classifier__max_features": ["sqrt", 0.7],
        "classifier__class_weight": ["balanced", "balanced_subsample"],
    },
    "CatBoost": {
        "classifier__iterations": [250, 400],
        "classifier__depth": [3, 4, 5, 6],
        "classifier__learning_rate": [0.03, 0.06],
        "classifier__l2_leaf_reg": [3, 7, 10],
    },
}

RANDOM_SEARCH_MODELS = {"Random Forest", "CatBoost"}


def load_splits(project_dir: Path) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    data = pd.read_csv(project_dir / "loan_applications.csv")
    labeled = data.loc[data["actual_outcome"].isin(["repaid", "defaulted"])].copy()
    target = (labeled["actual_outcome"] == "defaulted").astype(int)
    features = add_features(labeled)
    x_train_validation, x_test, y_train_validation, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=TRAIN_TEST_SEED,
        stratify=target,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=0.25,
        random_state=TRAIN_VALIDATION_SEED,
        stratify=y_train_validation,
    )
    return x_train, y_train, x_validation, y_validation, x_test, y_test


def search_model(
    model_name: str,
    estimator: object,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: RepeatedStratifiedKFold,
) -> tuple[object, dict[str, object]]:
    common = {
        "estimator": estimator,
        "scoring": "average_precision",
        "cv": cv,
        "n_jobs": 1,
        "refit": True,
        "return_train_score": False,
        "error_score": "raise",
    }
    if model_name in RANDOM_SEARCH_MODELS:
        search = RandomizedSearchCV(
            **common,
            param_distributions=PARAMETER_SEARCHES[model_name],
            n_iter=6,
            random_state=MODEL_SEED,
        )
    else:
        search = GridSearchCV(
            **common,
            param_grid=PARAMETER_SEARCHES[model_name],
        )

    search.fit(x_train, y_train)
    best_index = search.best_index_
    summary: dict[str, object] = {
        "model": model_name,
        "cv_splits": cv.get_n_splits(x_train, y_train),
        "candidate_count": len(search.cv_results_["params"]),
        "cv_mean_average_precision": search.best_score_,
        "cv_std_average_precision": search.cv_results_["std_test_score"][best_index],
        "best_params": json.dumps(search.best_params_, sort_keys=True),
    }
    return search.best_estimator_, summary


def evaluate_model(
    model_name: str,
    split_name: str,
    estimator: object,
    x: pd.DataFrame,
    y: pd.Series,
    threshold: float,
) -> tuple[dict[str, object], np.ndarray]:
    probabilities = estimator.predict_proba(x)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = _collect_metrics(
        model_name,
        split_name,
        y,
        predictions,
        probabilities,
        threshold,
    )
    return metrics, probabilities


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "reports_tuned"
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_validation, y_validation, x_test, y_test = load_splits(
        project_dir
    )
    all_data = pd.read_csv(project_dir / "loan_applications.csv")
    labeled_data = all_data.loc[
        all_data["actual_outcome"].isin(["repaid", "defaulted"])
    ]
    test_data = labeled_data.loc[x_test.index]
    applicant_ids = test_data["applicant_id"].to_numpy()
    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=2,
        random_state=MODEL_SEED,
    )

    metric_rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []
    test_predictions: list[pd.DataFrame] = []
    test_risk_predictions: dict[str, np.ndarray] = {}

    for model_name, model_factory in MODEL_FACTORIES.items():
        print(f"\nTuning {model_name} with repeated stratified CV...", flush=True)
        default_model = model_factory()
        default_cv_scores = cross_val_score(
            default_model,
            x_train,
            y_train,
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )
        default_cv_mean = float(default_cv_scores.mean())
        default_cv_std = float(default_cv_scores.std(ddof=1))

        tuned_model, search_summary = search_model(
            model_name,
            clone(default_model),
            x_train,
            y_train,
            cv,
        )
        tuned_validation_probabilities = tuned_model.predict_proba(x_validation)[:, 1]
        tuned_threshold = _choose_threshold(
            y_validation,
            tuned_validation_probabilities,
        )
        tuned_validation_metrics, _ = evaluate_model(
            f"{model_name} (tuned)",
            "validation",
            tuned_model,
            x_validation,
            y_validation,
            tuned_threshold,
        )
        tuned_test_metrics, tuned_test_probabilities = evaluate_model(
            f"{model_name} (tuned)",
            "test",
            tuned_model,
            x_test,
            y_test,
            tuned_threshold,
        )

        default_model.fit(x_train, y_train)
        default_validation_probabilities = default_model.predict_proba(x_validation)[:, 1]
        default_threshold = _choose_threshold(
            y_validation,
            default_validation_probabilities,
        )
        default_validation_metrics, _ = evaluate_model(
            f"{model_name} (default)",
            "validation",
            default_model,
            x_validation,
            y_validation,
            default_threshold,
        )
        default_test_metrics, default_test_probabilities = evaluate_model(
            f"{model_name} (default)",
            "test",
            default_model,
            x_test,
            y_test,
            default_threshold,
        )

        search_rows.append(
            {
                **search_summary,
                "default_cv_mean_average_precision": default_cv_mean,
                "default_cv_std_average_precision": default_cv_std,
                "cv_average_precision_change": (
                    search_summary["cv_mean_average_precision"] - default_cv_mean
                ),
                "selected_validation_threshold": tuned_threshold,
                "default_validation_threshold": default_threshold,
            }
        )
        metric_rows.extend(
            [
                tuned_validation_metrics,
                tuned_test_metrics,
                default_validation_metrics,
                default_test_metrics,
            ]
        )
        test_risk_predictions[f"{model_name} (tuned)"] = (
            tuned_test_probabilities >= tuned_threshold
        ).astype(int)
        test_risk_predictions[f"{model_name} (default)"] = (
            default_test_probabilities >= default_threshold
        ).astype(int)
        test_predictions.append(
            pd.DataFrame(
                {
                    "applicant_id": applicant_ids,
                    "model": f"{model_name} (tuned)",
                    "actual_default": y_test.to_numpy(),
                    "default_probability": tuned_test_probabilities,
                    "threshold": tuned_threshold,
                    "predicted_default": test_risk_predictions[
                        f"{model_name} (tuned)"
                    ],
                }
            )
        )
        print(
            f"{model_name}: CV AP {default_cv_mean:.3f} -> "
            f"{search_summary['cv_mean_average_precision']:.3f}; "
            f"test F1 {default_test_metrics['f1']:.3f} -> "
            f"{tuned_test_metrics['f1']:.3f}",
            flush=True,
        )


    applicant_ids = test_data["applicant_id"].to_numpy()
    fairness = _make_fairness_table(
        test_data,
        y_test,
        test_risk_predictions,
    )
    metric_frame = pd.DataFrame(metric_rows)
    search_frame = pd.DataFrame(search_rows)
    prediction_frame = pd.concat(test_predictions, ignore_index=True)

    metric_frame.to_csv(output_dir / "tuned_model_metrics.csv", index=False)
    search_frame.to_csv(output_dir / "tuning_search_summary.csv", index=False)
    fairness.to_csv(output_dir / "tuned_fairness_metrics.csv", index=False)
    prediction_frame.to_csv(output_dir / "tuned_test_predictions.csv", index=False)

    print("\nUntuned vs tuned test performance:")
    print(
        metric_frame.loc[metric_frame["split"] == "test"][
            [
                "model",
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
    print(f"\nTuning reports saved to {output_dir}")
    print(
        f"Test set remained held out. Seeds: train/test={TRAIN_TEST_SEED}, "
        f"train/validation={TRAIN_VALIDATION_SEED}, CV/model={MODEL_SEED}."
    )


if __name__ == "__main__":
    main()
