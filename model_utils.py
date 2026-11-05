from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_SEED = 42
TRAIN_TEST_SEED = 2025
TRAIN_VALIDATION_SEED = 2026
MODEL_SEED = 2027
FEATURE_EXCLUSIONS = {
    "applicant_id",
    "actual_outcome",
    "days_to_default",
    "rule_based_score",
    "rule_based_decision",
}


def generate_dataset(n: int = 2000) -> pd.DataFrame:
    rng = np.random.RandomState(DATA_SEED)
    employment = rng.choice(
        ["employed", "self_employed", "unemployed"], n, p=[0.6, 0.3, 0.1]
    )
    stated_income = np.where(
        employment == "employed",
        rng.normal(5000, 1500, n),
        np.where(
            employment == "self_employed",
            rng.normal(4500, 2000, n),
            rng.normal(1500, 800, n),
        ),
    )
    stated_income = np.clip(stated_income, 500, 25000).round(0)

    has_docs = rng.random(n) > 0.15
    misrepresents = rng.random(n) < 0.05
    doc_income = np.where(
        has_docs,
        np.where(
            misrepresents,
            stated_income * rng.uniform(0.2, 0.4, n),
            stated_income * rng.uniform(0.9, 1.05, n),
        ),
        np.nan,
    )
    loan_amount = rng.choice(
        [300, 500, 1000, 1500, 2000, 3000, 5000],
        n,
        p=[0.1, 0.15, 0.25, 0.2, 0.15, 0.1, 0.05],
    )
    bank_balance = rng.normal(2000, 1500, n).clip(0, 15000).round(0)
    has_overdrafts = rng.random(n) < np.where(bank_balance < 500, 0.6, 0.1)
    consistent_deposits = rng.random(n) < np.where(employment == "employed", 0.8, 0.4)
    monthly_deposits = np.where(has_docs, doc_income, stated_income) * rng.uniform(
        0.85, 1.0, n
    )
    monthly_withdrawals = monthly_deposits * rng.uniform(0.3, 0.95, n)
    num_docs = np.where(has_docs, rng.choice([1, 2], n, p=[0.4, 0.6]), 0)

    income_verification_score = np.where(
        ~has_docs,
        0,
        np.where(
            misrepresents,
            0,
            np.where(
                np.abs(stated_income - doc_income) / stated_income <= 0.1,
                95,
                40,
            ),
        ),
    )
    income_level_score = np.where(
        stated_income >= 3 * loan_amount,
        90,
        np.where(stated_income >= 2 * loan_amount, 45, 0),
    )
    account_stability_score = (
        np.where(bank_balance > 500, 40, 10)
        + np.where(~has_overdrafts, 30, 0)
        + np.where(consistent_deposits, 30, 0)
    ).clip(0, 100)
    employment_score = np.where(
        employment == "employed",
        100,
        np.where(employment == "self_employed", 60, 20),
    )
    debt_to_income_score = np.where(
        monthly_deposits > 0,
        np.clip(100 - monthly_withdrawals / monthly_deposits * 100, 0, 100),
        50,
    )
    rule_score = (
        income_verification_score * 0.30
        + income_level_score * 0.25
        + account_stability_score * 0.20
        + employment_score * 0.15
        + debt_to_income_score * 0.10
    ).round(1)
    rule_decision = np.where(
        rule_score >= 75,
        "approved",
        np.where(rule_score >= 50, "flagged_for_review", "denied"),
    )

    real_risk = (
        0.25 * np.where(has_docs & ~misrepresents, 0, 1)
        + 0.25 * np.where(loan_amount <= stated_income * 0.3, 0, 1)
        + 0.20 * np.where(has_overdrafts, 1, 0)
        + 0.10 * np.where(employment == "unemployed", 1, 0)
        + 0.10 * np.where(bank_balance < 500, 1, 0)
        + 0.10
        * np.where(
            monthly_withdrawals / np.maximum(monthly_deposits, 1) > 0.8, 1, 0
        )
    )
    default_probability = 1 / (1 + np.exp(-5 * (real_risk - 0.45)))
    defaults = rng.random(n) < default_probability
    ongoing = rng.random(n) < 0.08
    actual_outcome = np.where(
        ongoing, "ongoing", np.where(defaults, "defaulted", "repaid")
    )
    days_to_default = np.where(
        actual_outcome == "defaulted", rng.randint(15, 180, n), np.nan
    )

    return pd.DataFrame(
        {
            "applicant_id": [f"APP-{index:04d}" for index in range(n)],
            "stated_monthly_income": stated_income,
            "documented_monthly_income": doc_income.round(0),
            "loan_amount": loan_amount,
            "employment_status": employment,
            "bank_ending_balance": bank_balance,
            "bank_has_overdrafts": has_overdrafts,
            "bank_has_consistent_deposits": consistent_deposits,
            "monthly_withdrawals": monthly_withdrawals.round(0),
            "monthly_deposits": monthly_deposits.round(0),
            "num_documents_submitted": num_docs,
            "rule_based_score": rule_score,
            "rule_based_decision": rule_decision,
            "actual_outcome": actual_outcome,
            "days_to_default": days_to_default,
        }
    )


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data.drop(
        columns=[column for column in FEATURE_EXCLUSIONS if column in data.columns]
    ).copy()
    features["documented_income_missing"] = (
        data["documented_monthly_income"].isna().astype(int)
    )
    features["income_document_gap_ratio"] = (
        (data["stated_monthly_income"] - data["documented_monthly_income"]).abs()
        / data["stated_monthly_income"].clip(lower=1)
    )
    features["loan_to_income_ratio"] = (
        data["loan_amount"] / data["stated_monthly_income"].clip(lower=1)
    )
    features["withdrawal_to_deposit_ratio"] = (
        data["monthly_withdrawals"] / data["monthly_deposits"].clip(lower=1)
    )
    return features


def make_pipeline(
    classifier: ClassifierMixin, scale_numeric: bool = False
) -> Pipeline:
    numeric_columns = [
        "stated_monthly_income",
        "documented_monthly_income",
        "loan_amount",
        "bank_ending_balance",
        "bank_has_overdrafts",
        "bank_has_consistent_deposits",
        "monthly_withdrawals",
        "monthly_deposits",
        "num_documents_submitted",
        "documented_income_missing",
        "income_document_gap_ratio",
        "loan_to_income_ratio",
        "withdrawal_to_deposit_ratio",
    ]
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_columns),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["employment_status"],
            ),
        ],
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def _choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    _, _, thresholds = precision_recall_curve(y_true, probabilities)
    if not thresholds.size:
        return 0.5
    scores = np.asarray(
        [
            f1_score(y_true, probabilities >= threshold, zero_division=0)
            for threshold in thresholds
        ]
    )
    maximum_f1 = scores.max()
    return float(thresholds[scores == maximum_f1].max())


def _collect_metrics(
    model_name: str,
    split_name: str,
    y_true: pd.Series,
    risk_predictions: np.ndarray,
    risk_scores: np.ndarray,
    threshold: float,
    decisions: pd.Series | None = None,
) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y_true, risk_predictions, labels=[0, 1]).ravel()
    approved = risk_predictions == 0
    good_applicants = y_true.to_numpy() == 0
    wrongly_denied = (
        int(((decisions == "denied").to_numpy() & good_applicants).sum())
        if decisions is not None
        else int(fp)
    )
    return {
        "model": model_name,
        "split": split_name,
        "threshold": threshold,
        "precision": precision_score(y_true, risk_predictions, zero_division=0),
        "recall": recall_score(y_true, risk_predictions, zero_division=0),
        "f1": f1_score(y_true, risk_predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, risk_scores),
        "pr_auc": average_precision_score(y_true, risk_scores),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "good_applicants_wrongly_denied": wrongly_denied,
        "good_applicants_flagged_or_denied": int(fp),
        "defaults_missed": int(fn),
        "defaults_caught": int(tp),
        "approval_rate": float(approved.mean()),
        "approved_default_rate": (
            float(y_true.to_numpy()[approved].mean()) if approved.any() else np.nan
        ),
        "explicit_denial_count": (
            int((decisions == "denied").sum())
            if decisions is not None
            else int((~approved).sum())
        ),
        "applicant_count": int(len(y_true)),
    }


def _shap_values(
    model: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    train_matrix = preprocessor.transform(x_train)
    test_matrix = preprocessor.transform(x_test)
    if hasattr(train_matrix, "toarray"):
        train_matrix = train_matrix.toarray()
        test_matrix = test_matrix.toarray()

    feature_names = preprocessor.get_feature_names_out().tolist()
    background = train_matrix[: min(100, len(train_matrix))]
    if classifier.__class__.__name__ == "LogisticRegression":
        explainer = shap.LinearExplainer(classifier, background)
        values = np.asarray(explainer.shap_values(test_matrix))
    elif classifier.__class__.__name__ == "XGBClassifier":
        background = background[: min(50, len(background))]
        explainer = shap.PermutationExplainer(
            lambda values: classifier.predict_proba(values)[:, 1],
            background,
            feature_names=feature_names,
            seed=MODEL_SEED,
        )
        values = np.asarray(
            explainer(
                test_matrix,
                max_evals=2 * len(feature_names) + 1,
                silent=True,
            ).values
        )
    elif classifier.__class__.__name__ == "TabPFNClassifier":
        background_frame = pd.DataFrame(
            background,
            columns=feature_names,
        ).sample(
            n=min(50, len(background)),
            random_state=MODEL_SEED,
        )

        def predict_tabular(rows: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(rows, columns=feature_names).astype(
                x_train.dtypes.to_dict()
            )
            return classifier.predict_proba(frame)[:, 1]

        explainer = shap.PermutationExplainer(
            predict_tabular,
            background_frame,
            feature_names=feature_names,
            seed=MODEL_SEED,
        )
        values = np.asarray(
            explainer(
                test_matrix,
                max_evals=2 * len(feature_names) + 1,
                silent=True,
            ).values
        )
    else:
        explainer = shap.TreeExplainer(classifier)
        values = np.asarray(explainer.shap_values(test_matrix))
        if isinstance(values, list):
            values = np.asarray(values[1])
        if values.ndim == 3:
            values = values[:, :, 1]
    return values, test_matrix, feature_names, np.asarray(
        classifier.predict_proba(test_matrix)[:, 1]
    )


def _make_local_reasons(
    shap_values: np.ndarray,
    transformed_test: object,
    feature_names: list[str],
    default_predictions: np.ndarray,
) -> list[str]:
    reasons: list[str] = []
    feature_values = (
        transformed_test.to_numpy()
        if isinstance(transformed_test, pd.DataFrame)
        else np.asarray(transformed_test)
    )
    for row_index, contributions in enumerate(shap_values):
        direction = 1 if default_predictions[row_index] == 1 else -1
        relevant = np.flatnonzero(contributions * direction > 0)
        if not relevant.size:
            relevant = np.arange(len(contributions))
        top_indices = relevant[np.argsort(np.abs(contributions[relevant]))[::-1][:3]]
        reasons.append(
            "; ".join(
                f"{feature_names[index]} ({contributions[index]:+.3f}; "
                f"value={feature_values[row_index, index]})"
                for index in top_indices
            )
        )
    return reasons


def _make_fairness_table(
    test_data: pd.DataFrame,
    y_test: pd.Series,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = test_data["employment_status"].reset_index(drop=True)
    actual = y_test.reset_index(drop=True)
    baseline_decisions = test_data["rule_based_decision"].reset_index(drop=True)

    for group in sorted(groups.unique()):
        mask = groups == group
        group_actual = actual[mask]
        baseline_group_approved = baseline_decisions[mask] == "approved"
        rows.append(
            {
                "model": "Rule-based baseline",
                "employment_status": group,
                "applicant_count": int(mask.sum()),
                "observed_default_rate": float(group_actual.mean()),
                "approval_rate": float(baseline_group_approved.mean()),
                "approved_default_rate": (
                    float(group_actual[baseline_group_approved].mean())
                    if baseline_group_approved.any()
                    else np.nan
                ),
                "explicit_denial_rate": float(
                    (baseline_decisions[mask] == "denied").mean()
                ),
                "review_rate": float(
                    (baseline_decisions[mask] == "flagged_for_review").mean()
                ),
            }
        )
        for model_name, group_predictions in predictions.items():
            group_predictions = group_predictions[mask.to_numpy()]
            group_approved = group_predictions == 0
            rows.append(
                {
                    "model": model_name,
                    "employment_status": group,
                    "applicant_count": int(mask.sum()),
                    "observed_default_rate": float(group_actual.mean()),
                    "approval_rate": float(group_approved.mean()),
                    "approved_default_rate": (
                        float(group_actual.to_numpy()[group_approved].mean())
                        if group_approved.any()
                        else np.nan
                    ),
                    "explicit_denial_rate": float((~group_approved).mean()),
                    "review_rate": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _save_plots(
    data: pd.DataFrame,
    y_test: pd.Series,
    scores: dict[str, np.ndarray],
    fairness: pd.DataFrame,
    output_dir: Path,
) -> None:
    order = ["repaid", "defaulted", "ongoing"]
    counts = data["actual_outcome"].value_counts().reindex(order, fill_value=0)
    figure, axis = plt.subplots(figsize=(7, 4))
    counts.plot.bar(ax=axis, color=["#4C9F70", "#D95D39", "#8A8F98"])
    axis.set(title="Generated loan outcomes", xlabel="Outcome", ylabel="Applications")
    figure.tight_layout()
    figure.savefig(output_dir / "outcome_distribution.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    for model_name, risk_scores in scores.items():
        precision, recall, _ = precision_recall_curve(y_test, risk_scores)
        axis.plot(recall, precision, label=model_name)
    axis.set(
        title="Test precision-recall curves",
        xlabel="Recall (default detection)",
        ylabel="Precision (predicted defaults)",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "test_precision_recall_curves.png", dpi=150)
    plt.close(figure)

    approval_rates = fairness.pivot(
        index="employment_status", columns="model", values="approval_rate"
    )
    figure, axis = plt.subplots(figsize=(11, 5))
    approval_rates.plot.bar(ax=axis)
    axis.set(
        title="Test approval rate by employment status",
        xlabel="Employment status",
        ylabel="Share approved",
        ylim=(0, 1),
    )
    axis.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(output_dir / "employment_approval_rates.png", dpi=150)
    plt.close(figure)


def _run_model(
    model_name: str,
    model_factory: Callable[..., Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    x_test: pd.DataFrame,
) -> tuple[
    Pipeline,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
]:
    arguments: dict[str, float] = {}
    if model_name == "XGBoost":
        arguments["positive_class_weight"] = float(
            (y_train == 0).sum() / (y_train == 1).sum()
        )
    model = model_factory(**arguments)
    model.fit(x_train, y_train)
    validation_scores = model.predict_proba(x_validation)[:, 1]
    threshold = _choose_threshold(y_validation, validation_scores)
    test_scores = model.predict_proba(x_test)[:, 1]
    test_predictions = (test_scores >= threshold).astype(int)
    shap_values, transformed_test, feature_names, _ = _shap_values(model, x_train, x_test)
    reasons = _make_local_reasons(
        shap_values, transformed_test, feature_names, test_predictions
    )
    return (
        model,
        threshold,
        validation_scores,
        test_scores,
        test_predictions,
        shap_values,
        transformed_test,
        feature_names,
        reasons,
    )


def run_all_models(
    model_factories: dict[str, Callable[..., Pipeline]],
    data_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        data = pd.read_csv(data_path)
    else:
        data = generate_dataset()
        data.to_csv(data_path, index=False)
        print(f"Generated {len(data)} applications at {data_path}")

    required = {
        "applicant_id",
        "stated_monthly_income",
        "documented_monthly_income",
        "loan_amount",
        "employment_status",
        "bank_ending_balance",
        "bank_has_overdrafts",
        "bank_has_consistent_deposits",
        "monthly_withdrawals",
        "monthly_deposits",
        "num_documents_submitted",
        "rule_based_score",
        "rule_based_decision",
        "actual_outcome",
        "days_to_default",
    }
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    labeled = data.loc[data["actual_outcome"].isin(["repaid", "defaulted"])].copy()
    ongoing_count = int((data["actual_outcome"] == "ongoing").sum())
    target = (labeled["actual_outcome"] == "defaulted").astype(int)
    if target.nunique() != 2:
        raise ValueError("Both repaid and defaulted outcomes are required.")
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

    metrics: list[dict[str, object]] = []
    risk_predictions: dict[str, np.ndarray] = {}
    risk_scores: dict[str, np.ndarray] = {}
    predictions_output: list[pd.DataFrame] = []
    local_explanations: list[pd.DataFrame] = []
    global_importances: list[pd.DataFrame] = []

    validation_baseline_decisions = labeled.loc[
        x_validation.index, "rule_based_decision"
    ]
    validation_baseline_predictions = (
        validation_baseline_decisions != "approved"
    ).astype(int).to_numpy()
    validation_baseline_scores = (
        100 - labeled.loc[x_validation.index, "rule_based_score"].to_numpy()
    )
    metrics.append(
        _collect_metrics(
            "Rule-based baseline",
            "validation",
            y_validation,
            validation_baseline_predictions,
            validation_baseline_scores,
            25.0,
            validation_baseline_decisions,
        )
    )
    test_data = labeled.loc[x_test.index]
    test_baseline_decisions = test_data["rule_based_decision"]
    test_baseline_predictions = (
        test_baseline_decisions != "approved"
    ).astype(int).to_numpy()
    test_baseline_scores = 100 - test_data["rule_based_score"].to_numpy()
    risk_predictions["Rule-based baseline"] = test_baseline_predictions
    risk_scores["Rule-based baseline"] = test_baseline_scores
    metrics.append(
        _collect_metrics(
            "Rule-based baseline",
            "test",
            y_test,
            test_baseline_predictions,
            test_baseline_scores,
            25.0,
            test_baseline_decisions,
        )
    )

    for model_name, model_factory in model_factories.items():
        (
            model,
            threshold,
            validation_scores,
            test_scores,
            test_predictions,
            shap_values,
            transformed_test,
            feature_names,
            reasons,
        ) = _run_model(
            model_name,
            model_factory,
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
        )
        risk_predictions[model_name] = test_predictions
        risk_scores[model_name] = test_scores
        metrics.append(
            _collect_metrics(
                model_name,
                "validation",
                y_validation,
                (validation_scores >= threshold).astype(int),
                validation_scores,
                threshold,
            )
        )
        metrics.append(
            _collect_metrics(
                model_name,
                "test",
                y_test,
                test_predictions,
                test_scores,
                threshold,
            )
        )

        preprocessor = model.named_steps["preprocessor"]
        transformed_feature_names = preprocessor.get_feature_names_out().tolist()
        global_importances.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "feature": transformed_feature_names,
                    "mean_absolute_shap": np.abs(shap_values).mean(axis=0),
                    "mean_shap": shap_values.mean(axis=0),
                }
            ).sort_values("mean_absolute_shap", ascending=False)
        )
        local_frame = pd.DataFrame(
            {
                "applicant_id": np.repeat(
                    test_data["applicant_id"].to_numpy(), len(feature_names)
                ),
                "model": model_name,
                "row_index": np.repeat(np.arange(len(x_test)), len(feature_names)),
                "feature": [
                    feature_names[index]
                    for _row in range(len(x_test))
                    for index in range(len(feature_names))
                ],
                "feature_value": np.asarray(transformed_test).reshape(-1),
                "shap_value": shap_values.reshape(-1),
            }
        )
        local_explanations.append(local_frame)
        predictions_output.append(
            pd.DataFrame(
                {
                    "applicant_id": test_data["applicant_id"].to_numpy(),
                    "actual_outcome": test_data["actual_outcome"].to_numpy(),
                    "employment_status": test_data["employment_status"].to_numpy(),
                    "model": model_name,
                    "default_probability": test_scores,
                    "threshold": threshold,
                    "predicted_default": test_predictions,
                    "predicted_decision": np.where(
                        test_predictions == 1, "deny_or_review", "approve"
                    ),
                    "top_local_reasons": reasons,
                }
            )
        )

    fairness = _make_fairness_table(
        test_data,
        y_test,
        {
            name: predictions
            for name, predictions in risk_predictions.items()
            if name != "Rule-based baseline"
        },
    )
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.concat(predictions_output, ignore_index=True)
    explanation_frame = pd.concat(local_explanations, ignore_index=True)
    importance_frame = pd.concat(global_importances, ignore_index=True)
    metric_frame.to_csv(output_dir / "model_metrics.csv", index=False)
    fairness.to_csv(output_dir / "fairness_metrics.csv", index=False)
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    explanation_frame.to_csv(output_dir / "local_explanations.csv", index=False)
    importance_frame.to_csv(output_dir / "feature_importance.csv", index=False)
    data["actual_outcome"].value_counts().rename_axis("outcome").reset_index(
        name="count"
    ).to_csv(output_dir / "outcome_summary.csv", index=False)
    pd.DataFrame(
        {
            "metric": [
                "total_applications",
                "labeled_applications",
                "ongoing_excluded_from_evaluation",
                "default_rate_among_labeled",
                "missing_documented_income_rate",
                "training_rows",
                "validation_rows",
                "test_rows",
                "data_seed",
                "train_test_split_seed",
                "train_validation_split_seed",
                "model_seed",
            ],
            "value": [
                len(data),
                len(labeled),
                ongoing_count,
                float(target.mean()),
                float(data["documented_monthly_income"].isna().mean()),
                len(x_train),
                len(x_validation),
                len(x_test),
                DATA_SEED,
                TRAIN_TEST_SEED,
                TRAIN_VALIDATION_SEED,
                MODEL_SEED,
            ],
        }
    ).to_csv(output_dir / "data_summary.csv", index=False)
    _save_plots(data, y_test, risk_scores, fairness, output_dir)

    print(f"Excluded ongoing applications: {ongoing_count}")
    print(
        f"Seeds: data={DATA_SEED}, train/test={TRAIN_TEST_SEED}, "
        f"train/validation={TRAIN_VALIDATION_SEED}, model={MODEL_SEED}"
    )
    print(
        metric_frame[
            [
                "model",
                "split",
                "threshold",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "false_positive_rate",
                "false_negative_rate",
                "approval_rate",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print(f"\nReports and plots saved to {output_dir}")
