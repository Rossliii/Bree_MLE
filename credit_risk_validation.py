from __future__ import annotations

import numpy as np
import pandas as pd

from model_utils import generate_dataset


SCENARIOS = (
    "baseline",
    "lower_default_rate",
    "population_drift",
    "novel_risk_mechanism",
)


def _recompute_rule_policy(
    data: pd.DataFrame,
    misrepresents: np.ndarray,
) -> pd.DataFrame:
    result = data.copy()
    has_documents = result["documented_monthly_income"].notna().to_numpy()
    income_gap = (
        (
            result["stated_monthly_income"]
            - result["documented_monthly_income"]
        ).abs()
        / result["stated_monthly_income"].clip(lower=1)
    ).to_numpy()
    stated_income = result["stated_monthly_income"].to_numpy()
    loan_amount = result["loan_amount"].to_numpy()
    bank_balance = result["bank_ending_balance"].to_numpy()
    overdrafts = result["bank_has_overdrafts"].to_numpy()
    consistent_deposits = result["bank_has_consistent_deposits"].to_numpy()
    employment = result["employment_status"].to_numpy()
    deposits = result["monthly_deposits"].to_numpy()
    withdrawals = result["monthly_withdrawals"].to_numpy()

    if len(misrepresents) != len(result):
        raise ValueError("misrepresents must have one value per applicant.")
    income_verification_score = np.where(
        ~has_documents,
        0,
        np.where(
            misrepresents,
            0,
            np.where(income_gap <= 0.1, 95, 40),
        ),
    )
    income_level_score = np.where(
        stated_income >= 3 * loan_amount,
        90,
        np.where(stated_income >= 2 * loan_amount, 45, 0),
    )
    account_stability_score = (
        np.where(bank_balance > 500, 40, 10)
        + np.where(~overdrafts, 30, 0)
        + np.where(consistent_deposits, 30, 0)
    ).clip(0, 100)
    employment_score = np.where(
        employment == "employed",
        100,
        np.where(employment == "self_employed", 60, 20),
    )
    debt_to_income_score = np.where(
        deposits > 0,
        np.clip(100 - withdrawals / np.maximum(deposits, 1) * 100, 0, 100),
        50,
    )
    score = (
        income_verification_score * 0.30
        + income_level_score * 0.25
        + account_stability_score * 0.20
        + employment_score * 0.15
        + debt_to_income_score * 0.10
    ).round(1)
    result["rule_based_score"] = score
    result["rule_based_decision"] = np.where(
        score >= 75,
        "approved",
        np.where(score >= 50, "flagged_for_review", "denied"),
    )
    return result


def _apply_population_drift(
    data: pd.DataFrame,
    rng: np.random.RandomState,
) -> tuple[pd.DataFrame, np.ndarray]:
    result = data.copy()
    count = len(result)
    employment = rng.choice(
        ["employed", "self_employed", "unemployed"],
        count,
        p=[0.40, 0.35, 0.25],
    )
    stated_income = np.where(
        employment == "employed",
        rng.normal(4400, 1800, count),
        np.where(
            employment == "self_employed",
            rng.normal(3800, 2200, count),
            rng.normal(1300, 900, count),
        ),
    )
    stated_income = np.clip(stated_income, 500, 25000).round(0)
    has_documents = rng.random(count) > 0.22
    misrepresents = rng.random(count) < 0.08
    documented_income = np.where(
        has_documents,
        np.where(
            misrepresents,
            stated_income * rng.uniform(0.2, 0.4, count),
            stated_income * rng.uniform(0.9, 1.05, count),
        ),
        np.nan,
    )
    loan_amount = rng.choice(
        [300, 500, 1000, 1500, 2000, 3000, 5000],
        count,
        p=[0.05, 0.10, 0.20, 0.20, 0.20, 0.15, 0.10],
    )
    bank_balance = rng.normal(1300, 1800, count).clip(0, 15000).round(0)
    overdrafts = rng.random(count) < np.where(bank_balance < 500, 0.7, 0.16)
    consistent_deposits = rng.random(count) < np.where(
        employment == "employed",
        0.70,
        0.35,
    )
    deposits = np.where(has_documents, documented_income, stated_income) * rng.uniform(
        0.75,
        1.05,
        count,
    )
    withdrawals = deposits * rng.uniform(0.35, 1.05, count)
    document_count = np.where(
        has_documents,
        rng.choice([1, 2], count, p=[0.4, 0.6]),
        0,
    )

    result["employment_status"] = employment
    result["stated_monthly_income"] = stated_income
    result["documented_monthly_income"] = np.round(documented_income, 0)
    result["loan_amount"] = loan_amount
    result["bank_ending_balance"] = bank_balance
    result["bank_has_overdrafts"] = overdrafts
    result["bank_has_consistent_deposits"] = consistent_deposits
    result["monthly_deposits"] = np.round(deposits, 0)
    result["monthly_withdrawals"] = np.round(withdrawals, 0)
    result["num_documents_submitted"] = document_count
    return _recompute_rule_policy(result, misrepresents), misrepresents


def _risk_probability(
    data: pd.DataFrame,
    scenario: str,
    misrepresents: np.ndarray | None = None,
) -> np.ndarray:
    income_gap = (
        (
            data["stated_monthly_income"]
            - data["documented_monthly_income"]
        ).abs()
        / data["stated_monthly_income"].clip(lower=1)
    )
    documentation_risk = (
        data["documented_monthly_income"].isna().to_numpy()
        | (
            income_gap.to_numpy() > 0.11
            if misrepresents is None
            else misrepresents
        )
    )
    real_risk = (
        0.25
        * documentation_risk
        + 0.25 * (data["loan_amount"] / data["stated_monthly_income"].clip(lower=1) > 0.3)
        + 0.20 * data["bank_has_overdrafts"].astype(float)
        + 0.10 * (data["employment_status"] == "unemployed").astype(float)
        + 0.10 * (data["bank_ending_balance"] < 500)
        + 0.10
        * (
            data["monthly_withdrawals"]
            / data["monthly_deposits"].clip(lower=1)
            > 0.8
        )
    )
    log_odds = 5 * (real_risk.to_numpy() - 0.45)
    if scenario == "lower_default_rate":
        log_odds -= 0.9
    elif scenario == "novel_risk_mechanism":
        log_odds += 1.0 * (
            ~data["bank_has_consistent_deposits"].to_numpy(dtype=bool)
        )
    return 1 / (1 + np.exp(-log_odds))


def make_scenario_test_data(
    scenario: str,
    n: int,
    random_state: int,
) -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown synthetic scenario: {scenario}")
    if n < 1:
        raise ValueError("n must be positive.")

    data = generate_dataset(n=n, random_state=random_state)
    if scenario == "baseline":
        return data

    rng = np.random.RandomState(random_state + 104729)
    misrepresents: np.ndarray | None = None
    if scenario == "population_drift":
        data, misrepresents = _apply_population_drift(data, rng)

    defaults = rng.random(len(data)) < _risk_probability(
        data,
        scenario,
        misrepresents,
    )
    ongoing = rng.random(len(data)) < 0.08
    data["actual_outcome"] = np.where(
        ongoing,
        "ongoing",
        np.where(defaults, "defaulted", "repaid"),
    )
    data["days_to_default"] = np.where(
        data["actual_outcome"] == "defaulted",
        rng.randint(15, 180, len(data)),
        np.nan,
    )
    return data


def evaluate_at_approval_rate(
    y_true: pd.Series | np.ndarray,
    risk_scores: np.ndarray,
    approval_rate: float,
) -> dict[str, float | int]:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(risk_scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("Labels and risk scores must be matching one-dimensional arrays.")
    if not 0 < approval_rate < 1:
        raise ValueError("approval_rate must be strictly between 0 and 1.")
    if not np.isfinite(scores).all() or set(np.unique(labels)) - {0, 1}:
        raise ValueError("Scores must be finite and labels must be binary.")

    approved_count = int(np.floor(len(labels) * approval_rate))
    approval_order = np.argsort(scores, kind="stable")
    approved = np.zeros(len(labels), dtype=bool)
    approved[approval_order[:approved_count]] = True
    defaults = labels == 1
    good_applicants = ~defaults
    defaults_in_approved = int((approved & defaults).sum())
    return {
        "approval_rate": float(approved.mean()),
        "approved_default_rate": (
            float(defaults_in_approved / approved_count)
            if approved_count
            else np.nan
        ),
        "defaults_in_approved": defaults_in_approved,
        "defaults_captured": int((~approved & defaults).sum()),
        "good_applicants_rejected": int((~approved & good_applicants).sum()),
        "total_defaults": int(defaults.sum()),
        "approved_applicants": approved_count,
    }
