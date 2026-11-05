# Bree Credit Risk Modeling

This project compares synthetic loan-default models and evaluates them as credit
decision policies. The final experiment emphasizes repeated evaluation,
probability calibration, population shifts, and comparisons at a common approval
rate rather than relying on F1 alone.

> **Scope:** All applicants and outcomes in this repository are synthetic.
> Results are for a modeling demonstration, not evidence of real-world lending
> performance or a production credit policy.

## Reproduce the final experiment

Tested with Python 3.11.9. From the repository root:

```bash
python -m pip install -r requirements.txt
python MLE_Project.py
python run_credit_risk_validation.py
```

`MLE_Project.py` regenerates the 2,000-row `loan_applications.csv` synthetic
dataset using seed 42. The validation runner writes its CSV outputs to
`reports_credit_risk_validation/`. It performs repeated model searches and
CPU-based TabPFN v2 fitting, so a full run can take a while.

The experiment uses random seeds 42, 2025, and 2027. Four-fold stratified
cross-validation selects configurations for tuned Logistic Regression, WOE
Logistic Regression, CatBoost, and LightGBM, using average precision (PR-AUC).
TabPFN v2 is included with its configured default model and is **not** included
in the hyperparameter search. For each seed, data is split into training,
calibration, and threshold-selection partitions. A Platt scaler is fitted only
on the calibration partition; F1 thresholds are selected only on the separate
validation partition.

Each fitted model is tested on four independently generated scenarios:

| Scenario | Synthetic change |
|---|---|
| Baseline | Original data-generating process |
| Lower default rate | Lower default odds |
| Population drift | Changed applicant feature distributions and risk prevalence |
| Novel risk mechanism | Adds a risk effect for inconsistent deposits |

Ongoing applications are excluded from binary outcome metrics. Each scenario
contains up to 1,200 applicants before that exclusion. The rule-based policy is
also evaluated. At equal approval rates, applicants are ranked by risk score
and the same number is approved for each policy.

## Summary of results

The following are means across the three seeds. In the baseline scenario,
default prevalence was 30.2%.

| Model | F1 | Recall | PR-AUC | Brier score, raw to Platt |
|---|---:|---:|---:|---:|
| WOE Logistic Regression | **0.543** | 0.710 | **0.541** | 0.191 to 0.183 |
| TabPFN v2 | 0.540 | **0.760** | 0.540 | **0.181** to 0.182 |
| CatBoost | 0.542 | 0.735 | 0.532 | 0.202 to 0.184 |
| Tuned Logistic Regression | 0.532 | 0.743 | 0.532 | 0.184 to 0.184 |
| LightGBM | 0.516 | 0.633 | 0.517 | 0.195 to 0.187 |
| Rule-based baseline | 0.533 | 0.732 | 0.503 | Not applicable |

At the same approval rate in the baseline scenario:

| Target approval | Policy | Approved default rate | Defaults captured (mean count) | Good applicants rejected (mean count) |
|---:|---|---:|---:|---:|
| 45% | WOE Logistic Regression | **15.3%** | 255.7 | 348.3 |
| 45% | TabPFN v2 | 15.9% | 252.7 | 351.3 |
| 45% | Rule-based baseline | 16.6% | 249.0 | 355.0 |
| 70% | WOE Logistic Regression | 21.4% | 166.7 | 163.0 |
| 70% | TabPFN v2 | 21.5% | 166.0 | 163.7 |
| 70% | Rule-based baseline | 22.2% | 160.7 | 169.0 |

The leading model varies by scenario: TabPFN has the highest baseline recall,
while the rule baseline has the highest PR-AUC in the population-drift and
novel-risk scenarios. Calibration improvements are not uniform. These small
synthetic experiments do not establish that one model will be best on real
credit data.

### Earlier single-split experiments

The project also explored more models on one fixed train/validation/test split.
Those scores use a different experiment from the repeated results above and
should not be directly ranked against the multi-seed estimates.

| Model | F1 | Recall | PR-AUC |
|---|---:|---:|---:|
| Tuned Logistic Regression | **0.583** | 0.771 | **0.598** |
| TabPFN v2 | 0.557 | 0.606 | 0.592 |
| WOE Logistic Regression | 0.540 | 0.771 | 0.577 |
| Random Forest | 0.561 | 0.615 | 0.578 |
| Rule-based baseline | 0.550 | 0.789 | 0.563 |
| LightGBM (tuned) | 0.478 | 0.459 | 0.554 |
| CatBoost (tuned) | 0.535 | 0.771 | 0.557 |
| XGBoost | 0.530 | 0.569 | 0.525 |
| Qwen2.5 1.5B (few-shot) | 0.476 | 0.771 | 0.463 |
| Qwen2.5 1.5B (aLoRA) | 0.478 | **0.899** | 0.457 |
| Decision Tree | 0.550 | 0.862 | 0.466 |

The aLoRA run produced the highest recall in that split, with lower precision
and a substantially lower approval rate (18.2%). This illustrates why recall
alone is insufficient for comparing credit decisions.

Expected Loss is not reported because the project has no Loss Given Default
(LGD) or Exposure at Default (EAD) inputs or assumptions. Fairness outputs
include metrics grouped by employment status; this is a limited diagnostic, not
a complete fairness or compliance assessment.

## Generated CSV files

The full final-run outputs are committed under
[`reports_credit_risk_validation/`](reports_credit_risk_validation/):

- `robustness_summary.csv` - mean metrics and across-seed variability for each
  model and scenario.
- `robustness_metrics.csv` - per-seed test metrics, including precision,
  recall, F1, ROC-AUC, PR-AUC, approval rate, Brier score, and log loss.
- `equal_approval_summary.csv` and `equal_approval_comparison.csv` - decision
  outcomes at 45% and 70% target approval rates, aggregated and per seed.
- `calibration_curves.csv` - observed default rates by predicted-risk bin.
- `group_fairness_metrics.csv` - employment-group metrics, including recall,
  false-positive and false-negative rates.
- `model_search_summary.csv` - cross-validation search settings and selected
  parameters for the four tuned models.

`loan_applications.csv` is the generated input dataset. Per-model outputs from
earlier exploratory runs are intentionally not part of the final submission;
their scripts and summarized results remain available in the repository and
this README.

## Other experiments

The repository also contains standalone scripts for untuned and tuned Logistic
Regression, decision trees, random forests, XGBoost, CatBoost, LightGBM,
WOE Logistic Regression, TabPFN v2, and exploratory Qwen few-shot/aLoRA
experiments. They write their outputs to per-model `reports_*` directories.
The Qwen experiments require additional PyTorch, Transformers, and PEFT
dependencies and may download model weights; they are not needed to reproduce
the final validation CSVs above. Generated historical report directories and
model weights are excluded from Git.

## Main project files

- `MLE_Project.py` - generate the synthetic input CSV.
- `model_utils.py` - feature engineering, shared evaluation, and fairness
  helpers.
- `credit_risk_validation.py` - scenario generation and equal-approval
  evaluation.
- `run_credit_risk_validation.py` - final multi-seed model search, calibration,
  testing, and CSV report generation.
- `*_model.py` and `run_*.py` - model definitions and standalone experiments.
