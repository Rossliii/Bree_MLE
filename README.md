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
python main.py generate-data
python main.py validate
```

`main.py` is the project entry point; its default command is `validate`. It
provides these workflows:

| Command | Workflow |
|---|---|
| `python main.py generate-data` | Regenerate the 2,000-row synthetic input using seed 42 |
| `python main.py validate` | Run the final multi-seed, multi-scenario comparison |
| `python main.py baseline` | Run the original baseline model benchmark |
| `python main.py tune` | Run the original cross-validation tuning experiment |
| `python main.py woe` | Run WOE Logistic Regression |
| `python main.py lightgbm` | Run LightGBM |
| `python main.py tabpfn` | Run TabPFN v2 |
| `python main.py qwen` / `python main.py lora` | Run Qwen few-shot / fresh LoRA experiments |
| `python main.py lora-continue` | Continue from the saved LoRA adapter for three epochs |

`generate-data` regenerates `loan_applications.csv`; validation writes CSV
outputs to `reports_credit_risk_validation/`. The full validation command runs
repeated model searches and CPU-based TabPFN v2 fitting, so it can take a while.
The Qwen commands use Qwen3.5-2B (Apache-2.0 model weights); install its
optional dependencies first with `python -m pip install -r requirements-llm.txt`.
That model requires a Transformers version with Qwen3.5 architecture support.

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
| Decision Tree | 0.550 | 0.862 | 0.466 |

### Qwen3.5-2B evaluation

Both Qwen3.5 experiments were run on the fixed split described above
(data=42, train/test=2025, train/validation=2026, model=2027). Thresholds were
selected on validation data. The table shows held-out test metrics and does
not represent a multi-seed benchmark.

| Experiment | Model | F1 | Recall | PR-AUC | Approval rate |
|---|---|---:|---:|---:|---:|
| Few-shot | Qwen3.5-2B | **0.521** | 0.697 | **0.491** | 50.3% |
| LoRA | Qwen3.5-2B | **0.526** | 0.642 | **0.511** | 57.3% |

On the same test split, standard LoRA caught 70 of 109 defaults and approved
211 applicants; 39 approved applicants defaulted. It flagged or denied 87
good applicants. Compared with Qwen3.5 few-shot, LoRA has slightly higher F1
(0.526 vs. 0.521) and PR-AUC (0.511 vs. 0.491), lower recall (0.642 vs. 0.697),
and a higher approval rate (57.3% vs. 50.3%). These are single-split,
threshold-dependent results. The earlier Qwen3.5 aLoRA result is a different
adapter method and is not presented as a LoRA result.

The Qwen3.5 LoRA run uses rank 4, learning rate 1e-4, two epochs, and batch
size 8. `python main.py lora-continue` resumes its saved adapter, trains three
additional epochs, and writes separate outputs under
`reports_qwen_lora_continued/`. In this run, continuation validation PR-AUC
did not exceed the initial adapter's 0.4669, so the initial adapter was kept
as the best checkpoint and test metrics remained unchanged. Training history
and predictions are generated under `reports_qwen/`, `reports_qwen_lora/`,
and `reports_qwen_lora_continued/`; these per-model report directories are
excluded from Git. Results are one-run estimates on synthetic data, not
evidence of real-world lending performance.

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

## Repository structure

```text
main.py                         Main workflow entry point
data/                           Synthetic data generation
common/                         Shared features, metrics, paths, scenarios
models/
  linear/                       Logistic Regression and WOE Logistic
  trees/                        Decision Tree, Random Forest, XGBoost,
                                CatBoost, LightGBM
  foundation/                   TabPFN v2 and Qwen experiments
experiments/
  validation/                   Multi-seed scenario evaluation
  tuning/                       Shared tuning utilities and experiments
  baseline/                     Original model benchmark
  linear/                       WOE Logistic experiment
  trees/                        LightGBM experiment
  foundation/                   TabPFN and Qwen experiments
reports_credit_risk_validation/ Final experiment CSV deliverables
```

Run a standalone experiment module directly from the repository root if
preferred, for example:

```bash
python -m experiments.baseline.run_catboost
python -m experiments.tuning.run_tuning
python -m experiments.foundation.run_tabpfn_v2
```

## Other experiments

The repository also contains standalone scripts for untuned and tuned Logistic
Regression, decision trees, random forests, XGBoost, CatBoost, LightGBM,
WOE Logistic Regression, TabPFN v2, and exploratory Qwen few-shot/LoRA
experiments. They write their outputs to per-model `reports_*` directories at
the repository root.
The Qwen experiments may download model weights; they are not needed to
reproduce the final validation CSVs above. Generated historical report
directories and model weights are excluded from Git.

The Qwen3.5-2B weights are published under the
[Apache 2.0 license](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE).
This is an open-weight model with a permissive open-source license.

## Main project files

- `main.py` - main CLI entry point.
- `data/generate_dataset.py` - generate the synthetic input CSV.
- `common/model_utils.py` - feature engineering, shared evaluation, and
  fairness helpers.
- `common/credit_risk_validation.py` - scenario generation and equal-approval
  evaluation.
- `experiments/validation/run_credit_risk_validation.py` - final multi-seed
  model search, calibration, testing, and CSV report generation.
- `models/` and `experiments/` - algorithms grouped by category and their
  standalone runners.
