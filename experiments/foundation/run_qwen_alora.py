import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from common.model_utils import (
    DATA_SEED,
    MODEL_SEED,
    TRAIN_TEST_SEED,
    TRAIN_VALIDATION_SEED,
    _choose_threshold,
    _collect_metrics,
    _make_fairness_table,
    add_features,
)
from common.paths import PROJECT_ROOT
from models.foundation.qwen_alora_model import QwenALoRAClassifier


def main() -> None:
    project_dir = PROJECT_ROOT
    data = pd.read_csv(project_dir / "loan_applications.csv")
    labeled = data.loc[data["actual_outcome"].isin(["repaid", "defaulted"])].copy()
    y = (labeled["actual_outcome"] == "defaulted").astype(int)
    x = add_features(labeled)

    x_train_validation, x_test, y_train_validation, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=TRAIN_TEST_SEED,
        stratify=y,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=0.25,
        random_state=TRAIN_VALIDATION_SEED,
        stratify=y_train_validation,
    )

    model = QwenALoRAClassifier(
        rank=4,
        learning_rate=1e-4,
        epochs=2,
        batch_size=8,
        random_state=MODEL_SEED,
    )
    model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
    )
    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    threshold = _choose_threshold(y_validation, validation_probabilities)
    test_probabilities = model.predict_proba(x_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    model_name = "Qwen2.5-1.5B (aLoRA)"
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

    output_dir = project_dir / "reports_qwen_alora"
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"
    model.model_.save_pretrained(adapter_dir)
    model.tokenizer_.save_pretrained(adapter_dir)
    metrics_frame.to_csv(output_dir / "qwen_alora_metrics.csv", index=False)
    pd.DataFrame(model.training_history_).to_csv(
        output_dir / "qwen_alora_training_history.csv",
        index=False,
    )

    test_data = labeled.loc[x_test.index]
    fairness = _make_fairness_table(
        test_data,
        y_test,
        {model_name: test_predictions},
    )
    fairness.to_csv(output_dir / "qwen_alora_fairness_metrics.csv", index=False)

    predictions = pd.DataFrame(
        {
            "applicant_id": test_data["applicant_id"].to_numpy(),
            "actual_outcome": test_data["actual_outcome"].to_numpy(),
            "employment_status": test_data["employment_status"].to_numpy(),
            "model": model_name,
            "default_probability": test_probabilities,
            "threshold": threshold,
            "predicted_default": test_predictions,
            "predicted_decision": np.where(
                test_predictions == 1, "deny_or_review", "approve"
            ),
        }
    )
    predictions.to_csv(output_dir / "qwen_alora_test_predictions.csv", index=False)

    baseline_path = project_dir / "reports_qwen" / "qwen_metrics.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        comparison = pd.concat(
            [
                baseline.loc[baseline["split"] == "test"],
                metrics_frame.loc[metrics_frame["split"] == "test"],
            ],
            ignore_index=True,
        )
        comparison.to_csv(output_dir / "qwen_test_comparison.csv", index=False)

    columns = [
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
        "defaults_caught",
        "good_applicants_flagged_or_denied",
        "approval_rate",
    ]
    print(metrics_frame[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(
        f"\nReports and adapter saved to {output_dir}\n"
        f"Seeds: data={DATA_SEED}, train/test={TRAIN_TEST_SEED}, "
        f"train/validation={TRAIN_VALIDATION_SEED}, model={MODEL_SEED}\n"
        f"CPU training only; CUDA available={torch.cuda.is_available()}"
    )


if __name__ == "__main__":
    main()
