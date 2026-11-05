from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenFewShotClassifier(ClassifierMixin, BaseEstimator):
    """Score default/repaid labels with a local instruction-tuned language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        examples_per_class: int = 3,
        batch_size: int = 8,
        random_state: int = 2027,
        max_input_tokens: int = 2048,
    ) -> None:
        self.model_name = model_name
        self.examples_per_class = examples_per_class
        self.batch_size = batch_size
        self.random_state = random_state
        self.max_input_tokens = max_input_tokens

    @staticmethod
    def _format_row(row: pd.Series) -> str:
        values = []
        for column, value in row.items():
            if pd.isna(value):
                rendered = "unknown"
            elif isinstance(value, (float, np.floating)):
                rendered = f"{value:.3g}"
            else:
                rendered = str(value)
            values.append(f"{column.replace('_', ' ')}: {rendered}")
        return "; ".join(values)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> QwenFewShotClassifier:
        self.classes_ = np.array([0, 1])
        labels = pd.Series(np.asarray(y, dtype=int), index=x.index)
        if set(labels.unique()) != {0, 1}:
            raise ValueError("Qwen few-shot classification requires both labels 0 and 1.")

        rng = np.random.default_rng(self.random_state)
        self.examples_: list[tuple[str, str]] = []
        for label, outcome in ((0, "Repaid"), (1, "Default")):
            class_rows = x.loc[labels == label]
            count = min(self.examples_per_class, len(class_rows))
            selected = rng.choice(class_rows.index.to_numpy(), size=count, replace=False)
            self.examples_.extend(
                (self._format_row(x.loc[index]), outcome) for index in selected
            )

        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer_.padding_side = "left"
        if self.tokenizer_.pad_token_id is None:
            self.tokenizer_.pad_token = self.tokenizer_.eos_token

        self.model_ = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype="auto",
            low_cpu_mem_usage=True,
        )
        self.model_.to("cpu")
        self.model_.eval()
        return self

    def _make_prompt(self, row: pd.Series) -> str:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "Classify a loan application using only the labeled examples. "
                    "The target is whether the application later defaulted. "
                    "Use the same fields and label pattern. Do not invent missing "
                    "information. Label 0 means repaid; label 1 means defaulted. "
                    "For each example, answer with only its label."
                ),
            }
        ]
        for example, outcome in self.examples_:
            messages.append(
                {
                    "role": "user",
                    "content": f"Application: {example}\nLabel:",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "0" if outcome == "Repaid" else "1",
                }
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Application: {self._format_row(row)}\n"
                    "Which outcome is more likely? Answer with label 0 or 1 only."
                ),
            }
        )
        return self.tokenizer_.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _score_candidate_batch(
        self,
        prompts: list[str],
    ) -> np.ndarray:
        tokenized_prompts = self.tokenizer_(
            prompts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        if tokenized_prompts["input_ids"].shape[1] > self.max_input_tokens:
            raise ValueError(
                f"Qwen prompt has {tokenized_prompts['input_ids'].shape[1]} tokens, "
                f"exceeding the configured limit of {self.max_input_tokens}."
            )
        with torch.inference_mode():
            logits = self.model_(
                input_ids=tokenized_prompts["input_ids"],
                attention_mask=tokenized_prompts["attention_mask"],
                logits_to_keep=1,
            ).logits[:, -1, :]

        candidate_token_ids = [
            self.tokenizer_.encode(label, add_special_tokens=False)
            for label in ("0", "1")
        ]
        if any(len(token_ids) != 1 for token_ids in candidate_token_ids):
            raise ValueError("Qwen labels 0 and 1 must each map to one tokenizer token.")
        return (
            logits[:, [token_ids[0] for token_ids in candidate_token_ids]]
            .float()
            .cpu()
            .numpy()
        )

    def predict_proba(
        self,
        x: pd.DataFrame,
        progress_label: str | None = None,
    ) -> np.ndarray:
        prompts = [self._make_prompt(row) for _, row in x.iterrows()]
        scores: list[np.ndarray] = []
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start : start + self.batch_size]
            scores.extend(self._score_candidate_batch(batch_prompts))
            completed = min(start + len(batch_prompts), len(prompts))
            if progress_label and (
                start == 0
                or completed == len(prompts)
                or completed // (self.batch_size * 10)
                > start // (self.batch_size * 10)
            ):
                print(
                    f"Qwen {progress_label}: scored {completed}/{len(prompts)}",
                    flush=True,
                )

        logits = np.asarray(scores)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(x)
        return self.classes_[np.argmax(probabilities, axis=1)]
