from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import average_precision_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_llm_model import QwenFewShotClassifier


ALORA_INVOCATION_TEXT = "\nALORA_CLASSIFY_LOAN_RISK_7D3A:\n"
SYSTEM_PROMPT = (
    "Classify whether a loan applicant later defaults. Use the supplied "
    "application fields and return exactly one label: 0 means repaid; "
    "1 means defaulted."
)


class QwenALoRAClassifier:
    """Fine-tune a small aLoRA adapter for binary loan outcome classification."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        rank: int = 4,
        learning_rate: float = 1e-4,
        epochs: int = 2,
        batch_size: int = 8,
        max_length: int = 512,
        random_state: int = 2027,
    ) -> None:
        self.model_name = model_name
        self.rank = rank
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.random_state = random_state

    def _format_prompt(self, row: pd.Series) -> list[int]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Application: {QwenFewShotClassifier._format_row(row)}"
                    f"{ALORA_INVOCATION_TEXT}"
                ),
            },
        ]
        return self.tokenizer_.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

    def _pad_left(self, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([len(sequence) for sequence in sequences])
        width = int(lengths.max())
        input_ids = torch.full(
            (len(sequences), width),
            self.tokenizer_.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            input_ids[index, width - len(sequence) :] = torch.tensor(sequence)
            attention_mask[index, width - len(sequence) :] = 1
        return input_ids, attention_mask

    def _score_token_batches(self, prompts: list[list[int]]) -> np.ndarray:
        candidate_ids = torch.tensor(
            [token_ids[0] for token_ids in self.candidate_token_ids_],
            dtype=torch.long,
        )
        scores: list[np.ndarray] = []
        self.model_.eval()
        with torch.inference_mode():
            for start in range(0, len(prompts), self.batch_size):
                input_ids, attention_mask = self._pad_left(
                    prompts[start : start + self.batch_size]
                )
                if input_ids.shape[1] > self.max_length:
                    raise ValueError(
                        f"Qwen prompt has {input_ids.shape[1]} tokens, exceeding "
                        f"the configured limit of {self.max_length}."
                    )
                logits = self.model_(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits[:, -1, :]
                scores.extend(logits[:, candidate_ids].float().cpu().numpy())
        return np.asarray(scores)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        prompts = [self._format_prompt(row) for _, row in x.iterrows()]
        logits = self._score_token_batches(prompts)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        validation_data: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> QwenALoRAClassifier:
        labels = np.asarray(y, dtype=np.int64)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("aLoRA classification requires both labels 0 and 1.")
        if self.epochs < 1 or self.batch_size < 1 or self.rank < 1:
            raise ValueError("epochs, batch_size, and rank must be positive.")
        if validation_data is not None and set(
            np.unique(np.asarray(validation_data[1], dtype=np.int64))
        ) != {0, 1}:
            raise ValueError("aLoRA validation data requires both labels 0 and 1.")

        torch.manual_seed(self.random_state)
        torch.set_num_threads(min(torch.get_num_threads(), 8))
        self.classes_ = np.array([0, 1])
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer_.padding_side = "left"
        if self.tokenizer_.pad_token_id is None:
            self.tokenizer_.pad_token = self.tokenizer_.eos_token

        invocation_ids = self.tokenizer_.encode(
            ALORA_INVOCATION_TEXT,
            add_special_tokens=False,
        )
        if not invocation_ids:
            raise ValueError("The aLoRA invocation text produced no tokens.")
        self.candidate_token_ids_ = [
            self.tokenizer_.encode(label, add_special_tokens=False)
            for label in ("0", "1")
        ]
        if any(len(token_ids) != 1 for token_ids in self.candidate_token_ids_):
            raise ValueError("Qwen labels 0 and 1 must each map to one tokenizer token.")

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        layer_count = base_model.config.num_hidden_layers
        base_model.config.use_cache = False
        self.model_ = get_peft_model(
            base_model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.rank,
                lora_alpha=2 * self.rank,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                layers_to_transform=list(range(max(0, layer_count - 2), layer_count)),
                layers_pattern="layers",
                alora_invocation_tokens=invocation_ids,
            ),
        )
        self.model_.to("cpu")

        eos_id = self.tokenizer_.eos_token_id
        sequences: list[list[int]] = []
        for (_, row), label in zip(x.iterrows(), labels, strict=True):
            prompt_ids = self._format_prompt(row)
            sequence = prompt_ids + [self.candidate_token_ids_[label][0], eos_id]
            if len(sequence) > self.max_length:
                raise ValueError(
                    f"Training example has {len(sequence)} tokens, exceeding "
                    f"the configured limit of {self.max_length}."
                )
            sequences.append(sequence)

        validation_prompts: list[list[int]] | None = None
        validation_labels: np.ndarray | None = None
        if validation_data is not None:
            x_validation, y_validation = validation_data
            validation_prompts = [
                self._format_prompt(row) for _, row in x_validation.iterrows()
            ]
            validation_labels = np.asarray(y_validation, dtype=np.int64)

        trainable_parameters = [
            parameter for parameter in self.model_.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("PEFT did not expose any trainable aLoRA parameters.")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        rng = np.random.default_rng(self.random_state)
        self.training_history_: list[dict[str, Any]] = []
        best_state: dict[str, torch.Tensor] | None = None
        best_validation_score = -np.inf

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            batches = 0
            started_at = time.perf_counter()
            order = rng.permutation(len(sequences))
            for start in range(0, len(order), self.batch_size):
                indices = order[start : start + self.batch_size]
                batch_sequences = [sequences[index] for index in indices]
                input_ids, attention_mask = self._pad_left(batch_sequences)
                if input_ids.shape[1] > self.max_length:
                    raise ValueError(
                        f"Training batch has {input_ids.shape[1]} tokens, exceeding "
                        f"the configured limit of {self.max_length}."
                    )
                optimizer.zero_grad(set_to_none=True)
                output = self.model_(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=3,
                )
                selected_logits = output.logits[:, -3:-1, :].float()
                targets = input_ids[:, -2:]
                loss = F.cross_entropy(
                    selected_logits.reshape(-1, selected_logits.shape[-1]),
                    targets.reshape(-1),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
                optimizer.step()
                epoch_loss += float(loss.detach())
                batches += 1

            epoch_result: dict[str, Any] = {
                "epoch": epoch + 1,
                "training_loss": epoch_loss / batches,
                "duration_seconds": time.perf_counter() - started_at,
            }
            if validation_prompts is not None and validation_labels is not None:
                validation_logits = self._score_token_batches(validation_prompts)
                probabilities = self._positive_probabilities(validation_logits)
                validation_score = float(
                    average_precision_score(validation_labels, probabilities)
                )
                epoch_result["validation_pr_auc"] = validation_score
                if validation_score > best_validation_score:
                    best_validation_score = validation_score
                    best_state = {
                        name: parameter.detach().cpu().clone()
                        for name, parameter in self.model_.named_parameters()
                        if parameter.requires_grad
                    }
            self.training_history_.append(epoch_result)
            print(
                f"aLoRA epoch {epoch_result['epoch']}/{self.epochs}: "
                f"loss={epoch_result['training_loss']:.4f}, "
                f"val_pr_auc={epoch_result.get('validation_pr_auc', float('nan')):.4f}, "
                f"time={epoch_result['duration_seconds']:.1f}s",
                flush=True,
            )

        if best_state is not None:
            with torch.no_grad():
                for name, parameter in self.model_.named_parameters():
                    if name in best_state:
                        parameter.copy_(best_state[name].to(parameter.device))
        self.model_.eval()
        return self

    @staticmethod
    def _positive_probabilities(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities[:, 1]
