from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class WeightOfEvidenceEncoder(TransformerMixin, BaseEstimator):
    """Encode each feature using training-fold default/repaid log odds."""

    def __init__(self, n_bins: int = 5, smoothing: float = 0.5) -> None:
        self.n_bins = n_bins
        self.smoothing = smoothing

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> WeightOfEvidenceEncoder:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("WeightOfEvidenceEncoder requires a pandas DataFrame.")
        if self.n_bins < 1:
            raise ValueError("n_bins must be positive.")
        if self.smoothing <= 0:
            raise ValueError("smoothing must be positive.")

        labels = np.asarray(y, dtype=np.int64)
        if len(labels) != len(x) or set(np.unique(labels)) != {0, 1}:
            raise ValueError("WOE fitting requires matching features and both labels 0 and 1.")

        self.feature_names_in_ = np.asarray(x.columns, dtype=object)
        self.numeric_features_ = [
            column
            for column in x.columns
            if pd.api.types.is_numeric_dtype(x[column].dtype)
        ]
        self.categorical_features_ = [
            column for column in x.columns if column not in self.numeric_features_
        ]
        self.bin_edges_: dict[str, np.ndarray] = {}
        self.woe_mappings_: dict[str, dict[int | str, float]] = {}
        total_defaults = int(labels.sum())
        total_repaid = len(labels) - total_defaults

        for column in self.feature_names_in_:
            if column in self.numeric_features_:
                numeric_values = pd.to_numeric(x[column], errors="raise").astype(float)
                nonmissing = numeric_values.dropna()
                quantiles = (
                    nonmissing.quantile(
                        np.linspace(0, 1, self.n_bins + 1)
                    ).to_numpy()
                    if not nonmissing.empty
                    else np.array([], dtype=float)
                )
                internal_edges = np.unique(quantiles[1:-1])
                edges = np.concatenate(([-np.inf], internal_edges, [np.inf]))
                self.bin_edges_[column] = edges
                values = self._numeric_bins(numeric_values, edges)
            else:
                values = self._categorical_bins(x[column])

            grouped = pd.DataFrame(
                {"bin": values, "target": labels},
                index=x.index,
            ).groupby("bin", dropna=False)["target"].agg(["sum", "count"])
            bin_count = len(grouped)
            default_distribution = (
                grouped["sum"] + self.smoothing
            ) / (total_defaults + self.smoothing * bin_count)
            repaid_distribution = (
                grouped["count"] - grouped["sum"] + self.smoothing
            ) / (total_repaid + self.smoothing * bin_count)
            self.woe_mappings_[column] = np.log(
                default_distribution / repaid_distribution
            ).to_dict()

        return self

    @staticmethod
    def _numeric_bins(values: pd.Series, edges: np.ndarray) -> pd.Series:
        bins = pd.cut(values, bins=edges, labels=False, include_lowest=True)
        return bins.fillna(-1).astype(int)

    @staticmethod
    def _categorical_bins(values: pd.Series) -> pd.Series:
        return values.astype("string").fillna("__WOE_MISSING__")

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["woe_mappings_", "feature_names_in_"])
        if not isinstance(x, pd.DataFrame):
            raise TypeError("WeightOfEvidenceEncoder requires a pandas DataFrame.")
        missing_columns = set(self.feature_names_in_) - set(x.columns)
        if missing_columns:
            raise ValueError(f"Missing WOE features: {sorted(missing_columns)}")

        encoded: dict[str, pd.Series] = {}
        for column in self.feature_names_in_:
            if column in self.numeric_features_:
                numeric_values = pd.to_numeric(x[column], errors="raise").astype(float)
                values = self._numeric_bins(numeric_values, self.bin_edges_[column])
            else:
                values = self._categorical_bins(x[column])
            encoded[column] = (
                values.map(self.woe_mappings_[column]).fillna(0.0).astype(float)
            )
        return pd.DataFrame(encoded, index=x.index)

    def get_feature_names_out(
        self,
        input_features: np.ndarray | None = None,
    ) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        if input_features is not None and not np.array_equal(
            input_features,
            self.feature_names_in_,
        ):
            raise ValueError("input_features must match the features seen during fit.")
        return self.feature_names_in_.copy()


def create_model(
    c: float = 0.1,
    class_weight: str | None = "balanced",
    n_bins: int = 5,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("woe", WeightOfEvidenceEncoder(n_bins=n_bins)),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    class_weight=class_weight,
                    max_iter=2000,
                    random_state=2027,
                ),
            ),
        ]
    )
