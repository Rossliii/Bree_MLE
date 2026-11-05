from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class LightGBMFeatureEncoder(TransformerMixin, BaseEstimator):
    """Keep numeric features numeric and encode categorical features natively."""

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
    ) -> LightGBMFeatureEncoder:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("LightGBMFeatureEncoder requires a pandas DataFrame.")
        self.feature_names_in_ = np.asarray(x.columns, dtype=object)
        self.categorical_features_ = [
            column
            for column in x.columns
            if not pd.api.types.is_numeric_dtype(x[column].dtype)
        ]
        self.categories_ = {
            column: pd.Index(x[column].dropna().unique())
            for column in self.categorical_features_
        }
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["feature_names_in_", "categories_"])
        if not isinstance(x, pd.DataFrame):
            raise TypeError("LightGBMFeatureEncoder requires a pandas DataFrame.")
        missing_columns = set(self.feature_names_in_) - set(x.columns)
        if missing_columns:
            raise ValueError(f"Missing LightGBM features: {sorted(missing_columns)}")

        transformed = x.loc[:, self.feature_names_in_].copy()
        for column in self.categorical_features_:
            transformed[column] = pd.Categorical(
                transformed[column],
                categories=self.categories_[column],
            )
        return transformed

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


def create_model(**overrides: object) -> Pipeline:
    classifier_parameters: dict[str, object] = {
        "objective": "binary",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": -1,
        "min_child_samples": 20,
        "reg_lambda": 1.0,
        "class_weight": "balanced",
        "random_state": 2027,
        "n_jobs": 1,
        "verbosity": -1,
    }
    classifier_parameters.update(overrides)
    return Pipeline(
        steps=[
            ("feature_encoder", LightGBMFeatureEncoder()),
            ("classifier", LGBMClassifier(**classifier_parameters)),
        ]
    )
