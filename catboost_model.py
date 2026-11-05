from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def create_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                FunctionTransformer(
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "classifier",
                CatBoostClassifier(
                    iterations=500,
                    depth=5,
                    learning_rate=0.03,
                    auto_class_weights="Balanced",
                    cat_features=("employment_status",),
                    random_seed=2027,
                    thread_count=1,
                    verbose=False,
                    allow_writing_files=False,
                ),
            ),
        ]
    )
