from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion


def create_model() -> Pipeline:
    classifier = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        random_state=2027,
        device="cpu",
        ignore_pretraining_limits=True,
    )
    return Pipeline(
        steps=[
            (
                "preprocessor",
                FunctionTransformer(
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("classifier", classifier),
        ]
    )
