from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion


def create_model(random_state: int = 2027) -> Pipeline:
    classifier = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        random_state=random_state,
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
