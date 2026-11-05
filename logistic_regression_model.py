from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from model_utils import make_pipeline


def create_model(scale_numeric: bool = True) -> Pipeline:
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=2027,
    )
    return make_pipeline(classifier, scale_numeric=scale_numeric)
