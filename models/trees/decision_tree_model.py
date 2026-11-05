from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common.model_utils import make_pipeline


def create_model(scale_numeric: bool = False) -> Pipeline:
    classifier = DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=4,
        min_samples_leaf=20,
        random_state=2027,
    )
    return make_pipeline(classifier, scale_numeric=scale_numeric)
