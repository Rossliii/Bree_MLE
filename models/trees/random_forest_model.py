from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from common.model_utils import make_pipeline


def create_model(scale_numeric: bool = False) -> Pipeline:
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=2027,
        n_jobs=1,
    )
    return make_pipeline(classifier, scale_numeric=scale_numeric)
