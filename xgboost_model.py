from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from model_utils import make_pipeline


def create_model(
    positive_class_weight: float,
    scale_numeric: bool = False,
) -> Pipeline:
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        scale_pos_weight=positive_class_weight,
        random_state=2027,
        n_jobs=1,
        verbosity=0,
    )
    return make_pipeline(classifier, scale_numeric=scale_numeric)
