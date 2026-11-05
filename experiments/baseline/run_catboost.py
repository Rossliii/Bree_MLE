from common.model_utils import run_all_models
from common.paths import PROJECT_ROOT
from models.linear.logistic_regression_model import (
    create_model as create_logistic_regression,
)
from models.trees.catboost_model import create_model as create_catboost
from models.trees.decision_tree_model import create_model as create_decision_tree
from models.trees.random_forest_model import create_model as create_random_forest
from models.trees.xgboost_model import create_model as create_xgboost


MODEL_FACTORIES = {
    "Logistic Regression": create_logistic_regression,
    "Decision Tree": create_decision_tree,
    "XGBoost": create_xgboost,
    "Random Forest": create_random_forest,
    "CatBoost": create_catboost,
}


def main() -> None:
    project_dir = PROJECT_ROOT
    run_all_models(
        model_factories=MODEL_FACTORIES,
        data_path=project_dir / "loan_applications.csv",
        output_dir=project_dir / "reports_catboost",
    )


if __name__ == "__main__":
    main()
