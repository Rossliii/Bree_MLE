from pathlib import Path

from catboost_model import create_model as create_catboost
from decision_tree_model import create_model as create_decision_tree
from logistic_regression_model import create_model as create_logistic_regression
from model_utils import run_all_models
from random_forest_model import create_model as create_random_forest
from xgboost_model import create_model as create_xgboost


MODEL_FACTORIES = {
    "Logistic Regression": create_logistic_regression,
    "Decision Tree": create_decision_tree,
    "XGBoost": create_xgboost,
    "Random Forest": create_random_forest,
    "CatBoost": create_catboost,
}


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    run_all_models(
        model_factories=MODEL_FACTORIES,
        data_path=project_dir / "loan_applications.csv",
        output_dir=project_dir / "reports_catboost",
    )


if __name__ == "__main__":
    main()
