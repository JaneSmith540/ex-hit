from models.base import BaseModel, ModelComparator
from models.linear_models import (
    LinearRegressionModel,
    LassoModel,
    RidgeModel,
    ElasticNetModel
)
from models.tree_models import RandomForestModel, LightGBMModel
from models.ensemble_models import (
    LinearEnsembleModel,
    TreeEnsembleModel,
    WeightedEnsembleModel
)

AVAILABLE_MODELS = {
    "linear": {
        "class_name": "LinearRegressionModel",
        "module": "models.linear_models",
        "type": "linear",
    },
    "lasso": {
        "class_name": "LassoModel",
        "module": "models.linear_models",
        "type": "linear",
    },
    "ridge": {
        "class_name": "RidgeModel",
        "module": "models.linear_models",
        "type": "linear",
    },
    "elastic_net": {
        "class_name": "ElasticNetModel",
        "module": "models.linear_models",
        "type": "linear",
    },
    "random_forest": {
        "class_name": "RandomForestModel",
        "module": "models.tree_models",
        "type": "tree",
    },
    "lightgbm": {
        "class_name": "LightGBMModel",
        "module": "models.tree_models",
        "type": "tree",
    },
    "linear_ensemble": {
        "class_name": "LinearEnsembleModel",
        "module": "models.ensemble_models",
        "type": "ensemble",
    },
    "tree_ensemble": {
        "class_name": "TreeEnsembleModel",
        "module": "models.ensemble_models",
        "type": "ensemble",
    },
}


def get_model_info(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}")
    return AVAILABLE_MODELS[model_name]

__all__ = [
    "BaseModel",
    "ModelComparator",
    "LinearRegressionModel",
    "LassoModel",
    "RidgeModel",
    "ElasticNetModel",
    "RandomForestModel",
    "LightGBMModel",
    "LinearEnsembleModel",
    "TreeEnsembleModel",
    "WeightedEnsembleModel",
    "AVAILABLE_MODELS",
    "get_model_info",
]
