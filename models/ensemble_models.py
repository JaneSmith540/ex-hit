import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from models.base import BaseModel
from models.linear_models import LassoModel, RidgeModel, ElasticNetModel
from models.tree_models import RandomForestModel

try:
    from models.tree_models import LightGBMModel
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from models.tree_models import XGBoostModel
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class LinearEnsembleModel(BaseModel, model_name="linear_ensemble"):
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        weights: Optional[List[float]] = None
    ):
        super().__init__(model_params, random_state)
        
        self.weights = weights or [1/3, 1/3, 1/3]
        
        self.models = {
            "lasso": LassoModel(
                model_params=self.model_params.get("lasso", {}),
                random_state=random_state
            ),
            "ridge": RidgeModel(
                model_params=self.model_params.get("ridge", {}),
                random_state=random_state
            ),
            "elastic_net": ElasticNetModel(
                model_params=self.model_params.get("elastic_net", {}),
                random_state=random_state
            )
        }
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[tuple] = None
    ) -> "LinearEnsembleModel":
        self.feature_names_ = X.columns.tolist()
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X, y, eval_set)
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.check_is_fitted()
        
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return weighted_pred
    
    def _get_builtin_importance(self) -> pd.DataFrame:
        self.check_is_fitted()
        
        all_importance = []
        for name, model in self.models.items():
            imp = model.get_feature_importance()
            imp = imp.rename(columns={"importance": f"importance_{name}"})
            all_importance.append(imp)
        
        merged = all_importance[0]
        for imp in all_importance[1:]:
            merged = merged.merge(imp, on="feature")
        
        importance_cols = [c for c in merged.columns if c.startswith("importance_")]
        merged["importance"] = merged[importance_cols].mean(axis=1)
        
        return merged[["feature", "importance"]].sort_values(
            "importance", ascending=False
        )
    
    def get_individual_predictions(
        self, X: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        self.check_is_fitted()
        
        return {
            name: model.predict(X)
            for name, model in self.models.items()
        }
    
    def get_individual_importance(self) -> Dict[str, pd.DataFrame]:
        self.check_is_fitted()
        
        return {
            name: model.get_feature_importance()
            for name, model in self.models.items()
        }


class TreeEnsembleModel(BaseModel, model_name="tree_ensemble"):
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        weights: Optional[List[float]] = None
    ):
        super().__init__(model_params, random_state)
        
        self.models = {
            "random_forest": RandomForestModel(
                model_params=self.model_params.get("random_forest", {}),
                random_state=random_state
            )
        }
        
        if HAS_LIGHTGBM:
            self.models["lightgbm"] = LightGBMModel(
                model_params=self.model_params.get("lightgbm", {}),
                random_state=random_state
            )

        if HAS_XGBOOST:
            self.models["xgboost"] = XGBoostModel(
                model_params=self.model_params.get("xgboost", {}),
                random_state=random_state
            )

        self.weights = weights or [1.0 / len(self.models)] * len(self.models)
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[tuple] = None
    ) -> "TreeEnsembleModel":
        self.feature_names_ = X.columns.tolist()
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X, y, eval_set)
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.check_is_fitted()
        
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return weighted_pred
    
    def _get_builtin_importance(self) -> pd.DataFrame:
        self.check_is_fitted()
        
        all_importance = []
        for name, model in self.models.items():
            imp = model.get_feature_importance()
            imp = imp.rename(columns={"importance": f"importance_{name}"})
            all_importance.append(imp)
        
        merged = all_importance[0]
        for imp in all_importance[1:]:
            merged = merged.merge(imp, on="feature")
        
        importance_cols = [c for c in merged.columns if c.startswith("importance_")]
        merged["importance"] = merged[importance_cols].mean(axis=1)
        
        return merged[["feature", "importance"]].sort_values(
            "importance", ascending=False
        )
    
    def get_individual_predictions(
        self, X: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        self.check_is_fitted()
        
        return {
            name: model.predict(X)
            for name, model in self.models.items()
        }
    
    def get_individual_importance(self) -> Dict[str, pd.DataFrame]:
        self.check_is_fitted()
        
        return {
            name: model.get_feature_importance()
            for name, model in self.models.items()
        }


class WeightedEnsembleModel(BaseModel, model_name="weighted_ensemble"):
    
    def __init__(
        self,
        models: Dict[str, BaseModel],
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(model_params, random_state)
        
        self.models = models
        
        if weights is None:
            self.weights = {name: 1.0 / len(models) for name in models}
        else:
            self.weights = weights
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[tuple] = None
    ) -> "WeightedEnsembleModel":
        self.feature_names_ = X.columns.tolist()
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X, y, eval_set)
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.check_is_fitted()
        
        weighted_sum = np.zeros(len(X))
        total_weight = sum(self.weights.values())
        
        for name, model in self.models.items():
            pred = model.predict(X)
            weighted_sum += pred * self.weights[name]
        
        return weighted_sum / total_weight
    
    def _get_builtin_importance(self) -> pd.DataFrame:
        self.check_is_fitted()
        
        all_importance = []
        for name, model in self.models.items():
            imp = model.get_feature_importance()
            imp = imp.rename(columns={"importance": f"importance_{name}"})
            all_importance.append(imp)
        
        merged = all_importance[0]
        for imp in all_importance[1:]:
            merged = merged.merge(imp, on="feature")
        
        importance_cols = [c for c in merged.columns if c.startswith("importance_")]
        
        weights_array = np.array([
            self.weights.get(c.replace("importance_", ""), 1.0)
            for c in importance_cols
        ])
        weights_array = weights_array / weights_array.sum()
        
        merged["importance"] = sum(
            merged[c] * w for c, w in zip(importance_cols, weights_array)
        )
        
        return merged[["feature", "importance"]].sort_values(
            "importance", ascending=False
        )
    
    def set_weights(self, weights: Dict[str, float]):
        self.weights = weights
    
    def optimize_weights(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "grid_search"
    ) -> Dict[str, float]:
        self.check_is_fitted()
        
        predictions = {
            name: model.predict(X)
            for name, model in self.models.items()
        }
        
        if method == "grid_search":
            from sklearn.model_selection import GridSearchCV
            from sklearn.base import BaseEstimator, RegressorMixin
            
            class WeightedPredictor(BaseEstimator, RegressorMixin):
                def __init__(self, preds_dict, weights=None):
                    self.preds_dict = preds_dict
                    self.weights = weights or {k: 1.0 for k in preds_dict}
                
                def predict(self, X):
                    result = np.zeros(len(X))
                    total_w = sum(self.weights.values())
                    for name, pred in self.preds_dict.items():
                        result += pred * self.weights[name]
                    return result / total_w
            
            best_score = -np.inf
            best_weights = self.weights.copy()
            
            model_names = list(self.models.keys())
            n_models = len(model_names)
            
            weight_options = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            from itertools import product
            
            for weights_tuple in product(weight_options, repeat=n_models):
                if abs(sum(weights_tuple) - 1.0) > 0.01:
                    continue
                
                test_weights = {
                    name: w for name, w in zip(model_names, weights_tuple)
                }
                
                pred = np.zeros(len(X))
                for name, p in predictions.items():
                    pred += p * test_weights[name]
                
                from scipy.stats import spearmanr
                ic, _ = spearmanr(y, pred)
                
                if ic > best_score:
                    best_score = ic
                    best_weights = test_weights.copy()
            
            self.weights = best_weights
            return best_weights
        
        return self.weights
