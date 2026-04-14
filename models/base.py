from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


class BaseModel(ABC):
    
    _registry: Dict[str, Type["BaseModel"]] = {}
    
    def __init__(
        self, 
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        self.model_params = model_params or {}
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.feature_names_: Optional[List[str]] = None
    
    def __init_subclass__(cls, model_name: str = None):
        if model_name:
            BaseModel._registry[model_name] = cls
    
    @classmethod
    def get_registry(cls) -> Dict[str, Type["BaseModel"]]:
        return cls._registry
    
    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> "BaseModel":
        if model_name not in cls._registry:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[model_name](**kwargs)
    
    @abstractmethod
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        eval_set: Optional[tuple] = None
    ) -> "BaseModel":
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    def fit_predict(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X)
    
    def get_feature_importance(
        self, 
        method: str = "built_in",
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        if method == "built_in":
            return self._get_builtin_importance()
        elif method == "permutation":
            if X is None or y is None:
                raise ValueError("X and y are required for permutation importance")
            return self._get_permutation_importance(X, y)
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    @abstractmethod
    def _get_builtin_importance(self) -> pd.DataFrame:
        pass
    
    def _get_permutation_importance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_repeats: int = 10
    ) -> pd.DataFrame:
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            self.model, X, y,
            n_repeats=n_repeats,
            random_state=self.random_state
        )
        
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std
        })
        
        return importance_df.sort_values("importance_mean", ascending=False)
    
    def get_params(self) -> Dict[str, Any]:
        return self.model_params.copy()
    
    def set_params(self, **params) -> "BaseModel":
        self.model_params.update(params)
        if self.model is not None:
            self.model.set_params(**params)
        return self
    
    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "model_params": self.model_params,
                "is_fitted": self.is_fitted,
                "feature_names_": self.feature_names_
            }, f)
    
    def load(self, path: str) -> "BaseModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self.model_params = data["model_params"]
        self.is_fitted = data["is_fitted"]
        self.feature_names_ = data["feature_names_"]
        
        return self
    
    def check_is_fitted(self):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")


class ModelComparator:
    
    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self.results: Dict[str, Dict[str, float]] = {}
    
    def add_model(self, name: str, model: BaseModel):
        self.models[name] = model
    
    def fit_all(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train, eval_set=eval_set)
    
    def evaluate_all(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        metrics: Dict[str, callable]
    ) -> pd.DataFrame:
        results = []
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            result = {"model": name}
            for metric_name, metric_func in metrics.items():
                result[metric_name] = metric_func(y_test, y_pred)
            
            results.append(result)
            self.results[name] = result
        
        return pd.DataFrame(results)
    
    def get_feature_importance_comparison(
        self,
        method: str = "built_in",
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None
    ) -> Dict[str, pd.DataFrame]:
        importance_dict = {}
        
        for name, model in self.models.items():
            try:
                importance = model.get_feature_importance(
                    method=method, X=X, y=y
                )
                importance_dict[name] = importance
            except Exception as e:
                print(f"Could not get importance for {name}: {e}")
        
        return importance_dict
    
    def get_best_model(self, metric: str = "mse", ascending: bool = True) -> str:
        if not self.results:
            raise ValueError("No evaluation results. Run evaluate_all() first.")
        
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1][metric],
            reverse=not ascending
        )
        
        return sorted_results[0][0]
