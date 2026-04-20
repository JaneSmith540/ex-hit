import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestRegressor
from models.base import BaseModel

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class RandomForestModel(BaseModel, model_name="random_forest"):
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "n_jobs": 1
        }
        if model_params:
            default_params.update(model_params)
        
        super().__init__(default_params, random_state)
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[tuple] = None
    ) -> "RandomForestModel":
        self.feature_names_ = X.columns.tolist()
        
        self.model = RandomForestRegressor(
            random_state=self.random_state,
            **self.model_params
        )
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.check_is_fitted()
        return self.model.predict(X)
    
    def _get_builtin_importance(self) -> pd.DataFrame:
        self.check_is_fitted()
        
        importance = self.model.feature_importances_
        
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importance
        }).sort_values("importance", ascending=False)


class LightGBMModel(BaseModel, model_name="lightgbm"):
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is not installed. Install it with: pip install lightgbm")
        
        default_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "verbosity": -1
        }
        if model_params:
            default_params.update(model_params)
        
        super().__init__(default_params, random_state)
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[tuple] = None
    ) -> "LightGBMModel":
        self.feature_names_ = X.columns.tolist()
        
        callbacks = []
        
        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = [eval_set]
            callbacks.append(lgb.early_stopping(50, verbose=False))
        
        self.model = lgb.LGBMRegressor(
            random_state=self.random_state,
            **self.model_params
        )
        
        self.model.fit(
            X, y,
            callbacks=callbacks,
            **fit_params
        )
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.check_is_fitted()
        return self.model.predict(X)
    
    def _get_builtin_importance(self) -> pd.DataFrame:
        self.check_is_fitted()
        
        importance = self.model.feature_importances_
        
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importance
        }).sort_values("importance", ascending=False)
    
    def get_split_importance(self) -> pd.DataFrame:
        self.check_is_fitted()
        
        importance = self.model.booster_.feature_importance(importance_type="split")
        
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importance
        }).sort_values("importance", ascending=False)
    
    def get_gain_importance(self) -> pd.DataFrame:
        self.check_is_fitted()
        
        importance = self.model.booster_.feature_importance(importance_type="gain")
        
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importance
        }).sort_values("importance", ascending=False)


class XGBoostModel(BaseModel, model_name="xgboost"):

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed. Install it with: pip install xgboost")

        default_params = {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": -1,
        }
        if model_params:
            default_params.update(model_params)

        super().__init__(default_params, random_state)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[tuple] = None
    ) -> "XGBoostModel":
        self.feature_names_ = X.columns.tolist()

        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = [eval_set]
            fit_params["verbose"] = False

        self.model = xgb.XGBRegressor(
            random_state=self.random_state,
            **self.model_params
        )
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.check_is_fitted()
        return self.model.predict(X)

    def _get_builtin_importance(self) -> pd.DataFrame:
        self.check_is_fitted()

        importance = self.model.feature_importances_

        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importance
        }).sort_values("importance", ascending=False)
