import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from models.base import BaseModel


class LinearRegressionModel(BaseModel, model_name="linear"):
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        use_standardization: bool = True
    ):
        super().__init__(model_params, random_state)
        self.use_standardization = use_standardization
        self.scaler = StandardScaler() if use_standardization else None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[tuple] = None
    ) -> "LinearRegressionModel":
        self.feature_names_ = X.columns.tolist()
        
        X_processed = self._preprocess(X, fit=True)
        
        self.model = LinearRegression(**self.model_params)
        self.model.fit(X_processed, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.check_is_fitted()
        X_processed = self._preprocess(X, fit=False)
        return self.model.predict(X_processed)
    
    def _preprocess(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        X_array = X.values
        
        if self.use_standardization:
            if fit:
                X_array = self.scaler.fit_transform(X_array)
            else:
                X_array = self.scaler.transform(X_array)
            return pd.DataFrame(X_array, columns=X.columns, index=X.index)

        return X
    
    def _get_builtin_importance(self) -> pd.DataFrame:
        self.check_is_fitted()
        
        importance = np.abs(self.model.coef_)
        
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importance
        }).sort_values("importance", ascending=False)


class LassoModel(BaseModel, model_name="lasso"):
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        use_standardization: bool = True
    ):
        default_params = {"alpha": 0.001, "max_iter": 10000}
        if model_params:
            default_params.update(model_params)
        
        super().__init__(default_params, random_state)
        self.use_standardization = use_standardization
        self.scaler = StandardScaler() if use_standardization else None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[tuple] = None
    ) -> "LassoModel":
        self.feature_names_ = X.columns.tolist()
        
        X_processed = self._preprocess(X, fit=True)
        
        self.model = Lasso(**self.model_params)
        self.model.fit(X_processed, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.check_is_fitted()
        X_processed = self._preprocess(X, fit=False)
        return self.model.predict(X_processed)
    
    def _preprocess(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        X_array = X.values
        
        if self.use_standardization:
            if fit:
                X_array = self.scaler.fit_transform(X_array)
            else:
                X_array = self.scaler.transform(X_array)
            return pd.DataFrame(X_array, columns=X.columns, index=X.index)

        return X
    
    def _get_builtin_importance(self) -> pd.DataFrame:
        self.check_is_fitted()
        
        importance = np.abs(self.model.coef_)
        
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importance
        }).sort_values("importance", ascending=False)
    
    def get_selected_features(self, threshold: float = 1e-5) -> list:
        self.check_is_fitted()
        
        selected_mask = np.abs(self.model.coef_) > threshold
        return [f for f, s in zip(self.feature_names_, selected_mask) if s]


class RidgeModel(BaseModel, model_name="ridge"):
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        use_standardization: bool = True
    ):
        default_params = {"alpha": 1.0, "max_iter": 10000}
        if model_params:
            default_params.update(model_params)
        
        super().__init__(default_params, random_state)
        self.use_standardization = use_standardization
        self.scaler = StandardScaler() if use_standardization else None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[tuple] = None
    ) -> "RidgeModel":
        self.feature_names_ = X.columns.tolist()
        
        X_processed = self._preprocess(X, fit=True)
        
        self.model = Ridge(**self.model_params)
        self.model.fit(X_processed, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.check_is_fitted()
        X_processed = self._preprocess(X, fit=False)
        return self.model.predict(X_processed)
    
    def _preprocess(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        X_array = X.values
        
        if self.use_standardization:
            if fit:
                X_array = self.scaler.fit_transform(X_array)
            else:
                X_array = self.scaler.transform(X_array)
            return pd.DataFrame(X_array, columns=X.columns, index=X.index)

        return X
    
    def _get_builtin_importance(self) -> pd.DataFrame:
        self.check_is_fitted()
        
        importance = np.abs(self.model.coef_)
        
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importance
        }).sort_values("importance", ascending=False)


class ElasticNetModel(BaseModel, model_name="elastic_net"):
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        use_standardization: bool = True
    ):
        default_params = {
            "alpha": 0.1,
            "l1_ratio": 0.5,
            "max_iter": 10000
        }
        if model_params:
            default_params.update(model_params)
        
        super().__init__(default_params, random_state)
        self.use_standardization = use_standardization
        self.scaler = StandardScaler() if use_standardization else None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[tuple] = None
    ) -> "ElasticNetModel":
        self.feature_names_ = X.columns.tolist()
        
        X_processed = self._preprocess(X, fit=True)
        
        self.model = ElasticNet(**self.model_params)
        self.model.fit(X_processed, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.check_is_fitted()
        X_processed = self._preprocess(X, fit=False)
        return self.model.predict(X_processed)
    
    def _preprocess(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        X_array = X.values
        
        if self.use_standardization:
            if fit:
                X_array = self.scaler.fit_transform(X_array)
            else:
                X_array = self.scaler.transform(X_array)
            return pd.DataFrame(X_array, columns=X.columns, index=X.index)

        return X
    
    def _get_builtin_importance(self) -> pd.DataFrame:
        self.check_is_fitted()
        
        importance = np.abs(self.model.coef_)
        
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importance
        }).sort_values("importance", ascending=False)
    
    def get_selected_features(self, threshold: float = 1e-5) -> list:
        self.check_is_fitted()
        
        selected_mask = np.abs(self.model.coef_) > threshold
        return [f for f, s in zip(self.feature_names_, selected_mask) if s]
