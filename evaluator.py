import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelEvaluator:
    
    def __init__(self, config: Any):
        self.config = config
        self.metrics_registry: Dict[str, Callable] = {}
        self._register_default_metrics()
    
    def _register_default_metrics(self):
        self.register_metric("mse", self._mse)
        self.register_metric("rmse", self._rmse)
        self.register_metric("mae", self._mae)
        self.register_metric("r2", self._r2)
        self.register_metric("ic", self._ic)
        self.register_metric("rank_ic", self._rank_ic)
        self.register_metric("icir", self._icir)
    
    def register_metric(self, name: str, func: Callable):
        self.metrics_registry[name] = func
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        metrics = metrics or self.config.metrics
        
        results = {}
        for metric_name in metrics:
            if metric_name in self.metrics_registry:
                try:
                    results[metric_name] = self.metrics_registry[metric_name](
                        y_true, y_pred
                    )
                except Exception as e:
                    print(f"Error calculating {metric_name}: {e}")
                    results[metric_name] = np.nan
        
        return results
    
    def _mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    def _rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(self._mse(y_true, y_pred))
    
    def _mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))
    
    def _r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    def _ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) < 2:
            return np.nan
        corr, _ = stats.pearsonr(y_true, y_pred)
        return corr
    
    def _rank_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) < 2:
            return np.nan
        corr, _ = stats.spearmanr(y_true, y_pred)
        return corr
    
    def _icir(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ic = self._ic(y_true, y_pred)
        if np.isnan(ic):
            return np.nan
        return ic / np.std(y_pred) if np.std(y_pred) > 0 else 0.0
    
    def analyze_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        method: str = "built_in",
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        importance_df = model.get_feature_importance(
            method=method, X=X, y=y
        )
        
        if len(importance_df) > top_n:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 8)
    ):
        plt.figure(figsize=figsize)
        
        if "importance_std" in importance_df.columns:
            plt.barh(
                importance_df["feature"],
                importance_df["importance_mean"],
                xerr=importance_df["importance_std"],
                capsize=3
            )
        else:
            plt.barh(
                importance_df["feature"],
                importance_df["importance"]
            )
        
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close()
    
    def compare_models(
        self,
        results_dict: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
        figsize: tuple = (12, 6)
    ) -> pd.DataFrame:
        comparison_df = pd.DataFrame(results_dict).T
        comparison_df.index.name = "model"
        comparison_df = comparison_df.reset_index()
        
        if self.config.plot_results:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            metrics_for_plot = ["mse", "mae", "r2"]
            available_metrics = [m for m in metrics_for_plot if m in comparison_df.columns]
            
            if available_metrics:
                comparison_df.plot(
                    x="model",
                    y=available_metrics,
                    kind="bar",
                    ax=axes[0]
                )
                axes[0].set_title("Model Performance Comparison")
                axes[0].set_ylabel("Score")
                axes[0].legend(loc="best")
            
            ic_metrics = ["ic", "rank_ic"]
            available_ic = [m for m in ic_metrics if m in comparison_df.columns]
            
            if available_ic:
                comparison_df.plot(
                    x="model",
                    y=available_ic,
                    kind="bar",
                    ax=axes[1]
                )
                axes[1].set_title("Information Coefficient Comparison")
                axes[1].set_ylabel("IC")
                axes[1].legend(loc="best")
            
            plt.tight_layout()
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches="tight")

            plt.close(fig)
        
        return comparison_df
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
        figsize: tuple = (12, 5)
    ):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")
        axes[0].set_xlabel("True Values")
        axes[0].set_ylabel("Predicted Values")
        axes[0].set_title(f"{model_name}: True vs Predicted")
        axes[0].legend()
        
        residuals = y_true - y_pred
        axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        axes[1].axvline(x=0, color="r", linestyle="--", label="Zero")
        axes[1].set_xlabel("Residuals")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(f"{model_name}: Residual Distribution")
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)
    
    def generate_report(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        importance_df: Optional[pd.DataFrame] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        results = self.evaluate(y_true, y_pred)
        
        report = {
            "model_name": model_name,
            "metrics": results,
            "n_samples": len(y_true),
            "prediction_stats": {
                "mean": float(np.mean(y_pred)),
                "std": float(np.std(y_pred)),
                "min": float(np.min(y_pred)),
                "max": float(np.max(y_pred))
            },
            "target_stats": {
                "mean": float(np.mean(y_true)),
                "std": float(np.std(y_true)),
                "min": float(np.min(y_true)),
                "max": float(np.max(y_true))
            }
        }
        
        if importance_df is not None:
            report["top_features"] = importance_df.head(10).to_dict("records")
        
        if save_dir and self.config.save_results:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            if self.config.plot_results:
                self.plot_predictions(
                    y_true, y_pred, model_name,
                    save_path=str(save_dir / f"{model_name}_predictions.png")
                )
                
                if importance_df is not None:
                    self.plot_feature_importance(
                        importance_df,
                        save_path=str(save_dir / f"{model_name}_importance.png")
                    )
        
        return report
    
    def cross_validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        from sklearn.model_selection import TimeSeriesSplit
        
        metrics = metrics or self.config.metrics
        kf = TimeSeriesSplit(n_splits=cv_folds)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model_copy = type(model)(
                model_params=model.model_params,
                random_state=model.random_state
            )
            model_copy.fit(X_train, y_train)
            y_pred = model_copy.predict(X_val)
            
            fold_metrics = self.evaluate(y_val.values, y_pred, metrics)
            fold_results.append(fold_metrics)
        
        cv_results = {
            "fold_results": fold_results,
            "mean": {
                metric: np.mean([r[metric] for r in fold_results])
                for metric in metrics
            },
            "std": {
                metric: np.std([r[metric] for r in fold_results])
                for metric in metrics
            }
        }
        
        return cv_results
