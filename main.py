import pandas as pd
import numpy as np
import polars as pl
from typing import Dict, List, Optional, Any
from pathlib import Path
import argparse
import json
import inspect

from data_processor import DataProcessor
from feature_engineer import FeatureEngineer
from evaluator import ModelEvaluator
from models import AVAILABLE_MODELS, get_model_info
from models.base import BaseModel, ModelComparator
from settings import load_config


class WritableDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return WritableDataFrame

    @property
    def values(self):
        return self.to_numpy(copy=True)


class WritableSeries(pd.Series):
    @property
    def _constructor(self):
        return WritableSeries

    def to_numpy(self, *args, **kwargs):
        kwargs["copy"] = True
        return super().to_numpy(*args, **kwargs)

    @property
    def values(self):
        return self.to_numpy(copy=True)


class Experiment:
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        extra_feature_registry: Optional[Dict[str, Any]] = None,
    ):
        resolved_config_path = config_path
        if resolved_config_path is None:
            default_config = Path.cwd() / "config.yaml"
            if default_config.exists():
                resolved_config_path = str(default_config)

        if not resolved_config_path:
            raise FileNotFoundError("No config.yaml found. Pass --config or add config.yaml in the repo root.")

        self.config = load_config(resolved_config_path)
        
        self.data_processor = DataProcessor(self.config.data)
        self.feature_engineer = FeatureEngineer(
            self.config.feature,
            extra_feature_registry=extra_feature_registry,
        )
        self.evaluator = ModelEvaluator(self.config.eval)
        
        self.model: Optional[BaseModel] = None
        self.comparator: Optional[ModelComparator] = None
        self.model_df: Optional[pl.DataFrame] = None
        self.feature_names: List[str] = []
        
        self.X_train: Optional[pl.DataFrame] = None
        self.y_train: Optional[pl.Series] = None
        self.X_test: Optional[pl.DataFrame] = None
        self.y_test: Optional[pl.Series] = None
    
    def load_and_process_data(self, event_df: Optional[pl.DataFrame] = None) -> Dict[str, Any]:
        if event_df is None:
            print("Building event dataset from day/tick/l2/min sources...")
            event_df = self.data_processor.build_event_dataset(self.feature_engineer)
        else:
            print(f"Using prebuilt event dataset with {event_df.height} rows.")
        if event_df.is_empty():
            print("No touch events found.")
            return {"n_train": 0, "n_test": 0, "n_features": 0}

        feature_cols = self.feature_engineer.get_feature_names()
        train_raw = event_df.filter(
            (pl.col("trade_date") >= self.config.data.train_start_date.replace("-", ""))
            & (pl.col("trade_date") <= self.config.data.train_end_date.replace("-", ""))
        )
        test_raw = event_df.filter(
            (pl.col("trade_date") >= self.config.data.test_start_date.replace("-", ""))
            & (pl.col("trade_date") <= self.config.data.test_end_date.replace("-", ""))
        )
        if train_raw.is_empty() or test_raw.is_empty():
            print("Train or test event set is empty after date split.")
            return {"n_train": train_raw.height, "n_test": test_raw.height, "n_features": 0}

        train_features = self._select_usable_feature_columns(train_raw, feature_cols, require_variance=True)
        available_features = train_features
        if not available_features:
            print("No usable training features found.")
            return {"n_train": train_raw.height, "n_test": test_raw.height, "n_features": 0}

        train_before_clean = train_raw.height
        test_before_clean = test_raw.height
        train_df = self.data_processor.clean_data(
            train_raw.select(["trade_date", *available_features, "next_open_return"]),
            remove_outliers=False,
        )
        test_df = self.data_processor.clean_data(
            test_raw.select(["trade_date", *available_features, "next_open_return"]),
            remove_outliers=False,
        )
        if train_df.is_empty() or test_df.is_empty():
            print("Train or test event set is empty after cleaning.")
            return {"n_train": train_df.height, "n_test": test_df.height, "n_features": len(available_features)}
        print(
            "Cleaning flow: "
            f"train {train_before_clean}->{train_df.height}, "
            f"test {test_before_clean}->{test_df.height}; "
            "removed only null/non-finite rows for model inputs."
        )

        self.model_df = pl.concat([train_df, test_df], how="vertical_relaxed").sort("trade_date")
        self.feature_names = available_features
        self.X_train = train_df.select(available_features)
        self.y_train = train_df.get_column("next_open_return")
        self.X_test = test_df.select(available_features)
        self.y_test = test_df.get_column("next_open_return")

        return {
            "n_train": self.X_train.height,
            "n_test": self.X_test.height,
            "n_features": len(available_features),
            "n_train_raw": train_before_clean,
            "n_test_raw": test_before_clean,
            "feature_names": available_features
        }

    def _require_model_dataset(self) -> pl.DataFrame:
        if self.model_df is None:
            raise ValueError("Data not loaded. Call load_and_process_data() first.")
        return self.model_df

    def _get_walk_forward_dates(self) -> List[str]:
        model_df = self._require_model_dataset()
        test_dates = (
            model_df.filter(
                (pl.col("trade_date") >= self.config.data.test_start_date.replace("-", ""))
                & (pl.col("trade_date") <= self.config.data.test_end_date.replace("-", ""))
            )
            .get_column("trade_date")
            .unique(maintain_order=True)
            .to_list()
        )
        return [str(date) for date in test_dates]

    def walk_forward_evaluate_model(
        self,
        model_name: str,
        model_params: Optional[Dict] = None,
        train_window_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        model_df = self._require_model_dataset()
        if not self.feature_names:
            raise ValueError("Feature set not prepared. Call load_and_process_data() first.")

        train_window_days = train_window_days or self.config.train.rolling_train_days
        unique_dates = [str(date) for date in model_df.get_column("trade_date").unique(maintain_order=True).to_list()]
        date_to_index = {date: idx for idx, date in enumerate(unique_dates)}

        daily_predictions: List[pd.DataFrame] = []
        for test_date in self._get_walk_forward_dates():
            test_idx = date_to_index.get(test_date)
            if test_idx is None:
                continue

            train_dates = unique_dates[max(0, test_idx - train_window_days):test_idx]
            if len(train_dates) < train_window_days:
                continue

            train_df = model_df.filter(pl.col("trade_date").is_in(train_dates))
            test_df = model_df.filter(pl.col("trade_date") == test_date)
            if train_df.is_empty() or test_df.is_empty():
                continue

            model = self.create_model(model_name, model_params)
            X_train_pd, y_train_pd = self._to_model_inputs(
                train_df.select(self.feature_names),
                train_df.get_column("next_open_return"),
            )
            X_test_pd, y_test_pd = self._to_model_inputs(
                test_df.select(self.feature_names),
                test_df.get_column("next_open_return"),
            )

            model.fit(X_train_pd, y_train_pd)
            y_pred = model.predict(X_test_pd)
            daily_frame = pd.DataFrame(
                {
                    "trade_date": test_date,
                    "y_true": y_test_pd.to_numpy(copy=True),
                    "y_pred": y_pred,
                }
            )
            daily_predictions.append(daily_frame)

        if not daily_predictions:
            raise ValueError("No valid walk-forward windows were generated.")

        predictions_df = pd.concat(daily_predictions, ignore_index=True)
        metrics = self.evaluator.evaluate(
            predictions_df["y_true"].to_numpy(),
            predictions_df["y_pred"].to_numpy(),
        )

        daily_metrics = []
        for trade_date, group in predictions_df.groupby("trade_date", sort=True):
            group_metrics = self.evaluator.evaluate(
                group["y_true"].to_numpy(),
                group["y_pred"].to_numpy(),
            )
            group_metrics["trade_date"] = trade_date
            group_metrics["n_samples"] = len(group)
            daily_metrics.append(group_metrics)

        return {
            "metrics": metrics,
            "daily_metrics": pd.DataFrame(daily_metrics),
            "predictions": predictions_df,
            "train_window_days": train_window_days,
            "n_test_days": predictions_df["trade_date"].nunique(),
            "n_test_samples": len(predictions_df),
        }

    def _select_usable_feature_columns(
        self,
        features_df: pl.DataFrame,
        feature_cols: List[str],
        require_variance: bool = True,
    ) -> List[str]:
        usable = []
        for col in feature_cols:
            if col not in features_df.columns:
                continue
            series = features_df.get_column(col)
            if series.dtype.is_numeric():
                stats = features_df.select(
                    [
                        pl.col(col).is_finite().sum().alias("finite_count"),
                        pl.col(col).filter(pl.col(col).is_finite()).std().alias("std"),
                    ]
                ).row(0, named=True)
                finite_count = stats["finite_count"]
                std = stats["std"]
                if finite_count <= 0:
                    continue
                if require_variance and (std is None or std <= 0):
                    continue
                if not require_variance or (std is not None and std > 0):
                    usable.append(col)
            elif series.null_count() < features_df.height:
                usable.append(col)
        return usable
    
    def _split_data(
        self,
        X: pl.DataFrame,
        y: pl.Series
    ) -> tuple:
        n_samples = X.height
        if n_samples == 0:
            return X, X, y, y
        if n_samples == 1:
            return X, X, y, y
        n_train = int(n_samples * (1 - self.config.train.validation_ratio))
        n_train = max(1, min(n_train, n_samples - 1))
        
        X_train = X[:n_train]
        X_test = X[n_train:]
        y_train = y[:n_train]
        y_test = y[n_train:]
        
        return X_train, X_test, y_train, y_test

    def _to_model_inputs(self, X: pl.DataFrame, y: Optional[pl.Series] = None):
        X_pd = WritableDataFrame(np.array(X.to_numpy(), copy=True), columns=X.columns)
        if y is None:
            return X_pd
        y_pd = WritableSeries(np.array(y.to_numpy(), copy=True), name=y.name, copy=True)
        return X_pd, y_pd
    
    def create_model(self, model_name: str, model_params: Optional[Dict] = None) -> BaseModel:
        model_info = get_model_info(model_name)
        
        module = __import__(
            model_info["module"],
            fromlist=[model_info["class_name"]]
        )
        model_class = getattr(module, model_info["class_name"])
        
        params = model_params or self.config.model.model_params
        
        init_kwargs = {
            "model_params": params,
            "random_state": self.config.model.random_state,
        }
        if "use_standardization" in inspect.signature(model_class.__init__).parameters:
            init_kwargs["use_standardization"] = self.config.model.use_standardization

        model = model_class(**init_kwargs)
        
        return model
    
    def train_single_model(
        self,
        model_name: str,
        model_params: Optional[Dict] = None
    ) -> BaseModel:
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_and_process_data() first.")
        
        print(f"Creating model: {model_name}")
        self.model = self.create_model(model_name, model_params)
        
        print(f"Training {model_name}...")
        X_train, y_train = self._to_model_inputs(self.X_train, self.y_train)
        self.model.fit(X_train, y_train)
        
        if self.config.train.save_model:
            save_path = Path(self.config.train.model_save_path) / f"{model_name}.pkl"
            self.model.save(str(save_path))
            print(f"Model saved to {save_path}")
        
        return self.model
    
    def evaluate_model(self) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not trained. Call train_single_model() first.")
        
        print("Evaluating model...")
        
        X_test_pd, y_test_pd = self._to_model_inputs(self.X_test, self.y_test)
        y_pred = self.model.predict(X_test_pd)
        
        metrics = self.evaluator.evaluate(
            y_test_pd.values,
            y_pred
        )
        
        importance_df = self.model.get_feature_importance(
            method=self.config.eval.feature_importance_method,
            X=X_test_pd,
            y=y_test_pd
        )
        
        report = self.evaluator.generate_report(
            model_name=self.config.model.model_name,
            y_true=y_test_pd.values,
            y_pred=y_pred,
            importance_df=importance_df,
            save_dir=self.config.eval.results_save_path
        )
        
        return {
            "metrics": metrics,
            "feature_importance": importance_df,
            "report": report
        }
    
    def compare_models(
        self,
        model_names: Optional[List[str]] = None,
        model_params_dict: Optional[Dict[str, Dict]] = None,
        X_train_override: Optional[pl.DataFrame] = None,
        y_train_override: Optional[pl.Series] = None,
        X_eval_override: Optional[pl.DataFrame] = None,
        y_eval_override: Optional[pl.Series] = None,
    ) -> pd.DataFrame:
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_and_process_data() first.")
        
        model_names = model_names or list(AVAILABLE_MODELS.keys())
        model_params_dict = model_params_dict or {}
        
        self.comparator = ModelComparator()
        
        train_X = X_train_override if X_train_override is not None else self.X_train
        train_y = y_train_override if y_train_override is not None else self.y_train
        eval_X = X_eval_override if X_eval_override is not None else self.X_test
        eval_y = y_eval_override if y_eval_override is not None else self.y_test

        X_train_pd, y_train_pd = self._to_model_inputs(train_X, train_y)
        X_test_pd, y_test_pd = self._to_model_inputs(eval_X, eval_y)

        for model_name in model_names:
            try:
                params = model_params_dict.get(model_name, {})
                model = self.create_model(model_name, params)
                self.comparator.add_model(model_name, model)
            except Exception as e:
                print(f"Could not create model {model_name}: {e}")
        
        print("Training all models...")
        self.comparator.fit_all(X_train_pd, y_train_pd)
        
        print("Evaluating all models...")
        metrics = {
            "mse": lambda y, p: np.mean((y - p) ** 2),
            "mae": lambda y, p: np.mean(np.abs(y - p)),
            "r2": lambda y, p: 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2),
            "ic": lambda y, p: np.corrcoef(y, p)[0, 1]
        }
        
        comparison_df = self.comparator.evaluate_all(
            X_test_pd,
            y_test_pd,
            metrics
        )
        
        if self.config.eval.plot_results:
            save_path = Path(self.config.eval.results_save_path) / "model_comparison.png"
            self.evaluator.compare_models(
                self.comparator.results,
                save_path=str(save_path)
            )
        
        importance_dict = self.comparator.get_feature_importance_comparison(
            method=self.config.eval.feature_importance_method,
            X=X_test_pd,
            y=y_test_pd
        )
        
        return {
            "comparison_df": comparison_df,
            "feature_importance": importance_dict,
            "best_model": self.comparator.get_best_model(metric="mse")
        }
    
    def cross_validate_model(
        self,
        model_name: str,
        cv_folds: Optional[int] = None
    ) -> Dict[str, Any]:
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_and_process_data() first.")
        
        cv_folds = cv_folds or self.config.model.cv_folds
        
        model = self.create_model(model_name)
        
        print(f"Running {cv_folds}-fold cross-validation for {model_name}...")
        X_all, y_all = self._to_model_inputs(self.X_train, self.y_train)
        cv_results = self.evaluator.cross_validate(
            model,
            X_all,
            y_all,
            cv_folds=cv_folds
        )
        
        return cv_results
    
    def run_full_experiment(
        self,
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        print("=" * 50)
        print("Starting Full Experiment")
        print("=" * 50)
        
        data_info = self.load_and_process_data()
        print(f"Data loaded: {data_info['n_train']} train, {data_info['n_test']} test")

        fit_X, val_X, fit_y, val_y = self._split_data(self.X_train, self.y_train)
        comparison_results = self.compare_models(
            model_names,
            X_train_override=fit_X,
            y_train_override=fit_y,
            X_eval_override=val_X,
            y_eval_override=val_y,
        )
        print("\nModel Comparison Results:")
        print(comparison_results["comparison_df"])
        
        print(f"\nBest model: {comparison_results['best_model']}")

        self.model = self.create_model(comparison_results["best_model"])
        X_train_pd, y_train_pd = self._to_model_inputs(self.X_train, self.y_train)
        self.model.fit(X_train_pd, y_train_pd)
        
        eval_results = self.evaluate_model()
        print("\nEvaluation Results:")
        print(f"Metrics: {eval_results['metrics']}")

        return {
            "data_info": data_info,
            "comparison": comparison_results,
            "evaluation": eval_results
        }


def main():
    parser = argparse.ArgumentParser(description="Limit-up prediction experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model to use"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all models"
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run cross-validation"
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run rolling walk-forward evaluation on the test window"
    )
    
    args = parser.parse_args()
    
    experiment = Experiment(args.config)
    selected_model = args.model or experiment.config.model.model_name
    
    if args.compare:
        results = experiment.run_full_experiment()
    else:
        experiment.load_and_process_data()
        if args.walk_forward:
            wf_results = experiment.walk_forward_evaluate_model(selected_model)
            print("\nWalk-forward evaluation results:")
            print(wf_results["metrics"])
            print("\nDaily metrics:")
            print(wf_results["daily_metrics"])
        else:
            experiment.train_single_model(selected_model)

        if args.cv and not args.walk_forward:
            cv_results = experiment.cross_validate_model(selected_model)
            print("\nCross-validation results:")
            print(f"Mean MSE: {cv_results['mean']['mse']:.6f} (+/- {cv_results['std']['mse']:.6f})")

        if not args.walk_forward:
            eval_results = experiment.evaluate_model()
            print("\nEvaluation results:")
            print(eval_results["metrics"])


if __name__ == "__main__":
    main()
