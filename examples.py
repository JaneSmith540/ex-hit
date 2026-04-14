from main import Experiment


def example_single_model():
    experiment = Experiment("config.yaml")
    
    experiment.load_and_process_data()
    
    experiment.train_single_model("lasso")
    
    results = experiment.evaluate_model()
    print("Metrics:", results["metrics"])
    print("\nTop 10 Features:")
    print(results["feature_importance"].head(10))


def example_compare_models():
    experiment = Experiment("config.yaml")
    
    experiment.load_and_process_data()
    
    results = experiment.compare_models(
        model_names=["linear", "lasso", "ridge", "elastic_net", "random_forest", "lightgbm"]
    )
    
    print("\nModel Comparison:")
    print(results["comparison_df"])
    
    print(f"\nBest model: {results['best_model']}")


def example_custom_model_params():
    experiment = Experiment("config.yaml")
    
    experiment.load_and_process_data()
    
    custom_params = {
        "lasso": {"alpha": 0.05},
        "ridge": {"alpha": 0.5},
        "lightgbm": {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 8
        }
    }
    
    results = experiment.compare_models(
        model_names=["lasso", "ridge", "lightgbm"],
        model_params_dict=custom_params
    )
    
    print(results["comparison_df"])


def example_cross_validation():
    experiment = Experiment("config.yaml")
    
    experiment.load_and_process_data()
    
    cv_results = experiment.cross_validate_model("lasso", cv_folds=5)
    
    print("Cross-validation results:")
    for fold_idx, fold_result in enumerate(cv_results["fold_results"]):
        print(f"  Fold {fold_idx + 1}: MSE={fold_result['mse']:.6f}, IC={fold_result['ic']:.4f}")
    
    print(f"\nMean MSE: {cv_results['mean']['mse']:.6f} (+/- {cv_results['std']['mse']:.6f})")
    print(f"Mean IC: {cv_results['mean']['ic']:.4f} (+/- {cv_results['std']['ic']:.4f})")


def example_feature_importance():
    experiment = Experiment("config.yaml")
    
    experiment.load_and_process_data()
    experiment.train_single_model("lightgbm")
    
    model = experiment.model
    
    builtin_importance = model.get_feature_importance(method="built_in")
    print("Built-in Importance (Top 10):")
    print(builtin_importance.head(10))
    
    permutation_importance = model.get_feature_importance(
        method="permutation",
        X=experiment.X_test,
        y=experiment.y_test
    )
    print("\nPermutation Importance (Top 10):")
    print(permutation_importance.head(10))
    
    if hasattr(model, "get_gain_importance"):
        gain_importance = model.get_gain_importance()
        print("\nGain Importance (Top 10):")
        print(gain_importance.head(10))


def example_ensemble():
    experiment = Experiment("config.yaml")
    
    experiment.load_and_process_data()
    
    experiment.train_single_model("linear_ensemble")
    
    results = experiment.evaluate_model()
    print("Linear Ensemble Metrics:", results["metrics"])
    
    individual_importance = experiment.model.get_individual_importance()
    for model_name, importance_df in individual_importance.items():
        print(f"\n{model_name} - Top 5 Features:")
        print(importance_df.head(5))


def example_full_experiment():
    experiment = Experiment("config.yaml")
    
    results = experiment.run_full_experiment(
        model_names=["lasso", "ridge", "elastic_net", "random_forest", "lightgbm"],
    )
    
    print("=" * 50)
    print("Full Experiment Results")
    print("=" * 50)
    
    print("\nData Info:")
    print(f"  Train samples: {results['data_info']['n_train']}")
    print(f"  Test samples: {results['data_info']['n_test']}")
    print(f"  Features: {results['data_info']['n_features']}")
    
    print("\nModel Comparison:")
    print(results["comparison"]["comparison_df"])
    
    print(f"\nBest Model: {results['comparison']['best_model']}")
    
    print("\nEvaluation Metrics:")
    for metric, value in results["evaluation"]["metrics"].items():
        print(f"  {metric}: {value:.6f}")


if __name__ == "__main__":
    print("Example 1: Single Model Training")
    print("-" * 40)
    example_single_model()
    
    print("\n\nExample 2: Model Comparison")
    print("-" * 40)
    example_compare_models()
    
    print("\n\nExample 3: Cross-Validation")
    print("-" * 40)
    example_cross_validation()
