from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import Experiment


DEFAULT_MODELS = [
    "linear",
    "lasso",
    "ridge",
    "elastic_net",
    "random_forest",
    "lightgbm",
    "xgboost",
    "linear_ensemble",
    "tree_ensemble",
]


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass
    return str(value)


def _capture_environment(out_path: Path) -> None:
    lines = [f"python={sys.version.replace(chr(10), ' ')}"]
    commands = [
        ["git", "rev-parse", "HEAD"],
        ["git", "status", "--short"],
        [sys.executable, "-m", "pip", "freeze"],
    ]
    for command in commands:
        lines.append("$ " + " ".join(command))
        try:
            lines.append(subprocess.check_output(command, text=True, stderr=subprocess.STDOUT))
        except Exception as exc:
            lines.append(f"ERROR: {exc}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _model_test_frame(experiment: Experiment, predictions) -> pd.DataFrame:
    if experiment.model_df is None or experiment.feature_names is None:
        return pd.DataFrame()

    test_df = experiment.model_df.filter(
        (pl.col("trade_date") >= experiment.config.data.test_start_date.replace("-", ""))
        & (pl.col("trade_date") <= experiment.config.data.test_end_date.replace("-", ""))
    )
    return pd.DataFrame(
        {
            "row_id": range(len(predictions)),
            "trade_date": test_df.get_column("trade_date").to_list(),
            "y_true": test_df.get_column("next_open_return").to_numpy(),
            "y_pred": predictions,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a paper-style model suite on one event dataset.")
    parser.add_argument("--config", required=True, help="Config YAML path.")
    parser.add_argument("--run-id", required=True, help="Stable run identifier.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names to run. Defaults to the full available suite.",
    )
    parser.add_argument(
        "--out-dir",
        default="/mnt/nvme_raid0/experiment_data/logs/metrics",
        help="Directory for metrics, feature importance, predictions, and environment files.",
    )
    parser.add_argument(
        "--importance-method",
        choices=["built_in", "permutation"],
        default="built_in",
        help="Feature importance method. Built-in is the default for full-suite runtime.",
    )
    parser.add_argument(
        "--events-path",
        default=None,
        help="Optional prebuilt event parquet. When set, skip event construction and train from this dataset.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _capture_environment(out_dir / f"{args.run_id}_env.txt")

    experiment = Experiment(config_path=args.config)
    event_df = pl.read_parquet(args.events_path) if args.events_path else None
    data_info = experiment.load_and_process_data(event_df=event_df)
    if data_info.get("n_train", 0) <= 0 or data_info.get("n_test", 0) <= 0:
        raise RuntimeError(f"Empty train/test sample after data processing: {data_info}")

    X_train, y_train = experiment._to_model_inputs(experiment.X_train, experiment.y_train)
    X_test, y_test = experiment._to_model_inputs(experiment.X_test, experiment.y_test)

    all_metrics = []
    all_payload: dict[str, Any] = {
        "run_id": args.run_id,
        "config": args.config,
        "models": args.models,
        "data_info": data_info,
        "results": {},
    }

    for model_name in args.models:
        print(f"=== model start: {model_name} ===", flush=True)
        try:
            model = experiment.create_model(model_name, {})
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = experiment.evaluator.evaluate(y_test.to_numpy(copy=True), y_pred)
            all_metrics.append({"model": model_name, **metrics})

            predictions_df = _model_test_frame(experiment, y_pred)
            predictions_df.to_csv(out_dir / f"{args.run_id}_{model_name}_predictions.csv", index=False)

            importance_df = None
            try:
                importance_df = model.get_feature_importance(
                    method=args.importance_method,
                    X=X_test,
                    y=y_test,
                )
                importance_df.to_csv(
                    out_dir / f"{args.run_id}_{model_name}_feature_importance.csv",
                    index=False,
                )
            except Exception as exc:
                print(f"Feature importance failed for {model_name}: {exc}", flush=True)

            all_payload["results"][model_name] = {
                "status": "ok",
                "metrics": metrics,
                "prediction_stats": {
                    "mean": float(y_pred.mean()),
                    "std": float(y_pred.std()),
                    "min": float(y_pred.min()),
                    "max": float(y_pred.max()),
                },
                "top_features": (
                    importance_df.head(20).to_dict(orient="records")
                    if importance_df is not None
                    else []
                ),
            }
        except Exception as exc:
            all_metrics.append({"model": model_name, "error": str(exc)})
            all_payload["results"][model_name] = {"status": "failed", "error": str(exc)}
            print(f"=== model failed: {model_name}: {exc} ===", flush=True)
            continue
        print(f"=== model done: {model_name} {metrics} ===", flush=True)

        pd.DataFrame(all_metrics).to_csv(out_dir / f"{args.run_id}_model_comparison.csv", index=False)
        (out_dir / f"{args.run_id}_suite_metrics.json").write_text(
            json.dumps(all_payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    comparison_df = pd.DataFrame(all_metrics)
    comparison_df.to_csv(out_dir / f"{args.run_id}_model_comparison.csv", index=False)
    (out_dir / f"{args.run_id}_suite_metrics.json").write_text(
        json.dumps(all_payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(all_payload, ensure_ascii=False, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
