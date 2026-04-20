from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from main import Experiment


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one experiment and save reproducible outputs.")
    parser.add_argument("--config", required=True, help="Config YAML path.")
    parser.add_argument("--model", required=True, help="Model name.")
    parser.add_argument("--run-id", required=True, help="Stable run identifier.")
    parser.add_argument(
        "--out-dir",
        default="/mnt/nvme_raid0/experiment_data/runs/metrics",
        help="Directory for metrics, feature importance, predictions, and environment files.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _capture_environment(out_dir / f"{args.run_id}_env.txt")

    experiment = Experiment(config_path=args.config)
    data_info = experiment.load_and_process_data()
    experiment.train_single_model(args.model)
    eval_results = experiment.evaluate_model()

    metrics_payload = {
        "run_id": args.run_id,
        "config": args.config,
        "model": args.model,
        "data_info": data_info,
        "metrics": eval_results.get("metrics", {}),
        "report": eval_results.get("report", {}),
    }
    (out_dir / f"{args.run_id}_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    feature_importance = eval_results.get("feature_importance")
    if feature_importance is not None:
        feature_importance.to_csv(out_dir / f"{args.run_id}_feature_importance.csv", index=False)

    if experiment.X_test is not None and experiment.y_test is not None and experiment.model is not None:
        X_test_pd, y_test_pd = experiment._to_model_inputs(experiment.X_test, experiment.y_test)
        predictions = experiment.model.predict(X_test_pd)
        pd.DataFrame(
            {
                "row_id": range(len(predictions)),
                "y_true": y_test_pd.to_numpy(copy=True),
                "y_pred": predictions,
            }
        ).to_csv(out_dir / f"{args.run_id}_predictions.csv", index=False)

    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
