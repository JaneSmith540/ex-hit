from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import Experiment


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return str(value)


def _summary(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    values = frame["next_open_return"].dropna().astype(float)
    if values.empty:
        return {
            "group": name,
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "win_rate": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
        }
    t_stat, p_value = stats.ttest_1samp(values, 0.0, nan_policy="omit")
    return {
        "group": name,
        "n": int(values.shape[0]),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0,
        "win_rate": float((values > 0).mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def _time_bucket(event_time: Any) -> str:
    ts = pd.to_datetime(event_time)
    minutes = ts.hour * 60 + ts.minute
    if minutes < 10 * 60:
        return "09:30-10:00"
    if minutes < 10 * 60 + 30:
        return "10:00-10:30"
    if minutes < 11 * 60:
        return "10:30-11:00"
    return "11:00-11:30"


def main() -> None:
    parser = argparse.ArgumentParser(description="Unconditional event study for morning limit-up boards.")
    parser.add_argument("--config", required=True, help="Config YAML path.")
    parser.add_argument("--run-id", required=True, help="Stable run identifier.")
    parser.add_argument(
        "--out-dir",
        default="/mnt/nvme_raid0/experiment_data/logs/metrics",
        help="Output directory.",
    )
    parser.add_argument(
        "--events-path",
        default=None,
        help="Optional prebuilt event parquet. When set, skip event construction and only rebuild summaries.",
    )
    parser.add_argument(
        "--reuse-events",
        action="store_true",
        help="Reuse <out-dir>/<run-id>_events.parquet if it exists.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment = Experiment(config_path=args.config)
    events_path = Path(args.events_path) if args.events_path else out_dir / f"{args.run_id}_events.parquet"
    if args.events_path or (args.reuse_events and events_path.exists()):
        event_df = pl.read_parquet(events_path)
        print(f"Loaded existing events from {events_path}")
    else:
        event_df = experiment.data_processor.build_event_dataset(experiment.feature_engineer)
        if event_df.is_empty():
            raise RuntimeError("No morning limit-up events were built.")
        event_df = event_df.filter(pl.col("next_open_return").is_not_null())
        event_df.write_parquet(events_path)

    pdf = event_df.select(["symbol", "trade_date", "event_time", "next_open_return"]).to_pandas()
    pdf["trade_date"] = pdf["trade_date"].astype(str)
    pdf["month"] = pdf["trade_date"].str[:6]
    pdf["split"] = np.where(
        pdf["trade_date"] <= experiment.config.data.train_end_date.replace("-", ""),
        "train",
        "test",
    )
    pdf["time_bucket"] = pdf["event_time"].map(_time_bucket)

    rows = [_summary(pdf, "all")]
    rows.extend(_summary(group, f"split={name}") for name, group in pdf.groupby("split", sort=True))
    rows.extend(_summary(group, f"month={name}") for name, group in pdf.groupby("month", sort=True))
    rows.extend(_summary(group, f"time_bucket={name}") for name, group in pdf.groupby("time_bucket", sort=True))

    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / f"{args.run_id}_event_study_summary.csv"
    detail_path = out_dir / f"{args.run_id}_event_study_events.csv"
    json_path = out_dir / f"{args.run_id}_event_study_summary.json"

    summary_df.to_csv(summary_path, index=False)
    pdf.to_csv(detail_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "config": args.config,
                "n_events": int(pdf.shape[0]),
                "summary": rows,
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {events_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {detail_path}")
    print(summary_df.to_csv(index=False))


if __name__ == "__main__":
    main()
