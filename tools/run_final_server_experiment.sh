#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/home/busanbusi/experiment
PY=/home/busanbusi/.virtualenvs/experiment/bin/python
METRICS_DIR=/mnt/nvme_raid0/experiment_data/logs/metrics
CONFIG=config_morning_board_eventstudy.yaml
RUN_ID=morning_board_final_20260420
MODEL_RUN_ID=morning_board_final_20260420_models

cd "$PROJECT_DIR"
mkdir -p "$METRICS_DIR"

echo "==== final experiment start $(date -Is) ===="

MPLCONFIGDIR=/tmp/matplotlib "$PY" tools/run_event_study.py \
  --config "$CONFIG" \
  --run-id "$RUN_ID" \
  --out-dir "$METRICS_DIR"

MPLCONFIGDIR=/tmp/matplotlib "$PY" tools/run_model_suite.py \
  --config "$CONFIG" \
  --run-id "$MODEL_RUN_ID" \
  --events-path "$METRICS_DIR/${RUN_ID}_events.parquet" \
  --out-dir "$METRICS_DIR"

"$PY" tools/analyze_prediction_outputs.py \
  --metrics-dir "$METRICS_DIR" \
  --run-id "$MODEL_RUN_ID" \
  --models random_forest xgboost tree_ensemble ridge \
  --top-frac 0.1

echo "==== final experiment done $(date -Is) ===="
echo "Event summary: $METRICS_DIR/${RUN_ID}_event_study_summary.csv"
echo "Model comparison: $METRICS_DIR/${MODEL_RUN_ID}_model_comparison.csv"
echo "Portfolio check: $METRICS_DIR/${MODEL_RUN_ID}_portfolio_check.csv"
