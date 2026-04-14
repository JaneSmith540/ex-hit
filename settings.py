from pathlib import Path, PurePosixPath
from typing import Any, Dict
import re

import yaml


PRIMARY_SAMPLE_ROOT = PurePosixPath("/media/busanbusi/新加卷/数据样例")
WINDOWS_SAMPLE_ROOT_PATTERN = re.compile(r"^[A-Za-z]:[/\\]数据样例(?:(?=[/\\])|$)")

REQUIRED_CONFIG_SECTIONS = {
    "data",
    "feature",
    "model",
    "train",
    "eval",
}

REQUIRED_CONFIG_KEYS = {
    "data": {
        "data_path",
        "day_path",
        "tick_path",
        "l2_order_path",
        "min_path",
        "adj_factor_path",
        "train_start_date",
        "train_end_date",
        "test_start_date",
        "test_end_date",
        "limit_up_threshold",
        "time_window_before_limit",
        "event_window_minutes",
        "min_volume_threshold",
        "tushare_token",
        "enable_remote_adj_factor_fallback",
    },
    "feature": {
        "price_features",
        "volume_features",
        "orderbook_features",
        "flow_features",
        "technical_features",
        "feature_params",
    },
    "model": {
        "model_name",
        "model_params",
        "cv_folds",
        "random_state",
        "use_standardization",
    },
    "train": {
        "validation_ratio",
        "early_stopping_rounds",
        "metric",
        "save_model",
        "model_save_path",
        "rolling_train_days",
    },
    "eval": {
        "metrics",
        "feature_importance_method",
        "plot_results",
        "save_results",
        "results_save_path",
    },
}


def _normalize_local_data_path(raw_path: str) -> str:
    existing_path = Path(raw_path)
    if existing_path.exists():
        return str(existing_path)

    normalized = raw_path.replace("\\", "/")
    if normalized.startswith(str(PRIMARY_SAMPLE_ROOT)):
        return normalized
    if WINDOWS_SAMPLE_ROOT_PATTERN.match(normalized):
        suffix = WINDOWS_SAMPLE_ROOT_PATTERN.sub("", normalized, count=1).lstrip("/")
        return str(PurePosixPath(PRIMARY_SAMPLE_ROOT, suffix)) if suffix else str(PRIMARY_SAMPLE_ROOT)
    return raw_path


def _validate_config(config_dict: Dict[str, Any], yaml_path: str) -> None:
    missing_sections = sorted(REQUIRED_CONFIG_SECTIONS - set(config_dict))
    if missing_sections:
        raise ValueError(
            f"Config file {yaml_path} is missing sections: {', '.join(missing_sections)}"
        )

    missing_keys = []
    for section, required_keys in REQUIRED_CONFIG_KEYS.items():
        section_data = config_dict.get(section)
        if not isinstance(section_data, dict):
            raise ValueError(f"Config section '{section}' in {yaml_path} must be a mapping.")

        section_missing = sorted(required_keys - set(section_data))
        if section_missing:
            missing_keys.append(f"{section}: {', '.join(section_missing)}")

    if missing_keys:
        raise ValueError(
            f"Config file {yaml_path} is missing required keys: {'; '.join(missing_keys)}"
        )


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({key: _to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


class AttrDict(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def load_config(yaml_path: str) -> AttrDict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    if not isinstance(config_dict, dict):
        raise ValueError(f"Config file {yaml_path} must contain a top-level mapping.")

    _validate_config(config_dict, yaml_path)

    for field_name in [
        "data_path",
        "day_path",
        "tick_path",
        "l2_order_path",
        "min_path",
        "adj_factor_path",
    ]:
        raw_value = config_dict["data"].get(field_name)
        if raw_value:
            config_dict["data"][field_name] = _normalize_local_data_path(raw_value)

    return _to_namespace(config_dict)
