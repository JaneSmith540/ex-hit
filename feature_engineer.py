from typing import Any, Callable, Dict, List, Optional
import inspect

import numpy as np
import polars as pl

class FeatureEngineer:
    def __init__(
        self,
        config: Any,
        extra_feature_registry: Optional[Dict[str, Callable]] = None,
    ):
        self.config = config
        self.feature_registry: Dict[str, Callable] = {}
        self.group_feature_names: List[str] = []
        self.event_feature_names: List[str] = []
        self._register_default_features()
        if extra_feature_registry:
            for name, func in extra_feature_registry.items():
                self.register_feature(name, func, requires_event_context=True)

    def _register_default_features(self):
        self.register_feature("return_rate", self._calculate_return_rate)
        self.register_feature("return_speed", self._calculate_return_speed)
        self.register_feature("volatility", self._calculate_volatility)
        self.register_feature("price_position", self._calculate_price_position)

        self.register_feature("volume_ratio", self._calculate_volume_ratio)
        self.register_feature("volume_surge", self._calculate_volume_surge)
        self.register_feature("turnover_rate", self._calculate_turnover_rate)

        self.register_feature("bid_ask_ratio", self._calculate_bid_ask_ratio)
        self.register_feature("order_imbalance", self._calculate_order_imbalance)
        self.register_feature("depth_ratio", self._calculate_depth_ratio)

        self.register_feature("active_buy_ratio", self._calculate_active_buy_ratio)
        self.register_feature("large_order_ratio", self._calculate_large_order_ratio)
        self.register_feature("net_inflow", self._calculate_net_inflow)

        self.register_feature("ma_slope", self._calculate_ma_slope)
        self.register_feature("rsi_short", self._calculate_rsi_short)
        self.register_feature("momentum", self._calculate_momentum)

        self.register_feature("paper_breadth", _paper_breadth, requires_event_context=True)
        self.register_feature("paper_volume_all", _paper_volume_all, requires_event_context=True)
        self.register_feature("paper_signed_volume", _paper_signed_volume_flow, requires_event_context=True)
        self.register_feature("paper_txn_imbalance", _paper_txn_imbalance, requires_event_context=True)
        self.register_feature("paper_buy_ratio", _paper_buy_ratio, requires_event_context=True)
        self.register_feature("paper_quote_spread", _paper_quoted_spread, requires_event_context=True)
        self.register_feature("paper_lob_imbalance", _paper_lob_imbalance, requires_event_context=True)
        self.register_feature("paper_effective_spread", _paper_effective_spread, requires_event_context=True)
        self.register_feature("paper_book_amount_ratio", _paper_book_amount_ratio, requires_event_context=True)
        self.register_feature("paper_vwap_balance", _paper_vwap_balance, requires_event_context=True)
        self.register_feature("paper_snap_max_slope", _paper_snap_max_slope, requires_event_context=True)
        self.register_feature("paper_price_bias", _paper_price_bias, requires_event_context=True)
        self.register_feature("paper_limit_gap", _paper_price_limit_gap, requires_event_context=True)
        self.register_feature("paper_trade_intensity", _paper_trade_intensity, requires_event_context=True)

    def register_feature(self, name: str, func: Callable, requires_event_context: bool = False):
        self.feature_registry[name] = func
        self.group_feature_names = [feature_name for feature_name in self.group_feature_names if feature_name != name]
        self.event_feature_names = [feature_name for feature_name in self.event_feature_names if feature_name != name]
        if requires_event_context:
            self.event_feature_names.append(name)
        else:
            self.group_feature_names.append(name)

    def calculate_features(
        self,
        df: pl.DataFrame,
        feature_names: Optional[List[str]] = None,
        group_col: str = "event_id",
    ) -> pl.DataFrame:
        feature_names = feature_names or self._resolve_default_features()
        if df.is_empty():
            return pl.DataFrame()

        result_rows = []
        for group in df.partition_by(group_col, maintain_order=True):
            group = group.sort("datetime") if "datetime" in group.columns else group
            event_id = group.get_column(group_col)[0]
            features = {"event_id": event_id}

            for feat_name in feature_names:
                if feat_name not in self.feature_registry or feat_name in self.event_feature_names:
                    continue
                try:
                    features[feat_name] = self.feature_registry[feat_name](
                        group,
                        **self.config.feature_params,
                    )
                except Exception as exc:
                    print(f"Error calculating {feat_name} for {event_id}: {exc}")
                    features[feat_name] = np.nan

            result_rows.append(features)

        return pl.DataFrame(result_rows)

    def _resolve_default_features(self) -> List[str]:
        configured_features = (
            self.config.price_features
            + self.config.volume_features
            + self.config.orderbook_features
            + self.config.flow_features
            + self.config.technical_features
        )
        default_features = [feature_name for feature_name in configured_features if feature_name in self.group_feature_names]
        if not default_features:
            return list(self.group_feature_names)
        return list(dict.fromkeys(default_features))

    def _column_values(self, df: pl.DataFrame, column: str) -> np.ndarray:
        return df.get_column(column).cast(pl.Float64).to_numpy()

    def _sanitize_market_window(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty() or "price" not in df.columns:
            return df

        predicates = [pl.col("price").is_finite() & (pl.col("price") > 0)]
        if "volume" in df.columns:
            predicates.append(pl.col("volume").is_finite() & (pl.col("volume") >= 0))

        return df.filter(pl.all_horizontal(predicates))

    def _select_ta_source(self, minute_df: pl.DataFrame, tick_df: pl.DataFrame, min_points: int) -> pl.DataFrame:
        if not minute_df.is_empty() and minute_df.height >= min_points:
            return minute_df
        return tick_df

    def _calculate_return_rate(self, df: pl.DataFrame, **kwargs) -> float:
        prices = self._column_values(df, "price")
        if len(prices) < 2 or prices[0] == 0:
            return np.nan
        return float((prices[-1] - prices[0]) / prices[0])

    def _calculate_return_speed(self, df: pl.DataFrame, **kwargs) -> float:
        if df.height < 2:
            return np.nan

        start_time = df.get_column("datetime")[0]
        end_time = df.get_column("datetime")[-1]
        time_delta = (end_time - start_time).total_seconds()
        if time_delta <= 0:
            return np.nan

        return_rate = self._calculate_return_rate(df)
        return float(return_rate / time_delta * 60)

    def _calculate_volatility(self, df: pl.DataFrame, **kwargs) -> float:
        prices = self._column_values(df, "price")
        if len(prices) < 2:
            return np.nan

        returns = np.diff(prices) / prices[:-1]
        if len(returns) == 0:
            return np.nan
        return float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0

    def _calculate_price_position(self, df: pl.DataFrame, **kwargs) -> float:
        prices = self._column_values(df, "price")
        if len(prices) < 2:
            return np.nan

        high = float(np.max(prices))
        low = float(np.min(prices))
        current = float(prices[-1])
        if high == low:
            return 0.5
        return (current - low) / (high - low)

    def _calculate_volume_ratio(self, df: pl.DataFrame, **kwargs) -> float:
        if "volume" not in df.columns or df.height < 2:
            return np.nan

        volume = self._column_values(df, "volume")
        avg_vol = float(np.mean(volume))
        if avg_vol == 0:
            return np.nan
        return float(volume[-1] / avg_vol)

    def _calculate_volume_surge(self, df: pl.DataFrame, **kwargs) -> float:
        if "volume" not in df.columns or df.height < 5:
            return np.nan

        volume = self._column_values(df, "volume")
        recent_vol = float(np.sum(volume[-5:]))
        earlier_vol = float(np.sum(volume[:-5]))
        if earlier_vol == 0:
            return np.nan
        return recent_vol / earlier_vol

    def _calculate_turnover_rate(self, df: pl.DataFrame, **kwargs) -> float:
        if "volume" not in df.columns or "shares_outstanding" not in df.columns:
            return np.nan

        total_volume = float(np.sum(self._column_values(df, "volume")))
        shares = float(self._column_values(df, "shares_outstanding")[0])
        if shares == 0:
            return np.nan
        return total_volume / shares

    def _calculate_bid_ask_ratio(self, df: pl.DataFrame, **kwargs) -> float:
        if "bid_volume" not in df.columns or "ask_volume" not in df.columns:
            return np.nan

        bid_vol = float(self._column_values(df, "bid_volume")[-1])
        ask_vol = float(self._column_values(df, "ask_volume")[-1])
        if ask_vol == 0:
            return np.nan
        return bid_vol / ask_vol

    def _calculate_order_imbalance(self, df: pl.DataFrame, **kwargs) -> float:
        if "bid_volume" not in df.columns or "ask_volume" not in df.columns:
            return np.nan

        bid_vol = float(self._column_values(df, "bid_volume")[-1])
        ask_vol = float(self._column_values(df, "ask_volume")[-1])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def _calculate_depth_ratio(self, df: pl.DataFrame, **kwargs) -> float:
        if "bid_depth" not in df.columns or "ask_depth" not in df.columns:
            return np.nan

        bid_depth = float(self._column_values(df, "bid_depth")[-1])
        ask_depth = float(self._column_values(df, "ask_depth")[-1])
        if ask_depth == 0:
            return np.nan
        return bid_depth / ask_depth

    def _calculate_active_buy_ratio(self, df: pl.DataFrame, **kwargs) -> float:
        if "buy_volume" not in df.columns or "volume" not in df.columns:
            return np.nan

        buy_vol = float(np.sum(self._column_values(df, "buy_volume")))
        total_vol = float(np.sum(self._column_values(df, "volume")))
        if total_vol == 0:
            return np.nan
        return buy_vol / total_vol

    def _calculate_large_order_ratio(self, df: pl.DataFrame, **kwargs) -> float:
        if "volume" not in df.columns or df.height < 1:
            return np.nan

        volume = self._column_values(df, "volume")
        avg_vol = float(np.mean(volume))
        total_vol = float(np.sum(volume))
        if total_vol == 0:
            return np.nan
        large_orders = float(np.sum(volume[volume > avg_vol * 2]))
        return large_orders / total_vol

    def _calculate_net_inflow(self, df: pl.DataFrame, **kwargs) -> float:
        if "buy_amount" not in df.columns or "sell_amount" not in df.columns:
            return np.nan
        return float(np.sum(self._column_values(df, "buy_amount")) - np.sum(self._column_values(df, "sell_amount")))

    def _calculate_ma_slope(self, df: pl.DataFrame, ma_windows: List[int] = None, **kwargs) -> float:
        ma_windows = ma_windows or [5]
        window = ma_windows[0]
        prices = self._column_values(df, "price")
        if len(prices) < window + 1:
            return np.nan

        ma_prev = np.mean(prices[-window - 1 : -1])
        ma_last = np.mean(prices[-window:])
        return float(ma_last - ma_prev)

    def _calculate_rsi_short(self, df: pl.DataFrame, rsi_period: int = 14, **kwargs) -> float:
        prices = self._column_values(df, "price")
        if len(prices) < rsi_period + 1:
            return np.nan

        delta = np.diff(prices)
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        avg_gain = np.mean(gains[-rsi_period:])
        avg_loss = np.mean(losses[-rsi_period:])
        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def _calculate_momentum(self, df: pl.DataFrame, **kwargs) -> float:
        prices = self._column_values(df, "price")
        if len(prices) < 10:
            return np.nan
        return float(prices[-1] - prices[-10])

    def calculate_event_features(
        self,
        tick_window: pl.DataFrame,
        minute_window: pl.DataFrame,
        l2_window: pl.DataFrame,
        event_snapshot: Dict,
        limit_price: float,
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}

        tick_features = tick_window.select(
            [
                pl.col("datetime"),
                pl.col("price"),
                pl.col("volume").cast(pl.Float64),
            ]
        ) if not tick_window.is_empty() else pl.DataFrame()
        tick_features = self._sanitize_market_window(tick_features)

        minute_features = minute_window.select(
            [
                pl.col("datetime"),
                pl.col("price"),
                pl.col("volume").cast(pl.Float64),
            ]
        ) if not minute_window.is_empty() else pl.DataFrame()
        minute_features = self._sanitize_market_window(minute_features)

        price_feature_names = ["return_rate", "return_speed", "volatility", "price_position", "volume_ratio", "volume_surge"]
        for name in price_feature_names:
            features[name] = self.feature_registry[name](tick_features, **self.config.feature_params) if not tick_features.is_empty() else np.nan

        features["turnover_rate"] = np.nan

        bid_volume = float(event_snapshot.get("b1_v", 0.0) or 0.0)
        ask_volume = float(event_snapshot.get("a1_v", 0.0) or 0.0)
        depth_df = pl.DataFrame(
            {
                "bid_volume": [bid_volume],
                "ask_volume": [ask_volume],
                "bid_depth": [bid_volume],
                "ask_depth": [ask_volume],
            }
        )
        features["bid_ask_ratio"] = self._calculate_bid_ask_ratio(depth_df)
        features["order_imbalance"] = self._calculate_order_imbalance(depth_df)
        features["depth_ratio"] = self._calculate_depth_ratio(depth_df)

        if not tick_window.is_empty():
            signed = tick_window.with_columns(
                [
                    pl.when(pl.col("b/s") == "b").then(pl.col("volume")).otherwise(0.0).alias("buy_volume"),
                    pl.when(pl.col("b/s") == "b").then(pl.col("money")).otherwise(0.0).alias("buy_amount"),
                    pl.when(pl.col("b/s") == "s").then(pl.col("money")).otherwise(0.0).alias("sell_amount"),
                ]
            )
            signed_features = signed.select(["buy_volume", "volume", "buy_amount", "sell_amount"])
            features["active_buy_ratio"] = self._calculate_active_buy_ratio(signed_features)
            features["net_inflow"] = self._calculate_net_inflow(signed_features)
        else:
            features["active_buy_ratio"] = np.nan
            features["net_inflow"] = np.nan

        if not l2_window.is_empty():
            l2_features = l2_window.rename({"Volume": "volume"})
            features["large_order_ratio"] = self._calculate_large_order_ratio(l2_features)
            features["limit_order_count"] = float(l2_window.filter(pl.col("Price") == int(round(limit_price * 100))).height)
            features["limit_order_volume"] = float(
                l2_window.filter(pl.col("Price") == int(round(limit_price * 100))).select(pl.col("Volume").sum()).item() or 0.0
            )
        else:
            features["large_order_ratio"] = np.nan
            features["limit_order_count"] = 0.0
            features["limit_order_volume"] = 0.0

        ma_window = (self.config.feature_params.get("ma_windows") or [5])[0]
        ma_source = self._select_ta_source(minute_features, tick_features, ma_window + 1)
        rsi_source = self._select_ta_source(
            minute_features,
            tick_features,
            self.config.feature_params.get("rsi_period", 14) + 1,
        )
        momentum_source = self._select_ta_source(minute_features, tick_features, 10)

        features["ma_slope"] = self._calculate_ma_slope(ma_source, **self.config.feature_params) if not ma_source.is_empty() else np.nan
        features["rsi_short"] = self._calculate_rsi_short(rsi_source, **self.config.feature_params) if not rsi_source.is_empty() else np.nan
        features["momentum"] = self._calculate_momentum(momentum_source, **self.config.feature_params) if not momentum_source.is_empty() else np.nan

        extra_context = {
            "tick_window": tick_window,
            "minute_window": minute_window,
            "l2_window": l2_window,
            "event_snapshot": event_snapshot,
            "limit_price": limit_price,
            "feature_config": self.config,
        }
        for name in self.event_feature_names:
            func = self.feature_registry.get(name)
            if func is None:
                continue
            try:
                parameters = inspect.signature(func).parameters
                if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
                    features[name] = func(**extra_context)
                else:
                    kwargs = {key: value for key, value in extra_context.items() if key in parameters}
                    features[name] = func(**kwargs)
            except Exception as exc:
                print(f"Error calculating event feature {name}: {exc}")
                features[name] = np.nan

        return features

    def get_feature_names(self) -> List[str]:
        return list(self.feature_registry.keys()) + ["limit_order_count", "limit_order_volume"]

    def aggregate_by_event(
        self,
        df: pl.DataFrame,
        agg_funcs: Optional[Dict[str, List[str]]] = None,
        group_col: str = "event_id",
    ) -> pl.DataFrame:
        if agg_funcs is None:
            numeric_cols = [
                col for col, dtype in df.schema.items() if dtype in pl.NUMERIC_DTYPES and col != group_col
            ]
            agg_exprs = []
            for col in numeric_cols:
                agg_exprs.extend(
                    [
                        pl.col(col).mean().alias(f"{col}_mean"),
                        pl.col(col).std().alias(f"{col}_std"),
                        pl.col(col).max().alias(f"{col}_max"),
                        pl.col(col).min().alias(f"{col}_min"),
                    ]
                )
            return df.group_by(group_col).agg(agg_exprs)

        agg_exprs = []
        for col, funcs in agg_funcs.items():
            for func in funcs:
                agg_exprs.append(getattr(pl.col(col), func)().alias(f"{col}_{func}"))
        return df.group_by(group_col).agg(agg_exprs)


def _sanitize_event_tick_window(tick_window: Optional[pl.DataFrame]) -> pl.DataFrame:
    if tick_window is None:
        return pl.DataFrame()
    if tick_window.is_empty():
        return tick_window

    columns = tick_window.columns
    predicates = [pl.col("datetime").is_not_null()]
    if "price" in columns:
        predicates.append(pl.col("price").is_finite() & (pl.col("price") > 0))
    if "volume" in columns:
        predicates.append(pl.col("volume").is_finite() & (pl.col("volume") >= 0))

    return tick_window.filter(pl.all_horizontal(predicates))


def _trade_mask(df: pl.DataFrame) -> pl.Expr:
    if "b/s" not in df.columns:
        return pl.lit(True)
    return pl.col("b/s").is_in(["b", "s"])


def _signed_direction() -> pl.Expr:
    return (
        pl.when(pl.col("b/s") == "b")
        .then(1.0)
        .when(pl.col("b/s") == "s")
        .then(-1.0)
        .otherwise(0.0)
    )


def _paper_volume_all(tick_window: Optional[pl.DataFrame] = None, **_: Dict[str, Any]) -> float:
    df = _sanitize_event_tick_window(tick_window)
    if df.is_empty() or "volume" not in df.columns:
        return np.nan
    return float(df.filter(_trade_mask(df)).select(pl.col("volume").sum()).item() or 0.0)


def _paper_breadth(tick_window: Optional[pl.DataFrame] = None, **_: Dict[str, Any]) -> float:
    df = _sanitize_event_tick_window(tick_window)
    if df.is_empty():
        return np.nan
    return float(df.filter(_trade_mask(df)).height)


def _paper_trade_intensity(tick_window: Optional[pl.DataFrame] = None, **_: Dict[str, Any]) -> float:
    df = _sanitize_event_tick_window(tick_window)
    if df.height < 2:
        return np.nan
    start_time = df.get_column("datetime")[0]
    end_time = df.get_column("datetime")[-1]
    span = max((end_time - start_time).total_seconds(), 1.0)
    return float(_paper_breadth(df) / span)


def _paper_signed_volume_flow(tick_window: Optional[pl.DataFrame] = None, **_: Dict[str, Any]) -> float:
    df = _sanitize_event_tick_window(tick_window)
    if df.is_empty() or "volume" not in df.columns or "b/s" not in df.columns:
        return np.nan
    signed = df.filter(_trade_mask(df)).with_columns((_signed_direction() * pl.col("volume")).alias("signed_volume"))
    return float(signed.select(pl.col("signed_volume").sum()).item() or 0.0)


def _paper_txn_imbalance(tick_window: Optional[pl.DataFrame] = None, **_: Dict[str, Any]) -> float:
    total = _paper_volume_all(tick_window)
    if total is None or not np.isfinite(total):
        return np.nan
    return float(_paper_signed_volume_flow(tick_window) / (total + 100.0))


def _paper_buy_ratio(tick_window: Optional[pl.DataFrame] = None, **_: Dict[str, Any]) -> float:
    df = _sanitize_event_tick_window(tick_window)
    if df.is_empty() or "volume" not in df.columns or "b/s" not in df.columns:
        return np.nan
    total = _paper_volume_all(df)
    if total <= 0:
        return np.nan
    buy_volume = float(df.filter(pl.col("b/s") == "b").select(pl.col("volume").sum()).item() or 0.0)
    return float(buy_volume / (total + 100.0))


def _paper_quoted_spread(tick_window: Optional[pl.DataFrame] = None, **_: Dict[str, Any]) -> float:
    df = _sanitize_event_tick_window(tick_window)
    required = {"a1_p", "b1_p"}
    if df.is_empty() or not required.issubset(df.columns):
        return np.nan
    spread_df = df.filter(
        pl.col("a1_p").is_finite()
        & pl.col("b1_p").is_finite()
        & (pl.col("a1_p") > 0)
        & (pl.col("b1_p") > 0)
    )
    if spread_df.is_empty():
        return np.nan
    return float(
        spread_df.select(
            ((pl.col("a1_p") - pl.col("b1_p")) / (((pl.col("a1_p") + pl.col("b1_p")) / 2.0) + 1e-9)).mean()
        ).item()
    )


def _paper_lob_imbalance(tick_window: Optional[pl.DataFrame] = None, **_: Dict[str, Any]) -> float:
    df = _sanitize_event_tick_window(tick_window)
    required = {"a1_v", "b1_v"}
    if df.is_empty() or not required.issubset(df.columns):
        return np.nan
    return float(
        df.select(
            ((pl.col("b1_v") - pl.col("a1_v")) / (pl.col("b1_v") + pl.col("a1_v") + 100.0)).mean()
        ).item()
    )


def _paper_effective_spread(tick_window: Optional[pl.DataFrame] = None, **_: Dict[str, Any]) -> float:
    df = _sanitize_event_tick_window(tick_window)
    required = {"price", "a1_p", "b1_p", "volume", "b/s"}
    if df.is_empty() or not required.issubset(df.columns):
        return np.nan

    trade_df = df.filter(
        _trade_mask(df)
        & pl.col("price").is_finite()
        & (pl.col("price") > 0)
        & pl.col("a1_p").is_finite()
        & pl.col("b1_p").is_finite()
        & (pl.col("a1_p") > 0)
        & (pl.col("b1_p") > 0)
    )
    if trade_df.is_empty():
        return np.nan

    weighted = trade_df.with_columns(
        [
            (((pl.col("a1_p") + pl.col("b1_p")) / 2.0) + 1e-9).alias("mid_price"),
            (pl.col("price") * pl.col("volume")).alias("turnover"),
            _signed_direction().alias("signed_dir"),
        ]
    )
    numerator = weighted.select(
        (pl.col("signed_dir") * np.log(pl.col("price") / pl.col("mid_price")) * pl.col("turnover")).sum()
    ).item()
    denominator = weighted.select(pl.col("turnover").sum()).item()
    if not denominator:
        return np.nan
    return float(numerator / (denominator + 100.0))


def _book_arrays(event_snapshot: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ask_prices = np.asarray([float(event_snapshot.get(f"a{i}_p", 0.0) or 0.0) for i in range(1, 6)], dtype=float)
    ask_sizes = np.asarray([float(event_snapshot.get(f"a{i}_v", 0.0) or 0.0) for i in range(1, 6)], dtype=float)
    bid_prices = np.asarray([float(event_snapshot.get(f"b{i}_p", 0.0) or 0.0) for i in range(1, 6)], dtype=float)
    bid_sizes = np.asarray([float(event_snapshot.get(f"b{i}_v", 0.0) or 0.0) for i in range(1, 6)], dtype=float)
    return ask_prices, ask_sizes, bid_prices, bid_sizes


def _paper_book_amount_ratio(event_snapshot: Optional[Dict[str, Any]] = None, **_: Dict[str, Any]) -> float:
    if not event_snapshot:
        return np.nan
    ask_prices, ask_sizes, bid_prices, bid_sizes = _book_arrays(event_snapshot)
    ask_amount = float(np.sum(ask_prices * ask_sizes))
    bid_amount = float(np.sum(bid_prices * bid_sizes))
    return float(np.log((ask_amount + 100.0) / (bid_amount + 100.0)))


def _paper_vwap_balance(event_snapshot: Optional[Dict[str, Any]] = None, **_: Dict[str, Any]) -> float:
    if not event_snapshot:
        return np.nan
    ask_prices, ask_sizes, bid_prices, bid_sizes = _book_arrays(event_snapshot)
    ask_denom = float(np.sum(ask_sizes))
    bid_denom = float(np.sum(bid_sizes))
    if ask_denom <= 0 or bid_denom <= 0:
        return np.nan
    ask_vwap = float(np.sum(ask_prices * ask_sizes) / ask_denom)
    bid_vwap = float(np.sum(bid_prices * bid_sizes) / bid_denom)
    return float((bid_vwap - ask_vwap) / (((bid_vwap + ask_vwap) / 2.0) + 1e-9))


def _paper_snap_max_slope(event_snapshot: Optional[Dict[str, Any]] = None, **_: Dict[str, Any]) -> float:
    if not event_snapshot:
        return np.nan
    ask_prices, _, bid_prices, _ = _book_arrays(event_snapshot)
    ask_slopes = []
    bid_slopes = []
    for i in range(4):
        if ask_prices[i] > 0 and ask_prices[i + 1] > 0:
            ask_slopes.append((ask_prices[i + 1] - ask_prices[i]) / ask_prices[i])
        if bid_prices[i] > 0 and bid_prices[i + 1] > 0:
            bid_slopes.append((bid_prices[i] - bid_prices[i + 1]) / bid_prices[i])
    slopes = ask_slopes + bid_slopes
    return float(max(slopes)) if slopes else np.nan


def _paper_price_bias(event_snapshot: Optional[Dict[str, Any]] = None, **_: Dict[str, Any]) -> float:
    if not event_snapshot:
        return np.nan
    last_price = float(event_snapshot.get("current", 0.0) or 0.0)
    ask1 = float(event_snapshot.get("a1_p", 0.0) or 0.0)
    bid1 = float(event_snapshot.get("b1_p", 0.0) or 0.0)
    if last_price <= 0 or ask1 <= 0 or bid1 <= 0:
        return np.nan
    mid = (ask1 + bid1) / 2.0
    return float((last_price - mid) / (mid + 1e-9))


def _paper_price_limit_gap(
    event_snapshot: Optional[Dict[str, Any]] = None,
    limit_price: Optional[float] = None,
    **_: Dict[str, Any],
) -> float:
    if not event_snapshot or not limit_price:
        return np.nan
    last_price = float(event_snapshot.get("current", 0.0) or 0.0)
    if last_price <= 0:
        return np.nan
    return float((limit_price - last_price) / limit_price)
