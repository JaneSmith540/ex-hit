from __future__ import annotations

from datetime import timedelta
from typing import Callable, Dict

import numpy as np
import polars as pl


def get_paper_feature_registry() -> Dict[str, Callable]:
    return {
        "paper_breadth": _from_tick_window(_breadth),
        "paper_volume_all": _from_tick_window(_volume_all),
        "paper_signed_volume": _from_tick_window(_signed_volume_flow),
        "paper_txn_imbalance": _from_tick_window(_txn_imbalance),
        "paper_buy_ratio": _from_tick_window(_buy_ratio),
        "paper_quote_spread": _from_tick_window(_quoted_spread),
        "paper_lob_imbalance": _from_tick_window(_lob_imbalance),
        "paper_effective_spread": _from_tick_window(_effective_spread),
        "paper_book_amount_ratio": _book_amount_ratio,
        "paper_vwap_balance": _vwap_balance,
        "paper_snap_max_slope": _snap_max_slope,
        "paper_price_bias": _price_bias,
        "paper_limit_gap": _price_limit_gap,
        "paper_trade_intensity": _from_tick_window(_trade_intensity),
    }


def _from_tick_window(func: Callable[[pl.DataFrame], float]) -> Callable:
    def wrapped(tick_window: pl.DataFrame | None = None, **_: dict) -> float:
        frame = tick_window if tick_window is not None else pl.DataFrame()
        return func(_sanitize_tick_window(frame))

    return wrapped


def _sanitize_tick_window(tick_window: pl.DataFrame) -> pl.DataFrame:
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


def _signed_direction(df: pl.DataFrame) -> pl.Expr:
    return (
        pl.when(pl.col("b/s") == "b")
        .then(1.0)
        .when(pl.col("b/s") == "s")
        .then(-1.0)
        .otherwise(0.0)
    )


def _volume_all(df: pl.DataFrame) -> float:
    if df.is_empty() or "volume" not in df.columns:
        return np.nan
    return float(df.filter(_trade_mask(df)).select(pl.col("volume").sum()).item() or 0.0)


def _breadth(df: pl.DataFrame) -> float:
    if df.is_empty():
        return np.nan
    return float(df.filter(_trade_mask(df)).height)


def _trade_intensity(df: pl.DataFrame) -> float:
    if df.height < 2:
        return np.nan
    start_time = df.get_column("datetime")[0]
    end_time = df.get_column("datetime")[-1]
    span = max((end_time - start_time).total_seconds(), 1.0)
    return float(_breadth(df) / span)


def _signed_volume_flow(df: pl.DataFrame) -> float:
    if df.is_empty() or "volume" not in df.columns or "b/s" not in df.columns:
        return np.nan
    signed = df.filter(_trade_mask(df)).with_columns((_signed_direction(df) * pl.col("volume")).alias("signed_volume"))
    return float(signed.select(pl.col("signed_volume").sum()).item() or 0.0)


def _txn_imbalance(df: pl.DataFrame) -> float:
    total = _volume_all(df)
    if total is None or not np.isfinite(total):
        return np.nan
    return float(_signed_volume_flow(df) / (total + 100.0))


def _buy_ratio(df: pl.DataFrame) -> float:
    if df.is_empty() or "volume" not in df.columns or "b/s" not in df.columns:
        return np.nan
    total = _volume_all(df)
    if total <= 0:
        return np.nan
    buy_volume = float(df.filter(pl.col("b/s") == "b").select(pl.col("volume").sum()).item() or 0.0)
    return float(buy_volume / (total + 100.0))


def _sell_ratio(df: pl.DataFrame) -> float:
    if df.is_empty() or "volume" not in df.columns or "b/s" not in df.columns:
        return np.nan
    total = _volume_all(df)
    if total <= 0:
        return np.nan
    sell_volume = float(df.filter(pl.col("b/s") == "s").select(pl.col("volume").sum()).item() or 0.0)
    return float(sell_volume / (total + 100.0))


def _lambda_signal(df: pl.DataFrame) -> float:
    if df.is_empty() or "price" not in df.columns:
        return np.nan
    total = _volume_all(df)
    if total is None or not np.isfinite(total):
        return np.nan
    max_price = float(df.select(pl.col("price").max()).item())
    min_price = float(df.select(pl.col("price").min()).item())
    return float((max_price - min_price) / (total + 100.0))


def _price_range(df: pl.DataFrame) -> float:
    if df.is_empty() or "price" not in df.columns:
        return np.nan
    max_price = float(df.select(pl.col("price").max()).item())
    min_price = float(df.select(pl.col("price").min()).item())
    if min_price <= 0:
        return np.nan
    return float(np.log((max_price + 0.01) / (min_price + 0.01)))


def _quoted_spread(df: pl.DataFrame) -> float:
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


def _lob_imbalance(df: pl.DataFrame) -> float:
    required = {"a1_v", "b1_v"}
    if df.is_empty() or not required.issubset(df.columns):
        return np.nan
    return float(
        df.select(
            ((pl.col("b1_v") - pl.col("a1_v")) / (pl.col("b1_v") + pl.col("a1_v") + 100.0)).mean()
        ).item()
    )


def _effective_spread(df: pl.DataFrame) -> float:
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
            _signed_direction(trade_df).alias("signed_dir"),
        ]
    )
    numerator = weighted.select((pl.col("signed_dir") * np.log(pl.col("price") / pl.col("mid_price")) * pl.col("turnover")).sum()).item()
    denominator = weighted.select(pl.col("turnover").sum()).item()
    if not denominator:
        return np.nan
    return float(numerator / (denominator + 100.0))


def _book_arrays(event_snapshot: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ask_prices = np.asarray([float(event_snapshot.get(f"a{i}_p", 0.0) or 0.0) for i in range(1, 6)], dtype=float)
    ask_sizes = np.asarray([float(event_snapshot.get(f"a{i}_v", 0.0) or 0.0) for i in range(1, 6)], dtype=float)
    bid_prices = np.asarray([float(event_snapshot.get(f"b{i}_p", 0.0) or 0.0) for i in range(1, 6)], dtype=float)
    bid_sizes = np.asarray([float(event_snapshot.get(f"b{i}_v", 0.0) or 0.0) for i in range(1, 6)], dtype=float)
    return ask_prices, ask_sizes, bid_prices, bid_sizes


def _book_amount_ratio(event_snapshot: dict | None = None, **_: dict) -> float:
    if not event_snapshot:
        return np.nan
    ask_prices, ask_sizes, bid_prices, bid_sizes = _book_arrays(event_snapshot)
    ask_amount = float(np.sum(ask_prices * ask_sizes))
    bid_amount = float(np.sum(bid_prices * bid_sizes))
    return float(np.log((ask_amount + 100.0) / (bid_amount + 100.0)))


def _log_quote_slope(event_snapshot: dict | None = None, **_: dict) -> float:
    if not event_snapshot:
        return np.nan
    ask_prices, ask_sizes, bid_prices, bid_sizes = _book_arrays(event_snapshot)
    valid = (ask_prices > 0) & (bid_prices > 0)
    if not np.any(valid):
        return np.nan
    numerator = np.sum(np.log(ask_prices[valid] + 1e-6) - np.log(bid_prices[valid] + 1e-6))
    denominator = np.log(bid_sizes[0] + 1.0) - np.log(ask_sizes[0] + 1.0)
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _vwap_balance(event_snapshot: dict | None = None, **_: dict) -> float:
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


def _snap_amount_sum(event_snapshot: dict | None = None, **_: dict) -> float:
    if not event_snapshot:
        return np.nan
    ask_prices, ask_sizes, bid_prices, bid_sizes = _book_arrays(event_snapshot)
    return float(np.sum(ask_prices * ask_sizes) + np.sum(bid_prices * bid_sizes))


def _snap_amount_std(event_snapshot: dict | None = None, **_: dict) -> float:
    if not event_snapshot:
        return np.nan
    ask_prices, ask_sizes, bid_prices, bid_sizes = _book_arrays(event_snapshot)
    amounts = np.concatenate([ask_prices * ask_sizes, bid_prices * bid_sizes])
    if len(amounts) < 2:
        return np.nan
    return float(np.std(amounts, ddof=1))


def _snap_amount_skew(event_snapshot: dict | None = None, **_: dict) -> float:
    if not event_snapshot:
        return np.nan
    ask_prices, ask_sizes, bid_prices, bid_sizes = _book_arrays(event_snapshot)
    amounts = np.concatenate([ask_prices * ask_sizes, bid_prices * bid_sizes])
    if len(amounts) < 3:
        return np.nan
    std = np.std(amounts, ddof=1)
    if std <= 0:
        return 0.0
    centered = amounts - np.mean(amounts)
    return float(np.mean((centered / std) ** 3))


def _snap_max_slope(event_snapshot: dict | None = None, **_: dict) -> float:
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


def _price_bias(event_snapshot: dict | None = None, **_: dict) -> float:
    if not event_snapshot:
        return np.nan
    last_price = float(event_snapshot.get("current", 0.0) or 0.0)
    ask1 = float(event_snapshot.get("a1_p", 0.0) or 0.0)
    bid1 = float(event_snapshot.get("b1_p", 0.0) or 0.0)
    if last_price <= 0 or ask1 <= 0 or bid1 <= 0:
        return np.nan
    mid = (ask1 + bid1) / 2.0
    return float((last_price - mid) / (mid + 1e-9))


def _price_limit_gap(event_snapshot: dict | None = None, limit_price: float | None = None, **_: dict) -> float:
    if not event_snapshot or not limit_price:
        return np.nan
    last_price = float(event_snapshot.get("current", 0.0) or 0.0)
    if last_price <= 0:
        return np.nan
    return float((limit_price - last_price) / limit_price)
