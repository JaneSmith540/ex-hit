from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import tushare as ts

class DataProcessor:
    def __init__(self, config: Any):
        self.config = config
        self.pro = ts.pro_api(self.config.tushare_token) if self.config.tushare_token else None
        self._tick_cache: Dict[Tuple[str, str], pl.DataFrame] = {}
        self._l2_cache: Dict[str, pl.DataFrame] = {}
        self._l2_path_cache: Dict[Tuple[str, str], List[Path]] = {}
        self._l2_interval_folder_cache: Dict[Path, List[Tuple[int, int, Path]]] = {}
        self._max_l2_cache_frames = 32
        self._min_cache: Dict[str, pl.DataFrame] = {}
        self._adj_factor_maps: Optional[
            Tuple[Dict[Tuple[str, str], float], Dict[str, float]]
        ] = None

    def load_data(self, data_path: Optional[str] = None) -> pl.DataFrame:
        path = Path(data_path or self.config.data_path)

        if path.is_dir():
            print(f"Loading all parquet files from {path}...")
            files = list(path.rglob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No parquet files found in {path}")
            df = self._read_parquet_files(files)
        elif path.suffix == ".parquet":
            df = self._read_parquet_files([path])
        elif path.suffix == ".csv":
            df = pl.read_csv(path.as_posix())
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return self._standardize_market_frame(df)

    def _read_parquet_files(self, files: List[Path], columns: Optional[List[str]] = None) -> pl.DataFrame:
        frames = []
        for file_path in files:
            try:
                frame = pl.read_parquet(file_path.as_posix(), columns=columns)
                frames.append(frame)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
        if not frames:
            raise ValueError("No valid parquet files could be loaded")
        if len(frames) == 1:
            return frames[0]
        return pl.concat(frames, how="diagonal_relaxed")

    def _standardize_market_frame(self, df: pl.DataFrame) -> pl.DataFrame:
        if "ts_code" in df.columns:
            df = df.rename({"ts_code": "symbol"})
        if "代码" in df.columns and "symbol" not in df.columns:
            df = df.rename({"代码": "symbol"})

        expressions = []
        if "trade_date" in df.columns:
            expressions.append(self._parse_datetime_column("trade_date").alias("datetime"))
        elif "时间" in df.columns:
            expressions.append(pl.col("时间").cast(pl.Utf8).str.to_datetime(strict=False).alias("datetime"))

        if "close" in df.columns and "price" not in df.columns:
            expressions.append(pl.col("close").alias("price"))
        if "vol" in df.columns and "volume" not in df.columns:
            expressions.append(pl.col("vol").alias("volume"))
        if "amount" in df.columns and "turnover" not in df.columns:
            expressions.append(pl.col("amount").alias("turnover"))

        if expressions:
            df = df.with_columns(expressions)

        sort_cols = [col for col in ["datetime", "symbol"] if col in df.columns]
        if sort_cols:
            df = df.sort(sort_cols)

        return df

    def _parse_datetime_column(self, column_name: str) -> pl.Expr:
        return pl.col(column_name).cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).cast(pl.Datetime)

    def load_day_data(self) -> pl.DataFrame:
        return self.load_data(self.config.day_path)

    def _symbol_digits(self, symbol: str) -> str:
        return symbol.split(".")[0]

    def _symbol_core(self, symbol: str) -> str:
        return self._symbol_digits(symbol).strip().lower()

    def _symbol_numeric_code(self, symbol: str) -> str:
        return "".join(ch for ch in self._symbol_digits(symbol) if ch.isdigit())

    def _load_adj_factor_maps(self) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
        if self._adj_factor_maps is not None:
            return self._adj_factor_maps

        date_factors: Dict[Tuple[str, str], float] = {}
        latest_by_symbol: Dict[str, Tuple[str, float]] = {}
        factor_path = getattr(self.config, "adj_factor_path", None)
        if not factor_path:
            self._adj_factor_maps = ({}, {})
            return self._adj_factor_maps

        try:
            factors = self.load_adjustment_factors(factor_path)
        except Exception as exc:
            print(f"Warning: Failed to load adjustment factors from {factor_path}: {exc}")
            self._adj_factor_maps = ({}, {})
            return self._adj_factor_maps

        if factors.is_empty():
            self._adj_factor_maps = ({}, {})
            return self._adj_factor_maps

        for row in factors.iter_rows(named=True):
            symbol_key = self._symbol_numeric_code(str(row["symbol"]))
            if not symbol_key:
                continue
            trade_date = pd.to_datetime(row["datetime"]).strftime("%Y%m%d")
            factor = float(row["adj_factor"])
            date_factors[(symbol_key, trade_date)] = factor
            latest = latest_by_symbol.get(symbol_key)
            if latest is None or trade_date > latest[0]:
                latest_by_symbol[symbol_key] = (trade_date, factor)

        latest_factors = {
            symbol_key: factor for symbol_key, (_, factor) in latest_by_symbol.items()
        }
        self._adj_factor_maps = (date_factors, latest_factors)
        return self._adj_factor_maps

    def _adj_factor_for_date(self, symbol: str, trade_date: str) -> Optional[float]:
        date_factors, _ = self._load_adj_factor_maps()
        key = self._symbol_numeric_code(symbol)
        date_key = pd.to_datetime(str(trade_date)).strftime("%Y%m%d")
        return date_factors.get((key, date_key))

    def _latest_adj_factor_for_symbol(self, symbol: str) -> Optional[float]:
        _, latest_factors = self._load_adj_factor_maps()
        return latest_factors.get(self._symbol_numeric_code(symbol))

    def _scale_price_columns(self, df: pl.DataFrame, columns: List[str], factor: Optional[float]) -> pl.DataFrame:
        if factor is None or df.is_empty():
            return df
        expressions = [
            (pl.col(col).cast(pl.Float64) * float(factor)).alias(col)
            for col in columns
            if col in df.columns
        ]
        return df.with_columns(expressions) if expressions else df

    def _symbol_market_prefix(self, symbol: str) -> str:
        core = self._symbol_core(symbol)
        if core.startswith(("sh", "sz", "bj")):
            return core

        digits = self._symbol_numeric_code(symbol)
        if digits.startswith(("6", "900")):
            return f"sh{digits}"
        if digits.startswith(("4", "8", "92")):
            return f"bj{digits}"
        return f"sz{digits}"

    def _min_file_stems(self, symbol: str) -> List[str]:
        core = self._symbol_core(symbol)
        digits = self._symbol_numeric_code(symbol)
        stems = [core]

        if digits:
            if digits.startswith(("6", "900")):
                stems.append(f"sh{digits}")
            elif digits.startswith(("4", "8", "92")):
                stems.append(f"bj{digits}")
            else:
                stems.append(f"sz{digits}")
            stems.append(digits)

        unique_stems = []
        for stem in stems:
            if stem and stem not in unique_stems:
                unique_stems.append(stem)
        return unique_stems

    def _limit_ratio(self, symbol: str) -> float:
        digits = self._symbol_numeric_code(symbol)
        if digits.startswith(("300", "301", "688")):
            return 0.20
        return 0.10

    def compute_limit_price(self, pre_close: float, symbol: str) -> float:
        return round(float(pre_close) * (1 + self._limit_ratio(symbol)), 2)

    def _tick_file_path(self, symbol: str, trade_date: str) -> Path:
        dt = pd.to_datetime(str(trade_date))
        return (
            Path(self.config.tick_path)
            / dt.strftime("%Y")
            / dt.strftime("%m")
            / dt.strftime("%Y-%m-%d")
            / f"{self._symbol_numeric_code(symbol)}.parquet"
        )

    def _l2_interval_folders(self, year_root: Path) -> List[Tuple[int, int, Path]]:
        if year_root in self._l2_interval_folder_cache:
            return self._l2_interval_folder_cache[year_root]

        intervals: List[Tuple[int, int, Path]] = []
        if year_root.exists():
            for folder in year_root.iterdir():
                if not folder.is_dir():
                    continue
                parts = folder.name.split("_")
                if len(parts) != 2 or not all(part.isdigit() for part in parts):
                    continue
                intervals.append((int(parts[0]), int(parts[1]), folder))

        self._l2_interval_folder_cache[year_root] = intervals
        return intervals

    def _resolve_l2_file_paths(self, symbol: str, trade_date: str) -> List[Path]:
        cache_key = (self._symbol_numeric_code(symbol) or self._symbol_core(symbol), pd.to_datetime(str(trade_date)).strftime("%Y%m%d"))
        if cache_key in self._l2_path_cache:
            return self._l2_path_cache[cache_key]

        dt = pd.to_datetime(str(trade_date))
        year = dt.strftime("%Y")
        trade_key = int(dt.strftime("%Y%m%d"))
        base = Path(self.config.l2_order_path)
        year_root = base / year if (base / year).exists() else base
        if not year_root.exists():
            self._l2_path_cache[cache_key] = []
            return []

        digits = self._symbol_numeric_code(symbol)
        core = self._symbol_core(symbol)
        search_terms = [term for term in [digits, core, self._symbol_market_prefix(symbol)] if term]

        direct_matches: List[Path] = []
        interval_folders = self._l2_interval_folders(year_root)
        for start_key, end_key, folder in interval_folders:
            if not (start_key <= trade_key < end_key):
                continue
            for term in search_terms:
                direct_path = folder / f"{term}.parquet"
                if direct_path.exists():
                    direct_matches.append(direct_path)
            self._l2_path_cache[cache_key] = direct_matches
            return direct_matches
        if interval_folders:
            self._l2_path_cache[cache_key] = []
            return []

        matches: List[Path] = []
        for term in search_terms:
            matches.extend(sorted(year_root.rglob(f"*{term}*.parquet")))

        if not matches and digits:
            matches.extend(sorted(year_root.rglob(f"*.parquet")))
            matches = [path for path in matches if digits in path.stem or core in path.stem]

        interval_matches: List[Path] = []
        has_interval_folders = False
        for path in matches:
            parts = path.parent.name.split("_")
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                continue
            has_interval_folders = True
            start_key, end_key = int(parts[0]), int(parts[1])
            if start_key <= trade_key < end_key:
                interval_matches.append(path)
        if interval_matches:
            matches = interval_matches
        elif has_interval_folders:
            self._l2_path_cache[cache_key] = []
            return []

        unique_matches: List[Path] = []
        seen = set()
        for path in matches:
            if path not in seen:
                seen.add(path)
                unique_matches.append(path)
        self._l2_path_cache[cache_key] = unique_matches
        return unique_matches

    def _min_file_path(self, symbol: str, trade_date: str) -> Path:
        year = pd.to_datetime(str(trade_date)).strftime("%Y")
        return Path(self.config.min_path) / f"{year}_1min" / f"{self._symbol_market_prefix(symbol)}_{year}.parquet"

    def _resolve_min_file_path(self, symbol: str, trade_date: str) -> Optional[Path]:
        year = pd.to_datetime(str(trade_date)).strftime("%Y")
        base = Path(self.config.min_path)
        year_root = base / f"{year}_1min"
        search_roots = [year_root, base]
        candidate_stems = self._min_file_stems(symbol)

        for root in search_roots:
            if not root.exists():
                continue
            for stem in candidate_stems:
                direct = root / f"{stem}_{year}.parquet"
                if direct.exists():
                    return direct
                direct_alt = root / f"{stem}.parquet"
                if direct_alt.exists():
                    return direct_alt

            for stem in candidate_stems:
                matches = sorted(root.rglob(f"{stem}_{year}.parquet"))
                if matches:
                    return matches[0]
                matches = sorted(root.rglob(f"{stem}.parquet"))
                if matches:
                    return matches[0]

        return None

    def load_tick_data(self, symbol: str, trade_date: str) -> pl.DataFrame:
        cache_key = (symbol, str(trade_date))
        if cache_key in self._tick_cache:
            return self._tick_cache[cache_key]

        path = self._tick_file_path(symbol, str(trade_date))
        if not path.exists():
            frame = pl.DataFrame()
        else:
            try:
                frame = pl.read_parquet(str(path)).with_columns(
                    pl.col("time").cast(pl.Utf8).str.strptime(pl.Datetime, "%Y%m%d%H%M%S", strict=False).alias("datetime")
                )
            except Exception as e:
                print(f"Warning: Failed to load tick file {path}: {e}")
                frame = pl.DataFrame()
        self._tick_cache[cache_key] = frame
        return frame

    def load_l2_data(self, symbol: str, trade_date: Optional[str] = None) -> pl.DataFrame:
        date_key = pd.to_datetime(str(trade_date)).strftime("%Y%m%d") if trade_date else "all"

        if trade_date is None:
            base = Path(self.config.l2_order_path)
            file_paths = sorted(base.rglob("*.parquet")) if base.exists() else []
        else:
            file_paths = self._resolve_l2_file_paths(symbol, trade_date)

        if not file_paths:
            return pl.DataFrame()

        cache_key = "|".join(str(path) for path in file_paths)
        if cache_key in self._l2_cache:
            frame = self._l2_cache.pop(cache_key)
            self._l2_cache[cache_key] = frame
            return frame

        try:
            l2_columns = ["TradingDay", "OrderTime", "LastPrice", "Price", "Volume", "OrderType"]
            frame = self._read_parquet_files(file_paths, columns=l2_columns)
        except Exception as e:
            print(f"Warning: Failed to load L2 files for {symbol} {date_key}: {e}")
            frame = pl.DataFrame()

        if len(self._l2_cache) >= self._max_l2_cache_frames:
            self._l2_cache.pop(next(iter(self._l2_cache)))
        self._l2_cache[cache_key] = frame
        return frame

    def _l2_order_datetime(self, trading_day: int, order_time: int) -> datetime:
        time_key = f"{int(order_time):09d}"
        return datetime(
            year=int(str(int(trading_day))[:4]),
            month=int(str(int(trading_day))[4:6]),
            day=int(str(int(trading_day))[6:8]),
            hour=int(time_key[:2]),
            minute=int(time_key[2:4]),
            second=int(time_key[4:6]),
            microsecond=int(time_key[6:9]) * 1000,
        )

    def _l2_datetime_expr(self) -> pl.Expr:
        trading_day = pl.col("TradingDay").cast(pl.Int64)
        order_time = pl.col("OrderTime").cast(pl.Int64)
        return pl.datetime(
            year=(trading_day // 10000).cast(pl.Int32),
            month=((trading_day // 100) % 100).cast(pl.Int32),
            day=(trading_day % 100).cast(pl.Int32),
            hour=(order_time // 10_000_000).cast(pl.Int32),
            minute=((order_time // 100_000) % 100).cast(pl.Int32),
            second=((order_time // 1_000) % 100).cast(pl.Int32),
            microsecond=((order_time % 1_000) * 1_000).cast(pl.Int32),
            time_unit="us",
        )

    def _l2_to_market_frame(self, l2_df: pl.DataFrame, trade_date_key: int) -> pl.DataFrame:
        required = {"TradingDay", "OrderTime", "LastPrice", "Price", "Volume", "OrderType"}
        if l2_df.is_empty() or not required.issubset(set(l2_df.columns)):
            return pl.DataFrame()

        day_l2 = l2_df.filter(pl.col("TradingDay") == trade_date_key)
        if day_l2.is_empty():
            return pl.DataFrame()

        return (
            day_l2.with_columns(
                [
                    self._l2_datetime_expr().alias("datetime"),
                    (pl.col("LastPrice").cast(pl.Float64) / 100.0).alias("current"),
                    (pl.col("Price").cast(pl.Float64) / 100.0).alias("order_price"),
                    pl.col("Volume").cast(pl.Float64).alias("volume"),
                    pl.when(pl.col("OrderType") >= 0).then(pl.lit("b")).otherwise(pl.lit("s")).alias("b/s"),
                ]
            )
            .with_columns(
                [
                    pl.col("current").alias("price"),
                    pl.col("order_price").alias("b1_p"),
                    pl.col("order_price").alias("a1_p"),
                    (pl.col("current") * pl.col("volume")).alias("money"),
                    pl.col("volume").alias("b1_v"),
                    pl.col("volume").alias("a1_v"),
                ]
            )
            .sort("datetime")
        )

    def _l2_to_minute_frame(self, l2_market_df: pl.DataFrame) -> pl.DataFrame:
        if l2_market_df.is_empty() or "datetime" not in l2_market_df.columns:
            return pl.DataFrame()

        return (
            l2_market_df.sort("datetime")
            .with_columns(pl.col("datetime").dt.truncate("1m").alias("minute"))
            .group_by("minute")
            .agg(
                [
                    pl.col("price").first().alias("open"),
                    pl.col("price").max().alias("high"),
                    pl.col("price").min().alias("low"),
                    pl.col("price").last().alias("close"),
                    pl.col("volume").sum().alias("volume"),
                    pl.col("money").sum().alias("amount"),
                ]
            )
            .rename({"minute": "datetime"})
            .with_columns(pl.col("close").alias("price"))
            .sort("datetime")
        )

    def load_min_data(self, symbol: str, trade_date: str) -> pl.DataFrame:
        cache_key = f"{symbol}_{pd.to_datetime(str(trade_date)).strftime('%Y')}"
        if cache_key in self._min_cache:
            return self._min_cache[cache_key]

        path = self._resolve_min_file_path(symbol, str(trade_date))
        if path is None or not path.exists():
            frame = pl.DataFrame()
        else:
            try:
                frame = pl.read_parquet(str(path)).rename(
                    {
                        "代码": "symbol_raw",
                        "时间": "datetime",
                        "开盘价": "open",
                        "收盘价": "close",
                        "最高价": "high",
                        "最低价": "low",
                        "成交量": "volume",
                        "成交额": "amount",
                        "涨幅": "pct_chg",
                        "振幅": "amplitude",
                    }
                ).with_columns(
                    [
                        pl.col("datetime").cast(pl.Utf8).str.to_datetime(strict=False),
                        pl.lit(symbol).alias("symbol"),
                        pl.col("close").alias("price"),
                    ]
                )
            except Exception as e:
                print(f"Warning: Failed to load min file {path}: {e}")
                frame = pl.DataFrame()
        self._min_cache[cache_key] = frame
        return frame

    def load_min_data(self, symbol: str, trade_date: str) -> pl.DataFrame:
        cache_key = f"{symbol}_{pd.to_datetime(str(trade_date)).strftime('%Y')}"
        if cache_key in self._min_cache:
            return self._min_cache[cache_key]

        path = self._resolve_min_file_path(symbol, str(trade_date))
        if path is None or not path.exists():
            frame = pl.DataFrame()
        else:
            try:
                raw_frame = pl.read_parquet(str(path))
                rename_map = {
                    "代码": "symbol_raw",
                    "浠ｇ爜": "symbol_raw",
                    "时间": "datetime",
                    "鏃堕棿": "datetime",
                    "开盘价": "open",
                    "寮€鐩樹环": "open",
                    "收盘价": "close",
                    "最高价": "high",
                    "鏈€楂樹环": "high",
                    "最低价": "low",
                    "鏈€浣庝环": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                    "涨幅": "pct_chg",
                    "娑ㄥ箙": "pct_chg",
                    "振幅": "amplitude",
                    "鎸箙": "amplitude",
                }
                frame = raw_frame.rename(
                    {old: new for old, new in rename_map.items() if old in raw_frame.columns}
                ).with_columns(
                    [
                        pl.col("datetime").cast(pl.Utf8).str.to_datetime(strict=False),
                        pl.lit(symbol).alias("symbol"),
                        pl.col("close").alias("price"),
                    ]
                )
            except Exception as e:
                print(f"Warning: Failed to load min file {path}: {e}")
                frame = pl.DataFrame()
        self._min_cache[cache_key] = frame
        return frame

    def _find_first_touch_snapshot(
        self,
        tick_df: pl.DataFrame,
        limit_price: float,
        trigger_mode: str = "last_price_or_order_price",
    ) -> Optional[Dict]:
        if tick_df.is_empty():
            return None

        mode = (trigger_mode or "last_price_or_order_price").lower()
        if mode in {"last_price", "last_price_only", "trade_price", "trade_price_only"}:
            trigger_expr = pl.col("current") >= limit_price
        elif mode in {"order_price", "order_price_only", "limit_order", "limit_order_only"}:
            trigger_expr = pl.col("b1_p") >= limit_price
        elif mode in {"last_price_or_order_price", "last_or_order", "current_or_order"}:
            trigger_expr = (pl.col("current") >= limit_price) | (pl.col("b1_p") >= limit_price)
        else:
            raise ValueError(
                "Unsupported event_trigger_mode "
                f"{trigger_mode!r}; expected last_price_only, order_price_only, "
                "or last_price_or_order_price."
            )

        tick_intraday = tick_df.filter(
            (pl.col("datetime").dt.hour() >= 9)
            & ((pl.col("datetime").dt.hour() > 9) | (pl.col("datetime").dt.minute() >= 30))
            & trigger_expr
        ).head(1)

        if tick_intraday.is_empty():
            return None

        return tick_intraday.row(0, named=True)

    def _limit_ratio_expr(self) -> pl.Expr:
        digits = pl.col("symbol").cast(pl.Utf8).str.extract(r"(\d{6})", 1).fill_null("")
        return (
            pl.when(
                digits.str.starts_with("300")
                | digits.str.starts_with("301")
                | digits.str.starts_with("688")
            )
            .then(0.20)
            .otherwise(0.10)
        )

    def _add_daily_event_context(self, day_df: pl.DataFrame) -> pl.DataFrame:
        feature_cols = [
            "prior_limit_up_streak",
            "board_position",
            "market_limit_up_count",
            "market_limit_up_ratio",
            "market_close_limit_up_count",
            "market_close_limit_up_ratio",
            "market_up_count",
            "market_up_ratio",
            "market_total_count",
            "prev_market_max_close_limit_streak",
        ]
        base_df = day_df.drop([col for col in feature_cols if col in day_df.columns])
        required = {"symbol", "trade_date", "pre_close", "high"}
        close_col = "close" if "close" in base_df.columns else "price" if "price" in base_df.columns else None
        if not required.issubset(set(base_df.columns)) or close_col is None:
            return base_df.with_columns([pl.lit(np.nan).alias(col) for col in feature_cols])

        limit_price = (pl.col("pre_close").cast(pl.Float64) * (1.0 + self._limit_ratio_expr())).round(2)
        prepared = base_df.with_columns(
            [
                limit_price.alias("__limit_price"),
                (pl.col("high").cast(pl.Float64) + 1e-9 >= limit_price).alias("__is_limit_touch"),
                (pl.col(close_col).cast(pl.Float64) + 1e-9 >= limit_price).alias("__is_close_limit_up"),
                (pl.col(close_col).cast(pl.Float64) > pl.col("pre_close").cast(pl.Float64)).alias("__is_up"),
            ]
        )

        prepared = (
            prepared.with_columns(
                pl.col("__is_close_limit_up")
                .not_()
                .cast(pl.Int64)
                .cum_sum()
                .over("symbol")
                .alias("__limit_break_id")
            )
            .with_columns(
                pl.col("__is_close_limit_up")
                .cast(pl.Int64)
                .cum_sum()
                .over(["symbol", "__limit_break_id"])
                .alias("__close_limit_streak")
            )
            .with_columns(
                pl.col("__close_limit_streak")
                .shift(1)
                .over("symbol")
                .fill_null(0)
                .cast(pl.Int64)
                .alias("prior_limit_up_streak")
            )
            .with_columns(
                pl.when(pl.col("__is_limit_touch"))
                .then(pl.col("prior_limit_up_streak") + 1)
                .otherwise(0)
                .cast(pl.Int64)
                .alias("board_position")
            )
        )

        market = (
            prepared.group_by("trade_date")
            .agg(
                [
                    pl.len().alias("market_total_count"),
                    pl.col("__is_limit_touch").cast(pl.Int64).sum().alias("market_limit_up_count"),
                    pl.col("__is_close_limit_up").cast(pl.Int64).sum().alias("market_close_limit_up_count"),
                    pl.col("__is_up").cast(pl.Int64).sum().alias("market_up_count"),
                    pl.col("__close_limit_streak").max().alias("__market_max_close_limit_streak"),
                ]
            )
            .sort("trade_date")
            .with_columns(
                [
                    (pl.col("market_limit_up_count") / pl.col("market_total_count")).alias("market_limit_up_ratio"),
                    (pl.col("market_close_limit_up_count") / pl.col("market_total_count")).alias("market_close_limit_up_ratio"),
                    (pl.col("market_up_count") / pl.col("market_total_count")).alias("market_up_ratio"),
                    pl.col("__market_max_close_limit_streak")
                    .shift(1)
                    .fill_null(0)
                    .cast(pl.Int64)
                    .alias("prev_market_max_close_limit_streak"),
                ]
            )
            .drop("__market_max_close_limit_streak")
        )

        helper_cols = [
            "__limit_price",
            "__is_limit_touch",
            "__is_close_limit_up",
            "__is_up",
            "__limit_break_id",
            "__close_limit_streak",
        ]
        return prepared.join(market, on="trade_date", how="left").drop(helper_cols)

    def _add_intraday_event_context(self, event_df: pl.DataFrame) -> pl.DataFrame:
        feature_cols = ["market_prior_touch_count", "market_prior_touch_ratio"]
        required = {"trade_date", "event_time"}
        if event_df.is_empty():
            return event_df
        if not required.issubset(set(event_df.columns)):
            return event_df.with_columns([pl.lit(np.nan).alias(col) for col in feature_cols])

        event_df = event_df.drop([col for col in feature_cols if col in event_df.columns])
        sort_cols = ["trade_date", "event_time"]
        if "symbol" in event_df.columns:
            sort_cols.append("symbol")

        sorted_df = event_df.sort(sort_cols)
        event_counts = (
            sorted_df.group_by(["trade_date", "event_time"])
            .agg(pl.len().alias("__events_at_time"))
            .sort(["trade_date", "event_time"])
            .with_columns(
                (
                    pl.col("__events_at_time").cum_sum().over("trade_date")
                    - pl.col("__events_at_time")
                )
                .cast(pl.Float64)
                .alias("market_prior_touch_count")
            )
            .drop("__events_at_time")
        )

        result = sorted_df.join(event_counts, on=["trade_date", "event_time"], how="left")
        if "market_total_count" in result.columns:
            result = result.with_columns(
                pl.when(pl.col("market_total_count").cast(pl.Float64) > 0)
                .then(pl.col("market_prior_touch_count") / pl.col("market_total_count").cast(pl.Float64))
                .otherwise(np.nan)
                .alias("market_prior_touch_ratio")
            )
        else:
            result = result.with_columns(pl.lit(np.nan).alias("market_prior_touch_ratio"))

        return result

    def _missing_report_paths(self) -> Tuple[Path, Path]:
        raw_path = getattr(self.config, "missing_report_path", None)
        if raw_path:
            base_path = Path(raw_path)
        else:
            base_path = Path("results") / "missing_data_report.csv"

        if base_path.suffix:
            detail_path = base_path.with_name(f"{base_path.stem}_detail{base_path.suffix}")
            summary_path = base_path.with_name(f"{base_path.stem}_summary{base_path.suffix}")
        else:
            detail_path = base_path / "missing_data_detail.csv"
            summary_path = base_path / "missing_data_summary.csv"
        return detail_path, summary_path

    def _write_missing_data_report(self, records: List[Dict], summary: Dict) -> None:
        detail_path, summary_path = self._missing_report_paths()
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        summary_frame = pl.DataFrame([summary])
        summary_frame.write_csv(summary_path)

        detail_schema = {
            "symbol": pl.Utf8,
            "trade_date": pl.Utf8,
            "source": pl.Utf8,
            "reason": pl.Utf8,
        }
        detail_frame = pl.DataFrame(records, schema=detail_schema) if records else pl.DataFrame(schema=detail_schema)
        detail_frame.write_csv(detail_path)
        print(f"Missing data summary written to {summary_path}")
        print(f"Missing data detail written to {detail_path}")

    def _daily_limit_candidate_frame(self, day_df: pl.DataFrame) -> Tuple[pl.DataFrame, int]:
        symbol_digits = pl.col("symbol").cast(pl.Utf8).str.replace_all(r"\D", "")
        is_bj = (
            symbol_digits.str.starts_with("4")
            | symbol_digits.str.starts_with("8")
            | symbol_digits.str.starts_with("92")
        )
        trade_date_key = (
            pl.col("trade_date")
            .cast(pl.Utf8)
            .str.replace_all(r"\D", "")
            .str.slice(0, 8)
        )
        limit_price = (pl.col("pre_close").cast(pl.Float64) * (1.0 + self._limit_ratio_expr())).round(2)

        enriched = day_df.with_columns(
            [
                is_bj.alias("_is_bj"),
                trade_date_key.alias("_trade_date"),
                trade_date_key.cast(pl.Int64, strict=False).alias("_trade_date_key"),
                limit_price.alias("_limit_price"),
            ]
        )
        non_hs_skipped = enriched.filter(pl.col("_is_bj")).height
        limit_candidates = enriched.filter(
            (~pl.col("_is_bj"))
            & pl.col("_trade_date_key").is_not_null()
            & (pl.col("high").cast(pl.Float64) + 1e-9 >= pl.col("_limit_price"))
        )
        return limit_candidates, non_hs_skipped

    def build_event_dataset(self, feature_engineer) -> pl.DataFrame:
        day_df = (
            self._add_daily_event_context(self.load_day_data().sort(["symbol", "datetime"]))
            .with_columns(
                [
                    pl.col("open").shift(-1).over("symbol").alias("next_open"),
                    pl.col("trade_date").shift(-1).over("symbol").alias("next_trade_date"),
                ]
            )
        )

        end_date = pd.to_datetime(self.config.test_end_date)
        candidate_df = day_df.filter(
            (pl.col("datetime") >= pd.to_datetime(self.config.train_start_date))
            & (pl.col("datetime") <= end_date)
            & pl.col("next_open").is_not_null()
        )
        limit_candidate_df, non_hs_skipped = self._daily_limit_candidate_frame(candidate_df)

        rows: List[Dict] = []
        missing_records: List[Dict] = []
        summary: Dict[str, int | str] = {
            "train_start_date": self.config.train_start_date,
            "train_end_date": self.config.train_end_date,
            "test_start_date": self.config.test_start_date,
            "test_end_date": self.config.test_end_date,
            "event_trigger_mode": str(getattr(self.config, "event_trigger_mode", "last_price_or_order_price")),
            "candidate_rows": candidate_df.height,
            "daily_limit_candidates": limit_candidate_df.height,
            "missing_tick": 0,
            "no_tick_touch": 0,
            "missing_minute": 0,
            "missing_l2": 0,
            "empty_l2_event_day": 0,
            "l2_event_source": 0,
            "tick_event_source": 0,
            "derived_minute_from_l2": 0,
            "non_hs_skipped": non_hs_skipped,
            "touch_time_filtered": 0,
            "l2_baseline_excluded": 0,
            "events_built": 0,
        }

        def add_missing(symbol_value: str, date_value: str, source: str, reason: str) -> None:
            summary[source] = int(summary.get(source, 0)) + 1
            missing_records.append(
                {
                    "symbol": symbol_value,
                    "trade_date": date_value,
                    "source": source,
                    "reason": reason,
                }
            )

        total_candidates = limit_candidate_df.height
        progress_interval = int(getattr(self.config, "event_progress_interval", 1000) or 1000)
        for scan_index, day_row in enumerate(limit_candidate_df.iter_rows(named=True), start=1):
            if scan_index % progress_interval == 0:
                print(
                    "Event scan progress: "
                    f"{scan_index}/{total_candidates} rows, "
                    f"daily_limit_candidates={summary['daily_limit_candidates']}, "
                    f"events_built={summary['events_built']}, "
                    f"l2_baseline_excluded={summary['l2_baseline_excluded']}, "
                    f"derived_minute_from_l2={summary['derived_minute_from_l2']}",
                    flush=True,
                )

            symbol = day_row["symbol"]
            trade_date = str(day_row["_trade_date"])
            trade_date_key = int(day_row["_trade_date_key"])
            limit_price = float(day_row["_limit_price"])

            l2_df = self.load_l2_data(symbol, trade_date)
            l2_market_df = self._l2_to_market_frame(l2_df, trade_date_key)
            if l2_market_df.is_empty():
                add_missing(symbol, trade_date, "missing_l2", "excluded from L2 baseline: no event-date L2 order rows")
                summary["l2_baseline_excluded"] = int(summary["l2_baseline_excluded"]) + 1
                continue

            tick_df = l2_market_df
            summary["l2_event_source"] = int(summary["l2_event_source"]) + 1
            if tick_df.is_empty() or "datetime" not in tick_df.columns:
                add_missing(symbol, trade_date, "missing_tick", "L2-derived tick frame missing datetime")
                continue

            touch = self._find_first_touch_snapshot(
                tick_df,
                limit_price,
                trigger_mode=getattr(self.config, "event_trigger_mode", "last_price_or_order_price"),
            )
            if touch is None:
                summary["no_tick_touch"] = int(summary["no_tick_touch"]) + 1
                continue

            event_dt = touch["datetime"]
            min_touch_time = getattr(self.config, "event_min_touch_time", None)
            max_touch_time = getattr(self.config, "event_max_touch_time", None)
            if min_touch_time or max_touch_time:
                event_time = event_dt.time()
                if min_touch_time and event_time < pd.to_datetime(min_touch_time).time():
                    summary["touch_time_filtered"] = int(summary["touch_time_filtered"]) + 1
                    continue
                if max_touch_time and event_time > pd.to_datetime(max_touch_time).time():
                    summary["touch_time_filtered"] = int(summary["touch_time_filtered"]) + 1
                    continue

            window_start = event_dt - timedelta(minutes=self.config.event_window_minutes)
            tick_window = tick_df.filter(
                (pl.col("datetime") >= window_start) & (pl.col("datetime") <= event_dt)
            ).with_columns(
                [
                    pl.col("current").alias("price"),
                    pl.col("volume").cast(pl.Float64),
                ]
            )

            min_df = self.load_min_data(symbol, trade_date)
            if min_df.is_empty() or "datetime" not in min_df.columns:
                derived_min_df = self._l2_to_minute_frame(l2_market_df)
                if derived_min_df.is_empty():
                    add_missing(symbol, trade_date, "missing_minute", "minute missing and L2 could not derive minute bars")
                    min_window = pl.DataFrame()
                else:
                    summary["derived_minute_from_l2"] = int(summary["derived_minute_from_l2"]) + 1
                    min_df = derived_min_df
                    event_minute = event_dt.replace(second=0, microsecond=0)
                    min_window = min_df.filter(
                        (pl.col("datetime") >= window_start.replace(second=0, microsecond=0))
                        & (pl.col("datetime") < event_minute)
                    )
            else:
                event_minute = event_dt.replace(second=0, microsecond=0)
                min_window = min_df.filter(
                    (pl.col("datetime") >= window_start.replace(second=0, microsecond=0))
                    & (pl.col("datetime") < event_minute)
                )

            l2_window = pl.DataFrame()
            if l2_df.is_empty() or "TradingDay" not in l2_df.columns:
                if not l2_market_df.is_empty():
                    add_missing(symbol, trade_date, "missing_l2", "L2 order file missing, unreadable, or missing TradingDay")
            else:
                day_l2 = l2_df.filter(pl.col("TradingDay") == trade_date_key)
                if day_l2.is_empty():
                    add_missing(symbol, trade_date, "empty_l2_event_day", "L2 files exist but contain no rows for event date")
                event_order_time = int(event_dt.strftime("%H%M%S")) * 1000
                start_order_time = int(window_start.strftime("%H%M%S")) * 1000
                l2_window = day_l2.filter(
                    (pl.col("OrderTime") >= start_order_time) & (pl.col("OrderTime") <= event_order_time)
                )

            features = feature_engineer.calculate_event_features(
                tick_window=tick_window,
                minute_window=min_window,
                l2_window=l2_window,
                event_snapshot=touch,
                limit_price=limit_price,
                day_context={
                    "prior_limit_up_streak": day_row.get("prior_limit_up_streak"),
                    "board_position": day_row.get("board_position"),
                    "market_limit_up_count": day_row.get("market_limit_up_count"),
                    "market_limit_up_ratio": day_row.get("market_limit_up_ratio"),
                    "market_close_limit_up_count": day_row.get("market_close_limit_up_count"),
                    "market_close_limit_up_ratio": day_row.get("market_close_limit_up_ratio"),
                    "market_up_count": day_row.get("market_up_count"),
                    "market_up_ratio": day_row.get("market_up_ratio"),
                    "market_total_count": day_row.get("market_total_count"),
                    "prev_market_max_close_limit_streak": day_row.get("prev_market_max_close_limit_streak"),
                },
            )
            features.update(
                {
                    "event_id": f"{symbol}_{trade_date}_{event_dt.strftime('%H%M%S')}",
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "event_time": event_dt,
                    "buy_price": limit_price,
                    "next_open_return": (float(day_row["next_open"]) - limit_price) / limit_price,
                }
            )
            rows.append(features)
            summary["events_built"] = int(summary["events_built"]) + 1

        self._write_missing_data_report(missing_records, summary)
        return self._add_intraday_event_context(pl.DataFrame(rows)) if rows else pl.DataFrame()

    def load_adjustment_factors(
        self,
        factor_path: str | Path,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        path = Path(factor_path)
        factor_files = self._discover_adjustment_factor_files(path)
        if not factor_files:
            return self._empty_adj_factors()

        frames: List[pl.DataFrame] = []
        for file_path in factor_files:
            if file_path.suffix == ".parquet":
                frame = pl.read_parquet(str(file_path))
            elif file_path.suffix == ".csv":
                frame = pl.read_csv(file_path)
            else:
                continue

            standardized = self._standardize_adjustment_frame(frame)
            if not standardized.is_empty():
                frames.append(standardized)

        if not frames:
            return self._empty_adj_factors()

        result = pl.concat(frames, how="diagonal_relaxed")
        if symbols:
            result = result.filter(pl.col("symbol").is_in(symbols))
        if start_date:
            result = result.filter(pl.col("datetime") >= pd.to_datetime(start_date))
        if end_date:
            result = result.filter(pl.col("datetime") <= pd.to_datetime(end_date))

        return (
            result.sort(["symbol", "datetime"])
            .unique(subset=["symbol", "datetime"], keep="last", maintain_order=True)
            .select(["symbol", "datetime", "adj_factor"])
        )

    def _empty_adj_factors(self) -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "datetime": pl.Datetime,
                "adj_factor": pl.Float64,
            }
        )

    def _discover_adjustment_factor_files(self, path: Path) -> List[Path]:
        if path.is_file():
            return [path]
        if not path.exists():
            return []

        patterns = [
            "*adj*.parquet",
            "*adj*.csv",
            "*factor*.parquet",
            "*factor*.csv",
            "*复权*.parquet",
            "*复权*.csv",
            "*hfq*.parquet",
            "*hfq*.csv",
            "*qfq*.parquet",
            "*qfq*.csv",
        ]

        files: List[Path] = []
        for pattern in patterns:
            files.extend(path.rglob(pattern))

        if files:
            return sorted(set(files))

        return sorted(
            child for child in path.iterdir() if child.is_file() and child.suffix in {".parquet", ".csv"}
        )

    def _standardize_adjustment_frame(self, df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
        frame = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df
        if "adj_factor" not in frame.columns:
            return self._empty_adj_factors()

        if "ts_code" in frame.columns and "symbol" not in frame.columns:
            frame = frame.rename({"ts_code": "symbol"})

        if "trade_date" in frame.columns:
            frame = frame.with_columns(self._parse_datetime_column("trade_date").alias("datetime"))
        elif "datetime" in frame.columns:
            frame = frame.with_columns(pl.col("datetime").cast(pl.Datetime))
        else:
            return self._empty_adj_factors()

        if "symbol" not in frame.columns:
            return self._empty_adj_factors()

        return frame.select(
            [
                pl.col("symbol").cast(pl.Utf8),
                pl.col("datetime").cast(pl.Datetime),
                pl.col("adj_factor").cast(pl.Float64),
            ]
        )

    def get_adjustment_factors(self, symbols: List[str], start_date: str, end_date: str) -> pl.DataFrame:
        if self.config.adj_factor_path:
            local_factors = self.load_adjustment_factors(
                self.config.adj_factor_path,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
            )
            if not local_factors.is_empty():
                return local_factors

        if not self.config.enable_remote_adj_factor_fallback:
            return self._empty_adj_factors()

        return self.fetch_adj_factors(symbols, start_date, end_date)

    def fetch_adj_factors(self, symbols: List[str], start_date: str, end_date: str) -> pl.DataFrame:
        if not self.pro:
            print("Warning: Tushare API not initialized. Skipping adjustment factors.")
            return self._empty_adj_factors()

        print(f"Fetching adjustment factors for {len(symbols)} symbols from {start_date} to {end_date}...")

        start_t = str(start_date).replace("-", "").replace("/", "")
        end_t = str(end_date).replace("-", "").replace("/", "")

        try:
            trade_cal = self.pro.trade_cal(exchange="", start_date=start_t, end_date=end_t)
            trade_dates = trade_cal[trade_cal["is_open"] == 1]["cal_date"].tolist()
        except Exception as exc:
            print(f"Error fetching trade calendar: {exc}")
            trade_dates = []

        if not trade_dates:
            print("No trading dates found in the range.")
            return self._empty_adj_factors()

        all_factors: List[pl.DataFrame] = []
        symbol_set = set(symbols)

        import time

        for i, date in enumerate(trade_dates):
            if i % 20 == 0:
                print(f"Progress: {i}/{len(trade_dates)} dates fetched...")

            try:
                df_adj = self.pro.adj_factor(trade_date=date)
                if not df_adj.empty:
                    df_adj = df_adj[df_adj["ts_code"].isin(symbol_set)]
                    all_factors.append(self._standardize_adjustment_frame(df_adj))
                time.sleep(0.12)
            except Exception as exc:
                print(f"Error fetching factors for {date}: {exc}")
                time.sleep(1)

        if not all_factors:
            return self._empty_adj_factors()

        return pl.concat(all_factors, how="diagonal_relaxed")

    def apply_adjustment(self, df: pl.DataFrame, adj_df: pl.DataFrame) -> pl.DataFrame:
        if adj_df.is_empty():
            print("No adjustment factors to apply.")
            return df

        print("Applying adjustment factors (post-adjustment)...")
        result = df.join(adj_df, on=["symbol", "datetime"], how="left").sort(["symbol", "datetime"])
        result = result.with_columns(pl.col("adj_factor").forward_fill().over("symbol"))
        result = result.with_columns(
            pl.col("adj_factor").backward_fill().over("symbol").fill_null(1.0)
        )

        price_cols = ["open", "high", "low", "close", "pre_close", "price"]
        price_exprs = [
            (pl.col(col) * pl.col("adj_factor")).alias(col) for col in price_cols if col in result.columns
        ]
        if "vol" in result.columns:
            price_exprs.append((pl.col("vol") / pl.col("adj_factor")).alias("vol"))
        if "volume" in result.columns:
            price_exprs.append((pl.col("volume") / pl.col("adj_factor")).alias("volume"))

        if price_exprs:
            result = result.with_columns(price_exprs)

        return result

    def filter_by_date(self, df: pl.DataFrame, start_date: str, end_date: str) -> pl.DataFrame:
        if "datetime" not in df.columns:
            raise ValueError("DataFrame must have 'datetime' column")

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        return df.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))

    def identify_limit_up_events(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        prev_close_col: str = "pre_close",
        threshold: Optional[float] = None,
    ) -> pl.DataFrame:
        threshold = threshold or self.config.limit_up_threshold

        if "pct_chg" in df.columns:
            result = df.with_columns((pl.col("pct_chg") / 100.0).alias("return"))
        else:
            result = df.with_columns(
                ((pl.col(price_col) - pl.col(prev_close_col)) / pl.col(prev_close_col)).alias("return")
            )

        return (
            result.filter(pl.col("return") >= threshold)
            .unique(subset=["datetime", "symbol"], keep="first", maintain_order=True)
            .sort(["datetime", "symbol"])
        )

    def extract_pre_limit_window(
        self,
        df: pl.DataFrame,
        limit_up_events: pl.DataFrame,
        window_seconds: Optional[int] = None,
    ) -> pl.DataFrame:
        _ = window_seconds or self.config.time_window_before_limit
        frames: List[pl.DataFrame] = []
        sorted_df = df.sort(["symbol", "datetime"])

        for event in limit_up_events.iter_rows(named=True):
            event_time = event["datetime"]
            symbol = event["symbol"]
            start_time = event_time - timedelta(days=5)

            window_df = sorted_df.filter(
                (pl.col("symbol") == symbol)
                & (pl.col("datetime") >= start_time)
                & (pl.col("datetime") < event_time)
            )
            if window_df.is_empty():
                continue

            frames.append(
                window_df.with_columns(
                    [
                        pl.lit(event_time).alias("limit_up_time"),
                        pl.lit(f"{symbol}_{event_time.strftime('%Y%m%d')}").alias("event_id"),
                    ]
                )
            )

        if not frames:
            return pl.DataFrame()

        return pl.concat(frames, how="diagonal_relaxed")

    def calculate_next_day_open_return(
        self,
        df: pl.DataFrame,
        limit_up_events: pl.DataFrame,
        open_price_col: str = "open",
        prev_close_col: str = "pre_close",
    ) -> pl.DataFrame:
        _ = prev_close_col
        results = []
        sorted_df = df.sort(["symbol", "datetime"])

        for event in limit_up_events.iter_rows(named=True):
            symbol = event["symbol"]
            event_date = event["datetime"]
            future_data = sorted_df.filter(
                (pl.col("symbol") == symbol) & (pl.col("datetime") > event_date)
            )

            if future_data.is_empty():
                continue

            next_day = future_data.row(0, named=True)
            prev_close = event.get("close", event.get("price"))
            if prev_close in (None, 0):
                continue

            next_open = next_day[open_price_col]
            open_return = (next_open - prev_close) / prev_close
            results.append(
                {
                    "event_id": f"{symbol}_{event_date.strftime('%Y%m%d')}",
                    "symbol": symbol,
                    "limit_up_date": event_date,
                    "next_open_return": open_return,
                    "limit_up_price": prev_close,
                    "limit_up_time": event_date,
                }
            )

        return pl.DataFrame(results) if results else pl.DataFrame()

    def split_train_test(
        self,
        df: pl.DataFrame,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        train_start = train_start or self.config.train_start_date
        train_end = train_end or self.config.train_end_date
        test_start = test_start or self.config.test_start_date
        test_end = test_end or self.config.test_end_date

        train_df = self.filter_by_date(df, train_start, train_end)
        test_df = self.filter_by_date(df, test_start, test_end)
        return train_df, test_df

    def clean_data(
        self,
        df: pl.DataFrame,
        remove_outliers: bool = True,
        outlier_threshold: float = 3.0,
    ) -> pl.DataFrame:
        result = df.drop_nulls()
        numeric_cols = [
            col for col, dtype in result.schema.items() if dtype in pl.NUMERIC_DTYPES and col not in {"symbol", "event_id"}
        ]
        if numeric_cols:
            valid_mask = pl.all_horizontal(
                [pl.col(col).is_finite() for col in numeric_cols]
            )
            result = result.filter(valid_mask)
        if not remove_outliers or result.is_empty():
            return result

        for col in numeric_cols:
            stats = result.select(pl.col(col).mean().alias("mean"), pl.col(col).std().alias("std")).row(0, named=True)
            mean = stats["mean"]
            std = stats["std"]
            if std and std > 0:
                result = result.filter(((pl.col(col) - mean).abs() / std) <= outlier_threshold)

        return result

    def prepare_model_data(
        self,
        features_df: pl.DataFrame,
        target_df: pl.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[pl.DataFrame, pl.Series]:
        if features_df.is_empty() or target_df.is_empty():
            return pl.DataFrame(), pl.Series("next_open_return", [])

        merged = features_df.join(
            target_df.select(["event_id", "next_open_return"]),
            on="event_id",
            how="inner",
        )
        if merged.is_empty():
            return pl.DataFrame(), pl.Series("next_open_return", [])

        X = merged.select(feature_cols)
        y = merged.get_column("next_open_return")
        return X, y
