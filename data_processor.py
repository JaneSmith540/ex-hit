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
        self._min_cache: Dict[str, pl.DataFrame] = {}

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

    def _read_parquet_files(self, files: List[Path]) -> pl.DataFrame:
        frames = []
        for file_path in files:
            try:
                frame = pl.read_parquet(file_path.as_posix())
                frames.append(frame)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
        if not frames:
            raise ValueError("No valid parquet files could be loaded")
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

    def _symbol_market_prefix(self, symbol: str) -> str:
        digits = self._symbol_digits(symbol)
        if symbol.endswith(".SH") or digits.startswith("6"):
            return f"sh{digits}"
        return f"sz{digits}"

    def _limit_ratio(self, symbol: str) -> float:
        digits = self._symbol_digits(symbol)
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
            / f"{self._symbol_digits(symbol)}.parquet"
        )

    def _l2_file_path(self, symbol: str) -> Optional[Path]:
        digits = self._symbol_digits(symbol)
        base = Path(self.config.l2_order_path)
        matches = sorted(base.glob(f"*/*{digits}.parquet"))
        return matches[0] if matches else None

    def _min_file_path(self, symbol: str, trade_date: str) -> Path:
        year = pd.to_datetime(str(trade_date)).strftime("%Y")
        return Path(self.config.min_path) / f"{self._symbol_market_prefix(symbol)}_{year}.parquet"

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

    def load_l2_data(self, symbol: str) -> pl.DataFrame:
        if symbol in self._l2_cache:
            return self._l2_cache[symbol]

        path = self._l2_file_path(symbol)
        if path is None or not path.exists():
            frame = pl.DataFrame()
        else:
            try:
                frame = pl.read_parquet(str(path))
            except Exception as e:
                print(f"Warning: Failed to load L2 file {path}: {e}")
                frame = pl.DataFrame()
        self._l2_cache[symbol] = frame
        return frame

    def load_min_data(self, symbol: str, trade_date: str) -> pl.DataFrame:
        cache_key = f"{symbol}_{pd.to_datetime(str(trade_date)).strftime('%Y')}"
        if cache_key in self._min_cache:
            return self._min_cache[cache_key]

        path = self._min_file_path(symbol, str(trade_date))
        if not path.exists():
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

    def _find_first_touch_snapshot(self, tick_df: pl.DataFrame, limit_price: float) -> Optional[Dict]:
        if tick_df.is_empty():
            return None

        tick_intraday = tick_df.filter(
            (pl.col("datetime").dt.hour() >= 9)
            & ((pl.col("datetime").dt.hour() > 9) | (pl.col("datetime").dt.minute() >= 30))
            & (
                (pl.col("current") >= limit_price)
                | (pl.col("a1_p") >= limit_price)
                | (pl.col("b1_p") >= limit_price)
            )
        ).sort("datetime")

        if tick_intraday.is_empty():
            return None

        return tick_intraday.row(0, named=True)

    def build_event_dataset(self, feature_engineer) -> pl.DataFrame:
        day_df = self.load_day_data().sort(["symbol", "datetime"]).with_columns(
            [
                pl.col("open").shift(-1).over("symbol").alias("next_open"),
                pl.col("trade_date").shift(-1).over("symbol").alias("next_trade_date"),
            ]
        )

        end_date = pd.to_datetime(self.config.test_end_date)
        candidate_df = day_df.filter(
            (pl.col("datetime") >= pd.to_datetime(self.config.train_start_date))
            & (pl.col("datetime") <= end_date)
            & pl.col("next_open").is_not_null()
        )

        rows: List[Dict] = []
        for day_row in candidate_df.iter_rows(named=True):
            symbol = day_row["symbol"]
            trade_date = str(day_row["trade_date"])
            limit_price = self.compute_limit_price(day_row["pre_close"], symbol)

            # Use daily high as a cheap pre-filter, then use tick to locate the real touch time.
            if float(day_row["high"]) + 1e-9 < limit_price:
                continue

            tick_df = self.load_tick_data(symbol, trade_date)
            touch = self._find_first_touch_snapshot(tick_df, limit_price)
            if touch is None:
                continue

            event_dt = touch["datetime"]
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
            min_window = min_df.filter(
                (pl.col("datetime") >= window_start.replace(second=0, microsecond=0))
                & (pl.col("datetime") <= event_dt)
            )

            l2_df = self.load_l2_data(symbol)
            l2_window = pl.DataFrame()
            if not l2_df.is_empty():
                day_l2 = l2_df.filter(pl.col("TradingDay") == int(trade_date))
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

        return pl.DataFrame(rows) if rows else pl.DataFrame()

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
