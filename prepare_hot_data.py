from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl
import tushare as ts
import yaml


SOURCE_MIN_ROOT = Path("/mnt/hdd_data/A股_分时数据/A股_分时数据_沪深/1分钟_前复权_按年汇总")
SOURCE_TICK_ROOT = Path("/mnt/hdd_data/tick/stock_tick_month")
SOURCE_L2_ROOT = Path("/mnt/hdd_data/l2/order")
HOT_ROOT = Path("/mnt/nvme_raid0/experiment_data")
HOT_DAY_ROOT = HOT_ROOT / "day"
HOT_TICK_ROOT = HOT_ROOT / "tick" / "stock_tick_month"
HOT_L2_ROOT = HOT_ROOT / "l2" / "order"
HOT_MIN_ROOT = HOT_ROOT / "min" / "1m_hfq"
HOT_ADJ_ROOT = HOT_ROOT / "adj_factor"
LOG_PATH = Path("/home/busanbusi/experiment/hot_data_prepare.log")

START_DATE = "2021-02-23"
END_DATE = "2023-05-29"
YEARS = ("2021", "2022", "2023")


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hot_data_prepare")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def ensure_dirs() -> None:
    for path in [HOT_DAY_ROOT, HOT_TICK_ROOT, HOT_L2_ROOT, HOT_MIN_ROOT, HOT_ADJ_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def load_tushare_token() -> str:
    config_path = Path("/home/busanbusi/experiment/config.yaml")
    if not config_path.exists():
        config_path = Path.cwd() / "config.yaml"
    if not config_path.exists():
        return ""
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    return str(config.get("data", {}).get("tushare_token", "") or "")


def rsync_dir(src: Path, dst: Path, logger: logging.Logger) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    logger.info("Syncing %s -> %s", src, dst)
    subprocess.run(["rsync", "-a", f"{src.as_posix()}/", f"{dst.as_posix()}/"], check=True)


def aggregate_year_daily(year: str, logger: logging.Logger) -> pl.DataFrame:
    year_dir = SOURCE_MIN_ROOT / f"{year}_1min"
    if not year_dir.exists():
        raise FileNotFoundError(f"Missing minute source directory: {year_dir}")

    logger.info("Aggregating daily bars from %s", year_dir)
    scan = pl.scan_parquet(str(year_dir / "*.parquet"))
    daily = (
        scan.with_columns(
            [
                pl.col("时间").cast(pl.Utf8).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).alias("datetime"),
                pl.col("代码").cast(pl.Utf8).alias("symbol"),
                pl.col("开盘价").cast(pl.Float64).alias("open"),
                pl.col("收盘价").cast(pl.Float64).alias("close"),
                pl.col("最高价").cast(pl.Float64).alias("high"),
                pl.col("最低价").cast(pl.Float64).alias("low"),
                pl.col("成交量").cast(pl.Float64).alias("volume"),
                pl.col("成交额").cast(pl.Float64).alias("amount"),
            ]
        )
        .with_columns(pl.col("datetime").dt.strftime("%Y%m%d").alias("trade_date"))
        .group_by(["symbol", "trade_date"])
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("amount").sum().alias("amount"),
            ]
        )
        .sort(["symbol", "trade_date"])
        .collect(engine="streaming")
    )
    logger.info("Year %s daily rows: %s", year, daily.height)
    return daily


def build_day_dataset(logger: logging.Logger) -> pl.DataFrame:
    yearly_frames = [aggregate_year_daily(year, logger) for year in YEARS]
    combined = pl.concat(yearly_frames, how="vertical_relaxed").sort(["symbol", "trade_date"])
    combined = combined.with_columns(
        [
            pl.col("close").shift(1).over("symbol").alias("pre_close"),
            (
                (pl.col("close") - pl.col("close").shift(1).over("symbol"))
                / pl.col("close").shift(1).over("symbol")
                * 100.0
            ).alias("pct_chg"),
            (
                (pl.col("high") - pl.col("low"))
                / pl.col("close").shift(1).over("symbol")
                * 100.0
            ).alias("amplitude"),
        ]
    )
    combined = combined.with_columns(
        pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d", strict=False).cast(pl.Datetime).alias("datetime")
    )
    combined = combined.select(
        [
            "symbol",
            "trade_date",
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "volume",
            "amount",
            "pct_chg",
            "amplitude",
        ]
    )
    return combined.filter(
        (pl.col("trade_date") >= START_DATE.replace("-", ""))
        & (pl.col("trade_date") <= END_DATE.replace("-", ""))
    )


def limit_ratio_expr() -> pl.Expr:
    digits = pl.col("symbol").str.replace_all(r"[^0-9]", "")
    return (
        pl.when(digits.str.starts_with("300") | digits.str.starts_with("301") | digits.str.starts_with("688"))
        .then(pl.lit(0.20))
        .otherwise(pl.lit(0.10))
    )


def candidate_events(day_df: pl.DataFrame) -> pl.DataFrame:
    return (
        day_df.filter(
            pl.col("pre_close").is_not_null()
            & pl.col("pre_close").is_finite()
            & (pl.col("pre_close") > 0)
            & pl.col("high").is_finite()
            & (pl.col("high") >= pl.col("pre_close") * (1 + limit_ratio_expr()))
        )
        .select(["symbol", "trade_date", "datetime", "high", "pre_close"])
        .unique(subset=["symbol", "trade_date"], maintain_order=True)
    )


def copy_tick_candidates(candidate_df: pl.DataFrame, logger: logging.Logger) -> Tuple[int, int]:
    copied = 0
    missing = 0
    seen: set[Path] = set()
    for row in candidate_df.iter_rows(named=True):
        trade_date = str(row["trade_date"])
        dt = datetime.strptime(trade_date, "%Y%m%d")
        symbol_digits = "".join(ch for ch in str(row["symbol"]) if ch.isdigit())
        src = SOURCE_TICK_ROOT / dt.strftime("%Y") / dt.strftime("%m") / dt.strftime("%Y-%m-%d") / f"{symbol_digits}.parquet"
        dst = HOT_TICK_ROOT / dt.strftime("%Y") / dt.strftime("%m") / dt.strftime("%Y-%m-%d") / f"{symbol_digits}.parquet"
        if src in seen:
            continue
        seen.add(src)
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
    logger.info("Tick copy complete. copied=%s missing=%s", copied, missing)
    return copied, missing


def month_range_dir(dt: datetime) -> str:
    start = dt.replace(day=1)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return f"{start:%Y%m%d}_{end:%Y%m%d}"


def copy_l2_candidates(candidate_df: pl.DataFrame, logger: logging.Logger) -> Tuple[int, int]:
    copied = 0
    missing = 0
    seen: set[Path] = set()
    months: Dict[str, set[str]] = {}

    for row in candidate_df.iter_rows(named=True):
        trade_date = str(row["trade_date"])
        symbol_digits = "".join(ch for ch in str(row["symbol"]) if ch.isdigit())
        months.setdefault(symbol_digits, set()).add(trade_date[:6])

    for symbol_digits, ym_set in months.items():
        for ym in sorted(ym_set):
            dt = datetime.strptime(f"{ym}01", "%Y%m%d")
            src = SOURCE_L2_ROOT / dt.strftime("%Y") / month_range_dir(dt) / f"{symbol_digits}.parquet"
            dst = HOT_L2_ROOT / dt.strftime("%Y") / month_range_dir(dt) / f"{symbol_digits}.parquet"
            if src in seen:
                continue
            seen.add(src)
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
            else:
                missing += 1

    logger.info("L2 copy complete. copied=%s missing=%s", copied, missing)
    return copied, missing


def fetch_adj_factors(logger: logging.Logger) -> pl.DataFrame:
    token = load_tushare_token()
    if not token:
        logger.warning("No tushare token found, skipping adj factor fetch.")
        return pl.DataFrame(schema={"symbol": pl.Utf8, "datetime": pl.Datetime, "adj_factor": pl.Float64})

    ts.set_token(token)
    pro = ts.pro_api(token)
    trade_cal = pro.trade_cal(exchange="", start_date=START_DATE.replace("-", ""), end_date=END_DATE.replace("-", ""))
    trade_dates = trade_cal[trade_cal["is_open"] == 1]["cal_date"].tolist()
    logger.info("Fetching adj factors for %s open dates", len(trade_dates))

    frames: List[pl.DataFrame] = []
    for idx, trade_date in enumerate(trade_dates, start=1):
        if idx % 50 == 0:
            logger.info("Adj factor progress %s/%s", idx, len(trade_dates))
        try:
            df = pro.adj_factor(trade_date=trade_date)
        except Exception as exc:
            logger.warning("Adj factor fetch failed for %s: %s", trade_date, exc)
            continue
        if df is None or len(df) == 0:
            continue
        frame = pl.from_pandas(df).rename({"ts_code": "symbol"}).with_columns(
            pl.col("trade_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).cast(pl.Datetime).alias("datetime")
        )
        frames.append(frame.select(["symbol", "datetime", "adj_factor"]))

    if not frames:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "datetime": pl.Datetime, "adj_factor": pl.Float64})

    return pl.concat(frames, how="vertical_relaxed").sort(["symbol", "datetime"]).unique(
        subset=["symbol", "datetime"], keep="last", maintain_order=True
    )


def report_sizes(logger: logging.Logger) -> None:
    for path in [HOT_ROOT, HOT_DAY_ROOT, HOT_TICK_ROOT, HOT_L2_ROOT, HOT_MIN_ROOT, HOT_ADJ_ROOT]:
        if path.exists():
            proc = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True, check=True)
            count = sum(1 for item in path.rglob("*") if item.is_file())
            logger.info("SIZE %s | files=%s | %s", path, count, proc.stdout.strip())


def main() -> None:
    logger = setup_logging()
    logger.info("Hot data preparation started")
    ensure_dirs()

    for year in YEARS:
        rsync_dir(SOURCE_MIN_ROOT / f"{year}_1min", HOT_MIN_ROOT / f"{year}_1min", logger)

    day_df = build_day_dataset(logger)
    day_out = HOT_DAY_ROOT / "day_2021_2023.parquet"
    day_df.write_parquet(day_out.as_posix(), compression="zstd")
    logger.info("Wrote day dataset to %s with %s rows", day_out, day_df.height)

    candidates = candidate_events(day_df)
    cand_out = HOT_DAY_ROOT / "limitup_candidates_2021_2023.parquet"
    candidates.write_parquet(cand_out.as_posix(), compression="zstd")
    logger.info(
        "Candidate events rows=%s symbols=%s",
        candidates.height,
        candidates.get_column("symbol").n_unique() if not candidates.is_empty() else 0,
    )

    copy_tick_candidates(candidates, logger)
    copy_l2_candidates(candidates, logger)

    adj_df = fetch_adj_factors(logger)
    if not adj_df.is_empty():
        adj_out = HOT_ADJ_ROOT / "adj_factor_2021_2023.parquet"
        adj_df.write_parquet(adj_out.as_posix(), compression="zstd")
        logger.info("Wrote adj factors to %s with %s rows", adj_out, adj_df.height)

    report_sizes(logger)
    logger.info("Hot data preparation finished")


if __name__ == "__main__":
    main()
