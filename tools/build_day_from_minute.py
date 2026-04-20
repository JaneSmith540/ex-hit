from __future__ import annotations

from pathlib import Path

import polars as pl


MIN_ROOT = Path("/mnt/nvme_raid0/experiment_data/min/1m_hfq")
DAY_OUT = Path("/mnt/nvme_raid0/experiment_data/day/day_2021_2023.parquet")
START_DATE = "2021-02-23"
END_DATE = "2023-05-29"


def build_year(year: int) -> pl.DataFrame:
    files = sorted((MIN_ROOT / f"{year}_1min").glob("*.parquet"))
    if not files:
        return pl.DataFrame()

    lazy_frames = []
    for file_path in files:
        lazy_frames.append(
            pl.scan_parquet(str(file_path))
            .rename(
                {
                    "时间": "datetime",
                    "代码": "symbol",
                    "开盘价": "open",
                    "收盘价": "close",
                    "最高价": "high",
                    "最低价": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                },
                strict=False,
            )
            .with_columns(
                [
                    pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
                    pl.col("symbol").cast(pl.Utf8),
                ]
            )
            .with_columns(pl.col("datetime").dt.strftime("%Y%m%d").alias("trade_date"))
            .filter(
                (pl.col("datetime").dt.date() >= pl.date(year if year > 2021 else 2021, 1, 1))
                & (pl.col("datetime").dt.strftime("%Y-%m-%d") >= START_DATE)
                & (pl.col("datetime").dt.strftime("%Y-%m-%d") <= END_DATE)
            )
            .select(["symbol", "trade_date", "datetime", "open", "high", "low", "close", "volume", "amount"])
            .group_by(["symbol", "trade_date"])
            .agg(
                [
                    pl.col("datetime").min().alias("datetime"),
                    pl.col("open").first().alias("open"),
                    pl.col("high").max().alias("high"),
                    pl.col("low").min().alias("low"),
                    pl.col("close").last().alias("close"),
                    pl.col("volume").sum().alias("vol"),
                    pl.col("amount").sum().alias("amount"),
                ]
            )
            .sort(["symbol", "trade_date"])
            .with_columns(pl.col("close").shift(1).over("symbol").alias("pre_close"))
        )

    return pl.concat(lazy_frames).collect(streaming=True)


def main() -> None:
    DAY_OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in [2021, 2022, 2023]:
        print(f"building {year}", flush=True)
        frame = build_year(year)
        print(f"{year}: rows={frame.height}", flush=True)
        if not frame.is_empty():
            frames.append(frame)

    if not frames:
        raise RuntimeError(f"No minute data found under {MIN_ROOT}")

    result = (
        pl.concat(frames, how="vertical_relaxed")
        .sort(["symbol", "trade_date"])
        .with_columns(pl.col("close").shift(1).over("symbol").alias("pre_close"))
        .filter(pl.col("pre_close").is_not_null())
    )
    result.write_parquet(str(DAY_OUT), compression="zstd")
    print(f"wrote {DAY_OUT} rows={result.height} symbols={result['symbol'].n_unique()}", flush=True)


if __name__ == "__main__":
    main()
