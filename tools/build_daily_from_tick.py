from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl


TICK_ROOT = Path("/mnt/nvme_raid0/experiment_data/tick/stock_tick_month")
DAY_ROOT = Path("/mnt/nvme_raid0/experiment_data/day")
START_DATE = "20210223"
END_DATE = "20230529"


def market_symbol(digits: str) -> str:
    if digits.startswith(("6", "9")):
        return f"sh{digits}"
    return f"sz{digits}"


def iter_tick_files() -> Iterable[Path]:
    for year in [2021, 2022, 2023]:
        year_root = TICK_ROOT / str(year)
        if not year_root.exists():
            continue
        for file_path in year_root.glob("*/*/*.parquet"):
            trade_date = file_path.parent.name.replace("-", "")
            if START_DATE <= trade_date <= END_DATE:
                yield file_path


def summarize_file(file_path: Path) -> dict | None:
    trade_date = file_path.parent.name.replace("-", "")
    digits = file_path.stem
    try:
        df = pl.read_parquet(
            str(file_path),
            columns=["time", "current", "high", "low", "volume", "total_volume", "total_money"],
        )
    except Exception as exc:
        print(f"warn read {file_path}: {exc}", flush=True)
        return None

    if df.is_empty():
        return None
    valid = df.filter(pl.col("current").is_finite() & (pl.col("current") > 0)).sort("time")
    if valid.is_empty():
        return None

    high_expr = pl.max_horizontal("current", "high").max().alias("high")
    low_expr = pl.min_horizontal(
        pl.when(pl.col("low") > 0).then(pl.col("low")).otherwise(pl.col("current")),
        "current",
    ).min().alias("low")
    row = valid.select(
        [
            pl.col("current").first().alias("open"),
            high_expr,
            low_expr,
            pl.col("current").last().alias("close"),
            pl.col("total_volume").max().alias("vol"),
            pl.col("total_money").max().alias("amount"),
        ]
    ).row(0, named=True)
    row.update(
        {
            "symbol": market_symbol(digits),
            "trade_date": trade_date,
        }
    )
    return row


def main() -> None:
    DAY_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    out_parts: list[Path] = []
    for idx, file_path in enumerate(iter_tick_files(), start=1):
        row = summarize_file(file_path)
        if row is not None:
            rows.append(row)
        if idx % 50000 == 0:
            out_path = DAY_ROOT / f"daily_raw_tick_part_{idx:08d}.parquet"
            pl.DataFrame(rows).write_parquet(str(out_path), compression="zstd")
            out_parts.append(out_path)
            rows.clear()
            print(f"processed={idx} parts={len(out_parts)}", flush=True)

    if rows:
        out_path = DAY_ROOT / "daily_raw_tick_part_final.parquet"
        pl.DataFrame(rows).write_parquet(str(out_path), compression="zstd")
        out_parts.append(out_path)

    if not out_parts:
        raise RuntimeError("No tick daily rows generated")

    daily = (
        pl.concat([pl.read_parquet(str(path)) for path in out_parts], how="vertical_relaxed")
        .sort(["symbol", "trade_date"])
        .with_columns(
            [
                pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").cast(pl.Datetime).alias("datetime"),
                pl.col("close").shift(1).over("symbol").alias("pre_close"),
            ]
        )
        .filter(pl.col("pre_close").is_not_null())
    )
    final_path = DAY_ROOT / "day_all_a_raw_from_tick_20210223_20230529.parquet"
    daily.write_parquet(str(final_path), compression="zstd")
    print(
        f"wrote {final_path} rows={daily.height} symbols={daily['symbol'].n_unique()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
