from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import polars as pl
import tushare as ts


TOKEN = os.environ["TUSHARE_TOKEN"]
OUT_DIR = Path("D:/experiment/generated_hfq")
INDEX_CODE = "000300.SH"
UNIVERSE_START = "20210101"
SAMPLE_START = "20210223"
DAILY_START = "20210222"
SAMPLE_END = "20230529"
DAILY_END = "20230530"
FACTOR_END = "20251231"


def to_symbol(ts_code: str) -> str:
    code, exchange = ts_code.split(".")
    if exchange == "SH":
        return f"sh{code}"
    if exchange == "SZ":
        return f"sz{code}"
    if exchange == "BJ":
        return f"bj{code}"
    return code


def fetch_with_retry(func, **kwargs) -> pd.DataFrame:
    for attempt in range(5):
        try:
            return func(**kwargs)
        except Exception as exc:
            if attempt == 4:
                raise
            print(f"retry {attempt + 1} {func.__name__} {kwargs}: {exc}", flush=True)
            time.sleep(2 + attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts.set_token(TOKEN)
    pro = ts.pro_api(TOKEN)

    print("fetch index weights", flush=True)
    weights = fetch_with_retry(
        pro.index_weight,
        index_code=INDEX_CODE,
        start_date=UNIVERSE_START,
        end_date=SAMPLE_END,
    )
    if weights.empty:
        raise RuntimeError("index_weight returned empty")
    universe = sorted(weights["con_code"].dropna().unique().tolist())
    pd.DataFrame({"ts_code": universe, "symbol": [to_symbol(code) for code in universe]}).to_parquet(
        OUT_DIR / "hs300_universe_20210101_20230529.parquet",
        index=False,
    )
    print(f"universe={len(universe)}", flush=True)

    daily_frames: list[pd.DataFrame] = []
    adj_frames: list[pd.DataFrame] = []
    for idx, ts_code in enumerate(universe, start=1):
        print(f"{idx}/{len(universe)} {ts_code}", flush=True)
        daily = fetch_with_retry(
            pro.daily,
            ts_code=ts_code,
            start_date=DAILY_START,
            end_date=DAILY_END,
        )
        adj = fetch_with_retry(
            pro.adj_factor,
            ts_code=ts_code,
            start_date=DAILY_START,
            end_date=FACTOR_END,
        )
        if not daily.empty:
            daily_frames.append(daily)
        if not adj.empty:
            adj_frames.append(adj)
        time.sleep(0.15)

    if not daily_frames:
        raise RuntimeError("No daily data fetched")
    if not adj_frames:
        raise RuntimeError("No adjustment factors fetched")

    daily_df = pd.concat(daily_frames, ignore_index=True)
    adj_df = pd.concat(adj_frames, ignore_index=True)
    adj_df.to_parquet(OUT_DIR / "adj_factor_hs300_20210222_20251231.parquet", index=False)

    merged = daily_df.merge(adj_df, on=["ts_code", "trade_date"], how="left")
    merged["adj_factor"] = merged.groupby("ts_code")["adj_factor"].ffill().bfill()
    for col in ["open", "high", "low", "close", "pre_close"]:
        merged[col] = merged[col].astype(float) * merged["adj_factor"].astype(float)
    merged["symbol"] = merged["ts_code"].map(to_symbol)
    merged["datetime"] = pd.to_datetime(merged["trade_date"], format="%Y%m%d")
    result = merged[
        (merged["trade_date"] >= SAMPLE_START) & (merged["trade_date"] <= SAMPLE_END)
    ][
        [
            "symbol",
            "ts_code",
            "trade_date",
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "vol",
            "amount",
            "adj_factor",
        ]
    ].copy()
    result = result.sort_values(["symbol", "trade_date"])
    pl.from_pandas(result).write_parquet(OUT_DIR / "day_hs300_hfq_20210223_20230529.parquet")
    print(
        f"wrote daily rows={len(result)} symbols={result['symbol'].nunique()} "
        f"adj_rows={len(adj_df)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
