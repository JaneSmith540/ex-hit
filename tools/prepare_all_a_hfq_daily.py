from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import polars as pl
import tushare as ts


TOKEN = os.environ["TUSHARE_TOKEN"]
OUT_DIR = Path("D:/experiment/generated_hfq_all_a")
ADJ_CHUNK_DIR = OUT_DIR / "adj_chunks"

DAILY_START = "20210222"
SAMPLE_START = "20210223"
SAMPLE_END = "20230529"
DAILY_END = "20230530"
FACTOR_END = "20251231"

DAILY_RAW_PATH = OUT_DIR / "daily_raw_all_a_20210222_20230530.parquet"
ADJ_PATH = OUT_DIR / "adj_factor_all_a_20210222_20251231.parquet"
DAY_HFQ_PATH = OUT_DIR / "day_all_a_hfq_20210223_20230529.parquet"
UNIVERSE_PATH = OUT_DIR / "all_a_universe_20210223_20230529.parquet"


def ymd_range(start: str, end: str) -> list[str]:
    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")
    dates: list[str] = []
    cur = start_dt
    while cur <= end_dt:
        dates.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return dates


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
    for attempt in range(6):
        try:
            return func(**kwargs)
        except Exception as exc:
            wait = 2 + attempt * 2
            print(f"retry {attempt + 1} {func.__name__} {kwargs}: {exc}; sleep {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed {func.__name__} {kwargs}")


def fetch_daily(pro) -> pd.DataFrame:
    if DAILY_RAW_PATH.exists():
        print(f"reuse {DAILY_RAW_PATH}", flush=True)
        return pd.read_parquet(DAILY_RAW_PATH)

    frames: list[pd.DataFrame] = []
    done_dates = set()
    for chunk_file in sorted(OUT_DIR.glob("daily_raw_*.parquet")):
        part = pd.read_parquet(chunk_file)
        if not part.empty:
            frames.append(part)
            done_dates.update(part["trade_date"].astype(str).unique().tolist())

    for idx, trade_date in enumerate(ymd_range(DAILY_START, DAILY_END), start=1):
        if trade_date in done_dates:
            continue
        df = fetch_with_retry(pro.daily, trade_date=trade_date)
        if not df.empty:
            df = df[df["ts_code"].str.endswith((".SH", ".SZ"))].copy()
            frames.append(df)
            done_dates.add(trade_date)
            print(f"daily {idx} {trade_date} rows={len(df)}", flush=True)
        else:
            print(f"daily {idx} {trade_date} empty", flush=True)
        if len(frames) and len(frames) % 50 == 0:
            tmp = pd.concat(frames, ignore_index=True)
            tmp.to_parquet(OUT_DIR / f"daily_raw_checkpoint_{len(done_dates):04d}.parquet", index=False)
        time.sleep(0.08)

    if not frames:
        raise RuntimeError("No daily data fetched")
    daily = pd.concat(frames, ignore_index=True)
    daily = daily.drop_duplicates(["ts_code", "trade_date"])
    daily.to_parquet(DAILY_RAW_PATH, index=False)
    print(f"wrote daily raw rows={len(daily)} symbols={daily['ts_code'].nunique()}", flush=True)
    return daily


def fetch_adj_factors(pro, symbols: list[str]) -> pd.DataFrame:
    if ADJ_PATH.exists():
        print(f"reuse {ADJ_PATH}", flush=True)
        return pd.read_parquet(ADJ_PATH)

    ADJ_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    done = {path.stem.replace("adj_", "") for path in ADJ_CHUNK_DIR.glob("adj_*.parquet")}

    for idx, ts_code in enumerate(symbols, start=1):
        if ts_code in done:
            continue
        print(f"adj {idx}/{len(symbols)} {ts_code}", flush=True)
        df = fetch_with_retry(
            pro.adj_factor,
            ts_code=ts_code,
            start_date=DAILY_START,
            end_date=FACTOR_END,
        )
        if not df.empty:
            df.to_parquet(ADJ_CHUNK_DIR / f"adj_{ts_code}.parquet", index=False)
        else:
            pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"]).to_parquet(
                ADJ_CHUNK_DIR / f"adj_{ts_code}.parquet",
                index=False,
            )
        time.sleep(0.12)

    frames = [pd.read_parquet(path) for path in sorted(ADJ_CHUNK_DIR.glob("adj_*.parquet"))]
    adj = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if adj.empty:
        raise RuntimeError("No adj_factor data fetched")
    adj = adj.drop_duplicates(["ts_code", "trade_date"])
    adj.to_parquet(ADJ_PATH, index=False)
    print(f"wrote adj rows={len(adj)} symbols={adj['ts_code'].nunique()}", flush=True)
    return adj


def build_hfq_daily(daily: pd.DataFrame, adj: pd.DataFrame) -> None:
    universe = sorted(daily["ts_code"].dropna().unique().tolist())
    pd.DataFrame({"ts_code": universe, "symbol": [to_symbol(code) for code in universe]}).to_parquet(
        UNIVERSE_PATH,
        index=False,
    )

    merged = daily.merge(adj, on=["ts_code", "trade_date"], how="left")
    merged = merged.sort_values(["ts_code", "trade_date"])
    merged["adj_factor"] = merged.groupby("ts_code")["adj_factor"].ffill().bfill()
    missing = merged["adj_factor"].isna().sum()
    if missing:
        print(f"warning missing adj_factor rows={missing}; dropping", flush=True)
        merged = merged.dropna(subset=["adj_factor"])

    for col in ["open", "high", "low", "close", "pre_close"]:
        merged[f"{col}_raw"] = merged[col].astype(float)
        merged[f"{col}_hfq"] = merged[col].astype(float) * merged["adj_factor"].astype(float)
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
            "open_hfq",
            "high_hfq",
            "low_hfq",
            "close_hfq",
            "pre_close_hfq",
            "vol",
            "amount",
            "adj_factor",
        ]
    ].copy()
    result = result.sort_values(["symbol", "trade_date"])
    pl.from_pandas(result).write_parquet(DAY_HFQ_PATH)
    print(
        f"wrote {DAY_HFQ_PATH} rows={len(result)} symbols={result['symbol'].nunique()}",
        flush=True,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts.set_token(TOKEN)
    pro = ts.pro_api(TOKEN)
    daily = fetch_daily(pro)
    symbols = sorted(daily["ts_code"].dropna().unique().tolist())
    adj = fetch_adj_factors(pro, symbols)
    build_hfq_daily(daily, adj)


if __name__ == "__main__":
    main()
