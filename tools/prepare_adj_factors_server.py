from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import polars as pl
import tushare as ts


TOKEN = os.environ.get("TUSHARE_TOKEN") or "d83605429b7463a375ed3b36271d1b2ff7b7bd937920eb8a30fcd690"
OUT_DIR = Path("/mnt/nvme_raid0/experiment_data/adj_factor")
CHUNK_DIR = OUT_DIR / "chunks"
OUT_FILE = OUT_DIR / "adj_factor_all_a_20210222_20260417.parquet"
FAILED_FILE = OUT_DIR / "adj_factor_failed_symbols.txt"
START_DATE = "20210222"
END_DATE = "20260417"


def fetch_with_retry(func, **kwargs) -> pd.DataFrame:
    for attempt in range(6):
        try:
            return func(**kwargs)
        except Exception as exc:
            wait = 2 + attempt * 2
            print(f"retry {attempt + 1} {kwargs}: {exc}; sleep {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed {kwargs}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    pro = ts.pro_api(TOKEN)

    stock_basic = fetch_with_retry(pro.stock_basic)
    universe = sorted(
        stock_basic[
            stock_basic["ts_code"].astype(str).str.endswith((".SH", ".SZ"))
        ]["ts_code"].dropna().unique().tolist()
    )
    print(f"universe={len(universe)}", flush=True)

    done = {path.stem.replace("adj_", "") for path in CHUNK_DIR.glob("adj_*.parquet")}
    failed = set()
    if FAILED_FILE.exists():
        failed = {line.strip() for line in FAILED_FILE.read_text().splitlines() if line.strip()}

    for idx, ts_code in enumerate(universe, start=1):
        if ts_code in done:
            continue
        if ts_code in failed:
            print(f"retry previously failed {ts_code}", flush=True)
        print(f"adj {idx}/{len(universe)} {ts_code}", flush=True)
        try:
            df = fetch_with_retry(
                pro.adj_factor,
                ts_code=ts_code,
                start_date=START_DATE,
                end_date=END_DATE,
            )
        except Exception as exc:
            print(f"skip {ts_code}: {exc}", flush=True)
            with FAILED_FILE.open("a", encoding="utf-8") as f:
                f.write(f"{ts_code}\n")
            continue
        if df.empty:
            df = pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
        df.to_parquet(CHUNK_DIR / f"adj_{ts_code}.parquet", index=False)
        failed.discard(ts_code)
        time.sleep(0.12)

    frames = [pl.read_parquet(str(path)) for path in sorted(CHUNK_DIR.glob("adj_*.parquet"))]
    if not frames:
        raise RuntimeError("no adj chunks generated")
    result = (
        pl.concat(frames, how="diagonal_relaxed")
        .unique(["ts_code", "trade_date"], keep="last")
        .sort(["ts_code", "trade_date"])
    )
    result.write_parquet(str(OUT_FILE), compression="zstd")
    print(f"wrote {OUT_FILE} rows={result.height}", flush=True)


if __name__ == "__main__":
    main()
