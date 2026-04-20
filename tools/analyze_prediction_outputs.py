from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _portfolio_rows(df: pd.DataFrame, model: str, top_frac: float) -> dict:
    daily = []
    for trade_date, group in df.groupby("trade_date", sort=True):
        group = group.sort_values("y_pred", ascending=False)
        n_select = max(1, int(len(group) * top_frac))
        top = group.head(n_select)
        bottom = group.tail(n_select)
        daily.append(
            {
                "trade_date": trade_date,
                "n": len(group),
                "top_mean": top["y_true"].mean(),
                "bottom_mean": bottom["y_true"].mean(),
                "spread": top["y_true"].mean() - bottom["y_true"].mean(),
                "top_win_rate": (top["y_true"] > 0).mean(),
                "all_mean": group["y_true"].mean(),
                "all_win_rate": (group["y_true"] > 0).mean(),
            }
        )
    daily_df = pd.DataFrame(daily)
    return {
        "model": model,
        "days": len(daily_df),
        "test_events": len(df),
        "all_mean": df["y_true"].mean(),
        "all_win_rate": (df["y_true"] > 0).mean(),
        "top_daily_mean": daily_df["top_mean"].mean(),
        "top_win_rate": daily_df["top_win_rate"].mean(),
        "bottom_daily_mean": daily_df["bottom_mean"].mean(),
        "spread_daily_mean": daily_df["spread"].mean(),
        "spread_positive_days": (daily_df["spread"] > 0).mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved model prediction files.")
    parser.add_argument("--metrics-dir", required=True, help="Directory containing *_predictions.csv files.")
    parser.add_argument("--run-id", required=True, help="Run id prefix used in prediction file names.")
    parser.add_argument("--models", nargs="+", required=True, help="Model names to analyze.")
    parser.add_argument("--top-frac", type=float, default=0.2, help="Top/bottom fraction for daily portfolio checks.")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    portfolio_rows = []
    monthly_rows = []

    for model in args.models:
        path = metrics_dir / f"{args.run_id}_{model}_predictions.csv"
        df = pd.read_csv(path)
        df["trade_date"] = df["trade_date"].astype(str)
        df["month"] = df["trade_date"].str[:6]
        portfolio_rows.append(_portfolio_rows(df, model, args.top_frac))

        for month, group in df.groupby("month", sort=True):
            if len(group) < 20:
                continue
            sorted_group = group.sort_values("y_pred", ascending=False)
            n_select = max(1, int(len(sorted_group) * args.top_frac))
            monthly_rows.append(
                {
                    "model": model,
                    "month": month,
                    "n": len(group),
                    "mean_y": group["y_true"].mean(),
                    "ic": group["y_true"].corr(group["y_pred"], method="pearson"),
                    "rank_ic": group["y_true"].corr(group["y_pred"], method="spearman"),
                    "top_spread": (
                        sorted_group.head(n_select)["y_true"].mean()
                        - sorted_group.tail(n_select)["y_true"].mean()
                    ),
                    "top_mean": sorted_group.head(n_select)["y_true"].mean(),
                }
            )

    portfolio_df = pd.DataFrame(portfolio_rows).sort_values("top_daily_mean", ascending=False)
    monthly_df = pd.DataFrame(monthly_rows).sort_values("rank_ic", ascending=False)

    portfolio_path = metrics_dir / f"{args.run_id}_portfolio_check.csv"
    monthly_path = metrics_dir / f"{args.run_id}_monthly_heterogeneity.csv"
    portfolio_df.to_csv(portfolio_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)

    print(f"Wrote {portfolio_path}")
    print(f"Wrote {monthly_path}")
    print(portfolio_df.to_csv(index=False))


if __name__ == "__main__":
    main()
