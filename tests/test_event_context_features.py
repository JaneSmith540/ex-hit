import unittest
from datetime import datetime
from types import SimpleNamespace

import polars as pl

from data_processor import DataProcessor
from feature_engineer import FeatureEngineer


class EventContextFeatureTests(unittest.TestCase):
    def test_daily_context_adds_board_position_and_market_heat(self):
        processor = DataProcessor(SimpleNamespace(tushare_token=None))
        day_df = pl.DataFrame(
            {
                "symbol": [
                    "600001.SH",
                    "600001.SH",
                    "600001.SH",
                    "000001.SZ",
                    "000001.SZ",
                    "000001.SZ",
                ],
                "trade_date": [
                    "20210701",
                    "20210702",
                    "20210705",
                    "20210701",
                    "20210702",
                    "20210705",
                ],
                "datetime": [
                    datetime(2021, 7, 1),
                    datetime(2021, 7, 2),
                    datetime(2021, 7, 5),
                    datetime(2021, 7, 1),
                    datetime(2021, 7, 2),
                    datetime(2021, 7, 5),
                ],
                "pre_close": [10.0, 11.0, 12.1, 10.0, 10.2, 10.0],
                "open": [10.0, 11.0, 12.1, 10.0, 10.1, 10.0],
                "high": [11.0, 12.1, 13.31, 10.3, 10.2, 10.25],
                "close": [11.0, 12.1, 12.4, 10.2, 10.0, 10.2],
            }
        )

        result = processor._add_daily_event_context(day_df).sort(["symbol", "trade_date"])
        sh_rows = result.filter(pl.col("symbol") == "600001.SH").sort("trade_date")

        self.assertEqual(sh_rows.get_column("prior_limit_up_streak").to_list(), [0, 1, 2])
        self.assertEqual(sh_rows.get_column("board_position").to_list(), [1, 2, 3])

        day_one = result.filter(pl.col("trade_date") == "20210701").row(0, named=True)
        self.assertEqual(day_one["market_total_count"], 2)
        self.assertEqual(day_one["market_limit_up_count"], 1)
        self.assertEqual(day_one["market_close_limit_up_count"], 1)
        self.assertEqual(day_one["market_up_count"], 2)
        self.assertAlmostEqual(day_one["market_limit_up_ratio"], 0.5)
        self.assertAlmostEqual(day_one["market_up_ratio"], 1.0)
        self.assertEqual(day_one["prev_market_max_close_limit_streak"], 0)

        day_two = result.filter(pl.col("trade_date") == "20210702").row(0, named=True)
        self.assertEqual(day_two["prev_market_max_close_limit_streak"], 1)

        day_three = result.filter(pl.col("trade_date") == "20210705").row(0, named=True)
        self.assertEqual(day_three["prev_market_max_close_limit_streak"], 2)

    def test_event_features_include_time_bucket_and_daily_context(self):
        config = SimpleNamespace(
            feature_params={"ma_windows": [5], "rsi_period": 14},
            price_features=[],
            volume_features=[],
            orderbook_features=[],
            flow_features=[],
            technical_features=[],
            event_features=[],
        )
        engineer = FeatureEngineer(config)

        features = engineer.calculate_event_features(
            tick_window=pl.DataFrame(),
            minute_window=pl.DataFrame(),
            l2_window=pl.DataFrame(),
            event_snapshot={
                "datetime": datetime(2021, 7, 1, 10, 15),
                "b1_v": 100.0,
                "a1_v": 50.0,
            },
            limit_price=11.0,
            day_context={
                "prior_limit_up_streak": 2,
                "board_position": 3,
                "market_limit_up_count": 48,
                "market_limit_up_ratio": 0.03,
                "market_up_count": 2800,
                "market_up_ratio": 0.62,
                "market_total_count": 4500,
                "prev_market_max_close_limit_streak": 4,
            },
        )

        self.assertEqual(features["touch_time_bucket"], 1.0)
        self.assertEqual(features["prior_limit_up_streak"], 2.0)
        self.assertEqual(features["board_position"], 3.0)
        self.assertEqual(features["market_limit_up_count"], 48.0)
        self.assertAlmostEqual(features["market_up_ratio"], 0.62)
        self.assertEqual(features["market_total_count"], 4500.0)
        self.assertEqual(features["prev_market_max_close_limit_streak"], 4.0)

    def test_intraday_context_counts_only_prior_touch_events(self):
        processor = DataProcessor(SimpleNamespace(tushare_token=None))
        event_df = pl.DataFrame(
            {
                "symbol": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
                "trade_date": ["20210701", "20210701", "20210701", "20210702"],
                "event_time": [
                    datetime(2021, 7, 1, 9, 45),
                    datetime(2021, 7, 1, 9, 45),
                    datetime(2021, 7, 1, 10, 5),
                    datetime(2021, 7, 2, 9, 40),
                ],
                "market_total_count": [100, 100, 100, 80],
            }
        )

        result = processor._add_intraday_event_context(event_df).sort(["trade_date", "event_time", "symbol"])

        self.assertEqual(result.get_column("market_prior_touch_count").to_list(), [0.0, 0.0, 2.0, 0.0])
        self.assertEqual(result.get_column("market_prior_touch_ratio").to_list(), [0.0, 0.0, 0.02, 0.0])


if __name__ == "__main__":
    unittest.main()
