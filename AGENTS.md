# Project Rules

## Defaults

- Default config file is `config.yaml`. Do not fall back to `data/tick_data.parquet` when running from this repo root.
- The default experiment window is:
  - train: `2025-01-01` to `2025-01-31`
  - test: `2025-02-01` to `2025-02-28`
- Primary data path is `/media/busanbusi/新加卷/数据样例/day/2025`.

## Data Pipeline

- Use `polars` as the primary dataframe engine for loading, preprocessing, event extraction, and feature calculation.
- Only convert to `pandas` at the model boundary where `scikit-learn` compatibility requires it.
- Before model training, drop rows containing `NaN`, `null`, or non-finite numeric values.
- Before model training, drop feature columns that are entirely non-finite on the current dataset.

## Feature Rules

- New factors should be added through `extra_feature_registry` first. Do not modify the main experiment flow unless the factor truly requires new shared infrastructure.
- For daily data, do not assume order-book, flow, or turnover fields exist. Feature availability must be inferred from actual columns and finite values.

## Runtime Rules

- Headless environments are normal. Plotting code must use a non-interactive backend and must not rely on `plt.show()`.
- When the user asks to "run it through", validate with the real command outside the sandbox if sandbox execution times out or is environment-blocked.

## Validation

- Preferred regression command:

```bash
MPLCONFIGDIR=/tmp/matplotlib python -m unittest discover -s ./tests -t . -v
```
