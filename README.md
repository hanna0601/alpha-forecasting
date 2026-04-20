# Multimodal Alpha Forecasting

This repo contains the code and notebooks for our 15.095 project on next-day stock return prediction and portfolio construction. We build a daily S&P 500 panel from market data, fundamentals, macro variables, and news, then test which signals help forecast the next trading day's return.

The final model output is not used directly as a trading rule. We pass the forecasts into a CVaR portfolio optimizer so that the portfolio is built with downside risk and trading constraints in mind.

## Project Structure

```text
processing/              raw data collection and first-stage cleaning
feature_engineering/     feature construction and dataset merge
model/                   baselines, XGBoost, LightGBM, CatBoost, LSTM experiments
optimization/            CVaR portfolio optimization and backtesting
data/                    local processed data files
results/                 saved models and SHAP plots
documents/               report, figures, and slides
achive/                  older news-processing notebooks
```

Most of the work is in notebooks because the project was exploratory. The cleanest path through the repo is:

1. `processing/`
2. `feature_engineering/`
3. `model/`
4. `optimization/`

## Data

The universe is S&P 500 stocks. The modeling panel is daily and keyed by:

```text
Date, tic
```

The main data sources are:

- daily OHLCV prices from Yahoo Finance
- company fundamentals from WRDS/Compustat
- macro indicators from FRED
- S&P 500 company and sector information
- financial news headlines and summaries from the FNSPID dataset

The final split is chronological:

```text
train: 2016-2021
validation: 2022
test: 2023
```

This split matters because the task is time-series forecasting. I did not want random train/test splits leaking future market regimes into training.

## Pipeline

### 1. Market Data

`processing/market.py` and `processing/market.ipynb` download and clean daily prices. Tickers such as `BRK.B` are converted to Yahoo's format, like `BRK-B`. The output is a tidy daily price panel with:

```text
Date, Ticker, Open, High, Low, Close, Volume
```

The market features later include returns, rolling returns, moving averages, volatility, volume features, RSI, and other price-based signals.

### 2. Fundamentals

The fundamentals pipeline starts from quarterly accounting data and converts it into daily features by forward-filling within each firm. This avoids using future filings for earlier dates.

The feature set includes variables such as:

- profitability: ROA, ROE, margins
- valuation: book-to-market, earnings yield, cash-flow yield
- balance sheet: leverage, liquidity, cash holdings
- growth and accrual measures

The processed output is saved as:

```text
data/model/fundamental_features.csv
```

### 3. Macro Features

`processing/macro.ipynb` pulls macro variables from FRED and aligns them to the daily panel. Missing values are forward-filled because many macro series are not released daily.

The processed macro file is:

```text
data/model/macro_features.parquet
```

### 4. News Features

News processing happens in two steps.

First, `processing/news_headlines.ipynb` filters the raw news data to S&P 500 companies and splits it into train, validation, and test windows.

Then, `feature_engineering/news_finbert.ipynb` uses FinBERT to create:

- sentiment labels and scores
- daily sentiment aggregates by ticker
- FinBERT text embeddings
- PCA-compressed embedding features

For each stock-day, the news features include:

```text
mean_sentiment
max_sentiment
min_sentiment
sum_sentiment
news_count
pca_emb_0 ... pca_emb_63
```

Most stock-days have no news. In those cases, sentiment, news count, and embedding features are filled with zero. This is treated as "no new information" rather than dropping the row.

### 5. Final Merge

`feature_engineering/combine.ipynb` merges the market, fundamental, macro, and news features into the final modeling files:

```text
data/model/final_train.parquet
data/model/final_val.parquet
data/model/final_test.parquet
```

The final dataset has about 963k rows and 109 columns. The target is:

```text
return_next_day
```

No rows are dropped at the final merge stage. Remaining missing values are filled with zero after alignment.

## Modeling

The modeling notebooks compare simple baselines against machine learning models. The main metric is directional accuracy, because for this project the first question is whether the model gets the sign of the next-day return right.

The baselines include:

- yesterday's return
- moving averages
- exponential moving averages
- month and day-of-week rules
- ticker/month/day-specific hybrid rules

The machine learning experiments include:

- linear regression
- XGBoost
- LightGBM
- CatBoost
- neural networks with ticker embeddings
- LSTM experiments

The strongest and most stable base signal was not a complicated price feature set. It was the simple DMS structure:

```text
day-of-week + month-of-year + sector
```

After that, I tested whether fundamentals, macro variables, news sentiment, and text embeddings improved the DMS model.

## Main Results

Daily stock return prediction is noisy, so the numbers are close to random guessing. Still, several models consistently stayed above 50% directional accuracy.

Some important results:

- DMS + categorical ticker with CatBoost reached about 53.5% test directional accuracy.
- LightGBM was the strongest model class overall.
- DMS + selected fundamentals had the best validation directional accuracy, around 52.2%.
- DMS + mean news sentiment had the best test directional accuracy, around 54.0%.
- FinBERT PCA embeddings did not improve results. They usually added noise at this horizon.

The main takeaway is that simple structure mattered more than high-dimensional text features. Calendar effects, sector information, selected fundamentals, and aggregate news sentiment were more useful than embedding-heavy models.

## Portfolio Optimization

The forecasting models are also tested inside a portfolio construction problem. The optimizer uses CVaR instead of only variance, because I wanted the portfolio to care about tail losses directly.

The optimization notebooks use:

- predicted next-day returns from the LightGBM models
- realized one-day-ahead returns
- rolling historical return scenarios
- long-only and position-size constraints
- daily rebalancing
- transaction cost assumptions

In the CVaR backtest, the strongest feature sets were:

| Feature set | Sharpe | Annualized return |
| --- | ---: | ---: |
| DMS + Fundamentals | 1.166 | 10.84% |
| DMS + Mean Sentiment | 1.142 | 10.58% |
| DMS + Mean Sentiment + Embeddings | 0.989 | 9.23% |
| DMS + News Momentum | 0.895 | 8.32% |
| DMS + News Momentum + Embeddings | 0.654 | 6.07% |

The portfolio results tell the same story as the prediction results: simpler signals were more stable. The embedding-based models looked more flexible, but they did not translate into better out-of-sample portfolios.

## Libraries

The main Python packages used are:

```text
pandas
numpy
scikit-learn
yfinance
fredapi
transformers
torch
xgboost
lightgbm
catboost
cvxpy
matplotlib
shap
```

Some data files are large and are treated as local project artifacts. If the processed files already exist under `data/model/`, the modeling and optimization notebooks can be run without rebuilding the raw data pipeline.

## Notes

- The project is focused on next-day returns, which are very hard to predict. Small gains over 50% directional accuracy are already meaningful in this setting.
- The notebooks include some experiments that were not part of the final model choice. I kept them because they show how we got to the final feature sets.
- `achive/` contains older news-processing work and is not the main path anymore.
- The final report is in `documents/report/report.pdf`.
