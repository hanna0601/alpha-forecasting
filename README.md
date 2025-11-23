# alpha-forecasting

# Dataset Construction Summary 

This document summarizes all steps starting from news processing, leading to the final merged dataset used for the forecasting model.

---

## 1. Daily News Feature Engineering

For each (Date, Ticker) pair, we aggregated all related news articles and generated:

### Sentiment Features
- mean_sentiment
- max_sentiment
- min_sentiment
- sum_sentiment
- news_count

### Embedding Features
- Extracted FinBERT embeddings for each article.
- Aggregated embeddings per ticker per day.
- Applied PCA to reduce embedding dimensionality to 64 components:

pca_emb_0 ... pca_emb_63

News datasets were generated separately for:
- training window
- validation window
- testing window

All news feature frames were standardized to the format:

Date, tic, sentiment features, PCA components

---

## 2. Normalization and Cleaning

### Standardization
- Stock_symbol renamed to tic
- Date converted to YYYY-MM-DD (timezone removed)

### Missing News Handling
Most stock-days have no news. This is expected and is handled by:

- Sentiment features → 0
- news_count → 0
- PCA embeddings → 0

This corresponds to “no new information”.

---

## 3. Merging All Data Sources

We merged four main datasets:

1. Daily market data (Open, High, Low, Close, Volume)
2. Fundamentals (quarterly, converted to daily through forward-fill)
3. Macro variables (daily FRED indicators, forward-filled)
4. News features (sentiment + PCA embeddings)

Merge key:

(Date, tic)

Resulting merged dataset:
- ~963,000 rows
- 109 columns
- Fully aligned across date–ticker pairs

---

## 4. Handling Missing Values

### Filled Completely
The following were filled with 0:
- All sentiment features
- All PCA features
- Macro variables
- sp500 (forward-filled then 0 where needed)
- Market data already complete

### Fundamentals
Some fundamental variables naturally contain missing values because:
- Fundamentals are quarterly
- Some firms do not report all fields

Examples:
- equity_growth
- roe_ttm
- bm
- accruals_ta
- current_ratio
- div_yield

Strategy:
- Filled all remaining gaps with 0
- Interpreted as “no update” or “not reported”
- Works for neural networks and tree-based models

No rows were dropped.

---

## 5. Final Dataset Description

Final dataset includes:
- Daily OHLCV market data
- Forward-filled fundamentals
- Daily macroeconomic indicators
- Daily news sentiment
- PCA news embeddings
- return_next_day target
- todo - Optional momentum features

Final shape:
- 963,223 rows
- 109 columns
- Zero missing values remaining



