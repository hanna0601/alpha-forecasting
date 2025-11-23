import yfinance as yf
import pandas as pd
import time
from tqdm import tqdm

# ------------------------------------------------------------
# 1. Convert BRK.B → BRK-B etc.
# ------------------------------------------------------------
def fix_ticker(t):
    return t.replace(".", "-")

# Load your ticker list (YOU MUST UPDATE THIS LINE)
# Example:
# tickers = ["AAPL", "MSFT", "GOOGL", ...]
tickers = pd.read_csv("data/sp500_companies.csv")["ticker"].tolist()
tickers_fixed = [fix_ticker(t) for t in tickers]


# ------------------------------------------------------------
# 2. Cache IPO years to avoid repeated Yahoo lookups
# ------------------------------------------------------------
ipo_year_cache = {}

def get_ipo_year(ticker):
    """Return earliest available year for this ticker from Yahoo."""
    if ticker in ipo_year_cache:
        return ipo_year_cache[ticker]

    try:
        df = yf.Ticker(ticker).history(period="max")
        if df.empty:
            ipo_year_cache[ticker] = None
            return None
        year = df.index.min().year
        ipo_year_cache[ticker] = year
        return year
    except:
        ipo_year_cache[ticker] = None
        return None


# ------------------------------------------------------------
# 3. Safe download (no multi-index, auto-adjust ON)
# ------------------------------------------------------------
def fetch_ticker_year(ticker, year, max_retries=6):
    """Download one ticker for one year, robust against rate limits."""
    start = f"{year}-01-01"
    end   = f"{year}-12-31"

    delay = 3

    for attempt in range(max_retries):
        try:
            df = yf.Ticker(ticker).history(
                start=start,
                end=end,
                auto_adjust=True
            )

            if df is None or df.empty:
                return None

            # Standardize tidy format
            df["Ticker"] = ticker
            df["Date"] = df.index
            df = df.reset_index(drop=True)
            df = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]

            return df

        except Exception as e:
            print(f"[{ticker} {year}] Error: {e} | retrying in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, 45)

    print(f"[{ticker} {year}] FAILED after retries.")
    return None


# ------------------------------------------------------------
# 4. Download all tickers for a single year
# ------------------------------------------------------------
def download_one_year(year, tickers):
    all_dfs = []

    for ticker in tqdm(tickers, desc=f"Year {year}"):
        ipo_year = get_ipo_year(ticker)

        if ipo_year is None:
            print(f"[{ticker}] No Yahoo history → skipping.")
            continue

        if year < ipo_year:
            print(f"[{ticker}] IPO={ipo_year} → skipping {year}.")
            continue

        df = fetch_ticker_year(ticker, year)

        if df is not None:
            all_dfs.append(df)

        # Slow down slightly to help avoid rate limits
        time.sleep(0.5)

    if len(all_dfs) == 0:
        print(f"Year {year} → no data.")
        return None

    return pd.concat(all_dfs, ignore_index=True)


# ------------------------------------------------------------
# 5. Full modeling window
# ------------------------------------------------------------
YEARS = list(range(2014, 2024))  # 2014–2023

for year in YEARS:
    print(f"\n=== Processing {year} ===")

    df_year = download_one_year(year, tickers_fixed)

    if df_year is not None:
        output_path = f"data/market/prices_{year}.parquet"
        df_year.to_parquet(output_path, index=False)
        print(f"Saved: {output_path} ({len(df_year)} rows)")
    else:
        print(f"No data for year {year}")
