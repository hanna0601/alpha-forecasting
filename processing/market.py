import yfinance as yf
import pandas as pd
import time
from tqdm import tqdm

# -----------------------------------------
# 1. Fix tickers like BRK.B → BRK-B
# -----------------------------------------
def fix_ticker(t):
    return t.replace(".", "-")

sp500 = pd.read_csv("data/sp500_companies.csv")
tickers = sorted(sp500['ticker'].unique())
tickers_fixed = [fix_ticker(t) for t in tickers]


# -----------------------------------------
# 2. Cache IPO years
# -----------------------------------------
ipo_year_cache = {}

def get_ipo_year(ticker):
    """Find first trading year for each ticker via Yahoo."""
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

    except Exception:
        ipo_year_cache[ticker] = None
        return None


# -----------------------------------------
# 3. Clean tidy single-ticker downloader
# -----------------------------------------
def fetch_ticker_year(ticker, year, max_retries=6):
    """
    Clean tidy download: always returns
    Date | Ticker | Open | High | Low | Close | Volume
    """
    start = f"{year}-01-01"
    end   = f"{year}-12-31"

    delay = 3

    for attempt in range(max_retries):
        try:
            # IMPORTANT: use .history() NOT yf.download()
            df = yf.Ticker(ticker).history(
                start=start,
                end=end,
                auto_adjust=True
            )

            if df is None or df.empty:
                return None

            # Force tidy format
            df = df[["Open","High","Low","Close","Volume"]].copy()
            df["Ticker"] = ticker
            df["Date"] = df.index

            df = df.reset_index(drop=True)
            df = df[["Date","Ticker","Open","High","Low","Close","Volume"]]

            return df

        except Exception as e:
            print(f"[{ticker} {year}] Error: {e} | retrying in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, 45)

    print(f"[{ticker} {year}] FAILED after {max_retries} retries.")
    return None


# -----------------------------------------
# 4. Download all tickers for one year
# -----------------------------------------
def download_one_year(year, tickers):
    frames = []

    for ticker in tqdm(tickers, desc=f"Year {year}"):

        # IPO check
        ipo_year = get_ipo_year(ticker)
        if ipo_year is None:
            print(f"[{ticker}] No Yahoo history → skip.")
            continue
        if year < ipo_year:
            print(f"[{ticker}] IPO {ipo_year} → skip {year}.")
            continue

        df = fetch_ticker_year(ticker, year)

        if df is not None:
            frames.append(df)

        time.sleep(0.8)  # prevent rate limits

    if len(frames) == 0:
        print(f"Year {year}: nothing downloaded.")
        return None

    return pd.concat(frames, ignore_index=True)


# -----------------------------------------
# 5. RUN ALL YEARS — SAVE CLEAN PARQUETS
# -----------------------------------------
YEARS = list(range(2016, 2024))  # 2016 → 2023

for year in YEARS:
    df_year = download_one_year(year, tickers_fixed)
    if df_year is not None:
        df_year.to_parquet(f"data/market/prices_{year}.parquet", index=False)
        print(f"Saved year {year}")
    else:
        print(f"NO DATA for {year}")
