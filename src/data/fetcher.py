import logging
from datetime import datetime
from typing import List

import pandas as pd
import yfinance as yf

from src.data.caching import disk_cache


# Cache API calls for 1 hour (3600 seconds)
@disk_cache(ttl_seconds=3600)
def fetch_price_data(
    tickers: List[str], start_date: datetime, end_date: datetime, retries: int = 3
) -> pd.DataFrame:
    logging.info(f"Fetching data for tickers: {tickers} from {start_date} to {end_date}")
    
    for attempt in range(retries):
        try:
            # Force auto_adjust=False to keep 'Adj Close' if you want it
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False
            )

            if data.empty:
                logging.warning("No data returned from yfinance.")
                continue

            # Prefer 'Adj Close', fall back to 'Close' if missing
            if "Adj Close" in data:
                price_data = data["Adj Close"]
            elif "Close" in data:
                price_data = data["Close"]
                logging.warning("'Adj Close' not found, using 'Close' instead.")
            else:
                logging.error("Neither 'Adj Close' nor 'Close' columns found in data.")
                continue

            # Ensure DataFrame even for single ticker
            if isinstance(price_data, pd.Series):
                price_data = price_data.to_frame(name=tickers[0])

            # Drop tickers with all NaNs
            price_data.dropna(axis=1, how="all", inplace=True)
            # Forward-fill and backward-fill missing values
            price_data.ffill(inplace=True)
            price_data.bfill(inplace=True)

            # Drop any remaining columns with NaNs
            if price_data.isnull().values.any():
                logging.warning(
                    f"Some tickers still have NaNs: {price_data.columns[price_data.isnull().any()].tolist()}"
                )
                price_data.dropna(axis=1, inplace=True)

            if not price_data.empty:
                return price_data

        except Exception as e:
            logging.error(f"Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt + 1 == retries:
                logging.error("All attempts to fetch data failed.")
                return pd.DataFrame()

    return pd.DataFrame()
