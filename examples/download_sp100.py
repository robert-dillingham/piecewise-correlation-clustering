"""Download hourly bars for the S&P 100 and save them as a Parquet file.

Needs ALPACA_API_KEY and ALPACA_SECRET_KEY in a local .env file.
Run once; the examples then read the saved file instead of calling the API.
"""

import os
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

URL = "https://en.wikipedia.org/wiki/S%26P_100"
OUT = Path(__file__).parent / "data" / "sp100_hourly.parquet"
START = datetime(2025, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 11, 1, tzinfo=timezone.utc)  # fixed so the file is reproducible
CHUNK = 25  # symbols per request


def get_symbols() -> list[str]:
    print(f"Downloading S&P 100 symbols from {URL}...")
    headers = {"User-Agent": os.environ["USER_AGENT"]}
    resp = requests.get(URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    table = next(t for t in tables if "Symbol" in t.columns)
    return table["Symbol"].tolist()


def main() -> None:
    load_dotenv()
    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    )
    symbols = get_symbols()

    frames = []
    print(f"Downloading hourly bars for {len(symbols)} symbols from {START} to {END}")
    print("This may take several minutes...")
    for i in range(0, len(symbols), CHUNK):
        req = StockBarsRequest(
            symbol_or_symbols=symbols[i : i + CHUNK],
            timeframe=TimeFrame.Hour,
            start=START,
            end=END,
            feed=DataFeed.IEX,
        )
        frames.append(client.get_stock_bars(req).df)

    df = pd.concat(frames).reset_index().sort_values(["symbol", "timestamp"])

    missing = sorted(set(symbols) - set(df["symbol"]))
    if missing:
        print(f"No data returned for: {missing}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"Saved {len(df):,} rows, {df['symbol'].nunique()} symbols -> {OUT}")


if __name__ == "__main__":
    main()
