# tools/data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def download_ohlcv(symbol="AAPL", period="60d", interval="1m"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError("No data fetched")
    df = df.dropna()
    df = df.rename(columns={"Close": "close", "Open": "open", "High":"high","Low":"low","Volume":"volume"})
    df.index = pd.to_datetime(df.index)
    return df[["open","high","low","close","volume"]]

def rolling_ticks_from_candles(df):
    """Yield tick-like prices from a candle frame for demo (simple)."""
    for ts, r in df.iterrows():
        # yield close as tick
        yield {"ts": ts.isoformat(), "price": float(r["close"])}

# small test
if __name__ == "__main__":
    df = download_ohlcv("AAPL", period="7d", interval="5m")
    print(df.tail())
