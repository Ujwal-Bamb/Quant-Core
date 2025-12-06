# models/backtest.py
import pandas as pd
import numpy as np
import joblib

def generate_signals(df, model_path="models/lgb_model.pkl"):
    model = joblib.load(model_path)
    from models.train import make_features
    X = make_features(df)
    probs = model.predict(X)
    # simple threshold strategy
    sig = pd.Series(index=X.index, data=0)
    sig[probs > 0.6] = 1
    sig[probs < 0.4] = -1
    return sig, probs

def backtest(df, sig):
    # assume 1 unit position, enter next bar's open = close of prev
    df = df.copy()
    df['signal'] = sig
    df['ret'] = df['close'].pct_change().shift(-1)  # simplified
    df['strategy_ret'] = df['signal'] * df['ret']
    df['cum'] = (1 + df['strategy_ret'].fillna(0)).cumprod()
    total_return = df['cum'].iloc[-2] if len(df)>2 else 1.0
    daily_returns = df['strategy_ret'].resample('1D').sum()
    sharpe = (daily_returns.mean() / (daily_returns.std()+1e-9)) * (252**0.5)
    return {"total_return": total_return, "sharpe": sharpe, "equity": df['cum']}

if __name__ == "__main__":
    from tools.data_loader import download_ohlcv
    df = download_ohlcv("AAPL", period="60d", interval="5m")
    sig, probs = generate_signals(df)
    res = backtest(df, sig)
    print("Backtest:", res["total_return"], "Sharpe:", res["sharpe"])
