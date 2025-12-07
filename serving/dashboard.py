import os
import sys
from datetime import datetime

# -------------------------
# Make project importable
# -------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Our helper modules
from scanners.stock_scanner import (
    scan_universe,
    download_candles,
    compute_features,
    NIFTY50_SAMPLE,
)
from options.option_utils import synthetic_chain, suggest_strike
from signals.entry_rules import entry_stop_target

# ML imports
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

from sklearn.ensemble import RandomForestClassifier

# ==========================================
# BASIC PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Quant-Core Indian Trading AI", layout="wide")
st.title("🚀 Quant-Core Indian Trading AI (MVP)")

st.markdown(
    "Paper-trading only. This is **not** investment advice. "
    "Use for education, testing and strategy research."
)

# ==========================================
# HELPERS
# ==========================================

@st.cache_data(ttl=60)
def load_ohlcv(symbol: str, period: str = "30d", interval: str = "5m") -> pd.DataFrame:
    """Load OHLCV candles via yfinance."""
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        return df
    df = df.rename(columns=str.lower)
    return df[["open", "high", "low", "close", "volume"]]


@st.cache_resource(show_spinner=False)
def train_price_model(symbol: str, period: str = "30d", interval: str = "5m", horizon: int = 5):
    """Train a simple classification model: will price be higher after N bars?"""
    df = load_ohlcv(symbol, period=period, interval=interval)
    if df.empty:
        raise RuntimeError("No data to train for symbol: " + symbol)

    feats = compute_features(df)

    future_close = df["close"].shift(-horizon)
    ret = (future_close / df["close"] - 1.0)
    y = (ret > 0).astype(int)

    df_train = pd.concat([feats, y.rename("y")], axis=1).dropna()
    if df_train.empty:
        raise RuntimeError("Not enough data after feature engineering.")

    X = df_train.drop(columns=["y"])
    y = df_train["y"]

    if HAS_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X, y)

    meta = {
        "symbol": symbol,
        "period": period,
        "interval": interval,
        "horizon": horizon,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "feature_cols": list(X.columns),
    }

    return model, meta


def predict_latest_prob(model, meta, df_raw: pd.DataFrame):
    feats = compute_features(df_raw)
    X = feats[meta["feature_cols"]].iloc[[-1]]
    proba = model.predict_proba(X)[0, 1]
    return float(proba)


def backtest_threshold(model, meta, df_raw: pd.DataFrame, prob_threshold: float = 0.55):
    feats = compute_features(df_raw)
    X = feats[meta["feature_cols"]].dropna()
    close = df_raw["close"].loc[X.index]

    probs = model.predict_proba(X)[:, 1]
    signal = (probs > prob_threshold).astype(int)
    ret = close.pct_change().fillna(0)

    strat_ret = signal * ret
    equity = (1 + strat_ret).cumprod()

    total_return = equity.iloc[-1] - 1
    sharpe = np.nan
    if strat_ret.std() > 0:
        sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252 * 78)

    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else None,
        "trades": int(signal.diff().abs().sum()),
        "equity": equity,
    }


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("Global Settings")

default_symbol = st.sidebar.text_input("Symbol", value="RELIANCE.NS")
intraday_interval = st.sidebar.selectbox("Intraday interval", ["5m", "15m", "30m"], index=0)
intraday_period = st.sidebar.selectbox("History period", ["5d", "30d", "60d"], index=1)

tab_overview, tab_options, tab_ai, tab_backtest = st.tabs(
    ["📊 Market Overview", "📈 Options", "🤖 AI Prediction", "📉 Backtest"]
)

# ==========================================
# TAB 1 - MARKET OVERVIEW
# ==========================================
with tab_overview:
    st.subheader("Intraday Scanner")

    if st.button("Run Scanner"):
        st.session_state["scan_df"] = scan_universe(NIFTY50_SAMPLE)

    if "scan_df" in st.session_state:
        st.dataframe(st.session_state["scan_df"])

    df_spot = load_ohlcv(default_symbol, intraday_period, intraday_interval)
    if not df_spot.empty:
        st.line_chart(df_spot["close"])


# ==========================================
# TAB 2 - OPTIONS
# ==========================================
with tab_options:
    price = st.number_input("Underlying Price", value=20000.0)

    if st.button("Generate Option Chain"):
        chain, px = synthetic_chain(price)
        st.session_state["opt_chain"] = chain

    if "opt_chain" in st.session_state:
        st.dataframe(st.session_state["opt_chain"])

        direction = st.selectbox("Direction", ["bullish", "bearish"])
        strike = suggest_strike(st.session_state["opt_chain"], direction)
        levels = entry_stop_target(price)

        st.write("Best Strike:", strike)
        st.json(levels)


# ==========================================
# TAB 3 - AI PREDICTION
# ==========================================
with tab_ai:
    horizon = st.slider("Prediction Horizon", 3, 30, 10)

    if st.button("Train Model"):
        model, meta = train_price_model(default_symbol, intraday_period, intraday_interval, horizon)
        st.session_state["ai_model"] = model
        st.session_state["ai_meta"] = meta
        st.success("Model trained")

    if "ai_model" in st.session_state:
        df_latest = load_ohlcv(default_symbol, intraday_period, intraday_interval)
        prob = predict_latest_prob(st.session_state["ai_model"], st.session_state["ai_meta"], df_latest)
        st.metric("Probability Price Goes Up", f"{prob:.2%}")


# ==========================================
# TAB 4 - BACKTEST
# ==========================================
with tab_backtest:
    if "ai_model" in st.session_state:
        df_bt = load_ohlcv(default_symbol, intraday_period, intraday_interval)
        res = backtest_threshold(st.session_state["ai_model"], st.session_state["ai_meta"], df_bt)

        st.metric("Total Return", f"{res['total_return']*100:.2f}%")
        st.metric("Sharpe", "N/A" if res["sharpe"] is None else f"{res['sharpe']:.2f}")
        st.line_chart(res["equity"])


st.caption("Quant-Core – Education Use Only")
